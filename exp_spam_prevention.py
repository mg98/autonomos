import random
import torch
import numpy as np
from swarmsearch.dart.utils import split_by_qids, split_dataset_by_qids, compute_feature_stats
from allrank.config import Config
from swarmsearch.dart.rank import evaluate, shapley_valuation
from swarmsearch.utils.data import compile_clickthrough_records
from swarmsearch.utils.cache import Cache
from swarmsearch.utils.db import get_ctrs
from swarmsearch.utils.attack import poison_ctrs, flip_label, rand_ctr
from swarmsearch.dart.utils import ClickThroughRecord
from swarmsearch.dart.types import Dataset, SplitDataset
from swarmsearch.datasets.aol import load_dataset
from argparse import ArgumentParser
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from swarmsearch.utils.experiment import Experiment
from swarmsearch.dart.rank import tracincp_valuation
from tqdm import tqdm

shapley_df = pd.read_csv('results/experiment_shapley.tsv', sep='\t', header=None)

def experiment_fn(exp: Experiment, user_id: str) -> list:
    # sample neighbors
    MAX_NEIGHBORS = 10
    neighbor_user_ids = random.sample([uid for uid in exp.cache.get("user_ids") if uid != user_id], MAX_NEIGHBORS)

    # load user's personal dataset
    ctrs = get_ctrs(user_id)
    user_ds = split_dataset_by_qids(ctrs, train_ratio=1/3, val_ratio=1/3)

    neighbor_datasets = {
        neighbor_id: get_ctrs(int(neighbor_id))
        for neighbor_id in neighbor_user_ids
    }

    local_mrr = evaluate(exp.config, user_ds)

    naive_mrrs = defaultdict(float)
    defense_mrrs = defaultdict(float)
    oracle_mrrs = defaultdict(float)

    # attack from 0% to 100%, in steps of 10%
    for attack_vol in range(11):

        # collect ctrs and poison according to attack_vol
        candidate_datasets = deepcopy(neighbor_datasets)
        for neighbor_id in neighbor_user_ids[:attack_vol]:
            candidate_datasets[neighbor_id] = poison_ctrs(flip_label, candidate_datasets[neighbor_id])
        shapley_values = shapley_valuation(exp.config, user_ds, candidate_datasets, 100)

        # prepare and poison datasets
        datasets = defaultdict(list)
        for neighbor_id, neighbor_dataset in neighbor_datasets.items():
            if neighbor_id in neighbor_user_ids[:attack_vol]:
                neighbor_dataset = poison_ctrs(flip_label, neighbor_dataset)
            datasets[neighbor_id] = neighbor_dataset

        # Naive: train on all datasets
        naive_mrrs[attack_vol] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [ctr for uid in datasets for ctr in datasets[uid]],
            vali=user_ds.vali,
            test=user_ds.test
        ))

        # Defense: train where shapley value is positive
        defense_mrrs[attack_vol] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [
                ctr for uid in datasets for ctr in datasets[uid] if shapley_values[uid] > 0
                ],
            vali=user_ds.vali,
            test=user_ds.test
        ))

        # Oracle: train on non-poisoned datasets
        oracle_mrrs[attack_vol] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [ctr for uid in datasets for ctr in datasets[uid] if uid not in neighbor_user_ids[:attack_vol]],
            vali=user_ds.vali,
            test=user_ds.test
        ))
        
    return [user_id, local_mrr] + [
        naive_mrrs[attack_vol] for attack_vol in range(11)
        ] + [
        defense_mrrs[attack_vol] for attack_vol in range(11)
        ] + [
        oracle_mrrs[attack_vol] for attack_vol in range(11)
        ]

    

if __name__ == "__main__":
    exp = Experiment(
        id="mrrs",
        fn=experiment_fn
    )
    exp.run(parallel=True)