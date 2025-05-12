import random
import torch
import numpy as np
from autonomos.dart.utils import split_by_qids, split_dataset_by_qids, compute_feature_stats
from allrank.config import Config
from autonomos.dart.rank import evaluate, shapley_valuation
from autonomos.utils.data import compile_clickthrough_records
from autonomos.utils.cache import Cache
from autonomos.utils.db import get_ctrs
from autonomos.utils.attack import poison_ctrs, flip_label, rand_ctr
from autonomos.dart.utils import ClickThroughRecord
from autonomos.dart.types import Dataset, SplitDataset
from autonomos.datasets.aol import load_dataset
from argparse import ArgumentParser
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from autonomos.utils.experiment import Experiment
from autonomos.dart.rank import tracincp_valuation
from tqdm import tqdm

shapley_df = pd.read_csv('results/experiment_shapley.tsv', sep='\t', header=None)

def experiment_fn(exp: Experiment, user_id: str) -> list:
    user_row = shapley_df[shapley_df[0] == int(user_id)].iloc[0]
    neighbor_user_ids = user_row[-10:].values.astype(int).astype(str)
    assert len(neighbor_user_ids) == 10

    shapley_values = user_row[24:-10].values.reshape(11, 10).T  # 11 attack volumes, 10 neighbors

    neighbor_shapley_values = {
        neighbor_user_ids[i]: shapley_values[i]
        for i in range(10)
    }

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

    for attack_vol in range(11):

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
                ctr for uid in datasets for ctr in datasets[uid] if neighbor_shapley_values[uid][attack_vol] > 0
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
        fn=experiment_fn,
        user_ids_source="results/experiment_shapley.tsv"
    )
    exp.run(parallel=True)