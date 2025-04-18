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

def experiment_fn(exp: Experiment, user_id: str) -> list:
    # number of neighbors -> mrr (0 is local)
    attack_blind_mrrs: dict[int] = defaultdict(float)
    attack_aware_mrrs: dict[int] = defaultdict(float)
    blocked_attackers: dict[int] = defaultdict(int)

    # load local dataset
    ctrs = get_ctrs(user_id)
    user_ds = split_dataset_by_qids(ctrs, train_ratio=1/3, val_ratio=1/3)

    # evaluate local performance
    local_mrr = evaluate(exp.config, user_ds)

    # sample neighbors
    MAX_NEIGHBORS = 10
    neighbor_user_ids = random.sample([uid for uid in exp.user_ids if uid != user_id], MAX_NEIGHBORS)

    healthy_candidate_datasets = {
        neighbor_id: get_ctrs(neighbor_id)
        for neighbor_id in neighbor_user_ids
    }

    for attack_vol in tqdm(range(11)):
        # collect ctrs and poison according to attack_vol
        candidate_datasets = deepcopy(healthy_candidate_datasets)
        for neighbor_id in neighbor_user_ids[:attack_vol]:
            candidate_datasets[neighbor_id] = poison_ctrs(flip_label, candidate_datasets[neighbor_id])
        
        # attack-blind
        attack_blind_mrrs[attack_vol] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [ctr for uid in candidate_datasets for ctr in candidate_datasets[uid]],
            vali=user_ds.vali,
            test=user_ds.test
        ))

        # attack-aware
        shapley_values = shapley_valuation(exp.config, user_ds, candidate_datasets, 100)
        
        # Remove users with negative scores from candidate_datasets
        candidate_datasets = {
            uid: ds
            for uid, ds in candidate_datasets.items()
            if shapley_values[uid] > 0
        }

        attack_aware_mrrs[attack_vol] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [ctr for uid in candidate_datasets for ctr in candidate_datasets[uid]],
            vali=user_ds.vali,
            test=user_ds.test
        ))
        blocked_attackers[attack_vol] = MAX_NEIGHBORS - len(candidate_datasets)

    return [user_id, local_mrr] + [
        attack_blind_mrrs[n_neighbors] for n_neighbors in range(MAX_NEIGHBORS+1)
        ] + [
        attack_aware_mrrs[n_neighbors] for n_neighbors in range(MAX_NEIGHBORS+1)
        ] + [
        blocked_attackers[n_neighbors] for n_neighbors in range(MAX_NEIGHBORS+1)
        ]

if __name__ == "__main__":
    exp = Experiment(
        id="shapley",
        fn=experiment_fn
    )
    exp.run(parallel=True)
   