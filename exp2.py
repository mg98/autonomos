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

def experiment_fn(exp: Experiment, user_id: str) -> list:
    # number of neighbors -> mrr (0 is local)
    mrrs: dict[int] = defaultdict(float)

    # load local dataset
    ctrs = get_ctrs(user_id)
    user_ds = split_dataset_by_qids(ctrs, train_ratio=1/3, val_ratio=1/3)

    # evaluate local performance
    mrrs[0] = evaluate(exp.config, user_ds)

    # sample neighbors
    MAX_NEIGHBORS = 10
    neighbor_user_ids = random.sample([uid for uid in exp.user_ids if uid != user_id], MAX_NEIGHBORS)
    
    candidate_datasets = defaultdict(list)
    for i, neighbor_id in enumerate(neighbor_user_ids):
        candidate_datasets[neighbor_id] = get_ctrs(neighbor_id)

        infl_scores = tracincp_valuation(exp.config, user_ds, candidate_datasets)
        proponents = [user_id for user_id, score in infl_scores.items() if user_id != '0' and score > 0]

        print("hello")
        mrrs[i+1] = evaluate(exp.config, SplitDataset(
            train=user_ds.train + [ctr for user_id in proponents for ctr in candidate_datasets[user_id]],
            vali=user_ds.vali,
            test=user_ds.test
        ))
        print(mrrs[i+1])

    return [user_id] + [mrrs[n_neighbors] for n_neighbors in range(MAX_NEIGHBORS+1)]

if __name__ == "__main__":
    exp = Experiment(
        id="tracincp_filtered_healthy_increase",
        fn=experiment_fn
    )
    exp.run(parallel=True)
   