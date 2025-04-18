import random
import torch
import numpy as np
from autonomos.dart.utils import split_by_qids, compute_feature_stats
from allrank.config import Config
from autonomos.dart.rank import evaluate
from autonomos.utils.data import compile_clickthrough_records
from autonomos.utils.cache import Cache
from autonomos.semantica.graph import get_neighbors
from autonomos.utils.db import get_ctrs
from autonomos.utils.attack import poison_ctrs, flip_label, rand_ctr
from autonomos.dart.utils import ClickThroughRecord
from autonomos.dart.types import SplitDataset
from autonomos.datasets.aol import load_dataset
from argparse import ArgumentParser
import pandas as pd
from copy import deepcopy

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'):
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def selective_training(local_mrr, _user_ds, neighbors_ctrs: list[list[ClickThroughRecord]]):
    """
    Selective integration of neighbors into local context.
    Only accept neighbors that improve local MRR.

    Args:
        local_mrr: MRR of the local context
        user_ds: Dataset object containing the local context
        neighbors_ctrs: List of lists of ClickThroughRecords, where each inner list contains the CTRs for a single neighbor

    Returns:
        mrr_on_test: MRR of the integrated context on the test set
        count_accepted_neighbors: Number of neighbors accepted into the local context
    """
    # Selective integration
    user_ds = deepcopy(_user_ds)
    count_accepted_neighbors = 0
    assert len(neighbors_ctrs) == 10

    for ctrs in neighbors_ctrs:
        # Split local context into train/val (used for training) and test (used for evaluation of neighbor's dataset)
        local_context_split = split_by_qids(user_ds.context)
        ds = SplitDataset(
            train=ctrs,
            vali=local_context_split.context, # merge neighbor's ctrs into local context
            test=local_context_split.test # use test set of local context for evaluation
        )
        mrr = evaluate(config, ds, feature_means, feature_stds)
        # Accept dataset if it improves local MRR
        if mrr > local_mrr:
            # Update local MRR and context
            local_mrr = mrr
            user_ds.context += ctrs
            count_accepted_neighbors += 1
     
    # Evaluate final MRR on independent test set
    mrr_on_test = evaluate(config, user_ds, feature_means, feature_stds)

    # Return final MRR
    return mrr_on_test, local_mrr,count_accepted_neighbors

def get_done_user_ids():
    try:
        results_df = pd.read_csv('results.tsv', sep='\t', header=None)
        return set(results_df[0].astype(int))
    except FileNotFoundError:
        return set()

if __name__ == "__main__":
    parser = ArgumentParser(description='Simulate individual user search performance under the influence of poisoning attacks')
    parser.add_argument('--user', '-u', type=int, help='User ID to simulate (if not specified, all users will be simulated sequentially)')
    parser.add_argument('--job-id', type=int, help='Slurm job ID')
    parser.add_argument('--job-count', type=int, help='Total number of jobs')
    
    args = parser.parse_args()
    cache = Cache()
    config = Config.from_json("./allRank_config.json")

    if cache.is_empty():
        # This is done for normalization of features later
        print("Precomputing feature statistics from sample...")
        df = load_dataset()
        sample_df = df.sample(n=1000, random_state=42)
        sample_ctrs = compile_clickthrough_records(sample_df)
        feature_means, feature_stds = compute_feature_stats(sample_ctrs)
        cache.set("user_ids", df["user_id"].unique())
        cache.set("feature_means", feature_means)
        cache.set("feature_stds", feature_stds)
        print("Cache set")
    
    user_ids = cache.get("user_ids")
    if args.user is not None:
        user_ids = [args.user]

    feature_means = cache.get("feature_means")
    feature_stds = cache.get("feature_stds")

    done_user_ids = get_done_user_ids()

    for idx, user_id in enumerate(user_ids):
        if args.job_id is not None and idx % args.job_count != args.job_id:
            continue

        if user_id in done_user_ids:
            continue

        # Get ctrs from user_ctrs.lmdb by user_id
        ctrs = get_ctrs(user_id)
        if ctrs is None:
            continue
        user_ds = split_by_qids(ctrs, context_ratio=0.8)

        # Evaluate local performance
        n_l = len(user_ds.context)
        mrr_l = evaluate(config, user_ds, feature_means, feature_stds)

        # Sample neighbors
        N = 10
        neighbor_user_ids = random.sample([uid for uid in user_ids if uid != user_id], N)

        # poison_ratio -> mrr
        lf_poisoned_mrrs: dict[int, float] = {}
        selective_lf_poisoned_mrrs: dict[int, float] = {}
        selective_lf_poisoned_val_mrrs: dict[int, float] = {}
        count_accepted_neighbors: dict[int, int] = {}

        for N_poisoned in range(N+1):
            neighbors_ctrs: list[list[ClickThroughRecord]] = []

            # Poison percentage of neighbors
            for neighbor_user_id in neighbor_user_ids:
                ctrs = get_ctrs(neighbor_user_id)
                if ctrs is None:
                    raise Exception(f"CTRs are None for user {neighbor_user_id}")
                if neighbor_user_id in neighbor_user_ids[:N_poisoned]:
                    ctrs = poison_ctrs(flip_label, ctrs)
                neighbors_ctrs.append(ctrs)

            ds = Dataset(
                user_ds.context + sum(neighbors_ctrs, []), 
                user_ds.test
            )

            lf_poisoned_mrrs[N_poisoned] = evaluate(config, ds, feature_means, feature_stds)

            selective_lf_poisoned_mrrs[N_poisoned], selective_lf_poisoned_val_mrrs[N_poisoned], count_accepted_neighbors[N_poisoned] = selective_training(mrr_l, user_ds, neighbors_ctrs)

        sep = "\t"
        user_result = f"{user_id}{sep}{mrr_l}{sep}" + \
              sep.join([f"{lf_poisoned_mrrs[N_poisoned]}" for N_poisoned in range(N+1)]) + sep + \
              sep.join([f"{selective_lf_poisoned_mrrs[N_poisoned]}" for N_poisoned in range(N+1)]) + sep + \
              sep.join([f"{selective_lf_poisoned_val_mrrs[N_poisoned]}" for N_poisoned in range(N+1)]) + sep + \
              sep.join([f"{count_accepted_neighbors[N_poisoned]}" for N_poisoned in range(N+1)])
        
        if args.job_id is not None:
            with open(f"results.tsv", "a") as f:
                f.write(user_result + "\n")
        else:
            print(user_result)
        
        
