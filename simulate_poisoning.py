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
from autonomos.dart.types import Dataset
from autonomos.datasets.aol import load_dataset
from argparse import ArgumentParser
import pandas as pd

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'):
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def selective_training(local_mrr, user_ds, neighbors_ctrs: list[list[ClickThroughRecord]]):
    # Selective integration
    mrr_x = local_mrr
    cur_mrr_l = local_mrr
    cur_context = user_ds.context
    count_accepted_neighbors = 0
    for ctrs in neighbors_ctrs:
        ds = Dataset(
            cur_context + ctrs, 
            user_ds.test
        )
        mrr = evaluate(config, ds, feature_means, feature_stds)
        if mrr > cur_mrr_l:
            # dataset accepted
            mrr_x = mrr
            cur_mrr_l = mrr_x
            cur_context = cur_context + ctrs
            count_accepted_neighbors += 1
    
    # Return final MRR
    return mrr_x, count_accepted_neighbors

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

            selective_lf_poisoned_mrrs[N_poisoned], count_accepted_neighbors[N_poisoned] = selective_training(mrr_l, user_ds, neighbors_ctrs)

        sep = "\t"
        user_result = f"{user_id}{sep}{mrr_l}{sep}" + \
              sep.join([f"{lf_poisoned_mrrs[N_poisoned]}" for N_poisoned in range(N+1)]) + sep + \
              sep.join([f"{selective_lf_poisoned_mrrs[N_poisoned]}" for N_poisoned in range(N+1)]) + sep + \
              sep.join([f"{count_accepted_neighbors[N_poisoned]}" for N_poisoned in range(N+1)])
        
        if args.job_id is not None:
            with open(f"results.tsv", "a") as f:
                f.write(user_result + "\n")
        else:
            print(user_result)
        
        
