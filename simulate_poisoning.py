import random
import torch
import numpy as np
from dart.utils import split_by_qids, compute_feature_stats
from allrank.config import Config
from dart.rank import evaluate
from utils.data import compile_clickthrough_records
from utils.cache import Cache
from semantica.graph import get_neighbors
from utils.db import get_ctrs, get_ctrs_from_users
from utils.attack import poison_ctrs, flip_label, rand_ctr
from dart.utils import ClickThroughRecord
from datasets.aol4ps import load_dataset
from argparse import ArgumentParser
from dart.types import Dataset
from random import sample

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
    for ctrs in neighbors_ctrs:
        ds = Dataset(
            cur_context + ctrs, 
            user_ds.test
        )
        mrr = evaluate(config, ds, feature_means, feature_stds)
        if mrr > cur_mrr_l:
            mrr_x = mrr
            cur_mrr_l = mrr_x
            cur_context = cur_context + ctrs
    
    # Return final MRR
    return mrr_x



if __name__ == "__main__":
    parser = ArgumentParser(description='Simulate individual user search performance under the influence of poisoning attacks')
    parser.add_argument('--user', '-u', type=int, help='User ID to simulate (if not specified, all users will be simulated sequentially)')
    
    args = parser.parse_args()
    cache = Cache()
    config = Config.from_json("./allRank_config.json")

    if cache.is_empty():
        print("Loading dataset...")
        df, queries_df, docs_df = load_dataset('AOL4PS')
        print("Dataset loaded.")
        
        # This is done for normalization of features later
        print("Precomputing feature statistics from sample...")
        sample_df = df.sample(n=1000, random_state=42)
        sample_ctrs = compile_clickthrough_records(sample_df, queries_df, docs_df)
        feature_means, feature_stds = compute_feature_stats(sample_ctrs)

        cache.set("user_ids", df['AnonID'].unique())
        cache.set("feature_means", feature_means)
        cache.set("feature_stds", feature_stds)
    
    user_ids = cache.get("user_ids")
    if args.user is not None:
        user_ids = [args.user]

    feature_means = cache.get("feature_means")
    feature_stds = cache.get("feature_stds")

    for user_id in user_ids:
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
        neighbor_user_ids = sample(get_neighbors(user_id), N)

        # poison_ratio -> mrr
        lf_poisoned_mrrs: dict[float, float] = {}
        selective_lf_poisoned_mrrs: dict[float, float] = {}

        for N_poisoned in range(N+1):
            neighbors_ctrs: list[list[ClickThroughRecord]] = []

            # Poison percentage of neighbors
            for neighbor_user_id in neighbor_user_ids:
                ctrs = get_ctrs(neighbor_user_id)
                if neighbor_user_id in neighbor_user_ids[:N_poisoned]:
                    ctrs = poison_ctrs(flip_label, ctrs)
                neighbors_ctrs.append(ctrs)

            ds = Dataset(
                user_ds.context + sum(neighbors_ctrs, []), 
                user_ds.test
            )
            lf_poisoned_mrrs[N_poisoned] = evaluate(config, ds, feature_means, feature_stds)

            selective_lf_poisoned_mrrs[N_poisoned] = selective_training(mrr_l, user_ds, neighbors_ctrs)

        # Print results
        sep = "\t"
        print(f"{mrr_l:.4f}", end=sep)
        
        for N_poisoned in range(N+1):
            print(f"{lf_poisoned_mrrs[N_poisoned]:.4f}", end=sep)
            print(f"{selective_lf_poisoned_mrrs[N_poisoned]:.4f}", end=sep)
        
        print()  # End the line
        
        
