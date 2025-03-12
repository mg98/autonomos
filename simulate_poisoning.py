import random
import torch
import numpy as np
from dart.utils import split_by_qids, compute_feature_stats
from allrank.config import Config
from dart.rank import evaluate
from utils.data import compile_clickthrough_records
from utils.cache import Cache
from semantica.graph import get_neighbors
from utils.db import get_ctrs, get_ctrs_from_users, get_user_embedding
import warnings
from dart.utils import ClickThroughRecord
from datasets.aol4ps import load_dataset
from argparse import ArgumentParser
from dart.types import Dataset
from numpy import dot
from numpy.linalg import norm

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

warnings.filterwarnings(
    "ignore",
    message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.",
    category=UserWarning,
    module="joblib.externals.loky.process_executor"
)

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
if hasattr(torch, 'mps'):
    torch.mps.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    parser = ArgumentParser(description='Simulate individual user search performance under the influence ofpoisoning attacks')
    parser.add_argument('--user', '-u', type=int, required=True, help='User ID to simulate')
    parser.add_argument('--poison', '-p', type=float, default=0.0, help='Poisoning ratio')
    
    args = parser.parse_args()
    cache = Cache()
    config = Config.from_json("./allRank_config.json")
    
    user_id = args.user
    poison_ratio = args.poison

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
    feature_means = cache.get("feature_means")
    feature_stds = cache.get("feature_stds")

    df, _, _ = load_dataset('AOL4PS')
    # Create a dictionary mapping each user_id to their set of clicked documents
    user_doc_map = {user_id: set(df[df['AnonID'] == user_id]['DocIndex'].unique()) for user_id in user_ids}

    poison_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print("UserID\tN\tMRR-L\t" + "\t".join([f"MRR-{ratio}" for ratio in poison_ratios]))

    # Track metrics for summary
    mrr_l_values = []
    mrr_r_values = []
    mrr_s_values = []
    mrr_c_values = []

    for user_id in user_ids:
        # Get ctrs from user_ctrs.lmdb by user_id
        ctrs = get_ctrs(user_id)
        if ctrs is None:
            continue

        user_ds = split_by_qids(ctrs, context_ratio=0.5)

        # Evaluate local performance
        n_l = len(user_ds.context)
        mrr_l = evaluate(config, user_ds, feature_means, feature_stds)
        
        # Evaluate Semantica performance
        semantica_neighbor_user_ids = get_neighbors(user_id)
        if len(semantica_neighbor_user_ids) == 0:
            continue
        neighbor_ctrs = get_ctrs_from_users(semantica_neighbor_user_ids)

        # Poison neighbor records
        mrr_s: dict[float, float] = {}
        for poison_ratio in poison_ratios:
            poison_count = int(len(neighbor_ctrs) * poison_ratio)
            keep_count = len(neighbor_ctrs) - poison_count
            poison_ctrs = list(map(lambda x: ClickThroughRecord(rel=not x.rel, qid=x.qid, feat=x.feat), neighbor_ctrs[keep_count:]))
            
            ds = Dataset(
                user_ds.context + neighbor_ctrs[:keep_count] + poison_ctrs, 
                user_ds.test
            )
            mrr_s[poison_ratio] = evaluate(config, ds, feature_means, feature_stds)

        # Store metrics for summary
        mrr_l_values.append(mrr_l)
        mrr_s_values.append(mrr_s)

        sep = "\t"
        print(f"{user_id}{sep}"
            f"{len(semantica_neighbor_user_ids)}{sep}"
            f"{mrr_l:.3f}{sep}"
            f"{sep.join([f'{mrr_s[ratio]:.3f}' for ratio in sorted(mrr_s.keys())])}")
        
