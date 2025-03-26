import random
import torch
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from autonomos.datasets.aol import load_dataset
from autonomos.utils.data import compile_clickthrough_records_as_arrays
import pickle
import lmdb
import warnings
import argparse

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
    parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
    parser.add_argument('--job-id', type=int, help='Slurm job ID')
    parser.add_argument('--job-count', type=int, help='Total number of jobs')
    args = parser.parse_args()
    
    df = load_dataset()
    unique_users = sorted(df['user_id'].unique())
    
    # job filtering
    user_ids = [u for i, u in enumerate(unique_users) if args.job_id is None or i % args.job_count == args.job_id]
    user_groups = df[df['user_id'].isin(user_ids)].groupby('user_id')
    del df, unique_users

    ctr_arrays = Parallel(n_jobs=-1, batch_size=64)(
        delayed(compile_clickthrough_records_as_arrays)(user_df) 
        for _, user_df in tqdm(
            user_groups,
            total=len(user_groups),
            desc=f"Processing users (job {args.job_id+1}/{args.job_count})" if args.job_id is not None else "Processing users"
        )
    )
    
    # Write to LMDB with user_id as key and list of ClickThroughRecords as value
    print("Writing results to LMDB...")

    with lmdb.open(f'data/ctrs_{args.job_id+1}.lmdb' if args.job_id is not None else 'data/ctrs.lmdb', map_size=2**46) as db:
        with db.begin(write=True) as txn:
            for user_id, ctrs in tqdm(zip(user_ids, ctr_arrays), total=len(user_ids), desc="Writing to LMDB"):
                txn.put(str(user_id).encode(), pickle.dumps(ctrs))
    
    print("Saved to disk")
