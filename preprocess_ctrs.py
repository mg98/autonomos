import random
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from datasets.aol4ps import load_dataset
from utils.data import compile_clickthrough_records
import pickle
import lmdb
import warnings
import os
import shutil

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
    print("Loading data...")
    df, queries_df, docs_df = load_dataset('AOL4PS')
    user_groups = df.groupby('AnonID')
    user_ids = list(user_groups.groups.keys())
    
    ctrs = Parallel(n_jobs=16, batch_size=64)(
        delayed(compile_clickthrough_records)(user_df, queries_df, docs_df) 
        for _, user_df in tqdm(
            user_groups, 
            total=len(user_groups), 
            desc="Processing users"
        )
    )

    # Write to LMDB with user_id as key and list of ClickThroughRecords as value
    print("Writing results to LMDB...")
    
    # Delete existing LMDB if it exists
    if os.path.exists('user_ctrs.lmdb'):
        shutil.rmtree('user_ctrs.lmdb')

    with lmdb.open('user_ctrs.lmdb', map_size=2**44) as db: # 64M
        with db.begin(write=True) as txn:
            for user_id, ctrs in tqdm(zip(user_ids, ctrs), desc="Writing to LMDB"):
                txn.put(str(user_id).encode(), pickle.dumps(ctrs))
    
    print("Saved to disk")
