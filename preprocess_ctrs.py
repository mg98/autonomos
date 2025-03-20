import random
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from autonomos.datasets.aol import load_dataset
from autonomos.utils.data import compile_clickthrough_records
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
    df = load_dataset()
    user_groups = df.groupby('user_id')
    user_ids = list(user_groups.groups.keys())

    ctrs = Parallel(n_jobs=16, batch_size=64)(
        delayed(compile_clickthrough_records)(user_df) 
        for _, user_df in tqdm(
            user_groups, 
            total=len(user_groups), 
            desc="Processing users"
        )
    )

    # Write to LMDB with user_id as key and list of ClickThroughRecords as value
    print("Writing results to LMDB...")
    
    # Delete existing LMDB if it exists
    if os.path.exists('data/ctrs.lmdb'):
        shutil.rmtree('data/ctrs.lmdb')

    with lmdb.open('data/ctrs.lmdb', map_size=2**44, sync=False) as db:
        with db.begin(write=True) as txn:
            for user_id, ctrs in tqdm(zip(user_ids, ctrs), desc="Writing to LMDB"):
                txn.put(str(user_id).encode(), pickle.dumps(ctrs))
    
    print("Saved to disk")
