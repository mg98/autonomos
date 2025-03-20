import json
import random
import torch
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
import lmdb
import warnings
import os
import shutil
import ir_datasets
from autonomos.semantica.embed import embed_batch
from autonomos.datasets.aol import load_dataset
from hashlib import md5
from ir_datasets.datasets.aol_ia import DID_LEN

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
    print("Loading dataset...")
    _, queries_df, _ = load_dataset("AOL4PS")

    print("Embedding queries...")
    qids = queries_df.index.tolist()

    # Call embed_batch. Inside embed_batch, each text is further split if needed.
    # The "batch_size" argument inside embed_batch controls how many chunks go through the GPU at once. Adjust if needed.
    embeddings = embed_batch(queries_df['Query'].tolist(), batch_size=256)

    if os.path.exists('data/query_embeddings.lmdb'):
        shutil.rmtree('data/query_embeddings.lmdb')
    
    print("Saving to disk...")

    with lmdb.open('data/query_embeddings.lmdb', map_size=2**33, sync=False) as db: # 2**26=64M
        with db.begin(write=True) as txn:
            for qid, emb in tqdm(zip(qids, embeddings), total=len(qids)):
                txn.put(str(qid).encode(), pickle.dumps(emb))

    print("Done.")
