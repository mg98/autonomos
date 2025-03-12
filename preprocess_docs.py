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
from semantica.embed import embed_batch
from datasets.aol4ps import load_dataset
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
    _, _, docs_df = load_dataset("AOL4PS")
    
    url_to_aol4ps_docid = {}
    for aol4ps_docid, doc_row in docs_df.iterrows():
        url = doc_row['Url']
        # See https://github.com/allenai/ir_datasets/blob/e20850ea54d37aeefca745b12a82eb608c7cb6a4/ir_datasets/datasets/aol_ia.py#L167.
        aolia_docid = md5(str(aol4ps_docid).encode()).hexdigest()[:DID_LEN]
        url_to_aol4ps_docid[url] = aol4ps_docid
    
    del docs_df

    dataset = ir_datasets.load("aol-ia")
    
    # Convert iterable docs into a list so we can batch them in Python.
    # NOTE: For extremely large datasets, you may not want to hold them all in memory.
    all_docs = list(dataset.docs_iter())

    doc_ids = []
    texts = []

    for doc in all_docs:
        # AOLIA has 1.5M docs, but AOL4PS only has 950K docs.
        if doc.url in url_to_aol4ps_docid:
            doc_ids.append(url_to_aol4ps_docid[doc.url])
            texts.append(doc.text)

    print("Embedding documents...")

    # Call embed_batch. Inside embed_batch, each text is further split if needed.
    # The "batch_size" argument inside embed_batch controls how many chunks go through the GPU at once. Adjust if needed.
    # By experience: 64 consumes around 3.5G of memory, 256 consumes around 10G.
    embeddings = embed_batch(texts, batch_size=256)

    if os.path.exists('doc_embeddings.lmdb'):
        shutil.rmtree('doc_embeddings.lmdb')
    
    print("Saving to disk...")

    with lmdb.open('doc_embeddings.lmdb', map_size=2**33) as db: # 2**26=64M
        with db.begin(write=True) as txn:
            for doc_id, emb in tqdm(zip(doc_ids, embeddings), total=len(doc_ids)):
                txn.put(str(doc_id).encode(), pickle.dumps(emb))

    print("Done.")
