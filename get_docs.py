import random
import torch
import numpy as np
from tqdm import tqdm
import pickle
import lmdb
import ir_datasets
from autonomos.semantica.embed import embed_batch
import argparse

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

    print("Loading dataset...")

    dataset = ir_datasets.load("aol-ia")

    start = args.job_id / args.job_count
    end = (args.job_id + 1) / args.job_count

    doc_id2content = {}
    for doc in dataset.docs_iter()[start:end]:
        doc_id2content[doc.doc_id] = f"{doc.url}\n{doc.title}\n{doc.text}"
    
    print("Embedding documents...")

    # Call embed_batch. Inside embed_batch, each text is further split if needed.
    # The "batch_size" argument inside embed_batch controls how many chunks go through the GPU at once. Adjust if needed.
    # By experience: 64 consumes around 3.5G of memory, 256 consumes around 10G.
    embeddings = embed_batch(doc_id2content.values(), batch_size=256)
    
    print("Saving to disk...")

    with lmdb.open('data/doc_embeddings.lmdb', map_size=2**33, sync=False) as db: # 2**26=64M
        with db.begin(write=True) as txn:
            for doc_id, emb in tqdm(zip(doc_id2content.keys(), embeddings), total=len(doc_id2content)):
                txn.put(str(doc_id).encode(), pickle.dumps(emb))

    print("Done.")
