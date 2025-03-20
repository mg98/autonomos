import pandas as pd
import argparse
from tqdm import tqdm
import ir_datasets
import math
from pyserini.search.lucene import LuceneSearcher

dataset = ir_datasets.load("aol-ia")

parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
parser.add_argument('--job-id', type=int, help='Slurm job ID')
parser.add_argument('--job-count', type=int, help='Total number of jobs')
args = parser.parse_args()

job_size = math.ceil(dataset.qlogs_count() / args.job_count)
start = job_size * args.job_id
end = start + job_size

print(f"[{args.job_id}/{args.job_count}] Processing range {start} to {end}")

searcher = LuceneSearcher('scripts/aol-dataset/indexes/docs_jsonl')

doc_ids = [doc.doc_id for doc in dataset.docs_iter()]
query_to_candidates = {}

def process_qlog(searcher, qlog, doc_ids):
    if len(qlog.items) == 0:
        return None
    
    if qlog.query.strip() == '':
        return None

    target_doc_id = qlog.items[0].doc_id
    if target_doc_id not in doc_ids:
        return None
    
    if qlog.query not in query_to_candidates.keys():
        top_doc_ids = [hit.docid for hit in searcher.search(qlog.query, k=10)]
        if len(top_doc_ids) < 10:
            return None

        if target_doc_id not in top_doc_ids:
            # Replace the last top_doc_id with the target_doc_id to ensure it's in the list
            top_doc_ids[-1] = target_doc_id
        
        query_to_candidates[qlog.query] = top_doc_ids
    
    return {
        'user_id': qlog.user_id,
        'time': qlog.time,
        'query': qlog.query,
        'doc_id': target_doc_id,
        'candidate_doc_ids': query_to_candidates[qlog.query]
    }

results = []
i = -1
for qlog in tqdm(dataset.qlogs_iter(), total=args.end):
    i += 1
    if i < start: continue
    if i >= end: break

    result = process_qlog(searcher, qlog, doc_ids)
    if result is not None:
        results.append(result)

df = pd.DataFrame(results)
print(f"Saving DataFrame with {len(df)} records to disk...")
df.to_csv(f'aol_{args.start}_{args.end}.csv', index=False)
print("DataFrame saved successfully.")
