import pandas as pd
import argparse
from tqdm import tqdm
import ir_datasets
from pyserini.search.lucene import LuceneSearcher

dataset = ir_datasets.load("aol-ia")

parser = argparse.ArgumentParser(description='Process AOL dataset with start and end indices')
parser.add_argument('--job-id', type=int, help='Slurm job ID')
parser.add_argument('--job-count', type=int, help='Total number of jobs')
args = parser.parse_args()

searcher = LuceneSearcher('indexes/docs_jsonl')

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
    
    candidate_doc_ids = []
    if qlog.query not in query_to_candidates.keys():
        candidate_doc_ids = [hit.docid for hit in searcher.search(qlog.query, k=10)]
        query_to_candidates[qlog.query] = candidate_doc_ids
    
    if len(candidate_doc_ids) < 10:
        return None

    if target_doc_id not in candidate_doc_ids:
        # Replace the last top_doc_id with the target_doc_id to ensure it's in the list
        candidate_doc_ids[-1] = target_doc_id
    
    return {
        'user_id': qlog.user_id,
        'time': qlog.time,
        'query': qlog.query,
        'doc_id': target_doc_id,
        'candidate_doc_ids': candidate_doc_ids
    }

results = []
for idx, qlog in tqdm(enumerate(dataset.qlogs_iter()), total=dataset.qlogs_count()):
    if args.job_id is not None and idx % args.job_count != args.job_id:
        continue

    result = process_qlog(searcher, qlog, doc_ids)
    if result is not None:
        results.append(result)

df = pd.DataFrame(results)

print(f"Saving DataFrame with {len(df)} records to disk...")
pickle_filename = f'aol_{args.job_id}_{args.job_count}.pkl' if args.job_id is not None else 'aol_raw_dataset.pkl'
df.to_pickle(pickle_filename)
print("Success!")
