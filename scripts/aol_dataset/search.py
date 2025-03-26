import ir_datasets
from pyserini.search.lucene import LuceneSearcher

dataset = ir_datasets.load("aol-ia")
searcher = LuceneSearcher('indexes/docs_jsonl')

import time

start_time = time.time()
hits = searcher.search('document', k=10000)
search_time = time.time() - start_time

for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')
    print(dataset.docs_store().get(hits[i].docid).title)

print(f"Search completed in {search_time:.4f} seconds")