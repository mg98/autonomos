from tqdm import tqdm
import ir_datasets
import json
import os

dataset = ir_datasets.load("aol-ia")

os.system("rm -rf indexes")
os.makedirs("docs_jsonl", exist_ok=True)
docs_output_file = "docs_jsonl/docs.jsonl"

# Write documents to JSONL file (one JSON object per line)
with open(docs_output_file, "w") as f:
    for doc in tqdm(dataset.docs_iter(), desc="Processing documents", total=dataset.docs_count()):
        json_doc = {
            "id": doc.doc_id,
            "contents": f"{doc.url}\n{doc.title}\n{doc.text}",
        }
        f.write(json.dumps(json_doc) + "\n")

# Now create the index using Pyserini
print("Creating index...")
os.system(f"""
python -m pyserini.index.lucene \\
  --collection JsonCollection \\
  --input docs_jsonl \\
  --index indexes/docs_jsonl \\
  --generator DefaultLuceneDocumentGenerator \\
  --threads 8 \\
  --storePositions --storeDocvectors --storeRaw
""")

os.system("rm -rf docs_jsonl")

print("Success!")
