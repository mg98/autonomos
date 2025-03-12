from datasets.aol4ps import load_dataset
from utils.db import get_doc_embedding
from semantica.graph import get_neighbors
import random
import numpy as np
from numpy import dot
from numpy.linalg import norm

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

df, queries_df, docs_df = load_dataset('AOL4PS')
all_users = df['AnonID'].unique()

def get_user_embedding(user_id):
    doc_ids = df[df['AnonID'] == user_id]['DocIndex'].unique()
    if len(doc_ids) == 0: return None
    return np.mean([get_doc_embedding(doc_id) for doc_id in doc_ids], axis=0)

print("user_id", "\t", "met", "\t", "rand", "\t", "sem", "\t", "diff")

cs_diffs = []
ed_diffs = []

try:
    for user_id in all_users:
        user_emb = get_user_embedding(user_id)

        semantica_neighbors_emb = [get_user_embedding(neighbor) for neighbor in get_neighbors(user_id)]
        semantica_neighbors_emb = [emb for emb in semantica_neighbors_emb if emb is not None]
        if len(semantica_neighbors_emb) == 0: continue

        random_neighbors_emb = [get_user_embedding(neighbor) for neighbor in random.sample(list(all_users), len(semantica_neighbors_emb))]
        if len(random_neighbors_emb) == 0: continue
        
        semantica_neighbors_avg_emb = np.mean(semantica_neighbors_emb, axis=0)
        random_neighbors_avg_emb = np.mean(random_neighbors_emb, axis=0)

        semantica_similarity = cosine_similarity(user_emb, semantica_neighbors_avg_emb)
        random_similarity = cosine_similarity(user_emb, random_neighbors_avg_emb)
        cs_diffs.append(semantica_similarity - random_similarity)

        padded_user_id = str(user_id).ljust(7, '_')
        print(padded_user_id, "\t", "cs", "\t", f"{random_similarity:.3f}", "\t", f"{semantica_similarity:.3f}", "\t", f"{semantica_similarity - random_similarity:.3f}")
        
        semantica_similarity = euclidean_distance(user_emb, semantica_neighbors_avg_emb)
        random_similarity = euclidean_distance(user_emb, random_neighbors_avg_emb)
        ed_diffs.append(semantica_similarity - random_similarity)
        
        print(padded_user_id, "\t", "ed", "\t", f"{random_similarity:.3f}", "\t", f"{semantica_similarity:.3f}", "\t", f"{semantica_similarity - random_similarity:.3f}")
finally:
    print("\nAverage differences:")
    print(f"Average cosine similarity difference: {np.mean(cs_diffs):.3f}")
    print(f"Average euclidean distance difference: {np.mean(ed_diffs):.3f}")