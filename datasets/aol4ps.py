from dataclasses import dataclass
import pandas as pd
from os.path import join as pjoin
import numpy as np
import ir_datasets

@dataclass
class Document:
    """
    AOL4PS document.
    """
    id: str
    title: str
    url: str
    # text_embedding: np.ndarray

def update_doc_titles(docs_df: pd.DataFrame):
    # Update Title in all records of docs_df, as AOL-IA has more accurate titles
    dataset = ir_datasets.load("aol-ia")
    # Create a hashmap for fast lookup
    url_to_title = {}
    for doc in dataset.docs_iter():
        url_to_title[doc.url] = doc.title
    
    # Create a reverse mapping from docs_df for faster access
    url_to_idx = {}
    for idx, url in zip(docs_df.index, docs_df['Url']):
        if url not in url_to_idx:
            url_to_idx[url] = []
        url_to_idx[url].append(idx)
    
    # Update titles using the hashmaps
    for url, title in url_to_title.items():
        if url in url_to_idx:
            for idx in url_to_idx[url]:
                if title != "":
                    docs_df.at[idx, 'Title'] = title

def load_dataset(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load AOL4PS dataset filtered by:
    - users with at least 5 clicks
    - docs that have a corresponding AOL-IA document
    - logs that point to docs (incl. candidate list) that have a corresponding AOL-IA document
    - no (user,query) duplicates
    """
    df = pd.read_csv(pjoin(path, 'data.csv'), sep='\t')
    queries_df = pd.read_csv(pjoin(path, 'query.csv'), sep='\t', index_col=1)
    docs_df = pd.read_csv(pjoin(path, 'doc.csv'), sep='\t', index_col=1)

    update_doc_titles(docs_df)

    df = df.drop_duplicates(subset=['AnonID', 'QueryIndex'], keep='last') # TODO: How to handle duplicates?
    df = df.sort_values('QueryTime')

    # filter out users with less than 5 clicks
    user_counts = df['AnonID'].value_counts()
    eligible_users = user_counts[user_counts >= 5].index
    # sampled_users = np.random.choice(eligible_users, size=min(1000, len(eligible_users)), replace=False)
    df = df[df['AnonID'].isin(eligible_users)]

    # filter out docs that don't have a corresponding AOL-IA document
    aolia_dataset = ir_datasets.load("aol-ia")
    aolia_urls = {doc.url for doc in aolia_dataset.docs_iter()}
    docs_df = docs_df[docs_df['Url'].isin(aolia_urls)]

    # filter out data that points to docs that we had to filter out
    df = df[df['DocIndex'].isin(docs_df.index)]

    # filter out data where CandiList contains documents that were filtered out
    def check_candilist(candilist):
        doc_ids = candilist.strip('"').split('\t')
        return all(doc_id in docs_df.index for doc_id in doc_ids)
    
    df = df[df['CandiList'].apply(check_candilist)]

    return df, queries_df, docs_df

