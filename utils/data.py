import pandas as pd
import numpy as np
from dart.types import ClickThroughRecord, FeatureVector
from datasets.aol4ps import Document
from joblib import Parallel, delayed
from tqdm import tqdm

def compile_clickthrough_records(df: pd.DataFrame, queries_df: pd.DataFrame, docs_df: pd.DataFrame, parallel: bool = False) -> list[ClickThroughRecord]:
    """
    Compile raw clickthrough records into LTR-format records.

    Args:
        df: Dataframe containing the clickthrough records
        queries_df: Dataframe containing all queries
        docs_df: Dataframe containing all documents
        parallel: Whether to use parallel processing for acceleration
    """
    
    def process_row(row: pd.Series) -> ClickThroughRecord:
        candidate_docids = row['CandiList'].split('\t')
        query = queries_df.loc[row['QueryIndex']]['Query'].strip().lower()
        candidate_docs = []

        # Create candidate docs
        for docid in candidate_docids:
            doc_record = docs_df.loc[docid]
            doc = Document(
                id=docid,
                title=doc_record['Title'].strip().lower(),
                body=doc_record['Body'].strip().lower(),
                url=doc_record['Url'].strip().lower()
            )
            candidate_docs.append(doc)

        # Create clickthrough records for this row
        ctrs = []
        for pos, doc in enumerate(candidate_docs):
            ctr = ClickThroughRecord(
                pos == row['ClickPos'],
                row['QueryIndex'],
                FeatureVector.make(candidate_docs, doc, query, row['QueryIndex'], row['AnonID'])
            )
            ctrs.append(ctr)
        
        return ctrs
    
    if parallel:
        ctrs = Parallel(n_jobs=-1, batch_size=1024)(
            delayed(process_row)(row) for _, row in tqdm(df.iterrows(), total=len(df))
        )
    else:
        ctrs = [process_row(row) for _, row in df.iterrows()]
    
    return [ctr for row_records in ctrs for ctr in row_records]