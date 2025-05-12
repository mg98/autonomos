import pandas as pd
import numpy as np
from swarmsearch.dart.types import ClickThroughRecord, FeatureVector, bounded_string_to_int_hash
from joblib import Parallel, delayed
from tqdm import tqdm
import ir_datasets
from ir_datasets.datasets.aol_ia import AolIaDoc
import warnings

def compile_clickthrough_records_as_arrays(df: pd.DataFrame, parallel: bool = False) -> list[np.ndarray]:
    ctrs = compile_clickthrough_records(df, parallel)
    return [ctr.to_array() for ctr in ctrs]

def compile_clickthrough_records(df: pd.DataFrame, parallel: bool = False) -> list[ClickThroughRecord]:
    """
    Compile raw clickthrough records into LTR-format records.

    Args:
        df: Dataframe containing the clickthrough records
        queries_df: Dataframe containing all queries
        docs_df: Dataframe containing all documents
        parallel: Whether to use parallel processing for acceleration
    """
    def process_row(row: pd.Series) -> ClickThroughRecord:
        dataset = ir_datasets.load("aol-ia")
        docs_store = dataset.docs_store()
        candidate_docs: list[AolIaDoc] = []

        # Create candidate docs
        for docid in row['candidate_doc_ids']:
            doc = docs_store.get(docid)
            candidate_docs.append(doc)

        # Create clickthrough records for this row
        ctrs = []
        for doc in candidate_docs:
            ctr = ClickThroughRecord(
                doc.doc_id == row['doc_id'], 
                bounded_string_to_int_hash(row['query']), 
                FeatureVector.make(candidate_docs, doc, row['query'])
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