from datasets.aol import load_dataset
from dart.utils import normalize_features
from utils.data import compile_clickthrough_records
from dart.utils import split_dataset_by_qids, write_records

if __name__ == "__main__":
    print("Loading data...")
    df, queries_df, docs_df = load_dataset('./AOL4PS')
    print("Generating clickthrough records...")
    print(f"Dataset size: {len(df)} rows")
    # Sample 10000 records from the dataframe
    df = df.sample(n=10000, random_state=42)
    ctrs = compile_clickthrough_records(df, queries_df, docs_df, parallel=True)
    dataset = split_dataset_by_qids(ctrs)
    write_records('./out', dataset)
    normalize_features('./out')
