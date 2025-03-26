import pandas as pd

def load_dataset() -> pd.DataFrame:
    df = pd.read_pickle('data/aol_dataset.pkl')
    return df
