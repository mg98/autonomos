import pandas as pd

def load_dataset(threshold: int = 3) -> pd.DataFrame:
    df = pd.read_pickle('data/aol_dataset.pkl')
    
    user_query_counts = df.groupby('user_id').size()
    users_with_enough_queries = user_query_counts[user_query_counts >= threshold].index
    df = df[df['user_id'].isin(users_with_enough_queries)]
    
    return df
