import pandas as pd
import ast

def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv('data/aol_dataset.csv', keep_default_na=False, parse_dates=[1])
    df = df.sort_values('time')
    df = df.drop_duplicates(subset=['user_id', 'query'], keep='last')
    df['candidate_doc_ids'] = df['candidate_doc_ids'].apply(ast.literal_eval)

    # filter out users with less than 5 clicks
    user_counts = df['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= 5].index
    df = df[df['user_id'].isin(eligible_users)]

    return df

