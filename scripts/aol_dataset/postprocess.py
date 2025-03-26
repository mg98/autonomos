import pandas as pd

df = pd.read_pickle('aol_raw_dataset.pkl')
df = pd.read_csv('aol_dataset.csv', keep_default_na=False, parse_dates=[1])
df = df.sort_values('time')
df = df.drop_duplicates(subset=['user_id', 'query'], keep='last')

# filter out users with less than 5 clicks
user_counts = df['user_id'].value_counts()
eligible_users = user_counts[user_counts >= 5].index
df = df[df['user_id'].isin(eligible_users)]

print("Saving processed dataset to pickle file...")
df.to_pickle('aol_dataset.pkl')
print("Success!")
