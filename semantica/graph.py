import pandas as pd
import os

def get_neighbors(user_id: str) -> list[str]:
    """
    Get neighbors of user.
    """
    csv_path = "semantica/directed_links_from_semantica_tree_delta003_exprounds10_new.csv"
    links_df = pd.read_csv(csv_path, header=0, names=['source', 'target'])
    neighbors = links_df[links_df['source'] == user_id]['target'].tolist()
    return neighbors
