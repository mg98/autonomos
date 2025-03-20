import os
import numpy as np
import pandas as pd
from autonomos.dart.types import ClickThroughRecord, Dataset, SplitDataset
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

def compute_feature_stats(sample: list[ClickThroughRecord]):
    """
    Compute means and standard deviations of features in a sample.
    """
    features = np.array([ctr.feat.features for ctr in sample])
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    return means, stds

def normalize_features(ds_path: str, means: np.array = None, stds: np.array = None):
    """
    Normalize features in the dataset.
    Adapted from https://github.com/allegro/allRank/blob/master/reproducibility/normalize_features.py.

    Args:
        ds_path: Path to the dataset
        means: List of means for each feature; if None, means will be computed from the dataset
        stds: List of standard deviations for each feature; if None, stds will be computed from the dataset
    """
    # features that are already between 0 and 1
    features_without_logarithm = list(range(27)) + [28, 32, 33] + list(range(34, 61)) + [62, 66, 67] + list(range(69, 69+2*768))
    # features that are negative
    features_negative = []
    
    x_train, y_train, query_ids_train = load_svmlight_file(os.path.join(ds_path, "train.txt"), query_id=True)
    x_test, y_test, query_ids_test = load_svmlight_file(os.path.join(ds_path, "test.txt"), query_id=True)
    x_vali, y_vali, query_ids_vali = load_svmlight_file(os.path.join(ds_path, "vali.txt"), query_id=True)

    x_train_transposed = x_train.toarray().T
    x_test_transposed = x_test.toarray().T
    x_vali_transposed = x_vali.toarray().T

    x_train_normalized = np.zeros_like(x_train_transposed)
    x_test_normalized = np.zeros_like(x_test_transposed)
    x_vali_normalized = np.zeros_like(x_vali_transposed)

    eps_log = 1e-2
    eps = 1e-6

    for i, feat in enumerate(x_train_transposed):
        feature_vector_train = feat
        feature_vector_test = x_test_transposed[i, ]
        feature_vector_vali = x_vali_transposed[i, ]

        if i in features_negative:
            feature_vector_train = (-1) * feature_vector_train
            feature_vector_test = (-1) * feature_vector_test
            feature_vector_vali = (-1) * feature_vector_vali

        if i not in features_without_logarithm:
            # log only if all values >= 0
            if np.all(feature_vector_train >= 0) & np.all(feature_vector_test >= 0) & np.all(feature_vector_vali >= 0):
                feature_vector_train = np.log(feature_vector_train + eps_log)
                feature_vector_test = np.log(feature_vector_test + eps_log)
                feature_vector_vali = np.log(feature_vector_vali + eps_log)
            else:
                print("Some values of feature no. {} are still < 0 which is why the feature won't be normalized".format(i))

        mean = np.mean(feature_vector_train) if means is None or means[i] is None else means[i]
        std = np.std(feature_vector_train) if stds is None or stds[i] is None else stds[i]
        feature_vector_train = (feature_vector_train - mean) / (std + eps)
        feature_vector_test = (feature_vector_test - mean) / (std + eps)
        feature_vector_vali = (feature_vector_vali - mean) / (std + eps)
        x_train_normalized[i, ] = feature_vector_train
        x_test_normalized[i, ] = feature_vector_test
        x_vali_normalized[i, ] = feature_vector_vali

    ds_normalized_path = os.path.join(ds_path, "_normalized")
    os.makedirs(ds_normalized_path, exist_ok=True)

    train_normalized_path = os.path.join(ds_normalized_path, "train.txt")
    with open(train_normalized_path, "w"):
        dump_svmlight_file(x_train_normalized.T, y_train, train_normalized_path, query_id=query_ids_train)

    test_normalized_path = os.path.join(ds_normalized_path, "test.txt")
    with open(test_normalized_path, "w"):
        dump_svmlight_file(x_test_normalized.T, y_test, test_normalized_path, query_id=query_ids_test)

    vali_normalized_path = os.path.join(ds_normalized_path, "vali.txt")
    with open(vali_normalized_path, "w"):
        dump_svmlight_file(x_vali_normalized.T, y_vali, vali_normalized_path, query_id=query_ids_vali)

def write_records(export_path: str, split_ds: SplitDataset):
    os.makedirs(export_path, exist_ok=True)
    for role, records in split_ds.to_dict().items():
        with open(f"{export_path}/{role}.txt", "w") as f:
            f.writelines(str(record) + "\n" for record in records)

def split_by_qids(records: list[ClickThroughRecord], context_ratio=0.8) -> Dataset:
    """
    Split records into context and test sets based on query IDs.

    Args:
        records: list containing the records
        context_ratio: Proportion of data for context (default 0.8)
    """
    records_df = pd.DataFrame([record.to_dict() for record in records])
    qids = records_df['qid'].unique()
    n_qids = len(qids)
    n_context = int(context_ratio * n_qids)
    context_qids = qids[:n_context]
    test_qids = qids[n_context:]

    context_records = [record for record in records if record.qid in context_qids]
    test_records = [record for record in records if record.qid in test_qids]

    return Dataset(context_records, test_records)

def split_dataset_by_qids(records: list[ClickThroughRecord], train_ratio=0.8, val_ratio=0.1) -> SplitDataset:
    """
    Split records into train/validation/test sets based on query IDs.
    Test set ratio equals 1 - train_ratio - val_ratio.
    
    Args:
        records: list containing the records
        train_ratio: Proportion of data for training (default 0.8)
        val_ratio: Proportion of data for validation (default 0.1)
    """
    records_df = pd.DataFrame([record.to_dict() for record in records])
    qids = records_df['qid'].unique()
    
    # Calculate split sizes
    n_qids = len(qids)
    train_size = int(train_ratio * n_qids)
    val_size = int(val_ratio * n_qids)
    
    # Split qids into train/val/test
    train_qids = qids[:train_size]
    val_qids = qids[train_size:train_size+val_size]
    test_qids = qids[train_size+val_size:]
    
    # Filter records by qid
    train_records_df = records_df[records_df['qid'].isin(train_qids)]
    val_records_df = records_df[records_df['qid'].isin(val_qids)]
    test_records_df = records_df[records_df['qid'].isin(test_qids)]
    
    # Convert to ClickThroughRecord objects
    train_records = [ClickThroughRecord(**record) for _, record in train_records_df.iterrows()]
    val_records = [ClickThroughRecord(**record) for _, record in val_records_df.iterrows()]
    test_records = [ClickThroughRecord(**record) for _, record in test_records_df.iterrows()]
    
    return SplitDataset(train_records, val_records, test_records)