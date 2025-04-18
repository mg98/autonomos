# %%
from autonomos.utils.cache import Cache
from autonomos.utils.db import get_ctrs
from autonomos.dart.rank import evaluate, tracincp_valuation
from autonomos.dart.utils import split_dataset_by_qids
from allrank.config import Config
from autonomos.dart.types import SplitDataset
from autonomos.utils.attack import poison_ctrs, flip_label
import torch
from torch.utils.data import DataLoader
from autonomos.dart.rank import loss_batch_gradients
from autonomos.dart.types import SplitDataset, ClickThroughRecord
from allrank.config import Config
from autonomos.dart.rank import Model
import shutil
import uuid
import os
from copy import deepcopy
from autonomos.dart.utils import write_records, normalize_features
from allrank.models.model_utils import get_torch_device
from tqdm import tqdm
from typing import List, Dict, Type
from torch import Tensor
import numpy as np
import random
from autonomos.dart.tracin import vectorized_calculate_tracin_score
from autonomos.datasets.aol import load_dataset

df = load_dataset(50)
cache = Cache()
cache.set("user_ids", df["user_id"].unique())
del df

for seed in range(10):
    print(f"========= SEED: {seed} =========")
    random.seed(seed)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if hasattr(torch, 'mps'):
        torch.mps.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    config = Config.from_json("./allRank_config.json")
    config.data.batch_size = 256
    user_ids = cache.get("user_ids")
    user_id = user_ids[0]
    ctrs = get_ctrs(user_id)
    user_ds = split_dataset_by_qids(ctrs, train_ratio=1/5, val_ratio=1/5)

    # Sample user IDs randomly from the available user_ids
    sampled_user_ids = random.sample(list(user_ids), 4)


    candidate_datasets = {
        f'good{i}': get_ctrs(user_id) for i, user_id in enumerate(sampled_user_ids[:-2])
    }
    candidate_datasets.update({
        f'evil{i}': poison_ctrs(flip_label, get_ctrs(user_id)) for i, user_id in enumerate(sampled_user_ids[-2:])
    })

    # Shuffle the order of candidate_datasets items
    items = list(candidate_datasets.items())
    random.shuffle(items)
    candidate_datasets = dict(items)

    # print("Total number of CTRs:")
    # print(f"User train: {len(user_ds.train)}")
    # print(f"User vali: {len(user_ds.vali)}")
    # print(f"User test: {len(user_ds.test)}")

    # Print total across all datasets
    total_ctrs = len(user_ds.train) + len(user_ds.vali) + len(user_ds.test) + sum(len(dataset) for dataset in candidate_datasets.values())

    # TRAIN CTRS: user first, then candidate datasets sequentially
    train_ctrs = deepcopy(user_ds.train)
    for ds in candidate_datasets.values():
        train_ctrs.extend(ds)

    ds_split = SplitDataset(
        train=train_ctrs,
        vali=user_ds.vali,
        test=user_ds.test
    )

    ds_split_for_checkpoints = deepcopy(ds_split)
    ds_split_for_checkpoints.shuffle()

    dataset_path = f'.tmp/{uuid.uuid4().hex}'
    write_records(dataset_path, ds_split_for_checkpoints)
    normalize_features(dataset_path)
    config.data.path = os.path.join(dataset_path, '_normalized')

    dart = Model(deepcopy(config))
    shutil.rmtree(dataset_path)
    result = dart.train(trace=True)
    checkpoints = result['checkpoints']

    score_matrix = vectorized_calculate_tracin_score(
        dart.model,
        dart.loss_func,
        list(map(lambda x: x['state_dict'], checkpoints)),
        dart.train_dl,
        dart.test_dl,
        0.0001,
        config.data.slate_length,
        get_torch_device(),
        use_nested_loop_for_dot_product=False,
        float_labels=True
    )

    assert len(user_ds.train) % 10 == 0, "user_ds.train should be divisible by 10"
    start = len(user_ds.train) // 10

    print("DATASET SCORES")
    all_ctrs = user_ds.train
    keep_ctrs = user_ds.train
    for name, dataset in candidate_datasets.items():
        score = score_matrix[start:start+len(dataset)//10].sum().item()
        print(name, score)
        start += len(dataset)//10

        all_ctrs.extend(dataset)
        if score > 0:
            keep_ctrs.extend(dataset)

    
    config.data.batch_size = 16
    mrr_before = evaluate(config, SplitDataset(train=all_ctrs, vali=user_ds.vali, test=user_ds.test))
    mrr_after = evaluate(config, SplitDataset(train=keep_ctrs, vali=user_ds.vali, test=user_ds.test))
    print(f"MRR BEFORE: {mrr_before} / MRR AFTER: {mrr_after}")
