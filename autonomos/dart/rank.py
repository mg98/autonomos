import os
import shutil
import uuid
import random
import sys

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, load_libsvm_dataset_role, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.training.train_utils import fit, loss_batch_gradients
from allrank.utils.python_utils import dummy_context_mgr

from joblib import Parallel, delayed
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from argparse import Namespace
from attr import asdict
from functools import partial
from torch import optim
from copy import deepcopy
import numpy as np
from autonomos.dart.types import FeatureVector, Dataset, SplitDataset, ClickThroughRecord
from autonomos.dart.utils import normalize_features, write_records
from autonomos.utils.cache import Cache

dev = get_torch_device()

def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    model_size_kb = total_params * 4 / 1024  # Assuming 4 bytes per parameter
    print(f"Model size: {model_size_kb:.2f} KB ({total_params:,} parameters)")

class Model:
    def __init__(self, config: Config):
        self.config = config
        n_features = FeatureVector.n_features()
        self.model = make_model(n_features=n_features, **asdict(config.model, recurse=False))
        self.model.to(dev)
        self.optimizer = getattr(optim, config.optimizer.name)(params=self.model.parameters(), **config.optimizer.args)
        self.loss_func = partial(getattr(losses, config.loss.name), **config.loss.args)
        if config.lr_scheduler.name:
            self.scheduler = getattr(optim.lr_scheduler, config.lr_scheduler.name)(self.optimizer, **config.lr_scheduler.args)
        else:
            self.scheduler = None

        self.train_ds, self.val_ds = load_libsvm_dataset(
            input_path=self.config.data.path,
            slate_length=self.config.data.slate_length,
            validation_ds_role=self.config.data.validation_ds_role,
        )
        self.train_dl, self.val_dl = create_data_loaders(
            self.train_ds, self.val_ds, 
            num_workers=self.config.data.num_workers, 
            batch_size=self.config.data.batch_size)

        self.test_ds = load_libsvm_dataset_role("test", self.config.data.path, self.config.data.slate_length)
        self.test_dl = DataLoader(self.test_ds, batch_size=self.config.data.batch_size, num_workers=self.config.data.num_workers)
    
    def train(self, trace: bool = False):
        self.model.train()
        
        with torch.autograd.detect_anomaly() if self.config.detect_anomaly else dummy_context_mgr():  # type: ignore
            result = fit(
                model=self.model,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_dl=self.train_dl,
                valid_dl=self.val_dl,
                config=self.config,
                device=dev,
                trace=trace,
                **asdict(self.config.training)
            )
        
        return result

    def test(self):
        self.model.eval()

        mrrs = []

        with torch.no_grad():
            for xb, yb, indices in self.test_dl:

                X = xb.type(torch.float32).to(device=dev)
                y_true = yb.to(device=dev)
                indices = indices.to(device=dev)

                input_indices = torch.ones_like(y_true).type(torch.long)
                mask = (y_true == losses.PADDED_Y_VALUE)
                scores = self.model.score(X, mask, input_indices)

                # Iterate over each query in the batch
                for i in range(scores.size(0)):
                    slate_scores = scores[i]
                    slate_indices = indices[i]
                    slate_mask = mask[i]
                    
                    valid_scores = slate_scores[~slate_mask]
                    valid_indices = slate_indices[~slate_mask]
                    
                    # Compute the rankings
                    _, sorted_idx = torch.sort(valid_scores, descending=True)
                    sorted_original_indices = valid_indices[sorted_idx]

                    # Compute MRR
                    relevant_positions = (y_true[i][~slate_mask] == 1).nonzero()
                    if len(relevant_positions) > 0:
                        # Get position of first relevant doc in sorted results
                        first_rel_sorted_pos = (sorted_idx == relevant_positions[0]).nonzero()[0]
                        # MRR is 1/(rank) where rank starts at 1
                        mrr = 1.0 / (first_rel_sorted_pos.item() + 1)
                    else:
                        mrr = 0.0
                    mrrs.append(mrr)

        return np.mean(mrrs)

def evaluate(config: Config, ds: SplitDataset):
    config = deepcopy(config)
    ds.shuffle()
    dataset_path = f'.tmp/{uuid.uuid4().hex}'
    write_records(dataset_path, ds)
    normalize_features(dataset_path)
    config.data.path = os.path.join(dataset_path, '_normalized')

    dart = Model(config)
    shutil.rmtree(dataset_path)
    dart.train()
    return dart.test()

def tracincp_valuation(config: Config, user_ds: SplitDataset, candidate_datasets: dict[str,list[ClickThroughRecord]]) -> dict[str, float]:
    config = deepcopy(config)
    dataset_path = f'.tmp/{uuid.uuid4().hex}'
    
    mapped_train_ctrs = [
        (user_id, ctr) for user_id in candidate_datasets for ctr in candidate_datasets[user_id]
    ]
    mapped_train_ctrs.extend([
        ('0', ctr) for ctr in user_ds.train
    ])

    ds_split = SplitDataset(
        train=[ctr for _, ctr in mapped_train_ctrs],
        vali=user_ds.vali,
        test=user_ds.test
    )
    ds_split.shuffle()
    write_records(dataset_path, ds_split)
    normalize_features(dataset_path)
    config.data.path = os.path.join(dataset_path, '_normalized')

    dart = Model(config)
    shutil.rmtree(dataset_path)
    result = dart.train(trace=True)
    
    influence_scores = torch.zeros(len(dart.train_ds), len(dart.test_ds))

    train_loader = DataLoader(dart.train_ds)
    test_loader = DataLoader(dart.test_ds)

    for checkpoint in tqdm(result['checkpoints'][::5]):
            
        w = checkpoint['state_dict']
        lr = checkpoint['lr']
        dart.model.load_state_dict(w)

        # precompute test gradients
        checkpoint_test_grads = []
        for j, (test_xb, test_yb, test_indices) in enumerate(test_loader):
            test_xb, test_yb, test_indices = test_xb.to(dev), test_yb.to(dev), test_indices.to(dev)
            test_grads = loss_batch_gradients(dart.model, dart.loss_func, test_xb, test_yb, test_indices)
            if test_grads is None:
                continue
            checkpoint_test_grads.append(test_grads)
        
        for i, (xb, yb, indices) in enumerate(train_loader):
            xb, yb, indices = xb.to(dev), yb.to(dev), indices.to(dev)
            train_grads = loss_batch_gradients(dart.model, dart.loss_func, xb, yb, indices)
            if train_grads is None:
                continue
            
            for test_grads in checkpoint_test_grads:
                # Calculate dot product of train_grads and test_grads
                dot_product = 0.0
                for train_grad, test_grad in zip(train_grads, test_grads):
                    dot_product += torch.sum(train_grad * test_grad).item()
                dot_product *= lr

                influence_scores[i,j] += dot_product

    # Average over both test samples (j) and checkpoints (idx)
    influence_scores = influence_scores.mean(dim=(1))

    dataset_scores = {
        id: [] for id, _ in mapped_train_ctrs
    }
    # Loop through influence scores and map them to dataset IDs
    for i, score in enumerate(influence_scores):
        dataset_id, _ = mapped_train_ctrs[i]
        dataset_scores[dataset_id].append(score.item())
    
    # Calculate average influence score per dataset
    avg_dataset_scores = {}
    for dataset_id, scores in dataset_scores.items():
        if scores:  # Only calculate average if there are scores
            avg_dataset_scores[dataset_id] = sum(scores) / len(scores)
        else:
            avg_dataset_scores[dataset_id] = 0.0
    
    return avg_dataset_scores
        


# def shapley_valuation(config: Config, user_ds: Dataset, candidate_datasets: dict[str,list[ClickThroughRecord]], num_samples: int) -> dict[str, float]:
#     shapley_values = {user_id: 0.0 for user_id in candidate_datasets.keys()}  # Initialize Shapley value estimates
#     perms = [random.sample(list(candidate_datasets.keys()), k=len(candidate_datasets)) for _ in range(num_samples)]

#     for perm in perms:
#         subset = []
#         prev_value = 0.0
#         for user_id in perm:
#             x = candidate_datasets[user_id]
#             subset.extend(x)
#             new_value = evaluate(config, Dataset(user_ds.context + subset, user_ds.test))
#             marginal_contribution = new_value - prev_value
#             shapley_values[user_id] += marginal_contribution
#             prev_value = new_value
#     return shapley_values

def shapley_valuation(config: Config, user_ds: SplitDataset, candidate_datasets: dict[str,list[ClickThroughRecord]], num_samples: int) -> dict[str, float]:
    shapley_values = {user_id: 0.0 for user_id in candidate_datasets.keys()}
    perms = [random.sample(list(candidate_datasets.keys()), k=len(candidate_datasets)) for _ in range(num_samples)]
    
    def process_permutation(perm, candidate_datasets, config, user_ds):
        subset = []
        prev_value = 0.0
        user_contributions = {user_id: 0.0 for user_id in candidate_datasets.keys()}
        
        for user_id in perm:
            x = candidate_datasets[user_id]
            subset.extend(x)
            
            new_value = evaluate(config, SplitDataset(user_ds.train + subset, user_ds.vali, user_ds.test))
            
            marginal_contribution = new_value - prev_value
            user_contributions[user_id] = marginal_contribution
            
            prev_value = new_value
        
        return user_contributions
    
    # Process permutations in parallel
    results = Parallel(n_jobs=8)(
        delayed(process_permutation)(perm, candidate_datasets, config, user_ds) 
        for perm in perms
    )
    
    # Aggregate results
    for result in results:
        for user_id, contribution in result.items():
            shapley_values[user_id] += contribution

    return shapley_values