import os
import shutil
import uuid
import random

import allrank.models.losses as losses
from allrank.config import Config
from allrank.data.dataset_loading import load_libsvm_dataset, load_libsvm_dataset_role, create_data_loaders
from allrank.models.model import make_model
from allrank.models.model_utils import get_torch_device
from allrank.training.train_utils import fit
from allrank.utils.python_utils import dummy_context_mgr

import torch
from torch.utils.data import DataLoader
from argparse import Namespace
from attr import asdict
from functools import partial
from torch import optim
from copy import deepcopy
import numpy as np
from dart.types import FeatureVector, Dataset
from dart.utils import normalize_features, write_records

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
    
    def train(self):
        self.model.train()

        train_ds, val_ds = load_libsvm_dataset(
            input_path=self.config.data.path,
            slate_length=self.config.data.slate_length,
            validation_ds_role=self.config.data.validation_ds_role,
        )
        train_dl, val_dl = create_data_loaders(
            train_ds, val_ds, 
            num_workers=self.config.data.num_workers, 
            batch_size=self.config.data.batch_size)
        
        with torch.autograd.detect_anomaly() if self.config.detect_anomaly else dummy_context_mgr():  # type: ignore
            fit(
                model=self.model,
                loss_func=self.loss_func,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                train_dl=train_dl,
                valid_dl=val_dl,
                config=self.config,
                device=dev,
                **asdict(self.config.training)
            )

    def test(self):
        self.model.eval()

        test_ds = load_libsvm_dataset_role("test", self.config.data.path, self.config.data.slate_length)
        test_dl = DataLoader(test_ds, batch_size=self.config.data.batch_size, num_workers=self.config.data.num_workers)
        
        mrrs = []

        with torch.no_grad():
            for xb, yb, indices in test_dl:

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

def evaluate(config: Config, ds: Dataset, feature_means: np.array, feature_stds: np.array):
    config = deepcopy(config)
    dataset_path = f'.tmp/{uuid.uuid4().hex}'
    write_records(dataset_path, ds.split())
    normalize_features(dataset_path, feature_means, feature_stds)
    config.data.path = os.path.join(dataset_path, '_normalized')

    try:
        dart = Model(config)
        dart.train()
        return dart.test()
    finally:
        # print(dataset_path)
        shutil.rmtree(dataset_path)