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

dev = get_torch_device()

def precompute_test_gradients(dart: Model, checkpoints, test_dl: DataLoader) -> Dict[int, Tensor]:
    test_gradients = {}
    for index, checkpoint in tqdm(
        enumerate(checkpoints), desc="--> Precompute test gradients using checkpoints"
    ):
        dart.model.load_state_dict(checkpoint['state_dict'])
        dart.model.train()
        test_grads = []
        
        for test_xb, test_yb, test_indices in test_dl:
            test_xb = test_xb.to(dev, non_blocking=True)
            test_yb = test_yb.to(dev, non_blocking=True)
            test_indices = test_indices.to(dev, non_blocking=True)

            batch_grads = loss_batch_gradients(dart.model, dart.loss_func, test_xb, test_yb, test_indices)
            if batch_grads is None:
                continue
            test_grads.extend([grad.detach().cpu() for grad in batch_grads])

            # losses = loss_batch_gradients(dart.model, dart.loss_func, test_xb, test_yb, test_indices)

            # for i in range(len(losses)):
            #     dart.model.zero_grad()
            #     losses[i].backward(retain_graph=True)
            #     test_grad = torch.cat(
            #         [param.grad.reshape(-1) for param in dart.model.parameters()]
            #     ).detach()
            #     test_grads.append(test_grad)

            # test_grads = torch.stack(test_grads)  # [total_test_samples, num_params]
            # test_gradients[index] = test_grads
        
        test_grads = torch.stack(test_grads)  # [total_test_samples, num_params]
        test_gradients[index] = test_grads
    
    return test_gradients

def precompute_train_gradients_per_checkpoint(dart: Model, checkpoints, xb, yb, indices):
    checkpoints_train_grads = []
    for checkpoint in checkpoints:
        dart.model.load_state_dict(checkpoint['state_dict'])
        train_grads = loss_batch_gradients(dart.model, dart.loss_func, xb, yb, indices)
        if train_grads is None:
            continue
        checkpoints_train_grads.append(train_grads)
    return checkpoints_train_grads

def tracincp(config: Config, user_ds: SplitDataset, candidate_datasets: dict[str,list[ClickThroughRecord]]):
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
    checkpoints = result['checkpoints']

    train_dl = dart.train_dl
    test_dl = dart.test_dl

    score_matrix = torch.zeros((len(train_dl.dataset), len(test_dl.dataset)))
    num_test_samples = len(test_dl.dataset)

    all_test_grads = precompute_test_gradients(dart, checkpoints, test_dl)

    train_samples_processed = 0
    # TRAIN BATCHES
    for train_id, (xb, yb, indices) in tqdm(
        enumerate(train_dl), total=len(train_dl), desc="--> Calculate train..."
    ):
        xb = xb.to(dev, non_blocking=True)
        yb = yb.to(dev, non_blocking=True)
        indices = indices.to(dev, non_blocking=True)

        checkpoints_train_grads = precompute_train_gradients_per_checkpoint(dart, checkpoints, xb, yb, indices)

        # TEST BATCHES
        for test_id in range(len(test_dl)):
            grad_sum = None

            for index, checkpoint in tqdm(
                enumerate(checkpoints), total=len(checkpoints), desc="--> Precompute test gradients using checkpoints"
            ):
                
                train_grads = checkpoints_train_grads[index]
                
                test_grads = all_test_grads[index][
                    test_id * test_dl.batch_size : (test_id + 1) * test_dl.batch_size
                ]

                # Start accumulation of gradients among different checkpoints
                if grad_sum is None:
                    current_train_batch_size = train_grads.shape[0]
                    current_test_batch_size = test_grads.shape[0]
                    grad_sum = torch.zeros(
                        (current_train_batch_size, current_test_batch_size)
                    ).to(dev)

                # Dot product between all train data gradients and test data gradients
                train_grads = train_grads.to(dev)
                test_grads = test_grads.to(dev)
                print("train_grads shape:", train_grads.shape)
                print("test_grads shape:", test_grads.shape)
                output = checkpoint['lr'] * torch.einsum(
                    "if,jf->ij", train_grads, test_grads
                )
                grad_sum += output

            train_start_idx = train_samples_processed
            train_end_idx = train_start_idx + current_train_batch_size
            test_start_idx = test_id * test_dl.batch_size
            test_end_idx = test_start_idx + current_test_batch_size

            if train_id == len(train_dl) - 1:
                train_start_idx = score_matrix.shape[0] - current_train_batch_size
                train_end_idx = score_matrix.shape[0]

            if test_id == len(test_dl) - 1:
                test_start_idx = score_matrix.shape[1] - current_test_batch_size
                test_end_idx = score_matrix.shape[1]

            print("score_matrix slice shape:", score_matrix[train_start_idx:train_end_idx, test_start_idx:test_end_idx].shape)
            print("grad_sum shape:", grad_sum.shape)
            # Filling accumulated gradient sum per particular slice of data
            score_matrix[
                train_start_idx:train_end_idx, test_start_idx:test_end_idx
            ] += grad_sum.cpu().numpy()

        # Increment after processing all test batches for the current train batch
        train_samples_processed += current_train_batch_size

    return score_matrix