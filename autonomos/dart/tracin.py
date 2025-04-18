import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from allrank.data.dataset_loading import PADDED_Y_VALUE
from typing import List, Dict, Type


def precompute_test_gradients(
        models: List[nn.Module],
        criterion: nn.Module,
        test_dataloader: DataLoader,
        device: torch.device,
        float_labels: bool
) -> Dict[int, Tensor]:
    """
    Precompute gradients of the test samples with respect to the model parameters.

    Args:
        models (List[nn.Module]): List of PyTorch models.
        criterion (nn.Module): Loss function.
        test_dataloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the computations on (e.g., torch.device('cuda') or torch.device('cpu')).
        float_labels (bool): Whether the labels are floating point numbers.

    Returns:
        List[torch.Tensor, ...]: List of tuples containing the gradients for each model.
    """
    test_gradients = {}

    for index, model in tqdm(
        enumerate(models), desc="--> Precompute test gradients using checkpoints"
    ):
        test_grads = []
        for x_test, y_test, indices in test_dataloader:
            x_test = x_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)

            mask = (y_test == PADDED_Y_VALUE)
            y_pred_test = model(x_test, mask, indices)
            losses_test = criterion(
                y_pred_test, y_test.float() if float_labels else y_test.long(), reduction='none'
            )

            for i in range(len(losses_test)):
                model.zero_grad()
                losses_test[i].backward(retain_graph=True)
                test_grad = torch.cat(
                    [param.grad.reshape(-1) for param in model.parameters()]
                ).detach()
                test_grads.append(test_grad)

        test_grads = torch.stack(test_grads)  # [total_test_samples, num_params]
        test_gradients[index] = test_grads

    return test_gradients


def load_checkpoint(fname: str) -> nn.Module:
    with open(fname, "rb") as f:
        if torch.cuda.is_available():
            ckpt = torch.load(f, map_location=lambda storage, loc: storage.cuda())
        else:
            ckpt = torch.load(f, map_location=torch.device("cpu"))

    return ckpt


def precompute_models_with_weights(
    model_class: Type[nn.Module],
    state_dicts: List,
    device: torch.device
) -> List[nn.Module]:
    """
    Load models with specified weights and set them to evaluation mode.

    Args:
        model_class (Type[nn.Module]): The class of the model to instantiate.
        weights_paths (List[str]): List of paths to the weight files.
        device (torch.device): Device to load the models on (e.g., torch.device('cuda') or torch.device('cpu')).

    Returns:
        List[nn.Module]: List of models loaded with the specified weights.
    """
    models = []
    for state_dict in state_dicts:
        model = model_class.to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models


def calculate_train_grads_per_checkpoint(
    models: List[nn.Module],
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    indices: torch.Tensor,
    float_labels: bool,
    criterion: nn.Module
) -> List[torch.Tensor]:
    """
    Calculate gradients of the training samples with respect to the model parameters
    for each checkpoint model.

    Args:
        models (List[nn.Module]): List of PyTorch models.
        x_train (torch.Tensor): Training input data.
        y_train (torch.Tensor): Training labels.
        indices (torch.Tensor): Indices of the training samples.
        float_labels (bool): Whether the labels are floating point numbers.
        criterion (nn.Module): Loss function.

    Returns:
        List[torch.Tensor]: List of gradients for each checkpoint model. Each tensor in
                            the list has shape [train_batch_size, num_params].
    """

    checkpoints_train_grads = []
    for index, model in enumerate(models):
        mask = (y_train == PADDED_Y_VALUE)
        y_pred_train = model(x_train, mask, indices)
        losses_train = criterion(
            y_pred_train, y_train.float() if float_labels else y_train.long(), reduction='none'
        )

        train_grads = []
        for i in range(len(losses_train)):
            model.zero_grad()
            losses_train[i].backward(retain_graph=True)  # Backward pass
            train_grad = torch.cat(
                [param.grad.reshape(-1) for param in model.parameters()]
            ).detach()
            train_grads.append(train_grad)

        train_grads = torch.stack(train_grads)  # [train_batch_size, num_params]
        checkpoints_train_grads.append(
            train_grads
        )  # [num_checkpoints, train_batch_size, num_params]

    return checkpoints_train_grads


def vectorized_calculate_tracin_score(
    model: torch.nn.Module,
    criterion,
    state_dicts,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    lr,
    slate_length,
    device,
    use_nested_loop_for_dot_product: bool = False,
    float_labels: bool = False,
) -> Tensor:
    """
    :param model: your model architecture
    :param criterion: your criterion with reduction=none as a mandatory step
    :param state_dicts: your checkpoints for particular architecture
    :param train_dataloader: your train loader
    :param test_dataloader: your test loader
    :param lr: scaling for dor product between train and test gradients
    :param slate_length: number of documents per query (slate size)
    :param device: your device
    :param use_nested_loop_for_dot_product: either einsum or simpler nested loop
    :param float_labels: either float or int labels
    :return: score matrix with shape [train_samples, test_samples], where each value is accumulated scaled sum of dot products between different checkpoints

    Preamble:
    grad_sum = None
    output = lr * torch.einsum('if,jf->ij', train_grads, test_grads)  # [train_batch_size, test_batch_size]
    grad_sum += output

    is the same as

    grad_sum = None
    for i in range(train_grads.size(0)):
        for j in range(test_grads.size(0)):
            grad_sum[i, j] += lr * torch.dot(train_grads[i], test_grads[j]).item()

    """
    
    num_train_queries = len(train_dataloader.dataset)
    num_test_queries = len(test_dataloader.dataset)
    num_train_docs = num_train_queries * slate_length
    num_test_docs = num_test_queries * slate_length
    
    score_matrix = torch.zeros(
        (num_train_queries, num_test_queries) # Initialize based on document counts
    )
    actual_train_batch_size = train_dataloader.batch_size # Queries per batch
    actual_test_batch_size = test_dataloader.batch_size # Queries per batch

    # Precompute models with weights
    models = precompute_models_with_weights(model, state_dicts, device)

    # Precompute test gradients (per document)
    # Assumes precompute_test_gradients returns Dict[chkpt_idx, Tensor(total_test_docs, num_params)]
    test_gradients = precompute_test_gradients(
        models, criterion, test_dataloader, device, float_labels
    )

    for train_id, (x_train, y_train, indices) in tqdm(
        enumerate(train_dataloader), desc="--> Calculate train influences (per doc)...", total=len(train_dataloader)
    ):
        x_train = x_train.to(device, non_blocking=True)
        y_train = y_train.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)

        # Precalculate train data gradients per each checkpoint (per document)
        # checkpoints_train_grads: List[Tensor(docs_in_batch, num_params)]
        checkpoints_train_grads = calculate_train_grads_per_checkpoint(
            models=models,
            x_train=x_train,
            y_train=y_train,
            indices=indices,
            float_labels=float_labels,
            criterion=criterion,
        )
        
        # Iterate through all test data batches
        for test_id in range(len(test_dataloader)):
            # Initialization of grad_sum for particular test data batch
            grad_sum = None

            # Iterate through all models checkpoints
            for index, model in enumerate(models):
                # Use precomputed train gradients under particular model checkpoint
                train_grads = checkpoints_train_grads[index]

                # Use precomputed test gradients under particular model checkpoint
                test_grads = test_gradients[index][
                    test_id * actual_test_batch_size : (test_id + 1)
                    * actual_test_batch_size
                ]  # [test_batch_size, num_params]

                # Start accumulation of gradients among different checkpoints
                if grad_sum is None:
                    current_train_batch_size = train_grads.shape[0]
                    current_test_batch_size = test_grads.shape[0]
                    grad_sum = torch.zeros(
                        (current_train_batch_size, current_test_batch_size)
                    ).to(device)

                # Dot product between all train data gradients and test data gradients
                if use_nested_loop_for_dot_product:
                    for i in range(train_grads.size(0)):
                        for j in range(test_grads.size(0)):
                            # Accumulation of gradient sum per [current_train_batch_size, current_test_batch_size]
                            grad_sum[i, j] += (
                                lr * torch.dot(train_grads[i], test_grads[j]).item()
                            )
                else:
                    output = lr * torch.einsum(
                        "if,jf->ij", train_grads, test_grads
                    )  # [current_train_batch_size, current_test_batch_size]

                    # Accumulation of gradient sum per [current_train_batch_size, current_test_batch_size]
                    grad_sum += output

            train_start_idx = train_id * actual_train_batch_size
            train_end_idx = train_start_idx + actual_train_batch_size
            test_start_idx = test_id * actual_test_batch_size
            test_end_idx = test_start_idx + actual_test_batch_size

            # if train_id == len(train_dataloader) - 1:
            #     train_start_idx = score_matrix.shape[0] - current_train_batch_size
            #     train_end_idx = score_matrix.shape[0]

            # if test_id == len(test_dataloader) - 1:
            #     test_start_idx = score_matrix.shape[1] - current_test_batch_size
            #     test_end_idx = score_matrix.shape[1]

            # Filling accumulated gradient sum per particular slice of data
            score_matrix[
                train_start_idx:train_end_idx, test_start_idx:test_end_idx
            ] += grad_sum.cpu().numpy()

    return score_matrix.mean(dim=1)