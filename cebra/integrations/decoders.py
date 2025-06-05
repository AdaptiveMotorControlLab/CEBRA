from typing import List, Optional, Tuple, Union

import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


def ridge_decoding(
    embedding_train: Union[torch.Tensor, dict],
    embedding_valid: Union[torch.Tensor, dict],
    label_train: Union[torch.Tensor, dict],
    label_valid: Union[torch.Tensor, dict],
    n_run: Optional[int] = None,
) -> Tuple[List[float], List[float], np.ndarray]:
    """
    Perform ridge regression decoding on training and validation embeddings.

    Args:
        embedding_train (Union[torch.Tensor, dict]): Training embeddings.
        embedding_valid (Union[torch.Tensor, dict]): Validation embeddings.
        label_train (Union[torch.Tensor, dict]): Training labels.
        label_valid (Union[torch.Tensor, dict]): Validation labels.
        n_run (Optional[int]): Optional run number for dataset definition.

    Returns:
        Training R2 scores, validation R2 scores, and validation predictions.
    """
    if isinstance(embedding_train, dict):  # only on run 1
        if n_run is None:
            raise ValueError(f"n_run must be specified, got {n_run}.")

        all_train_embeddings = np.concatenate(
            [
                embedding_train[i][n_run].cpu().numpy()
                for i in range(len(embedding_train))
            ],
            axis=0,
        )
        train = np.concatenate(
            [
                label_train[i].continuous.cpu().numpy()
                for i in range(len(label_train))
            ],
            axis=0,
        )
        all_val_embeddings = np.concatenate(
            [
                embedding_valid[i][n_run].cpu().numpy()
                for i in range(len(embedding_valid))
            ],
            axis=0,
        )
        valid = np.concatenate(
            [
                label_valid[i].continuous.cpu().numpy()
                for i in range(len(label_valid))
            ],
            axis=0,
        )
    else:
        all_train_embeddings = embedding_train.cpu().numpy()
        train = label_train.cpu().numpy()
        all_val_embeddings = embedding_valid.cpu().numpy()
        valid = label_valid.cpu().numpy()

    decoder = GridSearchCV(Ridge(), {"alpha": np.logspace(-4, 0, 9)})
    decoder.fit(all_train_embeddings, train)

    train_prediction = decoder.predict(all_train_embeddings)
    train_scores = sklearn.metrics.r2_score(train,
                                            train_prediction,
                                            multioutput="raw_values").tolist()
    valid_prediction = decoder.predict(all_val_embeddings)
    valid_scores = sklearn.metrics.r2_score(valid,
                                            valid_prediction,
                                            multioutput="raw_values").tolist()

    return train_scores, valid_scores, valid_prediction


class SingleLayerDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(SingleLayerDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class TwoLayersDecoder(nn.Module):
    """Supervised module to predict behaviors.

    Note:
        By default, the output dimension is 2, to predict x/y velocity
        (Perich et al., 2018).
    """

    def __init__(self, input_dim, output_dim=2):
        super(TwoLayersDecoder, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 32), nn.GELU(),
                                nn.Linear(32, output_dim))

    def forward(self, x):
        return self.fc(x)


def mlp_decoding(
    embedding_train: Union[dict, torch.Tensor],
    embedding_valid: Union[dict, torch.Tensor],
    label_train: Union[dict, torch.Tensor],
    label_valid: Union[dict, torch.Tensor],
    num_epochs: int = 20,
    lr: float = 0.001,
    batch_size: int = 500,
    device: str = "cuda",
    model_type: str = "SingleLayerMLP",
    n_run: Optional[int] = None,
):
    """ Perform MLP decoding on training and validation embeddings.

    Args:
        embedding_train (Union[dict, torch.Tensor]): Training embeddings.
        embedding_valid (Union[dict, torch.Tensor]): Validation embeddings.
        label_train (Union[dict, torch.Tensor]): Training labels.
        label_valid (Union[dict, torch.Tensor]): Validation labels.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.
        batch_size (int): Batch size for training.
        device (str): Device to run the model on ('cuda' or 'cpu').
        model_type (str): Type of MLP model to use ('SingleLayerMLP' or 'TwoLayersMLP').
        n_run (Optional[int]): Optional run number for dataset definition.

    Returns:
        Training R2 scores, validation R2 scores, and validation predictions.
    """
    if len(label_train.shape) == 1:
        label_train = label_train[:, None]
        label_valid = label_valid[:, None]

    if isinstance(embedding_train, dict):  # only on run 1
        if n_run is None:
            raise ValueError(f"n_run must be specified, got {n_run}.")

        all_train_embeddings = torch.cat(
            [embedding_train[i][n_run] for i in range(len(embedding_train))],
            axis=0)
        train = torch.cat(
            [label_train[i].continuous for i in range(len(label_train))],
            axis=0)
        all_val_embeddings = torch.cat(
            [embedding_valid[i][n_run] for i in range(len(embedding_valid))],
            axis=0)
        valid = torch.cat(
            [label_valid[i].continuous for i in range(len(label_valid))],
            axis=0)
    else:
        all_train_embeddings = embedding_train
        train = label_train
        all_val_embeddings = embedding_valid
        valid = label_valid

    dataset = TensorDataset(all_train_embeddings.to(device), train.to(device))
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = all_train_embeddings.shape[1]
    output_dim = train.shape[1]
    if model_type == "SingleLayerMLP":
        model = SingleLayerDecoder(input_dim=input_dim, output_dim=output_dim)
    elif model_type == "TwoLayersMLP":
        model = TwoLayersDecoder(input_dim=input_dim, output_dim=output_dim)
    else:
        raise NotImplementedError()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    model.eval()
    train_pred = model(all_train_embeddings.to(device))
    train_r2 = sklearn.metrics.r2_score(
        y_true=train.cpu().numpy(),
        y_pred=train_pred.cpu().detach().numpy(),
        multioutput="raw_values",
    ).tolist()

    valid_pred = model(all_val_embeddings.to(device))
    valid_r2 = sklearn.metrics.r2_score(
        y_true=valid.cpu().numpy(),
        y_pred=valid_pred.cpu().detach().numpy(),
        multioutput="raw_values",
    ).tolist()

    return train_r2, valid_r2, valid_pred
