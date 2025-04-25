"""
First, download the data using the brainsets CLI:
https://brainsets.readthedocs.io/en/latest/concepts/using_existing_data.html.
"""

import pathlib
import pickle
from typing import List, Optional, Tuple

import numpy as np

import cebra.data
from cebra.datasets import PerichDataset


def split_data(data):
    # Same masking as in the POYO paper
    train_mask = data.train_mask
    test_mask = data.test_mask
    valid_mask = data.valid_mask

    neural_train = data.neural[train_mask | test_mask]
    neural_test = data.neural[valid_mask]
    label_train = data.index[train_mask | test_mask]
    label_test = data.index[valid_mask]
    trial_id_train = data.trial_id[train_mask | test_mask]
    trial_id_test = data.trial_id[valid_mask]
    subtask_train = data.subtask_index[train_mask | test_mask]
    subtask_test = data.subtask_index[valid_mask]

    return (
        neural_train,
        neural_test,
        label_train,
        label_test,
        trial_id_train,
        trial_id_test,
        subtask_train,
        subtask_test,
    )


def zscore_across_sessions(datasets: List[PerichDataset]):
    all_data = np.concatenate([dataset.index for dataset in datasets])

    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)

    for dataset in datasets:
        dataset.index = (dataset.index - mean) / std


def get_data(
    datapath: pathlib.Path,
    zscore,
    dataset_type=None,
    drop_outliers: bool = True,
    n_datasets: int = None,
):
    datasets = {}
    for i, file in enumerate(datapath.iterdir()):
        if str(file.name).split(".")[0] != "description":
            # poyo kept t as a test subject: (str(file.name).split("_")[0] != "t")?
            print(f'"{file}",')
            dataset = PerichDataset(file)
            dataset_name = dataset.session

        if dataset_name not in datasets.keys():
            datasets[dataset_name] = []
        datasets[dataset_name].append(dataset)

        if n_datasets is not None and i >= n_datasets - 1:
            break

    all_datasets = []
    for dataset_name in datasets:
        if zscore:
            zscore_across_sessions(datasets[dataset_name])
        all_datasets.extend(datasets[dataset_name])

    return all_datasets


def create_datasets(
    datapath,
    n_neurons: Optional[int] = None,
    device="cuda",
    zscore=False,
    savedir: Optional[pathlib.Path] = None,
    n_datasets: int = None,
) -> Tuple[List[cebra.data.TensorDataset]]:
    if zscore:
        zscore_tag = "_zscore"
    else:
        zscore_tag = ""

    if savedir and ((savedir / f"train_datasets{zscore_tag}.pkl").exists() and
                    (savedir / f"valid_datasets{zscore_tag}.pkl").exists()):
        with open(savedir / f"train_datasets{zscore_tag}.pkl", "rb") as f:
            train_datasets = pickle.load(f)
        with open(savedir / f"valid_datasets{zscore_tag}.pkl", "rb") as f:
            valid_datasets = pickle.load(f)
    else:
        all_datasets = get_data(
            datapath=datapath,
            dataset_type="mp",
            zscore=zscore,
            drop_outliers=True,
            n_datasets=n_datasets,
        )

        datas_train, datas_val = [], []
        labels_train, labels_val = [], []
        ids_train, ids_val = [], []
        subtasks_train, subtasks_val = [], []

        for i in range(len(all_datasets)):
            (
                train_data,
                valid_data,
                train_label,
                valid_label,
                train_ids,
                valid_ids,
                train_subtask,
                valid_subtask,
            ) = split_data(all_datasets[i],)

            if n_neurons is not None:
                neurons_ids = np.random.choice(train_data.shape[1],
                                               size=min(n_neurons,
                                                        train_data.shape[1]))
            else:
                neurons_ids = np.arange(train_data.shape[1])

            datas_train.append(train_data[:, neurons_ids])
            datas_val.append(valid_data[:, neurons_ids])
            labels_train.append(train_label)
            labels_val.append(valid_label)
            ids_train.append(train_ids)
            ids_val.append(valid_ids)
            subtasks_train.append(train_subtask)
            subtasks_val.append(valid_subtask)

        train_datasets = [
            cebra.data.TensorDataset(
                neural=datas_train[j],
                continuous=labels_train[j],
                discrete=np.column_stack([ids_train[j], subtasks_train[j]]),
                device=device,
            ) for j in range(len(datas_train))
        ]

        valid_datasets = [
            cebra.data.TensorDataset(
                neural=datas_val[i],
                continuous=labels_val[i],
                discrete=np.column_stack([ids_val[i], subtasks_val[i]]),
                device=device,
            ) for i in range(len(datas_val))
        ]

        if savedir is not None:
            with open(savedir / f"train_datasets{zscore_tag}.pkl", "wb") as f:
                pickle.dump(train_datasets, f)

            with open(savedir / f"valid_datasets{zscore_tag}.pkl", "wb") as f:
                pickle.dump(valid_datasets, f)
            print(f"Datasets saved at {savedir}.")

    return train_datasets, valid_datasets
