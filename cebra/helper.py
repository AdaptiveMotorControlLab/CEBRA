"""Collection of helper functions that did not fit into own modules."""

from typing import List

import cebra.data


def get_loader_options(dataset: cebra.data.Dataset) -> List[str]:
    """Return all possible dataloaders for the given dataset."""

    loader_options = []
    if isinstance(dataset, cebra.data.SingleSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(cebra.data.ContinuousDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            loader_options.append(cebra.data.DiscreteDataLoader)
        else:
            mixed = False
        if mixed:
            loader_options.append(cebra.data.MixedDataLoader)
    elif isinstance(dataset, cebra.data.MultiSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(cebra.data.ContinuousMultiSessionDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            pass  # not implemented yet
        else:
            mixed = False
        if mixed:
            pass  # not implemented yet
    else:
        raise TypeError(f"Invalid dataset type: {dataset}")
    return loader_options
