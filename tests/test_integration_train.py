import itertools

import pytest
import torch
from torch import nn

import cebra
import cebra.config
import cebra.data
import cebra.datasets
import cebra.helper
import cebra.solver


def _init_single_session_solver(loader, args):
    """Train a single session CEBRA model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def _init_multi_session_solver(loader, args):
    """Train a multi session CEBRA model."""
    model = nn.ModuleList([
    ]).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


def _list_data_loaders():
    """Yield (data/loader) pairs."""
    loaders = [
    ]
    # TODO limit this to the valid combinations---however this
    # requires to adapt the dataset API slightly; it is currently
    # required to initialize the dataset to run cebra.helper.get_loader_options.
    prefixes = set()
    for dataset_name, loader in itertools.product(cebra.datasets.get_options(),
                                                  loaders):
        yield dataset_name, loader
        if prefix in prefixes:
            # TODO(stes) include all datasets again
            return
        prefixes.add(prefix)


@pytest.mark.requires_dataset
@pytest.mark.parametrize("dataset_name, loader_type", _list_data_loaders())
def test_train(dataset_name, loader_type):
    args = cebra.config.Config(num_steps=1, device="cuda").as_namespace()

    dataset = cebra.datasets.init(dataset_name)
    if loader_type not in cebra.helper.get_loader_options(dataset):
        # skip this test, since the data/loader combination is not valid.
        pytest.skip("Not a valid dataset/loader combination.")
    if isinstance(dataset, cebra.data.SingleSessionDataset):
        solver = _init_single_session_solver(loader, args)
    else:
        solver = _init_multi_session_solver(loader, args)
    batch = next(iter(loader))
    solver.step(batch)
