#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import itertools

import pytest
import torch
from torch import nn

import cebra
import cebra.config
import cebra.data
import cebra.datasets
import cebra.helper
import cebra.models
import cebra.solver


def _init_single_session_solver(loader, args):
    """Train a single session CEBRA model."""
    model = cebra.models.init("offset5-model", loader.dataset.input_dimension,
                              args.num_hidden_units, 3).to(args.device)
    loader.dataset.configure_for(model)
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return cebra.solver.SingleSessionSolver(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )


def _init_multi_session_solver(loader, args):
    """Train a multi session CEBRA model."""
    model = nn.ModuleList([
        cebra.models.init("offset5-model", dataset.input_dimension,
                          args.num_hidden_units, 3)
        for dataset in loader.dataset.iter_sessions()
    ]).to(args.device)
    for n, dataset in enumerate(loader.dataset.iter_sessions()):
        dataset.configure_for(model[n])
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    return cebra.solver.MultiSessionSolver(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )


def _list_data_loaders():
    """Yield (data/loader) pairs."""
    loaders = [
        cebra.data.ContinuousDataLoader,
        cebra.data.DiscreteDataLoader,
        cebra.data.MixedDataLoader,
        cebra.data.HybridDataLoader,
        cebra.data.FullDataLoader,
        cebra.data.ContinuousMultiSessionDataLoader,
    ]
    # TODO limit this to the valid combinations---however this
    # requires to adapt the dataset API slightly; it is currently
    # required to initialize the dataset to run cebra.helper.get_loader_options.
    prefixes = set()
    for dataset_name, loader in itertools.product(cebra.datasets.get_options(),
                                                  loaders):
        yield dataset_name, loader
        prefix = dataset_name.split("_", 1)[0]
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
    loader = loader_type(
        dataset,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
    )
    if isinstance(dataset, cebra.data.SingleSessionDataset):
        solver = _init_single_session_solver(loader, args)
    else:
        solver = _init_multi_session_solver(loader, args)
    batch = next(iter(loader))
    solver.step(batch)
