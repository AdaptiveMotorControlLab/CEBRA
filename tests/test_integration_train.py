#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import itertools
from typing import List

import pytest
import torch
from torch import nn

import cebra
import cebra.config
import cebra.data
import cebra.data.helper as cebra_data_helper
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
    # required to initialize the dataset to run cebra_data_helper.get_loader_options.
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
    if loader_type not in cebra_data_helper.get_loader_options(dataset):
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
