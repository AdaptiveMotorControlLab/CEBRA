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

import cebra.data
import cebra.datasets
import cebra.models
import cebra.solver

device = "cpu"

single_session_tests = []
for args in [
    ("demo-discrete", cebra.data.DiscreteDataLoader),
    ("demo-continuous", cebra.data.ContinuousDataLoader),
    ("demo-mixed", cebra.data.MixedDataLoader),
]:
    single_session_tests.append((*args, cebra.solver.SingleSessionSolver))

single_session_hybrid_tests = []
for args in [("demo-continuous", cebra.data.HybridDataLoader)]:
    single_session_hybrid_tests.append(
        (*args, cebra.solver.SingleSessionHybridSolver))

multi_session_tests = []
for args in [("demo-continuous-multisession",
              cebra.data.ContinuousMultiSessionDataLoader)]:
    multi_session_tests.append((*args, cebra.solver.MultiSessionSolver))
    # multi_session_tests.append((*args, cebra.solver.MultiSessionAuxVariableSolver))

print(single_session_tests)


def _get_loader(data_name, loader_initfunc):
    data = cebra.datasets.init(data_name)
    kwargs = dict(num_steps=10, batch_size=32)
    loader = loader_initfunc(data, **kwargs)
    return loader


def _make_model(dataset):
    # TODO flexible input dimension
    return nn.Sequential(
        nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
        nn.Flatten(start_dim=1, end_dim=-1),
    )


def _make_behavior_model(dataset):
    # TODO flexible input dimension
    return nn.Sequential(
        nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
        nn.Flatten(start_dim=1, end_dim=-1),
    )


@pytest.mark.parametrize("data_name, loader_initfunc, solver_initfunc",
                         single_session_tests)
def test_single_session(data_name, loader_initfunc, solver_initfunc):
    loader = _get_loader(data_name, loader_initfunc)
    model = _make_model(loader.dataset)
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    assert batch.reference.shape == (32, loader.dataset.input_dimension, 10)
    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)


@pytest.mark.parametrize("data_name, loader_initfunc, solver_initfunc",
                         single_session_tests)
def test_single_session_auxvar(data_name, loader_initfunc, solver_initfunc):
    return  # TODO

    loader = _get_loader(data_name, loader_initfunc)
    model = _make_model(loader.dataset)
    behavior_model = _make_behavior_model(loader.dataset)

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
    )

    batch = next(iter(loader))
    assert batch.reference.shape == (32, loader.dataset.input_dimension, 10)
    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)


@pytest.mark.parametrize("data_name, loader_initfunc, solver_initfunc",
                         single_session_hybrid_tests)
def test_single_session_hybrid(data_name, loader_initfunc, solver_initfunc):
    loader = _get_loader(data_name, loader_initfunc)
    model = cebra.models.init("offset10-model", loader.dataset.input_dimension,
                              32, 3)
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    inference = solver._inference(batch)
    assert len(inference) == 2
    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)


@pytest.mark.parametrize("data_name, loader_initfunc, solver_initfunc",
                         multi_session_tests)
def test_multi_session(data_name, loader_initfunc, solver_initfunc):
    loader = _get_loader(data_name, loader_initfunc)
    criterion = cebra.models.InfoNCE()
    model = nn.ModuleList(
        [_make_model(dataset) for dataset in loader.dataset.iter_sessions()])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    for session_id, dataset in enumerate(loader.dataset.iter_sessions()):
        assert batch[session_id].reference.shape == (32,
                                                     dataset.input_dimension,
                                                     10)
        assert batch[session_id].index is not None

    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)
