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
import copy
import tempfile

import numpy as np
import pytest
import torch
from torch import nn

import cebra.data
import cebra.datasets
import cebra.models
import cebra.solver

device = "cpu"


def _get_loader(data_name, loader_initfunc):
    data = cebra.datasets.init(data_name)
    kwargs = dict(num_steps=2, batch_size=32)
    loader = loader_initfunc(data, **kwargs)
    return loader, data


OUTPUT_DIMENSION = 3


def _make_model(dataset, model_architecture="offset10-model"):
    # TODO flexible input dimension
    # return nn.Sequential(
    #     nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
    #     nn.Flatten(start_dim=1, end_dim=-1),
    # )
    return cebra.models.init(model_architecture, dataset.input_dimension, 32,
                             OUTPUT_DIMENSION)


def _make_behavior_model(dataset):
    # TODO flexible input dimension
    return nn.Sequential(
        nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
        nn.Flatten(start_dim=1, end_dim=-1),
    )


def _assert_same_state_dict(first, second):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], torch.Tensor):
            assert torch.allclose(first[key], second[key]), key
        elif isinstance(first[key], dict):
            _assert_same_state_dict(first[key], second[key]), key
        else:
            assert first[key] == second[key]


def check_if_fit(model):
    """Check if a model was already fit.

    Args:
        model: The model to check.

    Returns:
        True if the model was already fit.
    """
    return hasattr(model, "n_features_")


def _assert_equal(original_solver, loaded_solver):
    for k in original_solver.model.state_dict():
        assert original_solver.model.state_dict()[k].all(
        ) == loaded_solver.model.state_dict()[k].all()
    assert check_if_fit(loaded_solver) == check_if_fit(original_solver)

    if check_if_fit(loaded_solver):
        _assert_same_state_dict(original_solver.state_dict_,
                                loaded_solver.state_dict_)
        X = np.random.normal(0, 1, (100, 1))

        if loaded_solver.num_sessions is not None:
            assert np.allclose(loaded_solver.transform(X, session_id=0),
                               original_solver.transform(X, session_id=0))
        else:
            assert np.allclose(loaded_solver.transform(X),
                               original_solver.transform(X))


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]])
def test_single_session(data_name, loader_initfunc, model_architecture,
                        solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer,
                             tqdm_on=False)

    batch = next(iter(loader))
    assert batch.reference.shape[:2] == (32, loader.dataset.input_dimension)
    log = solver.step(batch)
    assert isinstance(log, dict)

    X = loader.dataset.neural
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X)

    solver.fit(loader)

    assert solver.num_sessions is None
    assert solver.n_features == X.shape[1]

    embedding = solver.transform(X)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(torch.Tensor(X))
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, session_id=0)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (
            (X.shape[0] - len(offset)) // solver.model.resample_factor + 1,
            OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0] - len(offset) + 1,
                                   OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        solver.transform(X.numpy())
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X, session_id=2)

    for param in solver.parameters():
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)

    embedding = solver.transform(X)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(torch.Tensor(X))
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, session_id=0)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (X.shape[0] // solver.model.resample_factor,
                                   OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    if isinstance(solver.model, cebra.models.ResampleModelMixin):
        assert embedding.shape == (
            (X.shape[0] - len(offset)) // solver.model.resample_factor + 1,
            OUTPUT_DIMENSION)
    else:
        assert embedding.shape == (X.shape[0] - len(offset) + 1,
                                   OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        solver.transform(X.numpy())
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X, session_id=2)

    for param in solver.parameters():
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]])
def test_single_session_auxvar(data_name, loader_initfunc, model_architecture,
                               solver_initfunc):

    pytest.skip("Not yet supported")

    loader = _get_loader(data_name, loader_initfunc)
    model = _make_model(loader.dataset)
    behavior_model = _make_behavior_model(loader.dataset)  # noqa: F841

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


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [("demo-continuous", model, cebra.data.HybridDataLoader,
      cebra.solver.SingleSessionHybridSolver)
     for model in ["offset1-model", "offset10-model"]])
def test_single_session_hybrid(data_name, loader_initfunc, model_architecture,
                               solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()
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

    X = loader.dataset.neural
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X)

    solver.fit(loader)

    assert solver.num_sessions is None
    assert solver.n_features == X.shape[1]

    embedding = solver.transform(X)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(torch.Tensor(X))
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, session_id=0)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X.shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X.shape[0] - len(offset) + 1, OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        solver.transform(X.numpy())
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X, session_id=2)

    for param in solver.parameters():
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.MultiSessionSolver)
     for dataset, loader in [
         ("demo-discrete-multisession",
          cebra.data.DiscreteMultiSessionDataLoader),
         ("demo-continuous-multisession",
          cebra.data.ContinuousMultiSessionDataLoader),
     ]
     for model in ["offset1-model", "offset10-model"]])
def test_multi_session(data_name, loader_initfunc, model_architecture,
                       solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = nn.ModuleList([
        _make_model(dataset, model_architecture)
        for dataset in data.iter_sessions()
    ])
    data.configure_for(model)
    offset_length = len(model[0].get_offset())

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    for session_id, dataset in enumerate(loader.dataset.iter_sessions()):
        assert batch[session_id].reference.shape == (32,
                                                     dataset.input_dimension,
                                                     offset_length)
        assert batch[session_id].index is not None

    log = solver.step(batch)
    assert isinstance(log, dict)

    X = [
        loader.dataset.get_session(i).neural
        for i in range(loader.dataset.num_sessions)
    ]
    with pytest.raises(ValueError, match="not.*fitted"):
        solver.transform(X[0], session_id=0)

    solver.fit(loader)

    assert solver.num_sessions == 3
    assert solver.n_features == [X[i].shape[1] for i in range(len(X))]

    embedding = solver.transform(X[0], session_id=0)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[0].shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X[1], session_id=1)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[1].shape[0], OUTPUT_DIMENSION)
    embedding = solver.transform(X[0], session_id=0, pad_before_transform=False)
    assert isinstance(embedding, torch.Tensor)
    assert embedding.shape == (X[0].shape[0] -
                               len(solver.model[0].get_offset()) + 1,
                               OUTPUT_DIMENSION)

    with pytest.raises(ValueError, match="torch.Tensor"):
        embedding = solver.transform(X[0].numpy(), session_id=0)

    with pytest.raises(ValueError, match="shape"):
        embedding = solver.transform(X[1], session_id=0)
    with pytest.raises(ValueError, match="shape"):
        embedding = solver.transform(X[0], session_id=1)

    with pytest.raises(RuntimeError, match="No.*session_id"):
        embedding = solver.transform(X[0])
    with pytest.raises(ValueError, match="single.*session"):
        embedding = solver.transform(X)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X[0], session_id=5)
    with pytest.raises(RuntimeError, match="Invalid.*session_id"):
        embedding = solver.transform(X[0], session_id=-1)

    for param in solver.parameters(session_id=0):
        assert isinstance(param, torch.Tensor)

    fitted_solver = copy.deepcopy(solver)
    with tempfile.TemporaryDirectory() as temp_dir:
        solver.save(temp_dir)
        solver.load(temp_dir)
    _assert_equal(fitted_solver, solver)


def _make_val_data(dataset):
    if isinstance(dataset, cebra.datasets.demo.DemoDataset):
        return dataset.neural
    elif isinstance(dataset, cebra.datasets.demo.DemoDatasetUnified):
        return [session.neural for session in dataset.iter_sessions()], [
            session.continuous_index for session in dataset.iter_sessions()
        ]


@pytest.mark.parametrize(
    "data_name, model_architecture, loader_initfunc, solver_initfunc",
    [(dataset, model, loader, cebra.solver.UnifiedSolver)
     for dataset, loader in [
         ("demo-continuous-unified", cebra.data.UnifiedLoader),
     ]
     for model in ["offset1-model", "offset10-model"]])
def test_unified_session(data_name, model_architecture, loader_initfunc,
                         solver_initfunc):
    loader, data = _get_loader(data_name, loader_initfunc)
    model = _make_model(data, model_architecture)
    data.configure_for(model)
    offset = model.get_offset()

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    assert batch.reference.shape == (32, loader.dataset.input_dimension,
                                     len(offset))

    log = solver.step(batch)
    assert isinstance(log, dict)

    solver.fit(loader)
    data, labels = _make_val_data(loader.dataset)

    assert solver.num_sessions == 3
    assert solver.n_features == sum(
        [data[i].shape[1] for i in range(len(data))])

    for i in range(loader.dataset.num_sessions):
        emb = solver.transform(data, labels, session_id=i)
        assert emb.shape == (loader.dataset.num_timepoints, 3)

        emb = solver.transform(data, labels, session_id=i, batch_size=300)
        assert emb.shape == (loader.dataset.num_timepoints, 3)
