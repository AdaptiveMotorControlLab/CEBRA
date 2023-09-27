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

single_session_tests = []
for args in [
    ("demo-discrete", cebra.data.DiscreteDataLoader, "offset10-model"),
    ("demo-discrete", cebra.data.DiscreteDataLoader, "offset1-model"),
    ("demo-discrete", cebra.data.DiscreteDataLoader, "offset1-model"),
    ("demo-discrete", cebra.data.DiscreteDataLoader, "offset10-model"),
    ("demo-continuous", cebra.data.ContinuousDataLoader, "offset10-model"),
    ("demo-continuous", cebra.data.ContinuousDataLoader, "offset1-model"),
    ("demo-mixed", cebra.data.MixedDataLoader, "offset10-model"),
    ("demo-mixed", cebra.data.MixedDataLoader, "offset1-model"),
]:
    single_session_tests.append((*args, cebra.solver.SingleSessionSolver))

single_session_hybrid_tests = []
for args in [("demo-continuous", cebra.data.HybridDataLoader, "offset10-model"),
             ("demo-continuous", cebra.data.HybridDataLoader, "offset1-model")]:
    single_session_hybrid_tests.append(
        (*args, cebra.solver.SingleSessionHybridSolver))

multi_session_tests = []
for args in [
    ("demo-continuous-multisession",
     cebra.data.ContinuousMultiSessionDataLoader, "offset1-model"),
    ("demo-continuous-multisession",
     cebra.data.ContinuousMultiSessionDataLoader, "offset10-model"),
]:
    multi_session_tests.append((*args, cebra.solver.MultiSessionSolver))

# multi_session_tests.append((*args, cebra.solver.MultiSessionAuxVariableSolver))


def _get_loader(data, loader_initfunc):
    kwargs = dict(num_steps=5, batch_size=32)
    loader = loader_initfunc(data, **kwargs)
    return loader


OUTPUT_DIMENSION = 3


def _make_model(dataset, model_architecture="offset10-model"):
    # TODO flexible input dimension
    # return nn.Sequential(
    #     nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
    #     nn.Flatten(start_dim=1, end_dim=-1),
    # )
    return cebra.models.init(model_architecture, dataset.input_dimension, 32,
                             OUTPUT_DIMENSION)


# def _make_behavior_model(dataset):
#     # TODO flexible input dimension
#     return nn.Sequential(
#         nn.Conv1d(dataset.input_dimension, 5, kernel_size=10),
#         nn.Flatten(start_dim=1, end_dim=-1),
#     )


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
    "data_name, loader_initfunc, model_architecture, solver_initfunc",
    single_session_tests)
def test_single_session(data_name, loader_initfunc, model_architecture,
                        solver_initfunc):
    data = cebra.datasets.init(data_name)
    loader = _get_loader(data, loader_initfunc)
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

    assert solver.num_sessions == None
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


@pytest.mark.parametrize("data_name, loader_initfunc, model_architecture, solver_initfunc",
                         single_session_tests)
def test_single_session_auxvar(data_name, loader_initfunc, model_architecture, solver_initfunc):

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
    "data_name, loader_initfunc, model_architecture, solver_initfunc",
    single_session_hybrid_tests)
def test_single_session_hybrid(data_name, loader_initfunc, model_architecture,
                               solver_initfunc):
    data = cebra.datasets.init(data_name)
    loader = _get_loader(data, loader_initfunc)
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

    assert solver.num_sessions == None
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
    "data_name, loader_initfunc, model_architecture, solver_initfunc",
    multi_session_tests)
def test_multi_session(data_name, loader_initfunc, model_architecture,
                       solver_initfunc):
    data = cebra.datasets.init(data_name)
    loader = _get_loader(data, loader_initfunc)
    model = nn.ModuleList([
        _make_model(dataset, model_architecture)
        for dataset in data.iter_sessions()
    ])
    data.configure_for(model)
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)

    batch = next(iter(loader))
    for session_id, dataset in enumerate(loader.dataset.iter_sessions()):
        assert batch[session_id].reference.shape[:2] == (
            32, dataset.input_dimension)
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


@pytest.mark.parametrize(
    "inputs, add_padding, offset, start_batch_idx, end_batch_idx, expected_output",
    [
        # Test case 1: No padding
        (torch.tensor([[1, 2], [3, 4], [5, 6]]), False, cebra.data.Offset(
            0, 1), 0, 2, torch.tensor([[1, 2], [3, 4]])),  # first batch
        (torch.tensor([[1, 2], [3, 4], [5, 6]]), False, cebra.data.Offset(
            0, 1), 1, 3, torch.tensor([[3, 4], [5, 6]])),  # last batch
        (torch.tensor(
            [[1, 2], [3, 4], [5, 6], [7, 8]]), False, cebra.data.Offset(
                0, 1), 1, 3, torch.tensor([[3, 4], [5, 6]])),  # middle batch

        # Test case 2: First batch with padding
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(0, 1),
            0,
            2,
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(1, 1),
            0,
            3,
            torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),

        # Test case 3: Last batch with padding
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(0, 1),
            1,
            3,
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                          [13, 14, 15]]),
            True,
            cebra.data.Offset(1, 2),
            1,
            3,
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
        ),

        # Test case 4: Middle batch with padding
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            True,
            cebra.data.Offset(0, 1),
            1,
            3,
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            True,
            cebra.data.Offset(1, 1),
            1,
            3,
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12],
                          [13, 14, 15]]),
            True,
            cebra.data.Offset(0, 1),
            2,
            4,
            torch.tensor([[7, 8, 9], [10, 11, 12]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            True,
            cebra.data.Offset(0, 1),
            0,
            3,
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),

        # Examples that throw an error:

        # Padding without offset (should raise an error)
        (torch.tensor([[1, 2]]), True, None, 0, 2, ValueError),
        # Negative start_batch_idx or end_batch_idx (should raise an error)
        (torch.tensor([[1, 2]]), False, cebra.data.Offset(
            0, 1), -1, 2, ValueError),
        # out of bound indices because offset is too large
        (torch.tensor([[1, 2], [3, 4]]), True, cebra.data.Offset(
            5, 5), 1, 2, ValueError),
        # Batch length is smaller than offset.
        (torch.tensor([[1, 2], [3, 4]]), False, cebra.data.Offset(
            0, 1), 0, 1, ValueError),  # first batch
    ],
)
def test_get_batch(inputs, add_padding, offset, start_batch_idx, end_batch_idx,
                   expected_output):
    if expected_output == ValueError:
        with pytest.raises(ValueError):
            cebra.solver.base._get_batch(inputs, offset, start_batch_idx,
                                         end_batch_idx, add_padding)
    else:
        result = cebra.solver.base._get_batch(inputs, offset, start_batch_idx,
                                              end_batch_idx, add_padding)
        assert torch.equal(result, expected_output)


def create_model(model_name, input_dimension):
    return cebra.models.init(model_name,
                             num_neurons=input_dimension,
                             num_units=128,
                             num_output=OUTPUT_DIMENSION)


single_session_tests_select_model = []
single_session_hybrid_tests_select_model = []
for model_name in ["offset1-model", "offset10-model"]:
    for session_id in [None, 0, 5]:
        for args in [
            ("demo-discrete", model_name, session_id,
             cebra.data.DiscreteDataLoader),
            ("demo-continuous", model_name, session_id,
             cebra.data.ContinuousDataLoader),
            ("demo-mixed", model_name, session_id, cebra.data.MixedDataLoader),
        ]:
            single_session_tests_select_model.append(
                (*args, cebra.solver.SingleSessionSolver))
            single_session_hybrid_tests_select_model.append(
                (*args, cebra.solver.SingleSessionHybridSolver))

multi_session_tests_select_model = []
for model_name in ["offset10-model"]:
    for session_id in [None, 0, 1, 5, 2, 6, 4]:
        for args in [("demo-continuous-multisession", model_name, session_id,
                      cebra.data.ContinuousMultiSessionDataLoader)]:
            multi_session_tests_select_model.append(
                (*args, cebra.solver.MultiSessionSolver))


@pytest.mark.parametrize(
    "data_name, model_name ,session_id, loader_initfunc, solver_initfunc",
    single_session_tests_select_model + single_session_hybrid_tests_select_model
)
def test_select_model_single_session(data_name, model_name, session_id,
                                     loader_initfunc, solver_initfunc):
    dataset = cebra.datasets.init(data_name)
    model = create_model(model_name, dataset.input_dimension)
    dataset.configure_for(model)
    loader = _get_loader(dataset, loader_initfunc=loader_initfunc)
    offset = model.get_offset()
    solver = solver_initfunc(model=model, criterion=None, optimizer=None)

    with pytest.raises(ValueError):
        solver.n_features = 1000
        solver._select_model(inputs=dataset.neural, session_id=0)

    solver.n_features = dataset.neural.shape[1]
    if session_id is not None and session_id > 0:
        with pytest.raises(RuntimeError):
            solver._select_model(inputs=dataset.neural, session_id=session_id)
    else:
        model_, offset_ = solver._select_model(inputs=dataset.neural,
                                               session_id=session_id)
        assert offset.left == offset_.left and offset.right == offset_.right
        assert model == model_


@pytest.mark.parametrize(
    "data_name, model_name, session_id, loader_initfunc, solver_initfunc",
    multi_session_tests_select_model)
def test_select_model_multi_session(data_name, model_name, session_id,
                                    loader_initfunc, solver_initfunc):
    dataset = cebra.datasets.init(data_name)
    model = nn.ModuleList([
        create_model(model_name, dataset.input_dimension)
        for dataset in dataset.iter_sessions()
    ])
    dataset.configure_for(model)
    loader = _get_loader(dataset, loader_initfunc=loader_initfunc)

    offset = model[0].get_offset()
    solver = solver_initfunc(model=model,
                             criterion=cebra.models.InfoNCE(),
                             optimizer=torch.optim.Adam(model.parameters(),
                                                        lr=1e-3))

    loader_kwargs = dict(num_steps=10, batch_size=32)
    loader = cebra.data.ContinuousMultiSessionDataLoader(
        dataset, **loader_kwargs)
    solver.fit(loader)

    for i, (model, dataset_) in enumerate(zip(model, dataset.iter_sessions())):
        inputs = dataset_.neural

        if session_id is None or session_id >= dataset.num_sessions:
            with pytest.raises(RuntimeError):
                solver._select_model(inputs, session_id=session_id)
        elif i != session_id:
            with pytest.raises(ValueError):
                solver._select_model(inputs, session_id=session_id)
        else:
            model_, offset_ = solver._select_model(inputs,
                                                   session_id=session_id)
            assert offset.left == offset_.left and offset.right == offset_.right
            assert model == model_


models = [
    "offset1-model",
    "offset10-model",
    "offset40-model-4x-subsample",
    "offset1-model",
    "offset10-model",
]
batch_size_inference = [40_000, 99_990, 99_999]

single_session_tests_transform = []
for padding in [True, False]:
    for model_name in models:
        for batch_size in batch_size_inference:
            for args in [
                ("demo-discrete", model_name, padding, batch_size,
                 cebra.data.DiscreteDataLoader),
                ("demo-continuous", model_name, padding, batch_size,
                 cebra.data.ContinuousDataLoader),
                ("demo-mixed", model_name, padding, batch_size,
                 cebra.data.MixedDataLoader),
            ]:
                single_session_tests_transform.append(
                    (*args, cebra.solver.SingleSessionSolver))

single_session_hybrid_tests_transform = []
for padding in [True, False]:
    for model_name in models:
        for batch_size in batch_size_inference:
            for args in [("demo-continuous", model_name, padding, batch_size,
                          cebra.data.HybridDataLoader)]:
                single_session_hybrid_tests_transform.append(
                    (*args, cebra.solver.SingleSessionHybridSolver))


@pytest.mark.parametrize(
    "data_name, model_name, padding, batch_size_inference, loader_initfunc, solver_initfunc",
    single_session_tests_transform + single_session_hybrid_tests_transform)
def test_batched_transform_single_session(
    data_name,
    model_name,
    padding,
    batch_size_inference,
    loader_initfunc,
    solver_initfunc,
):
    dataset = cebra.datasets.init(data_name)
    model = create_model(model_name, dataset.input_dimension)
    dataset.offset = model.get_offset()
    loader_kwargs = dict(num_steps=10, batch_size=32)
    loader = loader_initfunc(dataset, **loader_kwargs)

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    solver.fit(loader)

    smallest_batch_length = loader.dataset.neural.shape[0] - batch_size
    offset_ = model.get_offset()
    padding_left = offset_.left if padding else 0

    if smallest_batch_length <= len(offset_):
        with pytest.raises(ValueError):
            solver.transform(inputs=loader.dataset.neural,
                             batch_size=batch_size,
                             pad_before_transform=padding)

    else:
        embedding_batched = solver.transform(inputs=loader.dataset.neural,
                                             batch_size=batch_size,
                                             pad_before_transform=padding)

        embedding = solver.transform(inputs=loader.dataset.neural,
                                     pad_before_transform=padding)

        assert embedding_batched.shape == embedding.shape
        assert np.allclose(embedding_batched, embedding, rtol=1e-02)


multi_session_tests_transform = []
for padding in [True, False]:
    for model_name in models:
        for batch_size in batch_size_inference:
            for args in [
                ("demo-continuous-multisession", model_name, padding,
                 batch_size, cebra.data.ContinuousMultiSessionDataLoader)
            ]:
                multi_session_tests_transform.append(
                    (*args, cebra.solver.MultiSessionSolver))


@pytest.mark.parametrize(
    "data_name, model_name,padding,batch_size_inference,loader_initfunc, solver_initfunc",
    multi_session_tests_transform)
def test_batched_transform_multi_session(data_name, model_name, padding,
                                         batch_size_inference, loader_initfunc,
                                         solver_initfunc):
    dataset = cebra.datasets.init(data_name)
    model = nn.ModuleList([
        create_model(model_name, dataset.input_dimension)
        for dataset in dataset.iter_sessions()
    ])
    dataset.offset = model[0].get_offset()

    n_samples = dataset._datasets[0].neural.shape[0]
    assert all(
        d.neural.shape[0] == n_samples for d in dataset._datasets
    ), "for this set all of the sessions need to have same number of samples."

    smallest_batch_length = n_samples - batch_size
    offset_ = model[0].get_offset()
    padding_left = offset_.left if padding else 0
    for d in dataset._datasets:
        d.offset = offset_
    loader_kwargs = dict(num_steps=10, batch_size=32)
    loader = loader_initfunc(dataset, **loader_kwargs)

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    solver.fit(loader)

    # Transform each session with the right model, by providing the corresponding session ID
    for i, inputs in enumerate(dataset.iter_sessions()):

        if smallest_batch_length <= len(offset_):
            with pytest.raises(ValueError):
                solver.transform(inputs=inputs.neural,
                                 batch_size=batch_size,
                                 session_id=i,
                                 pad_before_transform=padding)

        else:
            model_ = model[i]
            embedding = solver.transform(inputs=inputs.neural,
                                         session_id=i,
                                         pad_before_transform=padding)
            embedding_batched = solver.transform(inputs=inputs.neural,
                                                 session_id=i,
                                                 pad_before_transform=padding,
                                                 batch_size=batch_size)

            assert embedding_batched.shape == embedding.shape
            assert np.allclose(embedding_batched, embedding, rtol=1e-02)
