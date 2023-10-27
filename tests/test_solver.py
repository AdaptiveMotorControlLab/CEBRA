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


def create_model(model_name, input_dimension):
    return cebra.models.init(model_name,
                             num_neurons=input_dimension,
                             num_units=128,
                             num_output=5)


single_session_tests_transform = []
for padding in [True, False]:
    for model_name in ["offset1-model", "offset10-model"]:
        for args in [
            ("demo-discrete", model_name, padding,
             cebra.data.DiscreteDataLoader),
            ("demo-continuous", model_name, padding,
             cebra.data.ContinuousDataLoader),
            ("demo-mixed", model_name, padding, cebra.data.MixedDataLoader),
        ]:
            single_session_tests_transform.append(
                (*args, cebra.solver.SingleSessionSolver))

single_session_hybrid_tests_transform = []
for padding in [True, False]:
    for model_name in ["offset1-model", "offset10-model"]:
        for args in [("demo-continuous", model_name, padding,
                      cebra.data.HybridDataLoader)]:
            single_session_hybrid_tests_transform.append(
                (*args, cebra.solver.SingleSessionHybridSolver))

multi_session_tests_transform = []
for padding in [True, False]:
    for model_name in ["offset1-model", "offset10-model"]:
        for args in [("demo-continuous-multisession", model_name, padding,
                      cebra.data.ContinuousMultiSessionDataLoader)]:
            multi_session_tests_transform.append(
                (*args, cebra.solver.MultiSessionSolver))

single_session_tests_select_model = []
single_session_hybrid_tests_select_model = []
for model_name in ["offset1-model", "offset10-model"]:
    for session_id in [None, 0, 5]:
        for args in [
            ("demo-discrete", model_name, session_id),
            ("demo-continuous", model_name, session_id),
            ("demo-mixed", model_name, session_id),
        ]:
            single_session_tests_select_model.append(
                (*args, cebra.solver.SingleSessionSolver))
            single_session_hybrid_tests_select_model.append(
                (*args, cebra.solver.SingleSessionHybridSolver))

multi_session_tests_select_model = []
for model_name in ["offset1-model", "offset10-model"]:
    for session_id in [None, 0, 1, 4]:
        for args in [("demo-continuous-multisession", model_name, session_id)]:
            multi_session_tests_select_model.append(
                (*args, cebra.solver.MultiSessionSolver))


@pytest.mark.parametrize(
    "inputs, add_padding, offset, start_batch_idx, end_batch_idx, expected_output",
    [
        # Test case 1: No padding
        (torch.tensor([[1, 2], [3, 4]]), False, None, 0, 1,
         torch.tensor([[1, 2]])),  # first batch
        (torch.tensor([[1, 2], [3, 4]]), False, None, 0, 2,
         torch.tensor([[1, 2], [3, 4]])),  # first batch
        (torch.tensor([[1, 2], [3, 4]]), False, None, 1, 2,
         torch.tensor([[3, 4]])),  # last batch

        # Test case 2: First batch with padding
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(1, 1),
            0,
            2,
            torch.tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]]),
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
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(1, 3),
            1,
            3,
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [7, 8, 9], [7, 8, 9]
                         ]),
        ),

        # Test case 4: Middle batch with padding
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(0, 1),
            1,
            2,
            torch.tensor([[4, 5, 6]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(0, 2),
            1,
            2,
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(1, 1),
            1,
            2,
            torch.tensor([[1, 2, 3], [4, 5, 6]]),
        ),
        (
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            True,
            cebra.data.Offset(1, 2),
            1,
            2,
            torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        ),

        # Examples that throw an error:

        # Padding without offset (should raise an error)
        (torch.tensor([[1, 2]]), True, None, 0, 2, ValueError),
        # Negative start_batch_idx or end_batch_idx (should raise an error)
        (torch.tensor([[1, 2]]), False, None, -1, 2, ValueError),
        # out of bound indices because offset is too large
        (torch.tensor([[1, 2], [3, 4]]), True, cebra.data.Offset(
            5, 5), 1, 2, ValueError),
    ],
)
def test_process_batch(inputs, add_padding, offset, start_batch_idx,
                       end_batch_idx, expected_output):
    if expected_output == ValueError:
        with pytest.raises(ValueError):
            cebra.solver.base._process_batch(inputs, add_padding, offset,
                                             start_batch_idx, end_batch_idx)
    else:
        result = cebra.solver.base._process_batch(inputs, add_padding, offset,
                                                  start_batch_idx,
                                                  end_batch_idx)
        assert torch.equal(result, expected_output)


@pytest.mark.parametrize("data_name, model_name,session_id,solver_initfunc",
                         single_session_tests_select_model +
                         single_session_hybrid_tests_select_model)
def test_select_model_single_session(data_name, model_name, session_id,
                                     solver_initfunc):
    dataset = cebra.datasets.init(data_name)
    model = create_model(model_name, dataset.input_dimension)
    offset = model.get_offset()
    solver = solver_initfunc(model=model, criterion=None, optimizer=None)

    if session_id is not None and session_id > 0:
        with pytest.raises(RuntimeError):
            solver._select_model(dataset.neural, session_id=session_id)
    else:
        model_, offset_ = solver._select_model(dataset.neural,
                                               session_id=session_id)
        assert offset.left == offset_.left and offset.right == offset_.right
        assert model == model_


#@pytest.mark.parametrize(
#    "data_name, model_name,session_id,solver_initfunc",
#    single_session_tests_select_model + single_session_hybrid_tests_select_model)
#def test_select_model_multi_session(data_name, model_name, session_id, solver_initfunc):
#    dataset = cebra.datasets.init(data_name)
#    model = nn.ModuleList(
#             [create_model(model_name, dataset.input_dimension) for dataset in dataset.iter_sessions()])
#    offset = model[0].get_offset()
#    solver = solver_initfunc(model=model,
#                             criterion=None,
#                             optimizer=None)
#
#    if session_id is not None and session_id > 0:
#        with pytest.raises(RuntimeError):
#            solver._select_model(dataset.neural, session_id=session_id)
#    else:
#        model_, offset_ = solver._select_model(dataset.neural, session_id=session_id)
#        assert offset.left == offset_.left and offset.right == offset_.right
#        assert model == model_


@pytest.mark.parametrize(
    "data_name, model_name, padding, loader_initfunc, solver_initfunc",
    single_session_tests_transform + single_session_hybrid_tests_transform)
def test_batched_transform_singlesession(data_name, model_name, padding,
                                         loader_initfunc, solver_initfunc):
    batch_size = 1024
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

    if len(model.get_offset()) < 2 and padding:
        pytest.skip("not relevant for now.")
        with pytest.raises(ValueError):
            solver.transform(inputs=loader.dataset.neural,
                             pad_before_transform=padding)

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

        if padding:
            if isinstance(model, cebra.models.ConvolutionalModelMixin):
                assert embedding_batched.shape == embedding.shape
                assert embedding_batched.shape == embedding.shape

        else:
            if isinstance(model, cebra.models.ConvolutionalModelMixin):
                #TODO: what to check here exactly?
                pass
            else:
                #print(model)
                assert embedding_batched.shape == embedding.shape, (padding,
                                                                    model)
                assert np.allclose(embedding_batched, embedding, rtol=1e-02)


# def test_batched_transform_multisession(data_name, model_name, padding, loader_initfunc, solver_initfunc):
#     batch_size = 1024
#     dataset = cebra.datasets.init(data_name)
#     model = nn.ModuleList(
#             [create_model(model_name, dataset.input_dimension) for dataset in dataset.iter_sessions()])
#     dataset.offset = model[0].get_offset()
#     loader_kwargs = dict(num_steps=10, batch_size=32)
#     loader = loader_initfunc(dataset, **loader_kwargs)

#     criterion = cebra.models.InfoNCE()
#     optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#     solver = solver_initfunc(model=model,
#                              criterion=criterion,
#                              optimizer=optimizer)
#     solver.fit(loader)

# if len(model.get_offset()) < 2 and padding:
#     with pytest.raises(ValueError):
#         solver.transform(inputs=loader.dataset.neural,
#                             pad_before_transform=padding)

#     with pytest.raises(ValueError):
#         solver.transform(inputs=loader.dataset.neural,
#                          batch_size=batch_size,
#                          pad_before_transform=padding)
# else:
#     embedding_batched = solver.transform(inputs=loader.dataset.neural,
#                                          batch_size=batch_size,
#                                          pad_before_transform=padding)

#     embedding = solver.transform(inputs=loader.dataset.neural,
#                                 pad_before_transform=padding)

#     if padding:
#         if isinstance(model, cebra.models.ConvolutionalModelMixin):
#             assert embedding_batched.shape == embedding.shape
#             assert embedding_batched.shape == embedding.shape

#     else:
#         if isinstance(model, cebra.models.ConvolutionalModelMixin):
#             #TODO: what to check here exactly?
#             pass
#         else:
#             assert embedding_batched.shape == embedding.shape
#             assert np.allclose(embedding_batched, embedding, rtol=1e-02)
