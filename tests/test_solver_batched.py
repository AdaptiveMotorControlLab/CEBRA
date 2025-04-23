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
import numpy as np
import pytest
import torch
from torch import nn

import cebra.data
import cebra.datasets
import cebra.models
import cebra.solver

device = "cpu"

NUM_STEPS = 2

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
    ("demo-discrete-multisession", cebra.data.DiscreteMultiSessionDataLoader,
     "offset1-model"),
    ("demo-discrete-multisession", cebra.data.DiscreteMultiSessionDataLoader,
     "offset10-model"),
]:
    multi_session_tests.append((*args, cebra.solver.MultiSessionSolver))


def _get_loader(data, loader_initfunc):
    kwargs = dict(num_steps=NUM_STEPS, batch_size=32)
    loader = loader_initfunc(data, **kwargs)
    return loader


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
                             num_output=3)


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

    loader_kwargs = dict(num_steps=NUM_STEPS, batch_size=32)
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
]  #NOTE(rodrigo): there is an issue with "offset4-model-2x-subsample" because it's not a convolutional model.
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
    dataset.configure_for(model)
    loader_kwargs = dict(num_steps=NUM_STEPS, batch_size=32)
    loader = loader_initfunc(dataset, **loader_kwargs)

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    solver.fit(loader)

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
    dataset.configure_for(model)

    n_samples = dataset._datasets[0].neural.shape[0]
    assert all(
        d.neural.shape[0] == n_samples for d in dataset._datasets
    ), "for this set all of the sessions need to have same number of samples."

    loader_kwargs = dict(num_steps=NUM_STEPS, batch_size=32)
    loader = loader_initfunc(dataset, **loader_kwargs)

    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    solver = solver_initfunc(model=model,
                             criterion=criterion,
                             optimizer=optimizer)
    solver.fit(loader)

    # Transform each session with the right model, by providing
    # the corresponding session ID
    for i, inputs in enumerate(dataset.iter_sessions()):
        embedding = solver.transform(inputs=inputs.neural,
                                     session_id=i,
                                     pad_before_transform=padding)
        embedding_batched = solver.transform(inputs=inputs.neural,
                                             session_id=i,
                                             pad_before_transform=padding,
                                             batch_size=batch_size)

        assert embedding_batched.shape == embedding.shape
        assert np.allclose(embedding_batched, embedding, rtol=1e-02)
