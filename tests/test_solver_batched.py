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
BATCHES = [250, 500, 750]
MODELS = ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]


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


@pytest.mark.parametrize(
    "data_name, model_name, session_id, loader_initfunc, solver_initfunc",
    [(dataset, model, session_id, loader, cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in ["offset1-model", "offset10-model"]
     for session_id in [None, 0, 5]] +
    [(dataset, model, session_id, loader,
      cebra.solver.SingleSessionHybridSolver)
     for dataset, loader in [
         ("demo-continuous", cebra.data.HybridDataLoader),
     ]
     for model in ["offset1-model", "offset10-model"]
     for session_id in [None, 0, 5]])
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
    [(dataset, model, session_id, loader, cebra.solver.MultiSessionSolver)
     for dataset, loader in [
         ("demo-continuous-multisession",
          cebra.data.ContinuousMultiSessionDataLoader),
     ]
     for model in ["offset1-model", "offset10-model"]
     for session_id in [None, 0, 1, 5, 2, 6, 4]])
def test_select_model_multi_session(data_name, model_name, session_id,
                                    loader_initfunc, solver_initfunc):

    dataset = cebra.datasets.init(data_name)
    kwargs = dict(num_steps=NUM_STEPS, batch_size=32)
    loader = loader_initfunc(dataset, **kwargs)

    model = nn.ModuleList([
        create_model(model_name, dataset.input_dimension)
        for dataset in dataset.iter_sessions()
    ])
    dataset.configure_for(model)

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


@pytest.mark.parametrize(
    "data_name, model_name, padding, batch_size_inference, loader_initfunc, solver_initfunc",
    [(dataset, model, padding, batch_size, loader,
      cebra.solver.SingleSessionSolver)
     for dataset, loader in [("demo-discrete", cebra.data.DiscreteDataLoader),
                             ("demo-continuous", cebra.data.ContinuousDataLoader
                             ), ("demo-mixed", cebra.data.MixedDataLoader)]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]
     for padding in [True, False]
     for batch_size in BATCHES] +
    [(dataset, model, padding, batch_size, loader,
      cebra.solver.SingleSessionHybridSolver)
     for dataset, loader in [
         ("demo-continuous", cebra.data.HybridDataLoader),
     ]
     for model in MODELS
     for padding in [True, False]
     for batch_size in BATCHES])
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
                                         batch_size=batch_size_inference,
                                         pad_before_transform=padding)

    embedding = solver.transform(inputs=loader.dataset.neural,
                                 pad_before_transform=padding)

    assert embedding_batched.shape == embedding.shape
    assert np.allclose(embedding_batched, embedding, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "data_name, model_name,padding,batch_size_inference,loader_initfunc, solver_initfunc",
    [(dataset, model, padding, batch_size, loader,
      cebra.solver.MultiSessionSolver)
     for dataset, loader in [
         ("demo-continuous-multisession",
          cebra.data.ContinuousMultiSessionDataLoader),
     ]
     for model in
     ["offset1-model", "offset10-model", "offset40-model-4x-subsample"]
     for padding in [True, False]
     for batch_size in BATCHES])
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
                                             batch_size=batch_size_inference)

        assert embedding_batched.shape == embedding.shape
        assert np.allclose(embedding_batched, embedding, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "batch_start_idx, batch_end_idx, offset, num_samples, expected_exception",
    [
        # Valid indices
        (0, 5, cebra.data.Offset(1, 1), 10, None),
        (2, 8, cebra.data.Offset(2, 2), 10, None),
        # Negative indices
        (-1, 5, cebra.data.Offset(1, 1), 10, ValueError),
        (0, -5, cebra.data.Offset(1, 1), 10, ValueError),
        # Start index greater than end index
        (5, 3, cebra.data.Offset(1, 1), 10, ValueError),
        # End index out of bounds
        (0, 11, cebra.data.Offset(1, 1), 10, ValueError),
        # Batch size smaller than offset
        (0, 2, cebra.data.Offset(3, 3), 10, ValueError),
    ],
)
def test_check_indices(batch_start_idx, batch_end_idx, offset, num_samples,
                       expected_exception):
    if expected_exception:
        with pytest.raises(expected_exception):
            cebra.solver.base._check_indices(batch_start_idx, batch_end_idx,
                                             offset, num_samples)
    else:
        cebra.solver.base._check_indices(batch_start_idx, batch_end_idx, offset,
                                         num_samples)


@pytest.mark.parametrize(
    "batch_start_idx, batch_end_idx, num_samples, expected_exception",
    [
        # First batch
        (0, 6, 12, 8),
        # Last batch
        (6, 12, 12, 8),
        # Middle batch
        (3, 9, 12, 6),
        # Invalid start index
        (-1, 3, 4, ValueError),
        # Invalid end index
        (3, -10, 4, ValueError),
        # Start index greater than end index
        (5, 3, 4, ValueError),
    ],
)
def test_add_batched_zero_padding(batch_start_idx, batch_end_idx, num_samples,
                                  expected_exception):
    batched_data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
                                 [9.0, 10.0], [1.0, 2.0]])

    model = create_model(model_name="offset5-model",
                         input_dimension=batched_data.shape[1])
    offset = model.get_offset()

    if expected_exception == ValueError:
        with pytest.raises(expected_exception):
            result = cebra.solver.base._add_batched_zero_padding(
                batched_data, offset, batch_start_idx, batch_end_idx,
                num_samples)
    else:
        result = cebra.solver.base._add_batched_zero_padding(
            batched_data, offset, batch_start_idx, batch_end_idx, num_samples)
        assert result.shape[0] == expected_exception


@pytest.mark.parametrize(
    "pad_before_transform, expected_exception",
    [
        # Valid batched inputs
        (True, None),
        # No padding
        (False, None),
    ],
)
def test_transform(pad_before_transform, expected_exception):
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
                           [9.0, 10.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                           [7.0, 8.0], [9.0, 10.0], [1.0, 2.0], [3.0, 4.0],
                           [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    model = create_model(model_name="offset5-model",
                         input_dimension=inputs.shape[1])
    offset = model.get_offset()

    result = cebra.solver.base._not_batched_transform(
        model=model,
        inputs=inputs,
        pad_before_transform=pad_before_transform,
        offset=offset,
    )
    if pad_before_transform:
        assert result.shape[0] == inputs.shape[0]
    else:
        assert result.shape[0] == inputs.shape[0] - len(offset) + 1


@pytest.mark.parametrize(
    "batch_size, pad_before_transform, expected_exception",
    [
        # Valid batched inputs
        (6, True, None),
        # Invalid batch size (too large)
        (12, True, ValueError),
        # Invalid batch size (too small)
        (2, True, ValueError),
        # Last batch size incomplete
        (5, True, None),
        # No padding
        (6, False, None),
    ],
)
def test_batched_transform(batch_size, pad_before_transform,
                           expected_exception):
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],
                           [9.0, 10.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],
                           [7.0, 8.0], [9.0, 10.0], [1.0, 2.0], [3.0, 4.0],
                           [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    model = create_model(model_name="offset5-model",
                         input_dimension=inputs.shape[1])
    offset = model.get_offset()

    if expected_exception:
        with pytest.raises(expected_exception):
            cebra.solver.base._batched_transform(
                model=model,
                inputs=inputs,
                batch_size=batch_size,
                pad_before_transform=pad_before_transform,
                offset=offset,
            )
    else:
        result = cebra.solver.base._batched_transform(
            model=model,
            inputs=inputs,
            batch_size=batch_size,
            pad_before_transform=pad_before_transform,
            offset=offset,
        )
        if pad_before_transform:
            assert result.shape[0] == inputs.shape[0]
        else:
            assert result.shape[0] == inputs.shape[0] - len(offset) + 1
