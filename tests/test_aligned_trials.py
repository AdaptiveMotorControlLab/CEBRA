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
"""Aligned-trial sampling preserves group, trial and encoder boundaries."""

import numpy as np
import pytest
import torch

from cebra.data import Offset
from cebra.data.aligned_trials import AlignedTrialDataset
from cebra.data.aligned_trials import AlignedTrialLoader
from cebra.data.aligned_trials import AlignedWindowIndex


def dataset(controls=True, device='cpu'):
    trials = [
        torch.stack((torch.full((n,), i), torch.arange(n)), -1).float()
        for i, n in enumerate((12, 12, 18, 18, 18))
    ]
    control = [x[:, 1:] * 3 for x in trials] if controls else None
    result = AlignedTrialDataset(trials, ['a', 'a', 'b', 'b', 'b'],
                                 .02,
                                 controls=control,
                                 time_origin=-.25,
                                 device=device)
    result.offset = Offset(2, 3)
    return result


def test_windows_align_views_and_controls_without_crossing_boundaries():
    data = dataset()
    loader = AlignedTrialLoader(data,
                                num_steps=2,
                                batch_size=17,
                                batch_size_negatives=5,
                                rollout_steps=3)
    batch = next(iter(loader))
    assert batch.observations.shape == (17, 2, 8, 2)
    assert batch.negative.shape == (5, 8, 2)
    assert batch.controls.shape == (17, 2, 4, 1)
    assert batch.time.shape == (17, 2, 4)
    assert batch.negative_controls.shape == (5, 4, 1)
    # Feature 0 encodes trial ID; feature 1 encodes that trial's time index.
    torch.testing.assert_close(
        batch.observations[..., 0],
        batch.index.trials[..., None].expand(-1, -1, 8).float())
    assert torch.all(batch.index.trials[:, 0] != batch.index.trials[:, 1])
    group = data.group_index[batch.index.trials]
    assert torch.equal(group[:, 0], group[:, 1])
    times = batch.observations[..., 2:6, 1]
    torch.testing.assert_close(batch.controls[..., 0], times * 3)
    torch.testing.assert_close(batch.time, times * .02 - .25)
    torch.testing.assert_close(batch.observations[:, 0, :, 1],
                               batch.observations[:, 1, :, 1])
    assert batch.to('cpu') is batch
    assert batch.index.trials.device.type == 'cpu'
    assert len(list(loader)) == 2


def test_seed_is_private_reproducible_and_does_not_reset_each_iteration():
    kwargs = dict(num_steps=2, batch_size=16, rollout_steps=2, seed=7)
    a = AlignedTrialLoader(dataset(), **kwargs)
    b = AlignedTrialLoader(dataset(), **kwargs)
    first = a.get_indices(16)
    torch.manual_seed(891)
    same = b.get_indices(16)
    torch.testing.assert_close(first.trials, same.trials)
    torch.testing.assert_close(first.starts, same.starts)
    second = a.get_indices(16)
    assert not (torch.equal(first.trials, second.trials) and
                torch.equal(first.starts, second.starts))


def test_singleton_groups_are_explicit_single_view():
    data = AlignedTrialDataset([np.zeros((8, 2)), np.ones((12, 2))], [0, 1], .1)
    loader = AlignedTrialLoader(data,
                                num_steps=1,
                                batch_size=5,
                                num_views=1,
                                rollout_steps=3)
    batch = next(iter(loader))
    assert batch.observations.shape == (5, 1, 4, 2)
    assert batch.controls is None
    with pytest.raises(ValueError, match='distinct trials'):
        AlignedTrialLoader(data,
                           num_steps=1,
                           batch_size=5,
                           num_views=2,
                           rollout_steps=3)


def test_tensor_alignment_labels_have_value_semantics():
    trials = [np.zeros((12, 2))] * 4
    data = AlignedTrialDataset(trials, torch.tensor([0, 0, 1, 1]), .1)
    assert data.groups == ((0, 1), (2, 3))
    for labels in (torch.zeros(4, 1), [0, 0, float("nan"),
                                       float("nan")],
                   np.array([0, 0, np.nan, np.nan], dtype="float32")):
        with pytest.raises(ValueError, match="alignment IDs"):
            AlignedTrialDataset(trials, labels, .1)


def test_shortest_legal_window_and_invalid_context():
    data = AlignedTrialDataset([np.zeros((7, 2))] * 2, [0, 0], .1)
    data.offset = Offset(2, 2)
    loader = AlignedTrialLoader(data,
                                num_steps=1,
                                batch_size=9,
                                rollout_steps=3)
    batch = next(iter(loader))
    assert torch.all(batch.index.starts == 2)
    assert batch.observations.shape[-2] == 7
    with pytest.raises(ValueError, match='too short'):
        AlignedTrialLoader(data, num_steps=1, batch_size=9, rollout_steps=4)
    with pytest.raises(ValueError, match='boundary'):
        data[torch.tensor([[0, 0]])]
    with pytest.raises(ValueError, match='boundary'):
        data[torch.tensor([[0, 6]])]
    assert data[torch.tensor([[1, 2]])].shape == (1, 2, 4)


def test_manual_index_cannot_pair_groups_or_duplicate_trials():
    data = dataset()
    index = AlignedWindowIndex(torch.tensor([[0, 2]]), torch.tensor([3]),
                               torch.tensor([1]), torch.tensor([3]), 4)
    with pytest.raises(ValueError, match='same alignment group'):
        data.load_batch(index)
    index.trials = torch.tensor([[0, 0]])
    with pytest.raises(ValueError, match='independent views'):
        data.load_batch(index)
    index.trials = torch.tensor([[0, 1]])
    index.starts = torch.tensor([10])
    with pytest.raises(ValueError, match='boundary'):
        data.load_batch(index)


@pytest.mark.parametrize('trials,labels,controls', [
    ([np.zeros((8, 2)), np.zeros((9, 2))], [0, 0], None),
    ([np.zeros((8, 2))], [], None),
    ([np.zeros((8, 2)), np.zeros((8, 3))], [0, 1], None),
    ([np.full((8, 2), np.nan)], [0], None),
    ([np.zeros((8, 2))], [0], [np.zeros((7, 1))]),
    ([np.zeros((8, 2))], [[0]], None),
])
def test_invalid_data_is_rejected(trials, labels, controls):
    with pytest.raises(ValueError):
        AlignedTrialDataset(trials, labels, .02, controls=controls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_device_move_includes_controls_and_metadata():
    loader = AlignedTrialLoader(dataset(),
                                num_steps=1,
                                batch_size=3,
                                rollout_steps=2)
    loader.to('cuda')
    batch = next(iter(loader))
    for value in (batch.observations, batch.controls, batch.time,
                  batch.index.trials):
        assert value.device.type == 'cuda'
    loader.to('cpu')
    assert next(iter(loader)).observations.device.type == 'cpu'


def test_readonly_input_is_copied_without_mutation():
    data = np.zeros((8, 2), dtype=np.float32)
    data.setflags(write=False)
    result = AlignedTrialDataset([data], [0], .02)
    result.neural.fill_(3)
    assert np.all(data == 0)
