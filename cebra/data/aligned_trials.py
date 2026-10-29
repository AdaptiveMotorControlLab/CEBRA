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
"""Explicitly aligned repeated-trial windows for dynamics learning."""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import literate_dataclasses as dataclasses
import torch

from cebra.data.base import Dataset
from cebra.data.base import Loader


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _copy_float(value, device):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32).clone()
    return torch.tensor(value, dtype=torch.float32, device=device)


@dataclass
class AlignedWindowIndex:
    """Trial identities and core starts; negatives are independent windows."""

    trials: torch.Tensor
    starts: torch.Tensor
    negative_trials: torch.Tensor
    negative_starts: torch.Tensor
    core_length: int

    def to(self, device):
        """Move tensor indices to a device, retaining the window length."""
        for name in ('trials', 'starts', 'negative_trials', 'negative_starts'):
            setattr(self, name, getattr(self, name).to(device))
        return self


@dataclass
class AlignedTrialBatch:
    """Raw aligned windows and their core time/control nodes.

    ``observations`` has shape ``(batch, view, raw_time, feature)``;
    ``negative`` has shape ``(negative_batch, raw_time, feature)``. Raw time
    includes encoder context. Time and controls contain only core nodes.
    """

    observations: torch.Tensor
    negative: torch.Tensor
    time: torch.Tensor
    negative_time: torch.Tensor
    controls: Optional[torch.Tensor]
    negative_controls: Optional[torch.Tensor]
    index: AlignedWindowIndex

    def to(self, device):
        """Move data, controls and index metadata together."""
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, value.to(device))
        return self


class AlignedTrialDataset(Dataset):
    """Repeated trials with caller-specified temporal alignment groups.

    Args:
        trials: Sequence of finite ``(time, feature)`` arrays. Features share
            one observation space; the arrays are not heterogeneous sessions.
        alignment_ids: One hashable label per trial. Trials in each group must
            have equal lengths and their time indices must already align.
        sample_interval: Positive constant time between adjacent samples.
        controls: Optional sequence of observed ``(time, control)`` arrays.
        time_origin: Time represented by the first sample of every trial.
        device: Device holding observations, controls and indexing tensors.
    """

    def __init__(self,
                 trials: Sequence,
                 alignment_ids: Sequence,
                 sample_interval: float,
                 controls: Optional[Sequence] = None,
                 time_origin: float = 0.,
                 device: str = 'cpu'):
        super().__init__(device=device)
        if not math.isfinite(sample_interval) or sample_interval <= 0:
            raise ValueError('sample_interval must be finite and positive.')
        if not math.isfinite(time_origin):
            raise ValueError('time_origin must be finite.')
        if len(trials) == 0 or len(alignment_ids) != len(trials):
            raise ValueError(
                'Supply one alignment ID for every nonempty trial sequence.')
        arrays = [_copy_float(x, device) for x in trials]
        if any(x.ndim != 2 or min(x.shape) == 0 or not torch.isfinite(x).all()
               for x in arrays):
            raise ValueError(
                'Trials must be finite, nonempty (time, feature) arrays.')
        if any(x.shape[1] != arrays[0].shape[1] for x in arrays):
            raise ValueError(
                'Aligned trials must share an observation feature dimension.')
        grouped = {}
        for i, label in enumerate(alignment_ids):
            try:
                grouped.setdefault(label, []).append(i)
            except TypeError as error:
                raise ValueError('Alignment IDs must be hashable.') from error
        self.groups = tuple(tuple(indices) for indices in grouped.values())
        self.group_lengths = tuple(
            len(arrays[group[0]]) for group in self.groups)
        for group in self.groups:
            if len({len(arrays[i]) for i in group}) != 1:
                raise ValueError(
                    'Trials in each alignment group must have equal lengths.')
        self.neural = torch.cat(arrays)
        self.lengths = torch.tensor([len(x) for x in arrays], device=device)
        self.trial_starts = torch.cat(
            (self.lengths.new_zeros(1), self.lengths.cumsum(0)[:-1]))
        self.group_index = self.lengths.new_empty(len(arrays))
        for i, group in enumerate(self.groups):
            self.group_index[list(group)] = i
        self.sample_interval = float(sample_interval)
        self.time_origin = float(time_origin)
        self.controls = None
        if controls is not None:
            if len(controls) != len(arrays):
                raise ValueError('Supply controls for every trial.')
            control_arrays = [_copy_float(u, device) for u in controls]
            if any(u.ndim != 2 or u.shape[0] != len(x) or u.shape[1] == 0 or
                   not torch.isfinite(u).all()
                   for x, u in zip(arrays, control_arrays)):
                raise ValueError(
                    'Controls must be finite (trial_time, control) arrays.')
            if any(u.shape[1] != control_arrays[0].shape[1]
                   for u in control_arrays):
                raise ValueError('Control feature dimensions must match.')
            self.controls = torch.cat(control_arrays)

    @property
    def input_dimension(self):
        """Number of observation features shared by all trials."""
        return self.neural.shape[1]

    def __len__(self):
        """Total observed samples, retaining explicit trial boundaries."""
        return len(self.neural)

    def _indices(self, trials, starts, core_length, context=True):
        trials = torch.as_tensor(trials, device=self.neural.device)
        starts = torch.as_tensor(starts, device=self.neural.device)
        if trials.dtype != torch.long or starts.dtype != torch.long:
            raise ValueError('Trial and time indices must be int64 tensors.')
        if torch.any((trials < 0) | (trials >= len(self.lengths))):
            raise ValueError('Invalid trial index.')
        starts = torch.broadcast_to(starts, trials.shape)
        left = self.offset.left if context else 0
        right = self.offset.right if context else 1
        if torch.any(starts < left) or torch.any(starts + core_length - 1 +
                                                 right > self.lengths[trials]):
            raise ValueError(
                'Requested window or encoder context crosses a trial boundary.')
        return self.trial_starts[trials][..., None] + starts[
            ..., None] + torch.arange(
                -left, core_length - 1 + right, device=self.neural.device)

    def __getitem__(self, index):
        """Load individual encoder inputs from explicit (trial, time) pairs."""
        index = torch.as_tensor(index, device=self.neural.device)
        if index.ndim != 2 or index.shape[1] != 2:
            raise ValueError(
                'Expected explicit (trial, time) pairs with shape (batch, 2).')
        idx = self._indices(index[:, 0], index[:, 1], 1)
        return self.neural[idx].transpose(-1, -2)

    def load_batch(self, index: AlignedWindowIndex) -> AlignedTrialBatch:
        """Load aligned windows without crossing any trial boundaries."""
        _positive_int(index.core_length, 'core_length')
        index.to(self.device)
        if index.trials.ndim != 2 or index.starts.shape != index.trials.shape[:
                                                                              1]:
            raise ValueError(
                'Reference indices require (batch, view) trials and (batch,) starts.'
            )
        if index.negative_trials.ndim != 1 or index.negative_starts.shape != index.negative_trials.shape:
            raise ValueError(
                'Negative trial and start vectors must have matching shapes.')
        raw = self._indices(index.trials, index.starts[:, None],
                            index.core_length)
        neg = self._indices(index.negative_trials, index.negative_starts,
                            index.core_length)
        groups = self.group_index[index.trials]
        if not torch.all(groups == groups[:, :1]):
            raise ValueError(
                'Positive views must belong to the same alignment group.')
        if index.trials.shape[1] > 1 and torch.any(
                index.trials.sort(dim=1).values.diff(dim=1) == 0):
            raise ValueError(
                'Repeated copies of a trial cannot be used as independent views.'
            )
        core = self._indices(index.trials,
                             index.starts[:, None],
                             index.core_length,
                             context=False)
        neg_core = self._indices(index.negative_trials,
                                 index.negative_starts,
                                 index.core_length,
                                 context=False)
        relative = torch.arange(index.core_length, device=self.neural.device)
        time = self.time_origin + (index.starts[:, None] +
                                   relative) * self.sample_interval
        neg_time = self.time_origin + (index.negative_starts[:, None] +
                                       relative) * self.sample_interval
        return AlignedTrialBatch(
            self.neural[raw], self.neural[neg],
            time[:, None].expand(-1, index.trials.shape[1], -1), neg_time,
            None if self.controls is None else self.controls[core],
            None if self.controls is None else self.controls[neg_core], index)


@dataclasses.dataclass
class AlignedTrialLoader(Loader):
    """Uniform alignment-group sampling with distinct views and independent negatives.

    All groups must contain enough views and enough samples for the requested
    horizon and encoder context. Singleton groups are supported with
    ``num_views=1``. Negative windows use the same group-uniform marginal,
    independently of the reference draw; collisions are retained in metadata
    for the objective's candidate policy. No unrelated groups are paired as
    positive views. ``batch_size_negatives`` counts negative windows.

    Args:
        See dataclass fields and the base Loader.
    """

    time_offset: int = dataclasses.field(
        default=1, doc='Sample stride; only one is supported.')
    rollout_steps: int = dataclasses.field(
        default=8, doc='Number of transitions in a core window.')
    num_views: int = dataclasses.field(
        default=2, doc='Distinct repetitions per reference window.')
    seed: int = dataclasses.field(
        default=42, doc='Seed for a private CPU sampling generator.')
    candidate_mode: str = dataclasses.field(
        default='independent',
        doc='Independent negative windows or in_batch trajectory candidates.')

    def __post_init__(self):
        super().__post_init__()
        if self.candidate_mode not in ('independent', 'in_batch'):
            raise ValueError('candidate_mode must be independent or in_batch.')
        if not isinstance(self.dataset, AlignedTrialDataset):
            raise ValueError(
                'AlignedTrialLoader requires an AlignedTrialDataset.')
        for name in ('num_steps', 'batch_size', 'rollout_steps', 'num_views'):
            _positive_int(getattr(self, name), name)
        if self.batch_size_negatives is None:
            self.batch_size_negatives = self.batch_size
        _positive_int(self.batch_size_negatives, 'batch_size_negatives')
        if self.time_offset != 1:
            raise ValueError(
                'Aligned-trial sampling currently requires time_offset=1.')
        self.generator = torch.Generator(device='cpu').manual_seed(self.seed)
        self._check_groups()

    def _check_groups(self):
        required = self.rollout_steps + len(self.dataset.offset)
        for group, length in zip(self.dataset.groups,
                                 self.dataset.group_lengths):
            if len(group) < self.num_views:
                raise ValueError(
                    'Every group must contain num_views distinct trials; use num_views=1 for singletons.'
                )
            if length < required:
                raise ValueError(
                    'A trial group is too short for rollout and encoder context.'
                )

    def _sample(self, count, views):
        trials, starts = [], []
        for group_id in torch.randint(len(self.dataset.groups), (count,),
                                      generator=self.generator).tolist():
            group = self.dataset.groups[group_id]
            choice = torch.randperm(len(group),
                                    generator=self.generator)[:views].tolist()
            trials.append([group[i] for i in choice])
            lower = self.dataset.offset.left
            upper = self.dataset.group_lengths[
                group_id] - self.rollout_steps - self.dataset.offset.right
            starts.append(
                torch.randint(lower, upper + 1, (),
                              generator=self.generator).item())
        return torch.tensor(trials,
                            dtype=torch.long), torch.tensor(starts,
                                                            dtype=torch.long)

    def get_indices(self, num_samples: int, num_negatives: int = None):
        """Draw reproducible windows; return explicit trial identities and starts."""
        _positive_int(num_samples, 'num_samples')
        if num_negatives is None:
            num_negatives = self.batch_size_negatives
        _positive_int(num_negatives, 'num_negatives')
        self._check_groups()
        trials, starts = self._sample(num_samples, self.num_views)
        if self.candidate_mode == 'in_batch':
            negative_trials = torch.empty((0, 1), dtype=torch.long)
            negative_starts = torch.empty(0, dtype=torch.long)
        else:
            negative_trials, negative_starts = self._sample(num_negatives, 1)
        return AlignedWindowIndex(trials, starts, negative_trials[:, 0],
                                  negative_starts, self.rollout_steps + 1).to(
                                      self.dataset.device)
