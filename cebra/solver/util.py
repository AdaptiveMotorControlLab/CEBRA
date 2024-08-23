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
"""Utility functions for solvers and their training loops."""

from collections.abc import Iterable
from typing import Dict

import literate_dataclasses as dataclasses
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import cebra.data


def _description(stats: Dict[str, float]):
    stats_str = [f"{key}: {value: .4f}" for key, value in stats.items()]
    return " ".join(stats_str)


class Meter:
    """Track statistics of a metric."""

    __slots__ = ["_num_elements", "_total"]

    def __init__(self):
        self._num_elements = 0
        self._total = 0

    def add(self, value: float, num_elements: int = 1):
        """Add the value to the meter.

        Args:
            value: The value to add to the meter.
            num_elements: Optional, if the value was already obtained
                by summing multiple elements (for example, loss values
                within a batch of data samples) and the average should
                be computed with respect to this unit.
        """
        self._total += value
        self._num_elements += num_elements

    @property
    def average(self) -> float:
        """Return the average value of the tracked metric."""
        if self._num_elements == 0:
            return float("nan")
        return self._total / float(self._num_elements)

    @property
    def sum(self) -> float:
        """Return the sum of all tracked values."""
        return self._total


@dataclasses.dataclass
class ProgressBar:
    "Log and display values during training."

    loader: Iterable
    log_format: str

    _valid_formats = ["tqdm", "off"]

    @property
    def use_tqdm(self) -> bool:
        """Display ``tqdm`` as the progress bar."""
        return self.log_format == "tqdm"

    def __post_init__(self):
        if self.log_format not in self._valid_formats:
            raise ValueError(
                f"log_format must be one of {self._valid_formats}, "
                f"but got {self.log_formats}")

    def __iter__(self):
        self.iterator = self.loader
        if self.use_tqdm:
            self.iterator = tqdm.tqdm(self.iterator)
        for num_batch, batch in enumerate(self.iterator):
            yield num_batch, batch

    def set_description(self, stats: Dict[str, float]):
        """Update the progress bar description.

        The description is updated by computing a formatted string
        from the given ``stats`` in the format ``{key}: {value: .4f}``
        with a space as the divider between dictionary elements.

        Behavior depends on the selected progress bar.
        """
        if self.use_tqdm:
            self.iterator.set_description(_description(stats))


def get_batches_of_data(inputs: torch.Tensor,
                        batch_size: int,
                        padding: bool,
                        offset: cebra.data.Offset = None):
    batches = []

    class IndexDataset(Dataset):

        def __init__(self, inputs):
            self.inputs = inputs

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            return idx

    index_dataset = IndexDataset(inputs)
    index_dataloader = DataLoader(index_dataset, batch_size=batch_size)
    for batch_id, index_batch in enumerate(index_dataloader):

        start_batch_idx, end_batch_idx = index_batch[0], index_batch[-1]
        if padding:
            if offset is None:
                raise ValueError("offset needs to be set if padding is True.")

            if batch_id == 0:
                indices = start_batch_idx, (end_batch_idx + offset.right)
                batched_data = inputs[slice(*indices)]
                batched_data = np.pad(batched_data.cpu().numpy(),
                                      ((offset.left, 0), (0, 0)),
                                      mode="edge")

            elif batch_id == len(index_dataloader) - 1:
                indices = (start_batch_idx - offset.left), end_batch_idx
                batched_data = inputs[slice(*indices)]
                batched_data = np.pad(batched_data.cpu().numpy(),
                                      ((0, offset.right), (0, 0)),
                                      mode="edge")
            else:  # Middle batches
                indices = start_batch_idx - offset.left, end_batch_idx + offset.right
                batched_data = inputs[slice(*indices)]

        else:
            indices = start_batch_idx, end_batch_idx
            batched_data = inputs[slice(*indices)]

        batched_data = torch.from_numpy(batched_data) if isinstance(
            batched_data, np.ndarray) else batched_data
        batches.append(batched_data)

    return batches
