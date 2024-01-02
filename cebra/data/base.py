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
"""Base classes for datasets and loaders."""

import abc
import collections
from typing import List

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra.data.assets as cebra_data_assets
import cebra.distributions
import cebra.io
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex
from cebra.data.datatypes import Offset

__all__ = ["Dataset", "Loader"]


class Dataset(abc.ABC, cebra.io.HasDevice):
    """Abstract base class for implementing a dataset.

    The class attributes provide information about the shape of the data when
    indexing this dataset.

    Attributes:
        input_dimension: The input dimension of the signal in this dataset.
            Models applied on this this dataset should match this dimensionality.
        offset: The offset determines the shape of the data obtained with the
            ``__getitem__`` and :py:meth:`expand_index` methods.
    """

    def __init__(self,
                 device="cpu",
                 download=False,
                 data_url=None,
                 data_checksum=None,
                 location=None,
                 file_name=None):

        self.offset: Offset = cebra.data.Offset(0, 1)
        super().__init__(device)

        self.download = download
        self.data_url = data_url
        self.data_checksum = data_checksum
        self.location = location
        self.file_name = file_name

        if self.download:
            if self.data_url is None:
                raise ValueError(
                    "Missing data URL. Please provide the URL to download the data."
                )

            if self.data_checksum is None:
                raise ValueError(
                    "Missing data checksum. Please provide the checksum to verify the data integrity."
                )

            cebra_data_assets.download_file_with_progress_bar(
                url=self.data_url,
                expected_checksum=self.data_checksum,
                location=self.location,
                file_name=self.file_name)

    @property
    @abc.abstractmethod
    def input_dimension(self) -> int:
        raise NotImplementedError

    @property
    def continuous_index(self) -> torch.Tensor:
        """The continuous index, if available.

        The continuous index along with a similarity metric is used for drawing
        positive and/or negative samples.

        Returns:
            Tensor of shape ``(N,d)``, representing the
            index for all ``N`` samples in the dataset.
        """
        return None

    @property
    def discrete_index(self) -> torch.Tensor:
        """The discrete index, if available.

        The discrete index can be used for making an embedding invariant to
        a variable for to restrict positive samples to share the same index variable.
        To implement more complicated indexing operations (such as modeling similiarities
        between indices), it is better to transform a discrete into a continuous index.

        Returns:
            Tensor of shape ``(N,)``, representing the index
            for all ``N`` samples in the dataset.
        """
        return None

    def expand_index(self, index: torch.Tensor) -> torch.Tensor:
        """

        Args:
            index: A one-dimensional tensor of type long containing indices
                to select from the dataset.

        Returns:
            An expanded index of shape ``(len(index), len(self.offset))`` where
            the elements will be
            ``expanded_index[i,j] = index[i] + j - self.offset.left`` for all ``j``
            in ``range(0, len(self.offset))``.

        Note:
            Requires the :py:attr:`offset` to be set.
        """

        # TODO(stes) potential room for speed improvements by pre-allocating these tensors/
        # using non_blocking copy operation.
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)

        index = torch.clamp(index, self.offset.left,
                            len(self) - self.offset.right)

        return index[:, None] + offset[None, :]

    def expand_index_in_trial(self, index, trial_ids, trial_borders):
        """When the neural/behavior is in discrete trial, e.g) Monkey Reaching Dataset
        the slice should be defined within the trial.
        trial_ids is in size of a length of self.index and indicate the trial id of the index belong to.
        trial_borders is in size of a length of self.idnex and indicate the border of each trial.

        Todo:
            - rewrite
        """

        # TODO(stes) potential room for speed improvements by pre-allocating these tensors/
        # using non_blocking copy operation.
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)
        index = torch.tensor(
            [
                torch.clamp(
                    i,
                    trial_borders[trial_ids[i]] + self.offset.left,
                    trial_borders[trial_ids[i] + 1] - self.offset.right,
                ) for i in index
            ],
            device=self.device,
        )
        return index[:, None] + offset[None, :]

    @abc.abstractmethod
    def __getitem__(self, index: torch.Tensor) -> torch.Tensor:
        """Return samples at the given time indices.

        Args:
            index: An indexing tensor of type :py:attr:`torch.long`.

        Returns:
            Samples from the dataset matching the shape
            ``(len(index), self.input_dimension, len(self.offset))``
        """

        raise NotImplementedError

    @abc.abstractmethod
    def load_batch(self, index: BatchIndex) -> Batch:
        """Return the data at the specified index location.

        TODO: adapt signature to support Batches and List[Batch]
        """
        raise NotImplementedError()

    def configure_for(self, model: "cebra.models.Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        :py:attr:`offset` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        self.offset = model.get_offset()


@dataclasses.dataclass
class Loader(abc.ABC, cebra.io.HasDevice):
    """Base dataloader class.

    Args:
        See dataclass fields.

    Yields:
        Batches of the specified size from the given dataset object.

    Note:
        The ``__iter__`` method is non-deterministic, unless explicit seeding is implemented
        in derived classes. It is recommended to avoid global seeding in numpy
        and torch, and instead locally instantiate a ``Generator`` object for
        drawing samples.
    """

    dataset: Dataset = dataclasses.field(
        default=None,
        doc="""A dataset instance specifying a ``__getitem__`` function.""",
    )

    num_steps: int = dataclasses.field(
        default=None,
        doc=
        """The total number of batches when iterating over the dataloader.""",
    )

    batch_size: int = dataclasses.field(default=None,
                                        doc="""The total batch size.""")

    def __post_init__(self):
        if self.num_steps is None or self.num_steps <= 0:
            raise ValueError(
                f"num_steps cannot be less than or equal to zero or None. Got {self.num_steps}"
            )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(
                f"Batch size has to be None, or a non-negative value. Got {self.batch_size}."
            )

    def __len__(self):
        """The number of batches returned when calling as an iterator."""
        return self.num_steps

    def __iter__(self) -> Batch:
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield self.dataset.load_batch(index)

    @abc.abstractmethod
    def get_indices(self, num_samples: int):
        """Sample and return the specified number of indices.

        The elements of the returned `BatchIndex` will be used to index the
        `dataset` of this data loader.

        Args:
            num_samples: The size of each of the reference, positive and
                negative samples.

        Returns:
            batch indices for the reference, positive and negative sample.
        """
        raise NotImplementedError()
