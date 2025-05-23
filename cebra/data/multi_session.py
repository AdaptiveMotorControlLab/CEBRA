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
"""Datasets and loaders for multi-session training."""

import abc
from typing import List

import literate_dataclasses as dataclasses
import torch
import torch.nn as nn

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex

__all__ = [
    "MultiSessionDataset",
    "MultiSessionLoader",
    "ContinuousMultiSessionDataLoader",
    "DiscreteMultiSessionDataLoader",
    "MixedMultiSessionDataLoader",
]


class MultiSessionDataset(cebra_data.Dataset):
    """A dataset spanning multiple recording sessions.

    Multi session datasets share the same dimensionality across the index,
    but can have differing feature dimensions (e.g. number of neurons) between
    different sessions.

    Multi-session datasets where the number of neurons is constant across sessions
    should utilize the normal ``Dataset`` class with a ``MultisessionLoader`` for
    better efficiency when sampling.

    Attributes:
        offset: The offset determines the shape of the data obtained with the
            ``__getitem__`` and :py:meth:`.base.Dataset.expand_index` methods.
    """

    @property
    @abc.abstractmethod
    def num_sessions(self):
        """The number of sessions in the dataset."""
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def input_dimension(self):
        raise NotImplementedError(
            "Input dimension property not defined in for multisession. Use {get_input_dimension(session_id)} instead."
        )

    @abc.abstractmethod
    def get_input_dimension(self, session_index):
        """The feature dimension of a given session."""
        raise NotImplementedError

    def get_session(self, session_id: int) -> cebra_data.SingleSessionDataset:
        """Returns a dataset instance representing a given session."""
        raise NotImplementedError()

    @property
    def session_lengths(self) -> List[int]:
        return [len(session) for session in self.iter_sessions()]

    def iter_sessions(self):
        for i in range(self.num_sessions):
            yield self.get_session(i)

    def __getitem__(self, args) -> List[Batch]:
        """Return a set of samples from all sessions."""
        session_id, index = args
        return self.get_session(session_id).__getitem__(index)

    def load_batch(self, index: BatchIndex) -> List[Batch]:
        """Return the data at the specified index location."""
        return [
            cebra_data.Batch(
                reference=session[index.reference[session_id]],
                positive=session[index.positive[session_id]],
                negative=session[index.negative[session_id]],
                index=index.index,
                index_reversed=index.index_reversed,
            ) for session_id, session in enumerate(self.iter_sessions())
        ]

    def configure_for(self, model: "cebra.models.Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        :py:attr:`~.Dataset.offset` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        if not isinstance(model, nn.ModuleList):
            raise ValueError(
                "The model must be a nn.ModuleList to configure the dataset.")
        if len(model) != self.num_sessions:
            raise ValueError(
                f"The model must have {self.num_sessions} sessions, but got {len(model)}."
            )

        for i, session in enumerate(self.iter_sessions()):
            session.configure_for(model[i])


@dataclasses.dataclass
class MultiSessionLoader(cebra_data.Loader):
    """Dataloader for multi-session datasets.

    The loader will enforce a uniform distribution across the sessions.
    Note that if samples within different sessions share the same feature
    dimension, it is better to use a :py:class:`cebra.data.single_session.MixedDataLoader`.
    """

    def __post_init__(self):
        super().__post_init__()
        self.sampler = cebra.distributions.MultisessionSampler(
            self.dataset, self.time_offset)

    def get_indices(self, num_samples: int) -> List[BatchIndex]:
        ref_idx = self.sampler.sample_prior(self.batch_size)
        neg_idx = self.sampler.sample_prior(self.batch_size)
        pos_idx, idx, idx_rev = self.sampler.sample_conditional(ref_idx)

        ref_idx = torch.from_numpy(ref_idx)
        neg_idx = torch.from_numpy(neg_idx)
        pos_idx = torch.from_numpy(pos_idx)

        return BatchIndex(
            reference=ref_idx,
            positive=pos_idx,
            negative=neg_idx,
            index=idx,
            index_reversed=idx_rev,
        )


@dataclasses.dataclass
class ContinuousMultiSessionDataLoader(MultiSessionLoader):
    """Contrastive learning conditioned on a continuous behavior variable."""

    conditional: str = "time_delta"

    @property
    def index(self):
        return self.dataset.continuous_index


@dataclasses.dataclass
class DiscreteMultiSessionDataLoader(MultiSessionLoader):
    """Contrastive learning conditioned on a discrete behavior variable."""

    # Overwrite sampler with the discrete implementation
    # Generalize MultisessionSampler to avoid doing this?
    def __post_init__(self):
        self.sampler = cebra.distributions.DiscreteMultisessionSampler(
            self.dataset)

    @property
    def index(self):
        return self.dataset.discrete_index


@dataclasses.dataclass
class MixedMultiSessionDataLoader(MultiSessionLoader):
    pass
