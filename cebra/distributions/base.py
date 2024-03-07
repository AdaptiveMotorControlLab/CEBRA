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
"""Abstract base classes for distributions and indices.

Contrastive learning in CEBRA requires a prior and conditional distribution.
Distributions are defined in terms of _indices_ that reference samples within
the dataset.

The appropriate base classes are defined in this module: An :py:class:`Index` is the
part of the dataset used to inform the prior and conditional distributions;
and could for example be time, or information about an experimental condition.
"""

import abc
import functools

import torch

import cebra.io


class HasGenerator(cebra.io.HasDevice):
    """Base class for all distributions implementing seeding.

    Args:
        device: The device the instance resides on, can be ``cpu`` or ``cuda``.
        seed: The seed to use for initializing the random number generator.

    Note:
        This class is not fully functional yet. Functionality and API might slightly
        change in upcoming versions. Do not rely (yet) on seeding provided by this class,
        but start using it to integrate seeding in all parts of ``cebra.distributions``.
    """

    def __init__(self, device: str, seed: int):
        super().__init__(device=device)
        self._device = device
        # TODO temporary
        # if seed is None:
        self._seed = None
        self._seed = self.generator.seed()

    @property
    def generator(self) -> torch.Generator:
        """The generator object.

        Can be used in many sampling methods provided by ``torch``.
        """
        if not hasattr(self, "_generator"):
            self._generator = torch.Generator(device=self.device)
            if self.seed is not None:
                self._generator.manual_seed(self.seed)
        return self._generator

    @property
    def seed(self) -> int:
        """The seed used for generating random numbers in this class."""
        return self._seed

    def to(self, device: str):
        """Move the instance to the specified device."""
        state = self.generator.get_state()
        self._generator = torch.Generator(device=device)
        try:
            self._generator.set_state(state.to(device))
        except (TypeError, RuntimeError) as e:
            # TODO(https://discuss.pytorch.org/t/cuda-rng-state-does-not-change-when-re-seeding-why-is-that/47917/3)
            self._generator.manual_seed(self.seed)

        return super().to(device)

    def randint(self, *args, **kwargs) -> torch.Tensor:
        """Generate random integers.

        See docs of ``torch.randint`` for information on the arguments.
        """
        return torch.randint(*args,
                             **kwargs,
                             generator=self.generator,
                             device=self.device)

    @property
    def device(self) -> str:
        """The device of all attributes.

        Can be ``cpu`` or ``cuda``.
        """
        return self._device


class Index(abc.ABC, cebra.io.HasDevice):
    """Base class for indexed datasets.

    Indexes contain functionality to pass a query vector, and
    return the indices of the closest matches within the index.
    """

    @abc.abstractmethod
    def search(self, query) -> torch.Tensor:
        """Return index of entry closest to query.

        Args:
            query: The query tensor to look for. The index
                computes the closest element to this query tensor
                and returns its location within the index.
                (TODO: add type)

        Returns:
            The index of the element closest to the query in
            the dataset.
        """
        raise NotImplementedError()


class PriorDistribution(abc.ABC):
    """Mixin for all prior distributions.

    Prior distributions return a batch of indices. Indexing
    the dataset with these indices will return samples from
    the prior distribution.
    """

    @abc.abstractmethod
    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Return indices for the prior distribution samples

        Args:
            num_samples: The batch size

        Returns:
            A tensor of indices. Indexing the dataset with these
            indices will return samples from the desired prior
            distribution.
        """
        raise NotImplementedError()


class ConditionalDistribution(abc.ABC):
    """Mixin for all conditional distributions.

    Conditional distributions return a batch of indices, based on
    a given batch of indices. Indexing the dataset with these indices
    will return samples from the conditional distribution.
    """

    @abc.abstractmethod
    def sample_conditional(self, query: torch.Tensor) -> torch.Tensor:
        """Return indices for the conditional distribution samples

        Args:
            query: Indices of reference samples

        Returns:
            A tensor of indices. Indexing the dataset with these
            indices will return samples from the desired conditional
            distribution.
        """
        raise NotImplementedError()


class JointDistribution(PriorDistribution, ConditionalDistribution):
    """Mixin for joint distributions."""

    def sample_joint(self, num_samples: int):
        """Return indices from the joint distribution.

        Args:
            num_samples: Desired batch size

        Returns:
            tuple containing indices of the reference and positive samples
        """
        reference = self.sample_prior(num_samples)
        positive = self.sample_conditional(reference)
        return reference, positive
