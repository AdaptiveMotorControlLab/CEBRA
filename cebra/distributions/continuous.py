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
"""Distributions for sampling from continuously indexed datasets."""

from typing import Literal, Optional

import numpy as np
import torch

import cebra.data
import cebra.distributions
import cebra.distributions.base as abc_
from cebra.data.datatypes import Offset


class Prior(abc_.PriorDistribution, abc_.HasGenerator):
    """An empirical prior distribution for continuous datasets.

    Given the index, uniformly sample across time steps, i.e.,
    sample from the empirical distribution.

    Args:
        continuous: The multi-dimensional continuous index.
    """

    def __init__(self,
                 continuous: torch.Tensor,
                 device: Literal["cpu", "cuda"] = "cpu",
                 seed: int = 0):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)
        self.continuous = continuous
        self.num_samples = len(self.continuous)

    def sample_prior(self,
                     num_samples: int,
                     offset: Optional[Offset] = None) -> torch.Tensor:
        """Return uniformly sampled indices.

        Args:
            num_samples: The number of samples to draw from the prior
                distribution. This will be the length of the returned
                tensor.
            offset: The :py:class:`cebra.data.datatypes.Offset` offset
                to be respected when sampling indices. The minimum index
                sampled will be ``offset.left`` (inclusive), the maximum
                index will be the index length minus ``offset.right``
                (exclusive).

        Returns:
            An integer tensor of shape ``num_samples`` containing
            random indices
        """

        if offset is not None and self.num_samples <= len(self.offset):
            raise ValueError(
                f"Dataset is too small, got {self.num_samples} timepoints "
                f"to train a model with offset {self.offset}.")

        return self.randint(
            0 if offset is None else self.offset.left,
            self.num_samples - (0 if offset is None else self.offset.right),
            (num_samples,),
        )


class TimeContrastive(abc_.JointDistribution, abc_.HasGenerator):
    """Time contrastive learning.

    Positive samples will have a distance of exactly :py:attr:`time_offset`
    samples in time.

    Attributes:
        continuous: The multi-dimensional continuous index.
        time_offset: The time delay between samples that form a positive pair
        num_samples: TODO(stes) remove?
        device: Device (cpu or gpu)
        seed: The seed for sampling from the prior and negative distribution
            (TODO currentlty not used)

    TODO:
        Implement time contrastive learning across multiple time steps, e.g.
        by sampling the time offset in the conditional distribution.
    """

    def __init__(
        self,
        continuous: Optional[torch.Tensor] = None,
        time_offset: int = 1,
        num_samples: Optional[int] = None,
        device: Literal["cpu", "cuda"] = "cpu",
        seed: Optional[int] = None,
    ):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)
        if continuous is None and num_samples is None:
            raise ValueError(
                f"Supply either a continuous index (which will be used to infer the dataset size) "
                f"or alternatively the number of datapoints using the num_samples argument."
            )
        if continuous is not None and num_samples is not None:
            if len(continuous) != num_samples:
                raise ValueError(
                    f"It is sufficient to provide either the continuous index, or the num_samples "
                    f"argument. You provided both, and the length of continuous ({len(continuous)} does "
                    f"not match num_samples={num_samples} you provided.")
        self.num_samples = len(
            continuous) if num_samples is None else num_samples
        self.time_offset = time_offset
        if self.num_samples <= self.time_offset:
            raise ValueError(
                f"number of samples has to exceed the time offset, but got {self.num_samples} <= {self.time_offset}."
            )

    def sample_prior(self,
                     num_samples: int,
                     offset: Optional[Offset] = None) -> torch.Tensor:
        """Return a random index sample, respecting the given time offset.

        Prior samples are uniformly sampled from ``[0, T - t)`` where ``T`` is the total
        number of samples in the index, and ``t`` is the time offset used for sampling.

        Args:
            num_samples: Number of time steps to draw uniformly from the
                number of available time steps in the dataset
            offset: The model offset to respect for sampling from the prior.
                TODO not yet implemented

        Returns:
            A ``(num_samples,)`` shaped tensor containing time indices from the
            uniform prior distribution.
        """
        if offset is not None:
            raise NotImplementedError(
                f"It is not yet supported to use an offset for sampling from the index "
                f"prior for time contrastive learning. The offset is fixed to "
                f"the specified time_offset={self.time_offset}")
        return self.randint(0, self.num_samples - self.time_offset,
                            (num_samples,))

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Return samples from the time-contrastive conditional distribution.

        The returned indices will be given by incrementing the reference indices
        by the specified :py:attr:`time_offset`. When the reference indices are
        sampled with :py:meth:`sample_prior`, it is ensured that the indices all
        lie within the bounds of the dataset.

        Args:
            reference_idx: The time indices of the reference samples

        Returns:
            A ``(len(reference_idx),)`` shaped tensor containing time indices from the
            time-contrastive conditional distribution. The samples will be simply
            offset by :py:attr:`time_offset` from ``reference_idx``.
        """
        return reference_idx + self.time_offset


class DirectTimedeltaDistribution(TimeContrastive, abc_.HasGenerator):
    """Look up indices with

    Todo:
        - This class is work in progress.
    """

    def __init__(self, continuous: torch.Tensor, time_offset: int = 1):
        super().__init__(continuous=continuous, time_offset=time_offset)
        self.index = cebra.distributions.ContinuousIndex(self.data)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Samples from the conditional distribution.

        Todo:
            - class and this function is work in progress.
        """
        query_idx = super().sample_conditional(reference_idx)
        query = self.index[query_idx]
        # TODO(stes): This will by default simply return the query_idx. This should
        # be covered by a test and fixed for a future release.
        return self.index.search(query)


class TimedeltaDistribution(abc_.JointDistribution, abc_.HasGenerator):
    """Define a conditional distribution based on behavioral changes over time.

    Takes a continuous index, and uses the empirical distribution of differences
    between samples in this index.

    Args:
        continuous: The multidimensional, continuous index
        time_delta: The time delay between samples that should form a positive
            pair.
        device: TODO
        seed: TODO

    Note:
        For best results, the given continuous index should contain independent
        factors; positive pairs will be formed by adding a _random_ difference
        estimated within the dataset to the reference samples.
        Factors should ideally also be within the same range (since the Euclidean
        distance is used in the search). A simple solution is to perform a PCA
        or ICA, or apply CEBRA first before passing the index to this function.
    """

    def __init__(self,
                 continuous,
                 time_delta: int = 1,
                 device: Literal["cpu", "cuda"] = "cpu",
                 seed: Optional[int] = None):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)
        self.data = continuous
        self.time_delta = time_delta
        self.time_difference = torch.zeros_like(self.data, device=self.device)
        self.time_difference[time_delta:] = (self.data[time_delta:] -
                                             self.data[:-time_delta])
        self.index = cebra.distributions.ContinuousIndex(self.data)
        self.prior = Prior(self.data, device=device, seed=seed)

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """See :py:meth:`.Prior.sample_prior`."""
        return self.prior.sample_prior(num_samples)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Return indices from the conditional distribution."""

        if reference_idx.dim() != 1:
            raise ValueError(
                f"Reference indices have wrong shape: {reference_idx.shape}. "
                "Pass a 1D array of indices of reference samples.")

        num_samples = reference_idx.size(0)
        diff_idx = self.randint(len(self.time_difference), (num_samples,))
        query = self.data[reference_idx] + self.time_difference[diff_idx]
        return self.index.search(query)


class DeltaNormalDistribution(abc_.JointDistribution, abc_.HasGenerator):
    """Define a conditional distribution based on behavioral changes over time.

    Takes a continuous index, and uses sample from Gaussian distribution to sample positive pairs.
    Note that if the continuous index is multidimensional, the Gaussian distribution will have
    isotropic covariance matrix i.e. Σ = sigma^2 * I.

    Args:
        continuous: The multidimensional, continuous index.
        delta: Standard deviation of Gaussian distribution to sample positive pair.

    """

    def __init__(self,
                 continuous: torch.Tensor,
                 delta: float = 0.1,
                 device: Literal["cpu", "cuda"] = "cpu",
                 seed: Optional[int] = None):
        abc_.HasGenerator.__init__(self, device=device, seed=seed)
        self.data = continuous
        self.std = delta
        self.index = cebra.distributions.ContinuousIndex(self.data)
        self.prior = Prior(self.data, device=device, seed=seed)

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """See :py:meth:`.Prior.sample_prior`."""
        return self.prior.sample_prior(num_samples)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Return indices from the conditional distribution."""

        if reference_idx.dim() != 1:
            raise ValueError(
                f"Reference indices have wrong shape: {reference_idx.shape}. "
                "Pass a 1D array of indices of reference samples.")

        # TODO(stes): Set seed
        mean = self.data[reference_idx]
        query = torch.distributions.Normal(
            loc=mean,
            scale=torch.ones_like(mean, device=self.device) * self.std,
        ).sample()

        query = query.unsqueeze(-1) if query.dim() == 1 else query
        return self.index.search(query)


class CEBRADistribution(abc_.JointDistribution):
    """Use CEBRA embeddings for defining a conditional distribution.

    TODO:
        - This class is not implemented yet. Contributions welcome!
    """

    pass


def _interleave(tensor: torch.Tensor, N: int) -> torch.Tensor:
    size = tensor.size()
    return tensor.reshape(N, N, -1).transpose(1, 0).reshape(*size)
