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
"""Distributions with a mix of continuous/discrete auxiliary variables.

TODO:
    * API in this package is not yet fully stable, and the docs are quite
      sparse because of this. Extend docs/finalize API.
"""
from typing import Literal

import numpy as np
import torch

import cebra.io
from cebra.distributions.continuous import TimedeltaDistribution
from cebra.distributions.discrete import DiscreteUniform
from cebra.distributions.index import ConditionalIndex


class ConfigurableDistribution:
    """Experimental. Do not use yet.

    TODO:
        * Add full implementation or decide to remove.
    """

    # Options for configuring the index
    {
        "discrete": [
            "uniform",  # resample the discrete labels to a uniform distribution
            "empirical",  # keep the discrete labels as-is
        ],
        "continuous": [
            "time"  # time contrastive learning
            "time_delta",  # use the expected temporal difference across continuous samples
        ],
    }

    def __init__(self):
        """Not implemented yet."""
        pass

    def configure_prior(self,
                        distribution: Literal["empirical",
                                              "uniform"] = "empirical"):
        """Not implemented yet."""
        pass

    def configure_conditional(self):
        """Not implemented yet."""
        pass


class Mixed(cebra.io.HasDevice):
    """Distribution over behavior variables.

    Class combines sampling across continuous and discrete variables.
    """

    def __init__(self, discrete: torch.Tensor, continuous: torch.Tensor):
        self.uniform_prior = False
        self.prior = DiscreteUniform(discrete)
        self.conditional = ConditionalIndex(discrete, continuous)

    def sample_conditional_discrete(self,
                                    discrete: torch.Tensor) -> torch.Tensor:
        """Sample conditional on the discrete samples, marginalized across continuous."""
        return self.prior.sample_conditional(discrete)

    def sample_conditional_continuous(self,
                                      continuous: torch.Tensor) -> torch.Tensor:
        """Sample conditional on the continuous samples, marginalized across discrete."""
        return self.conditional.search(continuous, discrete=None)

    def sample_conditional(self, discrete: torch.Tensor,
                           continuous: torch.Tensor) -> torch.Tensor:
        """Sample conditional on the continuous and discrete samples"""
        return self.conditional.search(continuous, discrete=discrete)

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Sample from the uniform prior distribution."""
        return self.prior.sample_prior(num_samples)


class MixedTimeDeltaDistribution(TimedeltaDistribution):
    """Combination of a time delta and discrete distribution for sampling.

    Sampling from the prior uses the :py:class:`.DiscreteUniform` distribution.
    For sampling the conditional, it is ensured that the positive pairs share their
    behavior variable, and are then sampled according to the :py:class:`.TimedeltaDistribution`.

    See also:
        * :py:class:`.TimedeltaDistribution` for the conditional distribution.
    """

    def __init__(self, discrete, continuous, time_delta: int = 1):
        super().__init__(continuous=continuous, time_delta=time_delta)
        self.prior = DiscreteUniform(discrete)
        self.index = ConditionalIndex(discrete, continuous)

        self._discrete = discrete
        self._continuous = continuous

    def sample_prior(self, num_samples: int) -> torch.Tensor:
        """Return indices from the uniform prior distribution.

        Args:
            num_samples: The number of samples

        Returns:
            The reference indices of shape ``(num_samples, )``.
        """
        return self.prior.sample_prior(num_samples)

    def sample_conditional(self, reference_idx: torch.Tensor) -> torch.Tensor:
        """Return indices from the conditional distribution.

        Args:
            reference_idx: The reference indices.

        Returns:
            The positive indices. The positive samples will match the reference
            samples in their discrete variable, and will otherwise be drawn from
            the :py:class:`.TimedeltaDistribution`.
        """

        if reference_idx.dim() != 1:
            raise ValueError(
                f"Reference indices have wrong shape: {reference_idx.shape}. "
                "Pass a 1D array of indices of reference samples.")

        num_samples = reference_idx.size(0)
        diff_idx = torch.randint(len(self.time_difference), (num_samples,))
        query = self.data[reference_idx] + self.time_difference[diff_idx]
        return self.index.search(query, discrete=self._discrete[reference_idx])
