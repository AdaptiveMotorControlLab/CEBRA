"""Discrete indices."""

from typing import Literal, Optional, Union
import numpy as np
import scipy.interpolate
import torch

import cebra.distributions.base as abc_


class Discrete(abc_.ConditionalDistribution, abc_.HasGenerator):
    """Resample 1-dimensional discrete data.

    The distribution is fully specified by an array of discrete samples.
    Samples can be drawn either from the dataset directly (i.e., output
    samples will have the same distribution of class labels as the samples
    used to specify the distribution), or from a resampled data distribution

    Args:
        samples: Discrete index used for sampling
    """

        if isinstance(samples, torch.Tensor):
            samples = samples.cpu().numpy()
        if samples.dtype not in (np.int32, np.int64):
            samples = samples.astype(int)
        return samples

        abc_.HasGenerator.__init__(self, device=device, seed=None)
        self._set_data(samples)
        self.sorted_idx = torch.from_numpy(np.argsort(self.samples))
        self._init_transform()

        samples = self._to_numpy_int(samples)
        if samples.ndim > 1:
            raise ValueError(
                f"Data dimensionality is {samples.shape}, but can only accept a single dimension."
            )
        self.samples = samples

    @property
    def num_samples(self) -> int:
        """Number of samples in the index."""
        return len(self.samples)

    def _init_transform(self):
        self.counts = np.bincount(self.samples)
        self.cdf = np.zeros((len(self.counts) + 1,))
        self.cdf[1:] = np.cumsum(self.counts)
        # NOTE(stes): This is the only use of a scipy function in the entire code
        # base for now. Replacing scipy.interpolate.interp1d with an equivalent
        # function from torch would make it possible to drop scipy as a dependency
        # of the package.
        self.transform = scipy.interpolate.interp1d(
            np.linspace(0, self.num_samples, len(self.cdf)), self.cdf)

        """Draw samples from the uniform distribution over values.
        This will change the likelihood of values depending on the values
        in the given (discrete) index. When reindexing the dataset with
        the returned indices, all values in the index will appear with
        equal probability.
        Args:
            num_samples: Number of uniform samples to be drawn.
        Returns:
            A batch of indices from the distribution. Reindexing the
            index samples of this instance with the returned in indices
            will yield a uniform distribution across the discrete values.
        """
        samples = np.random.uniform(0, self.num_samples, (num_samples,))
        samples = self.transform(samples).astype(int)
        return self.sorted_idx[samples]

        """Draw samples from the empirical distribution.
        Args:
            num_samples: Number of samples to be drawn.
        Returns:
            A batch of indices from the empirical distribution,
            which is the uniform distribution over ``[0, N-1]``.
        """
        samples = np.random.randint(0, self.num_samples, (num_samples,))
        return self.sorted_idx[samples]

        """Draw samples conditional on template samples.
        Args:
                prior distribution. Conditional samples will match
                the values of these indices
        Returns:
            batch of indices, whose values match the values
            corresponding to the given indices.
        """
        reference_index = self._to_numpy_int(reference_index)
        idx = np.random.uniform(0, 1, len(reference_index))
        idx *= self.cdf[reference_index + 1] - self.cdf[reference_index]
        idx += self.cdf[reference_index]
        idx = idx.astype(int)

        return self.sorted_idx[idx]


class DiscreteUniform(Discrete, abc_.PriorDistribution):
    """Re-sample the given indices and produce samples from a uniform distribution."""

        return self.sample_uniform(num_samples)


class DiscreteEmpirical(Discrete, abc_.PriorDistribution):
    """Draw samples from the empirical distribution defined by the passed index."""

        return self.sample_empirical(num_samples)


DiscreteUniform.sample_prior.__doc__ = Discrete.sample_uniform.__doc__
DiscreteEmpirical.sample_prior.__doc__ = Discrete.sample_empirical.__doc__
