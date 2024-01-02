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
"""Utilities for simulating spike counts for Poisson-like neuron models."""

import dataclasses

import joblib
import numpy as np
import scipy.stats
import torch
import torch.distributions
from torch import nn


class PoissonNeuronTransform(nn.Module):
    """Transform spike rates into expected spike counts.

    This is an implementation for transforming arrays or tensors containing spike
    rates into expected spike counts.

    Args:
        num_neurons: The number of neurons to simulate. Needs to match the
            dimensions of the array passed to :py:meth:`__call__`.
        refractory_period: The neuron's absolute refractory period, in seconds.
            The absolute refactory period is the lower bound for the inter-spike
            interval for each neuron.

    References:
        https://neuronaldynamics.epfl.ch/online/Ch7.S3.html
    """

    def __init__(self, num_neurons: int, refractory_period: float = 0.0):
        super().__init__()
        if refractory_period < 0:
            raise ValueError(
                f"Refractory period needs to be non-negative, but got {refractory_period}"
            )
        if num_neurons <= 0:
            raise ValueError(
                f"num_neurons needs to be positive, but got {num_neurons}")
        self.refractory_period = refractory_period
        self.num_neurons = num_neurons

    def __call__(self, spike_rates: torch.Tensor) -> torch.Tensor:
        """Sample spike counts from spike rates

        Args:
            spike_rates: The non-negative spike rates for each neuron, in a
                tensor with shape ``neurons x trials x timesteps``. The number
                of neurons needs to match :py:attr:`num_neurons`.

        Returns:
            A tensor of same shape as the input array, containing a sample
            of spike counts.
        """
        n_neurons, n_trials, n_timesteps = spike_rates.shape
        assert n_neurons == self.num_neurons

        time_interval = 1

        for n_sigmas in [4, 8, 12, 16, 24, 48, 96]:
            num_spikes = int(spike_rates.max() * (time_interval *
                                                  (1 + n_sigmas)))

            # NOTE(stes): See https://pytorch.org/docs/stable/distributions.html#exponential, the
            # parameter of the distribution is directly 1/scale.
            delta_distribution = torch.distributions.Exponential(
                rate=spike_rates)

            # NOTE(stes): The sample dimension is appended to the front of the array (dim 0)
            deltas = delta_distribution.sample(
                (num_spikes,)) + self.refractory_period

            spike_times = torch.cumsum(deltas, dim=0)

            # At least 2 spikes should occur after the window to avoid bias
            if (spike_times[-2] > time_interval).all():
                break
        else:
            raise ValueError()
        return (spike_times < time_interval).sum(dim=0)


def _sample_batch(spike_rates: torch.Tensor, refractory_period: float = 0):
    sample_poisson = PoissonNeuronTransform(num_neurons=spike_rates.shape[0],
                                            refractory_period=refractory_period)
    return sample_poisson(spike_rates)


def sample_parallel(spike_rates,
                    refractory_period: float = 0.0,
                    n_jobs: int = 10):
    """Generate spike counts from the specified spike rates.

    Args:
        spike_rates: The (non-negative) spike rates, with shape ``neurons x trials x time.``
            The number of neurons needs to be divisible by ``n_jobs``.
        n_jobs: The number of parallel jobs for generating the spike trains.

    Returns:
        A tensor of shape ``neurons x trials x time`` which contains the spike counts.
    """

    if any(spike_rates.flatten() < 0):
        raise ValueError("Need to pass non-negative values as the spike rate.")

    spike_rates = spike_rates.view(n_jobs, -1, *spike_rates.shape[1:])
    _sample_batch_delayed = joblib.delayed(_sample_batch)
    spike_counts = joblib.Parallel(n_jobs)(
        _sample_batch_delayed(batch, refractory_period=refractory_period)
        for batch in spike_rates)
    spike_counts = torch.cat(spike_counts, dim=0)

    return spike_counts


@dataclasses.dataclass
class PoissonNeuron:
    """A Poisson neuron model with different sampling methods.

    This is a reference implementation that can be used for testing generation of
    synthetic datasets with Poisson spiking.

    References:
      https://neuronaldynamics.epfl.ch/online/Ch7.S3.html
    """

    # the neuron's spike rate, in Hz
    spike_rate: float = 40.0

    # the length in seconds of the time interval to consider
    # for estimating the spike count. The count
    # will be the total count for the whole duration.
    time_interval: float = 1

    # the number of repeats. If you see irregularities, increase
    # this value.
    num_repeats: int = 10000

    # how many sigmas of the poisson distribution to consider.
    # you can leave this typically in the range of 1-3 sigmas
    data_range_sigmas: float = 10

    @property
    def _expected_count(self):
        """The expected number of spike counts, estimated as lambda * (1 + data_range_sigmas)"""
        return int(self.spike_rate * self.time_interval *
                   (1 + self.data_range_sigmas))

    def _get_counts(self, refractory_period: float = 0.0):
        """Estimate spike counts by sampling individual spikes"""
        max_spike_count = self._expected_count
        deltas = scipy.stats.expon(scale=1.0 / self.spike_rate).rvs(
            (self.num_repeats, max_spike_count))
        deltas += refractory_period
        spike_times = np.cumsum(deltas, axis=-1)

        latest_spike_time = spike_times.max(axis=-1).min()
        if latest_spike_time < self.time_interval:
            raise ValueError(
                f"The simulated number of spikes were not sufficient to complete the simulation. "
                f"You specified a time interval of {self.time_interval}s, but the last spike occurred "
                f"at {latest_spike_time}s. We simulated {max_spike_count} spikes in total. "
                f"Try to either increase data_range_sigmas (={self.data_range_sigmas}) or to decrease "
                f"the time_interval.")

        return (spike_times < self.time_interval).sum(axis=-1)

    def sample_spikes(self, refractory_period: float = 0.0):
        """Estimate count histogram by actually simulating spikes.

        Args:
            refractory_period: The refractory period (= cutoff of the exponential distribution
                we sample the interspike intervals from).

        Returns:
            histogram (bins and counts) of spike counts, estimated over the specified
            number of repeats.
        """
        total_counts = self._get_counts(refractory_period=refractory_period)
        histogram = np.bincount(total_counts)
        return np.arange(len(histogram)), histogram

    def sample_poisson(self, spike_rate=None, range_=None):
        """Compute count histogram with a homogeneous Poisson process.

        Returns:
            histogram (bins and counts) of spike counts, estimated over the specified
            number of repeats.
        """
        if spike_rate is None:
            spike_rate = self.spike_rate
        if range_ is None:
            range_ = (0, self._expected_count)
        t = np.arange(*range_)
        poisson = scipy.stats.poisson(spike_rate * self.time_interval).pmf
        return t, np.round(poisson(t) * self.num_repeats)

    def sample_poisson_estimate(self,
                                refractory_period: float = 0.0,
                                range_=None):
        """Compute count histogram with a Poisson distribution fitted to simulated data.

        The function first samples spikes using :py:meth:`sample_spikes`, then estimates the
        maximum likelihood fit of a Poisson distribution (by computing the mean number of spikes
        over all repeats).

        The count histogram is then computed from this estimated poisson distribution.

        Returns:
            histogram (bins and counts) of spike counts, estimated over the specified
            number of repeats.
        """
        total_counts = self._get_counts(
            refractory_period=refractory_period).mean()
        return self.sample_poisson(spike_rate=total_counts / self.time_interval,
                                   range_=range_)
