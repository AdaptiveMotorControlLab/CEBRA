#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import functools

import numpy as np
import pytest
import torch

import cebra.datasets as cebra_datasets
import cebra.distributions as cebra_distr


def assert_is_tensor(T, device=None):
    assert isinstance(T, torch.Tensor)
    if device is not None:
        assert T.device == device


def prepare(N=1000, n=128, d=5, probs=[0.3, 0.1, 0.6], device="cpu"):
    discrete = torch.from_numpy(np.random.choice([0, 1, 2], p=probs,
                                                 size=(N,))).to(device)
    continuous = torch.randn(N, d).to(device)

    rand = torch.from_numpy(np.random.randint(0, N, (n,))).to(device)
    qidx = discrete[rand].to(device)
    query = continuous[rand] + 0.1 * torch.randn(n, d).to(device)
    query = query.to(device)

    return discrete, continuous


def test_init_discrete():
    discrete, continuous = prepare()

    # Sampling operations on discrete data distributions
    cebra_distr.discrete.Discrete(discrete)
    cebra_distr.discrete.DiscreteUniform(discrete)
    cebra_distr.discrete.DiscreteEmpirical(discrete)


def test_distance_matrix():
    _, continuous = prepare()
    matrix = cebra_distr.index.DistanceMatrix(continuous)

    query = continuous[:10]
    dist = matrix(query)
    idx = torch.argmin(dist, dim=0)
    assert dist.shape == (len(continuous), 10)
    assert idx.shape == (10,)

    assert torch.allclose(continuous[idx], continuous[:10])
    assert torch.eq(idx, torch.arange(10)).all()


def test_init_index():
    discrete, continuous = prepare()

    # Indexing operations on continuous data
    idx = cebra_distr.index.ContinuousIndex(continuous)
    cidx = cebra_distr.index.ConditionalIndex(discrete, continuous)

    a = idx.search(continuous[:10])
    assert a.shape == (10,)
    assert torch.allclose(continuous[a], continuous[:10])
    assert torch.eq(a, torch.arange(10)).all()

    cidx = cebra_distr.index.ConditionalIndex(discrete, continuous)
    b = cidx.search(continuous[:10], discrete[:10])
    assert b.shape == (10,)
    assert torch.eq(b, torch.arange(10)).all()
    assert torch.allclose(continuous[b], continuous[:10])
    assert torch.eq(b, torch.arange(10)).all()


class _TestMixedBase:

    @functools.cached_property
    def ref_idx(self):
        return self.distribution.sample_prior(self.num_samples)

    def setup_method(self):
        self.num_samples = 15
        self.discrete, self.continuous = prepare()

    def test_prior(self):
        self.setup_method()
        assert_is_tensor(self.discrete)
        assert_is_tensor(self.continuous)
        assert_is_tensor(self.ref_idx)

    def test_conditional(self):
        self.setup_method()
        # The conditional distribution p(· | disc, cont) should yield
        # samples where the label exactly matches the reference sample.
        samples_both = self.distribution.sample_conditional(
            self.discrete[self.ref_idx], self.continuous[self.ref_idx])
        assert_is_tensor(samples_both)

        # the discrete labels should exactly match for all samples
        assert torch.eq(self.discrete[samples_both],
                        self.discrete[self.ref_idx]).all()

        # the returned indices should differ from the reference indices
        assert not torch.eq(samples_both, self.ref_idx).any()


class TestMixed(_TestMixedBase):

    def setup_method(self):
        super().setup_method()
        self.distribution = cebra_distr.mixed.Mixed(self.discrete,
                                                    self.continuous)

    def test_conditional(self):
        pytest.skip("Skipping")

    def test_conditional_continuous(self):
        pytest.skip("Skipping")

        self.setup_method()
        # Sampling only based on the continuous samples, p(· | cont) should reproduce
        # the empirical distribution of discrete values.
        samples_cont = self.distribution.sample_conditional_continuous(
            self.continuous[self.ref_idx])
        assert_is_tensor(samples_cont)

        # the returned indices should differ from the reference indices
        assert not torch.eq(samples_cont, self.ref_idx).any()

        # the discrete labels are not all same, since we marginalize across the
        # discrete values
        assert not torch.eq(self.discrete[samples_cont],
                            self.discrete[self.ref_idx]).all()

    def test_conditional_discrete(self):
        self.setup_method()
        samples_disc = self.distribution.sample_conditional_discrete(
            self.discrete[self.ref_idx])
        assert_is_tensor(samples_disc)

        # a few samples can be the same, but not all of them (extremely unlikely)
        assert not all(samples_disc == self.ref_idx)

        # the discrete labels should exactly match for all samples
        assert torch.eq(self.discrete[samples_disc],
                        self.discrete[self.ref_idx]).all()


def test_mixed():
    discrete, continuous = prepare()
    distribution = cebra_distr.mixed.MixedTimeDeltaDistribution(
        discrete, continuous)

    reference_idx = distribution.sample_prior(10)
    positive_idx = distribution.sample_conditional(reference_idx)

    # The conditional distribution p(· | disc, cont) should yield
    # samples where the label exactly matches the reference sample.
    samples_both = distribution.sample_conditional(reference_idx)
    assert_is_tensor(samples_both)

    # the discrete labels should exactly match for all samples
    assert torch.eq(discrete[samples_both], discrete[reference_idx]).all()

    # the returned indices should differ from the reference indices
    assert not torch.eq(samples_both, reference_idx).all()


def test_continuous(benchmark):
    discrete, continuous = prepare()

    def _test_distribution(dist):
        distribution = dist(continuous)
        reference_idx = distribution.sample_prior(10)
        positive_idx = distribution.sample_conditional(reference_idx)
        return distribution

    distribution = _test_distribution(
        cebra_distr.continuous.TimedeltaDistribution)

    def _conditional():
        reference_idx = distribution.sample_prior(1000)
        distribution.sample_conditional(reference_idx)

    benchmark(_conditional)

    distribution = _test_distribution(cebra_distr.continuous.TimeContrastive)


def test_discrete_uniform_discrete():
    """Functional test for empirical/uniform modes"""
    np.random.seed(0)
    N = 10000
    probs = [0.3, 0.1, 0.6]
    samples = np.random.choice([0, 1, 2], p=probs, size=(N,))
    dist = cebra_distr.Discrete(samples)
    resample_uni = samples[dist.sample_uniform(N)]
    resample_emp = samples[dist.sample_empirical(N)]

    assert samples.shape == (N,)
    assert resample_uni.shape == (N,)
    assert resample_emp.shape == (N,)

    assert np.allclose(
        np.bincount(samples) / N, np.array([0.3055, 0.0974, 0.5971]))
    assert np.allclose(
        np.bincount(resample_uni) / N, np.array([0.3424, 0.3278, 0.3298]))
    assert np.allclose(
        np.bincount(resample_emp) / N, np.array([0.2969, 0.098, 0.6051]))


@pytest.mark.parametrize("time_offset", [1, 5, 10])
def test_single_session_time_contrastive(time_offset):
    """Single session time-contrastive learning.

    The test checks if sampled pairs have the correct time offset.
    TODO: check kfold datasets.
    """

    index = torch.arange(100).unsqueeze(1)
    distribution = cebra_distr.TimeContrastive(index, time_offset=time_offset)

    num_samples = 5
    sample = distribution.sample_prior(num_samples)
    assert sample.shape == (num_samples,)

    positive = distribution.sample_conditional(sample)
    assert torch.eq(positive - sample, time_offset).all()
    assert positive.shape == (num_samples,)


@pytest.mark.parametrize("index_difference", [1, 5, 10])
def test_single_session_time_delta(index_difference):
    index = torch.cumsum(torch.ones(100) * index_difference,
                         dim=0).unsqueeze(1).float()
    distribution = cebra_distr.TimedeltaDistribution(index)

    num_samples = 5
    sample = distribution.sample_prior(num_samples)
    assert sample.shape == (num_samples,)

    positive = distribution.sample_conditional(sample)
    assert positive.shape == (num_samples,)

    assert not torch.eq(positive, sample).all(
    ), "No samples have time delta of 1, hence all indices should be different"

    sample = index[sample]
    positive = index[positive]

    assert (abs(positive - sample) <= index_difference).all()


def test_single_session_discrete():
    pass


def test_multi_session_time_delta():
    pass


@pytest.mark.parametrize("time_offset", [1, 5, 10])
def test_multi_session_time_contrastive(time_offset):
    dataset = cebra_datasets.init("demo-continuous-multisession")
    sampler = cebra_distr.MultisessionSampler(dataset, time_offset=time_offset)

    num_samples = 5
    sample = sampler.sample_prior(num_samples)
    assert sample.shape == (dataset.num_sessions, num_samples)

    positive, idx, rev_idx = sampler.sample_conditional(sample)
    assert positive.shape == (dataset.num_sessions, num_samples)
    assert idx.shape == (dataset.num_sessions * num_samples,)
    assert rev_idx.shape == (dataset.num_sessions * num_samples,)
    # NOTE(celia): test the private function ``_inverse_idx()``, with idx arrays flat
    assert (idx.flatten()[rev_idx.flatten()].all() == np.arange(
        len(rev_idx.flatten())).all())
