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
import functools
from typing import Literal, Optional

import numpy as np
import pytest
import torch

import cebra.data
import cebra.datasets as cebra_datasets
import cebra.distributions as cebra_distr
import cebra.distributions.base as cebra_distr_base


# ---- helpers ----


def assert_is_tensor(T, device=None):
    assert isinstance(T, torch.Tensor)
    if device is not None:
        assert T.device == device


def prepare(N=1000, n=128, d=5, probs=[0.3, 0.1, 0.6], device="cpu"):
    discrete = torch.from_numpy(np.random.choice([0, 1, 2], p=probs,
                                                 size=(N,))).to(device)
    continuous = torch.randn(N, d).to(device)

    rand = torch.from_numpy(np.random.randint(0, N, (n,))).to(device)
    _ = discrete[rand].to(device)
    query = continuous[rand] + 0.1 * torch.randn(n, d).to(device)
    query = query.to(device)

    return discrete, continuous


# ---- indices and distance matrices ----


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


# ---- mixed distributions ----


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
    _ = distribution.sample_conditional(reference_idx)

    # The conditional distribution p(· | disc, cont) should yield
    # samples where the label exactly matches the reference sample.
    samples_both = distribution.sample_conditional(reference_idx)
    assert_is_tensor(samples_both)

    # the discrete labels should exactly match for all samples
    assert torch.eq(discrete[samples_both], discrete[reference_idx]).all()

    # the returned indices should differ from the reference indices
    assert not torch.eq(samples_both, reference_idx).all()


# ---- continuous and discrete indices ----


def test_continuous(benchmark):
    discrete, continuous = prepare()

    def _test_distribution(dist):
        distribution = dist(continuous)
        reference_idx = distribution.sample_prior(10)
        _ = distribution.sample_conditional(reference_idx)
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


# ---- single-session samplers ----


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


# ---- multi-session samplers ----


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


def test_multi_session_discrete():
    dataset = cebra_datasets.init("demo-discrete-multisession")
    sampler = cebra_distr.DiscreteMultisessionSampler(dataset)

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

    # Check positive samples' labels match reference samples' labels
    sample_labels = sampler.all_data[(sample +
                                      sampler.lengths[:, None]).flatten()]
    sample_labels = sample_labels[idx.reshape(sample.shape[:2])].flatten()
    positive_labels = sampler.all_data[(positive +
                                        sampler.lengths[:, None]).flatten()]
    assert (sample_labels == positive_labels).all()


# ---- delta distributions ----


class OldDeltaDistribution(cebra_distr_base.JointDistribution,
                           cebra_distr_base.HasGenerator):
    """
    Old version of the Delta Distribution where it only works for 1d
    behavior variable.

    """

    def __init__(self,
                 continuous: torch.Tensor,
                 delta: float = 0.1,
                 device: Literal["cpu", "cuda"] = "cpu",
                 seed: Optional[int] = 1812):
        cebra_distr_base.HasGenerator.__init__(self, device=device, seed=seed)
        torch.manual_seed(seed)
        self.data = continuous
        self.std = delta
        self.index = cebra_distr.ContinuousIndex(self.data)
        self.prior = cebra_distr.Prior(self.data, device=device, seed=seed)

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
        query = torch.distributions.Normal(
            self.data[reference_idx].squeeze(),
            torch.ones_like(reference_idx, device=self.device) * self.std,
        ).sample()

        return self.index.search(query.unsqueeze(-1))


def test_old_vs_new_delta_normal_with_1Dindex():
    _, continuous = prepare()
    assert continuous.dim() == 2
    num_samples = len(continuous)
    reference_idx = torch.randint(0, num_samples, (num_samples,))

    new_distribution = cebra_distr.DeltaNormalDistribution(
        continuous=continuous[:, 0].unsqueeze(-1), delta=0.1)

    old_distribution = OldDeltaDistribution(
        continuous=continuous[:, 0].unsqueeze(-1), delta=0.1)

    torch.manual_seed(1812)
    old_positives = old_distribution.sample_conditional(reference_idx)
    torch.manual_seed(1812)
    new_positives = new_distribution.sample_conditional(reference_idx)

    assert not torch.equal(old_positives, reference_idx)
    assert not torch.equal(new_positives, reference_idx)
    assert torch.equal(old_positives, new_positives)


@pytest.mark.parametrize("delta,numerical_check", [(0.01, True), (0.025, True),
                                                   (1., False), (5., False)])
def test_new_delta_normal_with_multidimensional_index(delta, numerical_check):
    continuous = torch.rand(100_000, 3).to("cpu")
    num_samples = 1000
    delta_normal_multidim = cebra_distr.DeltaNormalDistribution(
        delta=delta, continuous=continuous)
    reference_idx = delta_normal_multidim.sample_prior(num_samples)
    positive_idx = delta_normal_multidim.sample_conditional(reference_idx)

    assert positive_idx.dim() == 1
    assert len(positive_idx) == num_samples
    assert not torch.equal(positive_idx, reference_idx)

    if numerical_check:
        reference_samples = continuous[reference_idx]
        positive_samples = continuous[positive_idx]
        diff = positive_samples - reference_samples
        #TODO(stes): Improve test, use lower error margin here
        assert torch.isclose(diff.std(), torch.tensor(delta), rtol=0.1)
    else:
        #TODO(stes): Add a warning message to the delta distribution.
        pytest.skip(
            "multivariate delta distribution can not accurately sample with the "
            "given parameters. TODO: Add a warning message for these cases.")


# ---- unified sampler ----


@pytest.mark.parametrize("time_offset", [1, 5, 10])
def test_unified_distribution(time_offset):
    dataset = cebra_datasets.init("demo-continuous-unified")
    sampler = cebra_distr.UnifiedSampler(dataset, time_offset=time_offset)

    num_samples = 5
    sample = sampler.sample_prior(num_samples)
    assert sample.shape == (dataset.num_sessions, num_samples)

    positive = sampler.sample_conditional(sample)
    assert positive.shape == (dataset.num_sessions, num_samples)


def _tiled_unified_dataset(n_frames=50, n_repeats=6, n_sessions=3, seed=0):
    """Sessions whose labels repeat exactly on every trial (same stimulus frame each repeat)."""
    gen = torch.Generator().manual_seed(seed)
    labels = torch.randn(n_frames, 4, generator=gen).repeat(n_repeats, 1)
    sessions = [
        cebra.data.TensorDataset(neural=torch.randn(n_frames * n_repeats, 3 + i, generator=gen),
                                 continuous=labels.clone()) for i in range(n_sessions)
    ]
    return cebra.data.UnifiedDataset(*sessions), labels


def test_unified_sampler_random_ties_default_is_true():
    dataset, _ = _tiled_unified_dataset()
    assert cebra_distr.UnifiedSampler(dataset, time_offset=1).random_ties is True


def test_unified_sampler_first_tie_when_random_ties_false():
    dataset, _ = _tiled_unified_dataset()
    sampler = cebra_distr.UnifiedSampler(dataset, time_offset=1, random_ties=False)
    ref = sampler.sample_prior(200, session_id=0)
    # every non-reference session is matched on the first trial: argmin returns the earliest tie
    assert (ref[1:] < 50).all()
    assert (sampler.sample_conditional(ref) < 50).all()


@pytest.mark.parametrize("set_after_init", [False, True])
def test_unified_sampler_random_ties_uses_every_repeat(set_after_init):
    n_frames, n_repeats = 50, 6
    dataset, labels = _tiled_unified_dataset(n_frames, n_repeats)
    if set_after_init:
        sampler = cebra_distr.UnifiedSampler(dataset, time_offset=1)
        sampler.random_ties = True
    else:
        sampler = cebra_distr.UnifiedSampler(dataset, time_offset=1, random_ties=True)
    np.random.seed(0)
    torch.manual_seed(0)
    ref = sampler.sample_prior(3000, session_id=0)
    pos = sampler.sample_conditional(ref)
    for idx in (ref[1:], pos):
        repeats = np.bincount((np.asarray(idx) // n_frames).ravel(), minlength=n_repeats)
        # spread over all repeats, roughly uniformly
        assert (repeats > 0).all()
        assert repeats.min() / repeats.max() > 0.7
    # matched samples are still exact label matches of the reference frame
    ref_frame = ref[0] % n_frames
    for s in range(1, ref.shape[0]):
        assert torch.equal(labels[ref[s]], labels[ref_frame])


# ---- unified sampler: nearest-label search (distinct labels, random ties) ----


def _tied_sessions(
        lengths=(60, 90, 75), n_channels=(3, 4, 5), n_frames=15, seed=0):
    """Sessions whose labels repeat exactly (ties), as in repeated stimulus presentations."""
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n_frames, 2, generator=gen)
    inputs, labels = [], []
    for length, n in zip(lengths, n_channels):
        reps = -(-length // n_frames)
        labels.append(base.repeat(reps, 1)[:length].clone())
        inputs.append(torch.randn(length, n, generator=gen))
    return inputs, labels


def _unified_sampler(labels, **kwargs):
    inputs = [torch.zeros(len(x), 1) for x in labels]
    dataset = cebra.data.UnifiedDataset(*[
        cebra.data.TensorDataset(i, continuous=l)
        for i, l in zip(inputs, labels)
    ])
    return cebra_distr.UnifiedSampler(dataset, time_offset=1, **kwargs)


def _brute(sampler, session_id, query):
    """Full row-by-row distance matrix and tie mask (reference implementation)."""
    dm = sampler.index[session_id].dist_matrix
    distance = dm(query)
    dmin = distance.min(dim=0, keepdim=True).values
    tied = distance <= dmin + sampler.tie_rtol * dmin.abs().clamp_min(1.0)
    return distance, tied


def _dup_labels(seed=0, n=120, n_unique=12, near=3):
    gen = torch.Generator().manual_seed(seed)
    base = torch.randn(n_unique, 4, generator=gen)
    labels = base[torch.randint(n_unique, (n, ), generator=gen)].clone()
    # near-duplicates: tiny perturbations, within tie_rtol of their source label
    idx = torch.randperm(n, generator=gen)[:near]
    labels[idx] += 1e-9
    return labels, base


@pytest.mark.parametrize("duplicates", [True, False])
def test_search_unique_labels_matches_brute_force(duplicates):
    if duplicates:
        labels, base = _dup_labels()
        query = torch.cat([base, base[:5] + 0.3, torch.randn(10, 4)])
        query = query[torch.randint(len(query), (200, ))]
    else:
        labels = torch.randn(80, 4)
        query = torch.randn(50, 4)
    sampler = _unified_sampler([labels, labels[:30]], random_ties=False)
    # distinct labels are only used when they are at most half of the rows
    num_labels = len(sampler._unique_labels(0)[2])
    assert num_labels == (12 if duplicates else len(labels))
    distance, tied = _brute(sampler, 0, query)
    # first tie: exactly argmin over all rows
    assert torch.equal(sampler._search(0, query), distance.argmin(dim=0))
    # random ties: always in the tie set, chunk size invariant, seed deterministic
    sampler.random_ties = True
    ref = sampler._search(0, query, torch.Generator().manual_seed(3))
    assert tied[ref, torch.arange(len(query))].all()
    for chunk_rows in (1, 7, 1000):
        sampler.max_search_elements = chunk_rows * len(labels)
        out = sampler._search(0, query, torch.Generator().manual_seed(3))
        assert torch.equal(out, ref)
    sampler.max_search_elements = 2**24
    # same u as the brute-force rank rule: identical draws
    u = torch.rand(len(query), generator=torch.Generator().manual_seed(3))
    k = tied.sum(0)
    rank = (u * k).floor().long().clamp_max(k - 1)
    expected = (tied.cumsum(0) <= rank[None]).sum(0)
    assert torch.equal(ref, expected)
    # very small distance blocks (unique-query sub-chunks) give the same result
    sampler.max_search_elements = 3 * len(labels)
    assert torch.equal(
        sampler._search(0, query,
                        torch.Generator().manual_seed(3)), ref)


def test_search_unique_labels_uniform_over_tied_rows():
    labels, base = _dup_labels(near=0)
    sampler = _unified_sampler([labels, labels[:30]], random_ties=True)
    query = base[:1].repeat(6000, 1)
    _, tied = _brute(sampler, 0, query[:1])
    tied_rows = tied[:, 0].nonzero().flatten()
    out = sampler._search(0, query, torch.Generator().manual_seed(0))
    counts = torch.bincount(out, minlength=len(labels))[tied_rows].double()
    assert counts.sum() == len(query)
    expected = len(query) / len(tied_rows)
    chi2 = ((counts - expected)**2 / expected).sum().item()
    # df = len(tied_rows) - 1; 99.9th percentile is well below df + 5 * sqrt(2 df) + 20
    df = len(tied_rows) - 1
    assert chi2 < df + 5 * (2 * df)**0.5 + 20


def test_sample_all_sessions_generator_argument():
    _, labels = _tied_sessions()
    sampler = _unified_sampler(labels, random_ties=True)
    ref = torch.arange(40)
    a = sampler.sample_all_sessions(ref,
                                    0,
                                    generator=torch.Generator().manual_seed(5))
    b = sampler.sample_all_sessions(ref,
                                    0,
                                    generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[0], ref)
