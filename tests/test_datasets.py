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
import numpy as np
import pytest
import torch
import tqdm

import cebra.data
import cebra.datasets
import cebra.registry
from cebra.datasets import poisson


def test_registry():
    """Check the registry: Are all functions defined and is the
    docstring correctly adapted?"""
    assert cebra.registry.is_registry(cebra.datasets)
    assert cebra.registry.is_registry(cebra.datasets, check_docs=True)


def test_factory():
    """Register a new dataset"""
    import cebra.datasets

    @cebra.datasets.register("test-data")
    class TestDataset:
        pass

    assert "test-data" in cebra.datasets.get_options()
    instance = cebra.datasets.init("test-data")
    assert isinstance(instance, TestDataset)


def test_demo():
    dataset = cebra.datasets.init("demo-discrete")
    indices = torch.arange(0, 5)
    batch = dataset[indices]

    assert len(batch) == len(indices)


@pytest.mark.requires_dataset
def test_hippocampus():
    from cebra.datasets import hippocampus

    pytest.skip("Outdated")
    dataset = cebra.datasets.init("rat-hippocampus-single")
    loader = cebra.data.ContinuousDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        conditional="time_delta",
    )
    for batch in loader:
        assert (len(batch.reference) == len(batch.positive)) and (len(
            batch.positive) == len(batch.negative))
        break

    dataset = cebra.datasets.init("rats-hippocampus-multisubjects")
    loader = cebra.data.ContinuousMultiSessionDataLoader(
        dataset=dataset,
        num_steps=10,
        batch_size=8,
        conditional="time_delta",
    )
    for batch in loader:
        for b in batch:
            assert (len(b.reference) == len(b.positive)) and (len(
                b.positive) == len(b.negative))
        break


@pytest.mark.requires_dataset
def test_monkey():
    from cebra.datasets import monkey_reaching

    dataset = cebra.datasets.init(
        "area2-bump-pos-active-passive",
        path=cebra.datasets.get_datapath("monkey_reaching_preload_smth_40/"),
    )
    indices = torch.randint(0, len(dataset), (10,))
    assert len(indices) == len(dataset[indices])


@pytest.mark.requires_dataset
def test_allen():
    from cebra.datasets import allen

    pytest.skip("Test takes too long")

    ca_dataset = cebra.datasets.init("allen-movie-one-ca-VISp-100-train-10-111")
    ca_loader = cebra.data.ContinuousDataLoader(
        dataset=ca_dataset,
        num_steps=10,
        batch_size=8,
        conditional="time_delta",
    )
    for batch in ca_loader:
        assert (len(batch.reference) == len(batch.positive)) and (len(
            batch.positive) == len(batch.negative))
        break
    joint_dataset = cebra.datasets.init(
        "allen-movie-one-ca-neuropixel-VISp-100-train-10-111")
    joint_loader = cebra.data.ContinuousMultiSessionDataLoader(
        dataset=joint_dataset,
        num_steps=10,
        batch_size=8,
        conditional="time_delta",
    )
    for batch in joint_loader:
        for b in batch:
            assert (len(b.reference) == len(b.positive)) and (len(
                b.positive) == len(b.negative))
        break


try:
    options = cebra.datasets.get_options("*")
    multisubject_options = cebra.datasets.get_options(
        "allen-movie1-ca-multi-session-*")
    multisubject_options.extend(
        cebra.datasets.get_options(
            "rat-hippocampus-multisubjects-3fold-trial-split*"))
except:
    options = []


@pytest.mark.requires_dataset
@pytest.mark.parametrize("options",
                         cebra.datasets.get_options("*",
                                                    expand_parametrized=False))
def test_options(options):
    assert len(options) > 0
    assert len(multisubject_options) > 0


@pytest.mark.requires_dataset
@pytest.mark.parametrize("dataset", options)
def test_all(dataset):
    import cebra.datasets

    data = cebra.datasets.init(dataset)
    assert (data.continuous_index is not None) or (data.discrete_index
                                                   is not None)
    assert isinstance(data, cebra.data.base.Dataset)


@pytest.mark.requires_dataset
@pytest.mark.parametrize("dataset", multisubject_options)
def test_all_multisubject(dataset):
    # NOTE(stes) In theory a duplicate of test_all, but allows to quickly double check #611 won't re-appear
    # in the future. Will keep it in, but fine to remove at a later point once dataset tests are optimized.
    import cebra.datasets
    data = cebra.datasets.init(dataset)
    assert (data.continuous_index is not None) or (data.discrete_index
                                                   is not None)
    assert isinstance(data, cebra.data.base.Dataset)


@pytest.mark.requires_dataset
@pytest.mark.parametrize("dataset", [
    "allen-movie1-ca-multi-session-leave2out-repeat-0-train",
    "allen-movie1-ca-multi-session-decoding-repeat-1-test",
    "rat-hippocampus-multisubjects-3fold-trial-split",
    "rat-hippocampus-multisubjects-3fold-trial-split-0"
])
def test_compat_fix611(dataset):
    """Check that confirm the fix applied in internal PR #611

    The PR removed the explicit continuous and discrete args from the 
    datasets used to parametrize this function. We manually check that
    the continuous index is available, and no discrete index is set.

    https://github.com/AdaptiveMotorControlLab/CEBRA-dev/pull/613
    """
    import cebra.datasets
    data = cebra.datasets.init(dataset)
    assert (data.continuous_index is not None)
    assert (data.discrete_index is None)
    assert isinstance(data, cebra.data.datasets.DatasetCollection)


def _assert_histograms_close(values, histogram):
    max_counts = max(max(values), len(histogram))

    value_mean = values.mean()
    histogram_mean = (histogram *
                      np.arange(len(histogram))).sum() / histogram.sum()
    assert np.isclose(value_mean, histogram_mean, rtol=0.05)

    value_histogram = np.bincount(values, minlength=max_counts)
    # NOTE(stes) normalize the histograms to be able to use the same atol values in the histogram
    # test below
    value_histogram = value_histogram / float(value_histogram.sum())
    histogram = histogram / float(histogram.sum())
    if len(histogram) < len(value_histogram):
        histogram = np.pad(
            histogram,
            pad_width=[(0, len(value_histogram) - len(histogram))],
            mode="constant",
            constant_values=(0, 0),
        )

    assert value_histogram.shape == histogram.shape
    # NOTE(stes) while the relative tolerance here is quite high (20%), this is a tradeoff vs. speed.
    # For lowering the tolerance, the number of samples drawn in the test methods needs to be increased.
    assert np.allclose(value_histogram, histogram, atol=0.05, rtol=0.25)


def test_poisson_reference_implementation():
    spike_rate = 40
    num_repeats = 500

    neuron_model = poisson.PoissonNeuron(
        spike_rate=spike_rate,
        num_repeats=num_repeats,
    )

    def _check_histogram(bins, hist):
        assert len(bins) == len(hist)
        assert (hist >= 0).all()

    bins, hist = neuron_model.sample_spikes()
    _check_histogram(bins, hist)
    assert hist.sum() == num_repeats

    bins, hist = neuron_model.sample_poisson_estimate()
    _check_histogram(bins, hist)

    bins, hist = neuron_model.sample_poisson()
    _check_histogram(bins, hist)


@pytest.mark.parametrize("spike_rate", 10**np.linspace(0, 2.0))
def test_homogeneous_poisson_sampling(spike_rate):
    torch.manual_seed(0)
    np.random.seed(0)
    spike_rates = spike_rate * torch.ones((10, 2000, 1))
    spike_counts = poisson._sample_batch(spike_rates)
    assert spike_counts.shape == spike_rates.shape

    neuron_model = poisson.PoissonNeuron(spike_rate=spike_rate,
                                         num_repeats=spike_counts.numel())
    _, reference_counts = neuron_model.sample_poisson(
        range_=(0, spike_counts.max() + 1))

    _assert_histograms_close(spike_counts.flatten().numpy(), reference_counts)


@pytest.mark.parametrize(
    "spike_rate,refractory_period",
    [[10, 0.02], [30, 0.01], [50, 0.0], [80, 0.1], [100, 0.01], [1, 0.001],
     [2, 0.0]],
)
def test_poisson_sampling(spike_rate, refractory_period):
    torch.manual_seed(0)
    np.random.seed(0)
    spike_rates = spike_rate * torch.ones((10, 2000, 1))
    spike_counts = poisson._sample_batch(spike_rates,
                                         refractory_period=refractory_period)
    transform = poisson.PoissonNeuronTransform(
        num_neurons=10, refractory_period=refractory_period)
    spike_counts = transform(spike_rates)
    assert spike_counts.shape == spike_rates.shape

    neuron_model = poisson.PoissonNeuron(spike_rate=spike_rate,
                                         num_repeats=spike_counts.numel())

    _, reference_counts = neuron_model.sample_spikes(
        refractory_period=refractory_period)

    _assert_histograms_close(spike_counts.flatten().numpy(), reference_counts)
