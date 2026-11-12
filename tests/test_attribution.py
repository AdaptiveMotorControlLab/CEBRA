import numpy as np
import pytest
import torch

import cebra.attribution
import cebra.attribution._jacobian
import cebra.attribution.jacobian_attribution as jacobian_attribution
import cebra.models
from cebra.attribution import attribution_models
from cebra.models import Model


class DummyModel(Model):

    def __init__(self):
        super().__init__(num_input=10, num_output=5)
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def get_offset(self):
        return None


@pytest.fixture
def model():
    return DummyModel()


@pytest.fixture
def input_data():
    return torch.randn(100, 10)


def test_neuron_gradient_method(model, input_data):
    attribution = attribution_models.NeuronGradientMethod(model=model,
                                                          input_data=input_data,
                                                          output_dimension=5)

    result = attribution.compute_attribution_map()

    assert 'neuron-gradient' in result
    assert 'neuron-gradient-convabs' in result
    assert result['neuron-gradient'].shape == (100, 5, 10)


def test_neuron_gradient_shap_method(model, input_data):
    attribution = attribution_models.NeuronGradientShapMethod(
        model=model, input_data=input_data, output_dimension=5)

    result = attribution.compute_attribution_map(baselines="zeros")

    assert 'neuron-gradient-shap' in result
    assert 'neuron-gradient-shap-convabs' in result
    assert result['neuron-gradient-shap'].shape == (100, 5, 10)

    with pytest.raises(NotImplementedError):
        attribution.compute_attribution_map(baselines="invalid")


def test_feature_ablation_method(model, input_data):
    attribution = attribution_models.FeatureAblationMethod(
        model=model, input_data=input_data, output_dimension=5)

    result = attribution.compute_attribution_map()

    assert 'feature-ablation' in result
    assert 'feature-ablation-convabs' in result
    assert result['feature-ablation'].shape == (100, 5, 10)


def test_integrated_gradients_method(model, input_data):
    attribution = attribution_models.IntegratedGradientsMethod(
        model=model, input_data=input_data, output_dimension=5)

    result = attribution.compute_attribution_map()

    assert 'integrated-gradients' in result
    assert 'integrated-gradients-convabs' in result
    assert result['integrated-gradients'].shape == (100, 5, 10)


def test_batched_methods(model, input_data):
    # Test batched version of NeuronGradientMethod
    attribution = attribution_models.NeuronGradientMethodBatched(
        model=model, input_data=input_data, output_dimension=5)

    result = attribution.compute_attribution_map(batch_size=32)
    assert 'neuron-gradient' in result
    assert result['neuron-gradient'].shape == (100, 5, 10)

    # Test batched version of IntegratedGradientsMethod
    attribution = attribution_models.IntegratedGradientsMethodBatched(
        model=model, input_data=input_data, output_dimension=5)

    result = attribution.compute_attribution_map(batch_size=32)
    assert 'integrated-gradients' in result
    assert result['integrated-gradients'].shape == (100, 5, 10)


def test_compute_metrics():
    attribution = attribution_models.AttributionMap(model=None, input_data=None)

    attribution_map = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
    ground_truth = np.array([False, True, False, True, False])

    metrics = attribution.compute_metrics(attribution_map, ground_truth)

    assert 'max_connected' in metrics
    assert 'mean_connected' in metrics
    assert 'min_connected' in metrics
    assert 'max_nonconnected' in metrics
    assert 'mean_nonconnected' in metrics
    assert 'min_nonconnected' in metrics
    assert 'gap_max' in metrics
    assert 'gap_mean' in metrics
    assert 'gap_min' in metrics
    assert 'gap_minmax' in metrics
    assert 'max_jacobian' in metrics
    assert 'min_jacobian' in metrics


def test_compute_attribution_score():
    attribution = attribution_models.AttributionMap(model=None, input_data=None)

    attribution_map = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
    ground_truth = np.array([False, True, False, True, False])

    score = attribution.compute_attribution_score(attribution_map, ground_truth)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_jacobian_computation():
    # Create a simple model and input for testing
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(),
                                torch.nn.Linear(5, 3))
    input_data = torch.randn(100, 10, requires_grad=True)

    # Test basic Jacobian computation
    jf, jhatg = jacobian_attribution.get_attribution_map(model=model,
                                                         input_data=input_data,
                                                         double_precision=True,
                                                         convert_to_numpy=True)

    # Check shapes
    assert jf.shape == (100, 3, 10)  # (batch_size, output_dim, input_dim)
    assert jhatg.shape == (100, 10, 3)  # (batch_size, input_dim, output_dim)


def test_tensor_conversion():
    # Test CPU and double precision conversion
    test_tensors = [torch.randn(10, 5), torch.randn(5, 3)]

    converted = cebra.attribution._jacobian.tensors_to_cpu_and_double(
        test_tensors)

    for tensor in converted:
        assert tensor.device.type == "cpu"
        assert tensor.dtype == torch.float64

    # Only test CUDA conversion if CUDA is available
    if torch.cuda.is_available():
        cuda_tensors = cebra.attribution._jacobian.tensors_to_cuda(
            test_tensors, cuda_device="cuda")
        for tensor in cuda_tensors:
            assert tensor.is_cuda
    else:
        # Skip CUDA test with a message
        pytest.skip("CUDA not available - skipping CUDA conversion test")


def test_jacobian_with_hybrid_solver():
    # Test Jacobian computation with hybrid solver
    class HybridModel(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.fc2 = torch.nn.Linear(10, 3)

        def forward(self, x):
            return self.fc1(x), self.fc2(x)

    model = HybridModel()
    # Move model to CPU to ensure test works everywhere
    model = model.cpu()
    input_data = torch.randn(50, 10, requires_grad=True)

    # Ensure input is on CPU
    input_data = input_data.cpu()

    jacobian = cebra.attribution._jacobian.compute_jacobian(
        model=model,
        input_vars=[input_data],
        hybrid_solver=True,
        convert_to_numpy=True,
        cuda_device=None  # Explicitly set to None to use CPU
    )

    # Check shape (batch_size, output_dim, input_dim)
    assert jacobian.shape == (50, 8, 10)  # 8 = 5 + 3 concatenated outputs


def test_attribution_map_transforms():
    model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU(),
                                torch.nn.Linear(5, 3))
    input_data = torch.randn(100, 10)

    # Test different aggregation methods
    for aggregate in ["mean", "sum", "max"]:
        jf, jhatg = jacobian_attribution.get_attribution_map(
            model=model, input_data=input_data, aggregate=aggregate)
        assert isinstance(jf, np.ndarray)
        assert isinstance(jhatg, np.ndarray)


def _optimized_and_reference(model, input_data, batch_size, **kwargs):
    """Maps of ``jacobian-based-batched-optimized`` and of ``jacobian-based-batched``."""
    optimized = cebra.attribution.init("jacobian-based-batched-optimized",
                                       model=model,
                                       input_data=input_data,
                                       output_dimension=3)
    reference = cebra.attribution.init("jacobian-based-batched",
                                       model=model,
                                       input_data=input_data,
                                       output_dimension=3)
    return (optimized.compute_attribution_map(batch_size=batch_size,
                                              **kwargs),
            reference.compute_attribution_map(batch_size=batch_size))


_JACOBIAN_KEYS = ("jf", "jf-convabs", "jf-inv-lsq", "jf-inv-svd",
                  "jf-convabs-inv-lsq", "jf-convabs-inv-svd")


@pytest.mark.parametrize("conv", [False, True])
def test_jacobian_batched_optimized_matches_batched(conv):
    torch.manual_seed(0)
    if conv:
        model = cebra.models.init("offset10-model",
                                  num_neurons=6,
                                  num_units=8,
                                  num_output=3)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(),
                                    torch.nn.Linear(8, 3))
    input_data = torch.randn(70, 6, requires_grad=True)
    # batch_size does not divide the number of samples: the last batch is smaller.
    full, reference = _optimized_and_reference(model, input_data, batch_size=32)
    for key in _JACOBIAN_KEYS:
        np.testing.assert_allclose(full[key], reference[key], rtol=1e-5, atol=1e-6)

    mean, _ = _optimized_and_reference(model,
                                       input_data,
                                       batch_size=32,
                                       return_full=False)
    assert mean["num_samples"] == len(reference["jf"])
    for key in _JACOBIAN_KEYS:
        np.testing.assert_allclose(mean[key],
                                   reference[key].mean(axis=0),
                                   rtol=1e-5,
                                   atol=1e-6)


def test_jacobian_batched_optimized_selected_keys_and_latent_batching():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 3))
    input_data = torch.randn(40, 6, requires_grad=True)
    full, reference = _optimized_and_reference(model,
                                               input_data,
                                               batch_size=16,
                                               selected_keys=["jf", "jf_inv_svd"],
                                               latent_batch_size=1)
    # "_" is read as "-"; only the selected maps and the timings are returned.
    assert set(full) == {"jf", "jf-inv-svd", "time_inversion_lsq", "time_inversion_svd"}
    assert full["time_inversion_lsq"] == 0.0
    np.testing.assert_allclose(full["jf"], reference["jf"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(full["jf-inv-svd"],
                               reference["jf-inv-svd"],
                               rtol=1e-5,
                               atol=1e-6)


def test_jacobian_batched_optimized_errors():
    model = torch.nn.Linear(6, 3)
    method = cebra.attribution.init("jacobian-based-batched-optimized",
                                    model=model,
                                    input_data=torch.randn(10, 6),
                                    output_dimension=3)
    with pytest.raises(ValueError, match="bigger than data"):
        method.compute_attribution_map(batch_size=11)
    with pytest.raises(ValueError, match="Unknown attribution result key"):
        method.compute_attribution_map(batch_size=5, selected_keys=["jf-abs"])


@pytest.mark.parametrize("output_batch_size", [1, 2, 3, 100])
def test_compute_jacobian_output_batching(output_batch_size):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh(),
                                torch.nn.Linear(8, 5))
    x = torch.randn(7, 6)
    kwargs = dict(double_precision=True)
    full = cebra.attribution._jacobian.compute_jacobian(
        model, [x.clone().requires_grad_(True)], **kwargs)
    batched = cebra.attribution._jacobian.compute_jacobian(
        model, [x.clone().requires_grad_(True)],
        output_batch_size=output_batch_size,
        **kwargs)
    assert full.shape == (7, 5, 6)
    np.testing.assert_allclose(batched, full, atol=1e-10)


def test_compute_jacobian_invalid_output_batch_size():
    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError):
        cebra.attribution._jacobian.compute_jacobian(
            model, [torch.randn(4, 3, requires_grad=True)],
            double_precision=True,
            output_batch_size=0)
