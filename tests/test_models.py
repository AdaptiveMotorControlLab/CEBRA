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
import itertools

import pytest
import torch
from torch import nn

import cebra.models
import cebra.models.model
import cebra.registry


def test_registry():
    assert cebra.registry.is_registry(cebra.models)
    assert cebra.registry.is_registry(cebra.models, check_docs=True)


@pytest.mark.parametrize(
    "model_name,batch_size,input_length",
    list(itertools.product(cebra.models.get_options(), [1, 2, 7], [100, 430])),
)
def test_offset_models(model_name, batch_size, input_length):
    model = cebra.models.init(model_name,
                              num_neurons=5,
                              num_output=3,
                              num_units=4)
    assert isinstance(model, cebra.models.Model)
    assert isinstance(model, cebra.models.HasFeatureEncoder)

    if isinstance(model, cebra.models.ClassifierModel):
        model.set_output_num(label_num=12, override=False)

    offset = model.get_offset()

    # batched input
    inputs = torch.randn((batch_size, 5, len(offset)))
    outputs = model.net(inputs)
    assert isinstance(outputs, torch.Tensor)
    assert outputs.shape == (batch_size, 3)

    # check that full model works as well
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        for output in outputs:
            assert isinstance(output, torch.Tensor)
            assert len(output) == batch_size
    else:
        assert isinstance(outputs, torch.Tensor)
        assert len(outputs) == batch_size

    # full length input
    if isinstance(model, cebra.models.ConvolutionalModelMixin):
        inputs = torch.randn((batch_size, 5, input_length))
        outputs = model.net(inputs)
        if isinstance(model, cebra.models.ResampleModelMixin):
            assert outputs.shape == (
                batch_size,
                3,
                (input_length - len(offset)) // model.resample_factor + 1,
            )
        else:
            assert outputs.shape == (batch_size, 3,
                                     input_length - len(offset) + 1)

        # check that full model works as well
        outputs = model(inputs)
        assert isinstance(outputs, torch.Tensor)
        assert len(outputs) == batch_size


def test_multiobjective():

    # NOTE(stes): This test is deprecated and will be removed in a future version.
    # As of CEBRA 0.6.0, the multi objective models are tested separately in
    # test_multiobjective.py.

    class TestModel(cebra.models.Model):

        def __init__(self):
            super().__init__(num_input=10, num_output=10)
            self._model = nn.Linear(self.num_input, self.num_output)

        def forward(self, x):
            return self._model(x)

        @property
        def get_offset(self):
            return None

    model = TestModel()

    multi_model_overlap = cebra.models.MultiobjectiveModel(
        model,
        dimensions=(4, 6),
        output_mode="overlapping",
        append_last_dimension=True)
    multi_model_separate = cebra.models.MultiobjectiveModel(
        model,
        dimensions=(4, 6),
        output_mode="separate",
        append_last_dimension=True)

    x = torch.randn(5, 10)

    assert model(x).shape == (5, 10)

    assert model.num_output == multi_model_overlap.num_output
    assert model.get_offset == multi_model_overlap.get_offset

    first, second, third = multi_model_overlap(x)
    assert first.shape == (5, 4)
    assert second.shape == (5, 6)
    assert third.shape == (5, 10)

    first, second, third = multi_model_separate(x)
    assert first.shape == (5, 4)
    assert second.shape == (5, 2)
    assert third.shape == (5, 4)


@pytest.mark.parametrize("version,raises", [
    ["1.12", False],
    ["2.", False],
    ["2.0.0rc", False],
    ["2.0", False],
    ["2.5", False],
    ["1.11.0rc1", True],
    ["1.10", True],
    ["1.2", True],
    ["1.0", True],
])
def test_version_check(version, raises):

    torch.__version__ = version
    assert cebra.models.model._check_torch_version(
        raise_error=False) == (not raises)
    if raises:
        with pytest.raises(ImportError):
            cebra.models.model._check_torch_version(raise_error=True)


def test_version_check_dropout_available():
    raises = cebra.models.model._check_torch_version(raise_error=False)
    if raises:
        assert len(cebra.models.get_options("*dropout*")) == 0
    else:
        assert len(cebra.models.get_options("*dropout*")) > 0


# Tests for parametrized offset models backward compatibility
from _reference_implementations import Offset5ModelReference
from _reference_implementations import Offset10ModelReference
from _reference_implementations import Offset15ModelReference
from _reference_implementations import Offset20ModelReference
from _reference_implementations import Offset36Reference
from _reference_implementations import Offset40Reference
from _reference_implementations import Offset50Reference


@pytest.mark.parametrize("offset_n,reference_class", [
    (5, Offset5ModelReference),
    (10, Offset10ModelReference),
    (15, Offset15ModelReference),
    (20, Offset20ModelReference),
    (36, Offset36Reference),
    (40, Offset40Reference),
    (50, Offset50Reference),
])
def test_parametrized_offset_models_match_reference(offset_n, reference_class):
    """Test that parametrized offset models produce identical output to reference hardcoded models."""

    num_neurons = 5
    num_units = 8
    num_output = 3
    normalize = True

    # Create reference model
    ref_model = reference_class(num_neurons,
                                num_units,
                                num_output,
                                normalize=normalize)

    # Create parametrized model using OffsetNModel
    param_model = cebra.models.init(f"offset{offset_n}-model",
                                    num_neurons=num_neurons,
                                    num_units=num_units,
                                    num_output=num_output)

    # Test 1: Check offsets match
    ref_offset = ref_model.get_offset()
    param_offset = param_model.get_offset()
    assert ref_offset.left == param_offset.left, \
        f"Offset left mismatch for offset{offset_n}: {ref_offset.left} != {param_offset.left}"
    assert ref_offset.right == param_offset.right, \
        f"Offset right mismatch for offset{offset_n}: {ref_offset.right} != {param_offset.right}"

    # Test 2: Check model architecture - same number of parameters
    ref_params = sum(p.numel() for p in ref_model.parameters())
    param_params = sum(p.numel() for p in param_model.parameters())
    assert ref_params == param_params, \
        f"Parameter count mismatch for offset{offset_n}: {ref_params} != {param_params}"

    # Test 3: Check output shape consistency
    batch_size = 2
    input_length = 100
    offset_len = len(ref_offset)

    test_input = torch.randn(batch_size, num_neurons, offset_len)

    with torch.no_grad():
        ref_output = ref_model.net(test_input)
        param_output = param_model.net(test_input)

    assert ref_output.shape == param_output.shape, \
        f"Output shape mismatch for offset{offset_n}: {ref_output.shape} != {param_output.shape}"

    # Test 4: For convolutional models, test on full length input
    if isinstance(param_model, cebra.models.ConvolutionalModelMixin):
        test_input_full = torch.randn(batch_size, num_neurons, input_length)

        with torch.no_grad():
            ref_output_full = ref_model.net(test_input_full)
            param_output_full = param_model.net(test_input_full)

        expected_length = input_length - len(ref_offset) + 1
        assert ref_output_full.shape == (batch_size, num_output, expected_length), \
            f"Reference model output shape unexpected for offset{offset_n}"
        assert param_output_full.shape == (batch_size, num_output, expected_length), \
            f"Parametrized model output shape unexpected for offset{offset_n}"


@pytest.mark.parametrize("offset_n", [5, 10, 15, 18, 20, 31, 36, 40, 50])
def test_parametrized_offset_models_exist(offset_n):
    """Test that all parametrized offset models can be instantiated."""
    model = cebra.models.init(f"offset{offset_n}-model",
                              num_neurons=5,
                              num_units=4,
                              num_output=3)
    assert isinstance(model, cebra.models.Model)
    assert isinstance(model, cebra.models.HasFeatureEncoder)
    assert isinstance(model, cebra.models.ConvolutionalModelMixin)


@pytest.mark.parametrize("offset_n,reference_class", [
    (5, Offset5ModelReference),
    (10, Offset10ModelReference),
    (15, Offset15ModelReference),
    (20, Offset20ModelReference),
    (36, Offset36Reference),
    (40, Offset40Reference),
    (50, Offset50Reference),
])
def test_parametrized_offset_models_forward_pass_identical(
        offset_n, reference_class):
    """Test that parametrized and reference models produce identical forward pass outputs.
    
    This test verifies that when both models are initialized with the same seed and weights,
    they produce identical outputs.
    """

    num_neurons = 5
    num_units = 8
    num_output = 3
    normalize = True
    batch_size = 2

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create reference model and get its state dict
    ref_model = reference_class(num_neurons,
                                num_units,
                                num_output,
                                normalize=normalize)
    ref_state_dict = {k: v.clone() for k, v in ref_model.state_dict().items()}

    # Create parametrized model
    param_model = cebra.models.init(f"offset{offset_n}-model",
                                    num_neurons=num_neurons,
                                    num_units=num_units,
                                    num_output=num_output)

    # Load the same weights into parametrized model
    param_model.load_state_dict(ref_state_dict)

    # Test with multiple input sizes
    offset = ref_model.get_offset()
    offset_len = len(offset)

    for input_length in [offset_len, offset_len * 2, 100]:
        test_input = torch.randn(batch_size, num_neurons, input_length)

        with torch.no_grad():
            ref_output = ref_model.net(test_input)
            param_output = param_model.net(test_input)

        # Check that outputs are identical
        assert torch.allclose(ref_output, param_output, rtol=1e-5, atol=1e-7), \
            f"Output mismatch for offset{offset_n} with input_length={input_length}"

        # Check that outputs have same device and dtype
        assert ref_output.device == param_output.device, \
            f"Device mismatch for offset{offset_n}"
        assert ref_output.dtype == param_output.dtype, \
            f"Dtype mismatch for offset{offset_n}"


@pytest.mark.parametrize("offset_n", [5, 10, 15, 18, 20, 31, 36, 40, 50])
def test_parametrized_offset_models_layer_structure(offset_n):
    """Test that parametrized models have the correct layer structure."""
    num_neurons = 4
    num_units = 8
    num_output = 3

    model = cebra.models.init(f"offset{offset_n}-model",
                              num_neurons=num_neurons,
                              num_units=num_units,
                              num_output=num_output)

    # Model should have Conv1d -> GELU -> Skip layers -> Conv1d structure
    # Extract the actual network layers
    layers = list(model.net.children())

    # First layer should be Conv1d
    assert isinstance(layers[0], nn.Conv1d), \
        f"First layer of offset{offset_n} model should be Conv1d"
    assert layers[0].in_channels == num_neurons
    assert layers[0].out_channels == num_units
    assert layers[0].kernel_size == (2,)

    # Last meaningful layer (before Norm and Squeeze) should be Conv1d
    # Find the second-to-last Conv1d layer
    conv_layers = [l for l in layers if isinstance(l, nn.Conv1d)]
    assert len(conv_layers) >= 2, \
        f"offset{offset_n} model should have at least 2 Conv1d layers"

    last_conv = conv_layers[-1]
    assert last_conv.out_channels == num_output

    # Check that offset is computed correctly
    offset = model.get_offset()
    expected_left = offset_n // 2
    expected_right = offset_n // 2 + offset_n % 2

    assert offset.left == expected_left, \
        f"Offset left for offset{offset_n} should be {expected_left}, got {offset.left}"
    assert offset.right == expected_right, \
        f"Offset right for offset{offset_n} should be {expected_right}, got {offset.right}"
