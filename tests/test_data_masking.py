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
import copy

import pytest
import torch

import cebra.data.mask
from cebra.data.masking import MaskedMixin

#### Tests for Mask class ####


@pytest.mark.parametrize("mask", [
    cebra.data.mask.RandomNeuronMask,
    cebra.data.mask.RandomTimestepMask,
    cebra.data.mask.NeuronBlockMask,
])
def test_random_mask(mask: cebra.data.mask.Mask):
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)
    mask = mask(masking_value=0.5)
    masked_data = mask.apply_mask(copy.deepcopy(data))

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert (masked_data <= 1).all() and (
        masked_data >= 0).all(), "Masked data should only contain values 0 or 1"
    assert torch.sum(masked_data) < torch.sum(
        data), "Masked data should have fewer active neurons than original data"


def test_timeblock_mask():
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)
    mask = cebra.data.mask.TimeBlockMask(masking_value=(0.035, 10))
    masked_data = mask.apply_mask(copy.deepcopy(data))

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert (masked_data <= 1).all() and (
        masked_data >= 0).all(), "Masked data should only contain values 0 or 1"
    assert torch.sum(masked_data) < torch.sum(
        data), "Masked data should have fewer active neurons than original data"


#### Tests for MaskedMixin class ####


def test_masked_mixin_no_masks():
    mixin = MaskedMixin()
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)
    masked_data = mixin.apply_mask(copy.deepcopy(data))

    assert torch.equal(
        data,
        masked_data), "Data should remain unchanged when no masks are applied"


@pytest.mark.parametrize(
    "mask", ["RandomNeuronMask", "RandomTimestepMask", "NeuronBlockMask"])
def test_masked_mixin_random_mask(mask):
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)

    mixin = MaskedMixin()
    assert mixin.masks == [], "Masks should be empty initially"

    mixin.set_masks({mask: 0.5})
    assert len(mixin.masks) == 1, "One mask should be set"
    assert isinstance(mixin.masks[0],
                      getattr(cebra.data.mask,
                              mask)), f"Mask should be of type {mask}"
    if isinstance(mixin.masks[0], cebra.data.mask.NeuronBlockMask):
        assert mixin.masks[
            0].mask_prop == 0.5, "Masking value should be set correctly"
    else:
        assert mixin.masks[
            0].mask_ratio == 0.5, "Masking value should be set correctly"

    masked_data = mixin.apply_mask(copy.deepcopy(data))
    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified when a mask is applied"

    mixin.set_masks({mask: [0.5, 0.1]})
    assert len(mixin.masks) == 1, "One mask should be set"
    assert isinstance(mixin.masks[0],
                      getattr(cebra.data.mask,
                              mask)), f"Mask should be of type {mask}"
    masked_data = mixin.apply_mask(copy.deepcopy(data))
    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified when a mask is applied"

    mixin.set_masks({mask: (0.3, 0.9, 0.05)})
    assert len(mixin.masks) == 1, "One mask should be set"
    assert isinstance(mixin.masks[0],
                      getattr(cebra.data.mask,
                              mask)), f"Mask should be of type {mask}"
    masked_data = mixin.apply_mask(copy.deepcopy(data))
    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified when a mask is applied"


def test_apply_mask_with_time_block_mask():
    mixin = MaskedMixin()

    with pytest.raises(AssertionError, match="sampled_rate.*masked_seq_len"):
        mixin.set_masks({"TimeBlockMask": 0.2})

    with pytest.raises(AssertionError, match="(sampled_rate.*masked_seq_len)"):
        mixin.set_masks({"TimeBlockMask": [0.2, 10]})

    with pytest.raises(AssertionError, match="between.*0.0.*1.0"):
        mixin.set_masks({"TimeBlockMask": (-2, 10)})

    with pytest.raises(AssertionError, match="between.*0.0.*1.0"):
        mixin.set_masks({"TimeBlockMask": (2, 10)})

    with pytest.raises(AssertionError, match="integer.*greater"):
        mixin.set_masks({"TimeBlockMask": (0.2, -10)})

    with pytest.raises(AssertionError, match="integer.*greater"):
        mixin.set_masks({"TimeBlockMask": (0.2, 5.5)})

    mixin.set_masks({"TimeBlockMask": (0.035, 10)})  # Correct usage
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)
    masked_data = mixin.apply_mask(copy.deepcopy(data))

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified when a mask is applied"


def test_multiple_masks_mixin():
    mixin = MaskedMixin()
    mixin.set_masks({"RandomNeuronMask": 0.5, "RandomTimestepMask": 0.3})
    data = torch.ones(
        (10, 20,
         30))  # Example tensor with shape (batch_size, n_neurons, offset)

    masked_data = mixin.apply_mask(copy.deepcopy(data))
    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data,
        masked_data), "Data should be modified when multiple masks are applied"

    masked_data2 = mixin.apply_mask(copy.deepcopy(masked_data))
    assert masked_data2.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data,
        masked_data2), "Data should be modified when multiple masks are applied"
    assert not torch.equal(
        masked_data, masked_data2
    ), "Masked data should be different for different iterations"


def test_single_dim_input():
    mixin = MaskedMixin()
    mixin.set_masks({"RandomNeuronMask": 0.5})
    data = torch.ones((10, 1, 30))  # Single neuron
    masked_data = mixin.apply_mask(copy.deepcopy(data))

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified even with a single neuron"

    mixin = MaskedMixin()
    mixin.set_masks({"RandomTimestepMask": 0.5})
    data = torch.ones((10, 20, 1))  # Single timestep
    masked_data = mixin.apply_mask(copy.deepcopy(data))

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data,
        masked_data), "Data should be modified even with a single timestep"


def test_apply_mask_with_invalid_input():
    mixin = MaskedMixin()
    mixin.set_masks({"RandomNeuronMask": 0.5})

    with pytest.raises(ValueError, match="Data must be a 3D tensor"):
        data = torch.ones(
            (10, 20, 30, 40))  # Invalid tensor shape (extra dimension)
        mixin.apply_mask(data)

    with pytest.raises(ValueError, match="Data must be a float32 tensor"):
        data = torch.ones((10, 20, 30), dtype=torch.int32)
        mixin.apply_mask(data)


def test_apply_mask_with_chunk_size():
    mixin = MaskedMixin()
    mixin.set_masks({"RandomNeuronMask": 0.5})
    data = torch.ones((10000, 20, 30))  # Large tensor to test chunking
    masked_data = mixin.apply_mask(copy.deepcopy(data), chunk_size=1000)

    assert masked_data.shape == data.shape, "Masked data shape should match input data shape"
    assert not torch.equal(
        data, masked_data), "Data should be modified when a mask is applied"
