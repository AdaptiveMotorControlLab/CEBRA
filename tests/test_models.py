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


def test_version_check():
    raises = not cebra.models.model._check_torch_version(raise_error=False)
    if raises:
        assert len(cebra.models.get_options("*dropout*")) == 0
    else:
        assert len(cebra.models.get_options("*dropout*")) > 0
