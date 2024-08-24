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
