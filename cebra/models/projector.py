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
"""Training CEBRA with projectors."""

import torch
from torch import nn

import cebra
import cebra.models.layers as cebra_models_layers
from cebra.models import register


class _Squeeze(nn.Module):

    def forward(self, inp):
        return inp.squeeze(2)


class PointwiseLinear(nn.Module):
    """Pointwise linear layer, mapping (d,i) -> (d,j) features."""

    def __init__(self, num_parallel, num_inputs, num_outputs):
        super().__init__()

        def uniform(a, b, size):
            r = torch.rand(size)
            return r * (b - a) + a

        weight = uniform(
            -(num_inputs**0.5),
            num_inputs**0.5,
            size=(num_parallel, num_inputs, num_outputs),
        )
        bias = uniform(-(num_inputs**0.5),
                       num_inputs**0.5,
                       size=(1, 1, num_outputs))

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, inputs):
        return torch.einsum("ndi,dij->ndj", inputs, self.weight) + self.bias


class PointwiseProjector(nn.Module):
    """Projector, applied pointwise to feature output"""

    def __init__(self, num_inputs, num_units):
        super().__init__()
        self.net = nn.Sequential(
            PointwiseLinear(num_inputs, 1, num_units),
            cebra_models_layers._Skip(
                PointwiseLinear(num_inputs, num_units, num_units),
                nn.GELU(),
            ),
            cebra_models_layers._Skip(
                PointwiseLinear(num_inputs, num_units, num_units),
                nn.GELU(),
            ),
            PointwiseLinear(num_inputs, num_units, 1),
        )

        self.norm = cebra_models_layers._Norm()

    def forward(self, inputs):
        return self.norm(self.net(inputs[:, :, None]).squeeze(2))


class FeatureExtractor(nn.Sequential):

    def __init__(self, num_neurons, num_units, num_output):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3),
                                      nn.GELU()),
            cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3),
                                      nn.GELU()),
            cebra_models_layers._Skip(nn.Conv1d(num_units, num_units, 3),
                                      nn.GELU()),
            nn.Conv1d(num_units, num_output, 3),
            _Squeeze(),
        )


class ModelView:

    def __init__(self, model, index):
        self.model = model
        self.index = index

    def __call__(self, inputs):
        features = self.model.features(inputs, self.index)
        return self.model.projector(features)

    def get_offset(self) -> cebra.data.Offset:
        return cebra.data.Offset(5, 5)


@register("offset10-projector-model")
class MultisessionProjectorModel(cebra.models.Model):

    def __init__(self, num_neurons, num_units, num_output):
        super().__init__()
        self._features = nn.ModuleList([
            FeatureExtractor(num_neurons_, num_units, num_output)
            for num_neurons_ in num_neurons
        ])
        self.projector = PointwiseProjector(num_output, num_output)

    def __getitem__(self, index):
        return ModelView(self, index)

    def features(self, inp, index):
        return self._features[index](inp)

    def forward(self, inp):
        raise NotImplemented()

    def get_offset(self) -> cebra.data.Offset:
        return cebra.data.Offset(5, 5)
