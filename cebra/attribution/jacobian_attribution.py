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
"""Tools for computing attribution maps."""

from typing import Literal

import numpy as np
import torch
from torch import nn

import cebra.attribution._jacobian

__all__ = ["get_attribution_map"]


def _prepare_inputs(inputs):
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs)
    inputs.requires_grad_(True)
    return inputs


def _prepare_model(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_attribution_map(
    model: nn.Module,
    input_data: torch.Tensor,
    double_precision: bool = True,
    convert_to_numpy: bool = True,
    aggregate: Literal["mean", "sum", "max"] = "mean",
    transform: Literal["none", "abs"] = "none",
    hybrid_solver=False,
):
    """Estimate attribution maps.
    The function estimates Jacobian matrices for each point in the model,
    computes the pseudo-inverse (for every sample), applies the `transform`
    function point-wise, and then aggregates with the `aggregate` function
    over the sample dimension.
    The result is a `(num_inputs, num_features)` attribution map.
    """
    assert aggregate in ["mean", "sum", "max"]

    input_data = _prepare_inputs(input_data)
    model = _prepare_model(model)

    # compute jacobian CEBRA model
    jf = cebra.attribution._jacobian.compute_jacobian(
        model,
        input_vars=[input_data],
        mode="autograd",
        double_precision=double_precision,
        convert_to_numpy=convert_to_numpy,
        hybrid_solver=hybrid_solver,
    )

    jhatg = np.linalg.pinv(jf)
    return jf, jhatg
