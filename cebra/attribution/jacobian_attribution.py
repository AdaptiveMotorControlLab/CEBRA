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
    hybrid_solver: bool = False,
):
    """Estimate attribution maps using the Jacobian pseudo-inverse.

    The function estimates Jacobian matrices for each point in the model,
    computes the pseudo-inverse (for every sample) and then aggregates
    the resulting matrices to compute an attribution map.

    Args:
        model: The neural network model for which to compute attributions.
        input_data: Input tensor or numpy array to compute attributions for.
        double_precision: If ``True``, use double precision for computation.
        convert_to_numpy: If ``True``, convert the output to numpy arrays.
        aggregate: Method to aggregate attribution values across samples.
            Options are ``"mean"``, ``"sum"``, or ``"max"``.
        transform: Transformation to apply to attribution values.
            Options are ``"none"`` or ``"abs"``.
        hybrid_solver: If ``True``, handle multi-objective models differently.

    Returns:
        A tuple containing:
            - jf: The Jacobian matrix of shape (num_samples, output_dim, input_dim)
            - jhatg: The pseudo-inverse of the Jacobian matrix
        The result is effectively a ``(num_inputs, num_features)`` attribution map.
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
