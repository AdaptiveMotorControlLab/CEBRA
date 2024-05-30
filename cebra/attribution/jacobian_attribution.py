#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
"""Tools for computing attribution maps."""

from typing import Literal

import numpy as np
import torch
from torch import nn

import cebra.attribution.jacobian

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
    agg = getattr(np, aggregate)

    input_data = _prepare_inputs(input_data)
    model = _prepare_model(model)

    # compute jacobian CEBRA model
    jf = cebra.attribution.jacobian.compute_jacobian(
        model,
        input_vars=[input_data],
        mode="autograd",
        double_precision=double_precision,
        convert_to_numpy=convert_to_numpy,
        hybrid_solver=hybrid_solver,
    )

    jhatg = np.linalg.pinv(jf)
    return jf, jhatg
