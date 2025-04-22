#
# CEBRA: Consistent EmBeddings of high-dimensional Recordings using Auxiliary variables
# © Mackenzie W. Mathis & Steffen Schneider (v0.4.0+)
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/blob/main/LICENSE.md
#
# Adapted from https://github.com/rpatrik96/nl-causal-representations/blob/master/care_nl_ica/dep_mat.py,
# licensed under the following MIT License:
#
#   MIT License
#
#   Copyright (c) 2022 Patrik Reizinger
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

"""Jacobian computation utilities for attribution analysis.

This module provides core functionality for computing Jacobian matrices, which are
essential for many attribution methods. The Jacobian matrix represents the first-order
partial derivatives of a model's output with respect to its input features.

The module includes:
- Utility functions for tensor manipulation
- Core Jacobian computation function
- Helper functions for matrix operations
"""

from typing import Union, Tuple, Optional

import numpy as np
import torch


def tensors_to_cpu_and_double(vars_: list[torch.Tensor]) -> list[torch.Tensor]:
    """Convert a list of tensors to CPU and double precision.

    This function ensures all tensors in the input list are moved to CPU and converted
    to double precision (float64) format. This is useful for operations requiring higher
    numerical precision.

    Args:
        vars_: List of PyTorch tensors to convert. Tensors can be on any device.

    Returns:
        List of tensors converted to CPU and double precision. The order of tensors
        in the output list matches the input list.

    Example:
        >>> tensors = [torch.randn(3, 3).cuda(), torch.randn(2, 2)]
        >>> cpu_tensors = tensors_to_cpu_and_double(tensors)
        >>> all(t.is_cpu for t in cpu_tensors)
        True
        >>> all(t.dtype == torch.float64 for t in cpu_tensors)
        True
    """
    cpu_vars = []
    for v in vars_:
        if v.is_cuda:
            v = v.to("cpu")
        cpu_vars.append(v.double())
    return cpu_vars


def tensors_to_cuda(vars_: list[torch.Tensor],
                    cuda_device: str) -> list[torch.Tensor]:
    """Convert a list of tensors to CUDA device.

    This function moves all tensors in the input list to the specified CUDA device.
    Tensors already on CUDA are left unchanged. This is useful for GPU-accelerated
    computations.

    Args:
        vars_: List of PyTorch tensors to convert. Tensors can be on any device.
        cuda_device: CUDA device identifier (e.g., "cuda:0", "cuda:1") to move
            tensors to.

    Returns:
        List of tensors moved to the specified CUDA device. The order of tensors
        in the output list matches the input list.

    Example:
        >>> tensors = [torch.randn(3, 3), torch.randn(2, 2)]
        >>> cuda_tensors = tensors_to_cuda(tensors, "cuda:0")
        >>> all(t.is_cuda for t in cuda_tensors)
        True
    """
    cpu_vars = []
    for v in vars_:
        if not v.is_cuda:
            v = v.to(cuda_device)
        cpu_vars.append(v)
    return cpu_vars


def compute_jacobian(
    model: torch.nn.Module,
    x: torch.Tensor,
    output_dim: Optional[int] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Compute the Jacobian matrix for a given model and input.

    The Jacobian matrix J is defined as:
        J[i,j] = ∂f(x)[i]/∂x[j]
    where f is the model function and x is the input tensor.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model for which to compute the Jacobian.
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim).
    output_dim : Optional[int]
        The dimension of the model's output. If None, it will be inferred from the model.
    device : Optional[torch.device]
        The device on which to perform computations. If None, uses the model's device.

    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (batch_size, output_dim, input_dim).

    Raises
    ------
    ValueError
        If the input tensor is not 2D or if the model's output is not compatible
        with the specified output_dim.

    Examples
    --------
    >>> model = torch.nn.Linear(10, 5)
    >>> x = torch.randn(32, 10)
    >>> jacobian = compute_jacobian(model, x)
    >>> print(jacobian.shape)  # (32, 5, 10)
    """
    if output_dim is None:
        output_dim = model(x).shape[1]
    if device is None:
        device = x.device

    if x.ndim != 2:
        raise ValueError("Input tensor must be 2D")
    if model(x).shape[1] != output_dim:
        raise ValueError("Model's output dimension must match the specified output_dim")

    model = model.to(device).float()
    x = x.to(device)

    jacobian = []
    for i in range(output_dim):
        grads = torch.autograd.grad(
            model(x)[:, i:i + 1],
            x,
            retain_graph=True,
            create_graph=False,
            grad_outputs=torch.ones(model(x)[:, i:i + 1].shape).to(device),
        )
        jacobian.append(torch.cat(grads, dim=1))

    jacobian = torch.stack(jacobian, dim=1)
    return jacobian


def _reshape_for_jacobian(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor to be compatible with Jacobian computation.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor of any shape.

    Returns
    -------
    torch.Tensor
        Reshaped tensor of shape (batch_size, -1).

    Notes
    -----
    This function ensures that the input tensor is properly flattened for
    Jacobian computation while preserving the batch dimension.
    """
    return tensor.view(tensor.shape[0], -1)


def _compute_jacobian_columns(
    model: torch.nn.Module,
    x: torch.Tensor,
    output_dim: int,
    device: torch.device
) -> torch.Tensor:
    """Compute Jacobian matrix column by column using automatic differentiation.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model.
    x : torch.Tensor
        Input tensor of shape (batch_size, input_dim).
    output_dim : int
        The dimension of the model's output.
    device : torch.device
        The device on which to perform computations.

    Returns
    -------
    torch.Tensor
        Jacobian matrix of shape (batch_size, output_dim, input_dim).

    Notes
    -----
    This function computes the Jacobian by iterating over input dimensions and
    using automatic differentiation to compute partial derivatives. It is more
    memory-efficient than computing the full Jacobian at once but may be slower
    for large input dimensions.
    """
    jacobian = []
    for i in range(output_dim):
        grads = torch.autograd.grad(
            model(x)[:, i:i + 1],
            x,
            retain_graph=True,
            create_graph=False,
            grad_outputs=torch.ones(model(x)[:, i:i + 1].shape).to(device),
        )
        jacobian.append(torch.cat(grads, dim=1))

    jacobian = torch.stack(jacobian, dim=1)
    return jacobian
