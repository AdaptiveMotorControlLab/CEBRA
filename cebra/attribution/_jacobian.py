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

from typing import Union

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
    input_vars: list[torch.Tensor],
    mode: str = "autograd",
    cuda_device: str = "cuda",
    double_precision: bool = False,
    convert_to_numpy: bool = True,
    hybrid_solver: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Compute the Jacobian matrix for a given model and input.

    This function computes the Jacobian matrix using PyTorch's autograd functionality.
    The Jacobian represents the first-order partial derivatives of the model's output
    with respect to its input parameters. It supports both CPU and CUDA computation,
    as well as single and double precision.

    Args:
        model: PyTorch model to compute Jacobian for. The model should be callable
            with the input variables.
        input_vars: List of input tensors to compute the Jacobian with respect to.
            Each tensor should have requires_grad=True if gradients are needed.
        mode: Computation mode. Currently only "autograd" is supported, which uses
            PyTorch's automatic differentiation.
        cuda_device: CUDA device identifier to use for computation (e.g., "cuda:0").
            Ignored if double_precision is True.
        double_precision: If True, use double precision (float64) for computation.
            This moves all tensors to CPU.
        convert_to_numpy: If True, convert the output Jacobian to a numpy array.
            If False, returns a PyTorch tensor.
        hybrid_solver: If True, concatenate multiple model outputs along dimension 1
            before computing the Jacobian. Useful for models with multiple outputs.

    Returns:
        Jacobian matrix as either a PyTorch tensor or numpy array. The shape is
        (batch_size, output_dim, input_dim), where:
        - batch_size is the size of the input batch
        - output_dim is the dimension of the model's output
        - input_dim is the total dimension of all input variables

    Example:
        >>> model = torch.nn.Linear(10, 5)
        >>> x = torch.randn(3, 10, requires_grad=True)
        >>> jacobian = compute_jacobian(model, [x])
        >>> jacobian.shape
        (3, 5, 10)
    """
    if double_precision:
        model = model.to("cpu").double()
        input_vars = tensors_to_cpu_and_double(input_vars)
        if hybrid_solver:
            output = model(*input_vars)
            output_vars = torch.cat(output, dim=1).to("cpu").double()
        else:
            output_vars = model(*input_vars).to("cpu").double()
    else:
        model = model.to(cuda_device).float()
        input_vars = tensors_to_cuda(input_vars, cuda_device=cuda_device)

        if hybrid_solver:
            output = model(*input_vars)
            output_vars = torch.cat(output, dim=1)
        else:
            output_vars = model(*input_vars)

    if mode == "autograd":
        jacob = []
        for i in range(output_vars.shape[1]):
            grads = torch.autograd.grad(
                output_vars[:, i:i + 1],
                input_vars,
                retain_graph=True,
                create_graph=False,
                grad_outputs=torch.ones(output_vars[:, i:i + 1].shape).to(
                    output_vars.device),
            )
            jacob.append(torch.cat(grads, dim=1))

        jacobian = torch.stack(jacob, dim=1)

    jacobian = jacobian.detach().cpu()

    if convert_to_numpy:
        jacobian = jacobian.numpy()

    return jacobian
