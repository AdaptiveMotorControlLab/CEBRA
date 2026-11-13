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

    Args:
        vars_: List of PyTorch tensors to convert

    Returns:
        List of tensors converted to CPU and double precision
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

    Args:
        vars_: List of PyTorch tensors to convert
        cuda_device: CUDA device to move tensors to

    Returns:
        List of tensors moved to specified CUDA device
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
    output_batch_size: int | None = None,
) -> Union[torch.Tensor, np.ndarray]:
    """Compute the Jacobian matrix for a given model and input.

    This function computes the Jacobian matrix using PyTorch's autograd functionality.
    It supports both CPU and CUDA computation, as well as single and double precision.

    Args:
        model: PyTorch model to compute Jacobian for
        input_vars: List of input tensors
        mode: Computation mode, currently only "autograd" is supported
        cuda_device: Device to use for CUDA computation
        double_precision: If True, use double precision
        convert_to_numpy: If True, convert output to numpy array
        hybrid_solver: If True, concatenate multiple outputs along dimension 1
        output_batch_size: Optional number of output/latent dimensions to process
            per autograd chunk. If ``None``, process all outputs in one retained
            graph as before. Smaller values reduce peak GPU memory for
            deterministic evaluation-mode models.

    Returns:
        Jacobian matrix as either PyTorch tensor or numpy array
    """
    if output_batch_size is not None and int(output_batch_size) <= 0:
        raise ValueError(
            f"output_batch_size must be positive when provided, got {output_batch_size}."
        )

    def _prepare_model_and_inputs():
        if double_precision:
            prepared_model = model.to("cpu").double()
            prepared_inputs = tensors_to_cpu_and_double(input_vars)
        else:
            prepared_model = model.to(cuda_device).float()
            prepared_inputs = tensors_to_cuda(input_vars, cuda_device=cuda_device)
        return prepared_model, prepared_inputs

    def _forward_outputs(prepared_model, prepared_inputs):
        if hybrid_solver:
            output = prepared_model(*prepared_inputs)
            if double_precision:
                return torch.cat(output, dim=1).to("cpu").double()
            return torch.cat(output, dim=1)
        if double_precision:
            return prepared_model(*prepared_inputs).to("cpu").double()
        return prepared_model(*prepared_inputs)

    if mode == "autograd":
        prepared_model, prepared_inputs = _prepare_model_and_inputs()
        output_vars = _forward_outputs(prepared_model, prepared_inputs)
        n_outputs = int(output_vars.shape[1])
        effective_output_batch_size = (
            n_outputs
            if output_batch_size is None or any(
                module.training for module in prepared_model.modules())
            else min(int(output_batch_size), n_outputs)
        )

        jacob_chunks = []
        for output_start in range(0, n_outputs, effective_output_batch_size):
            output_end = min(output_start + effective_output_batch_size, n_outputs)
            if output_start > 0:
                output_vars = _forward_outputs(prepared_model, prepared_inputs)

            chunk_rows = []
            for local_idx, output_idx in enumerate(range(output_start, output_end)):
                is_last_in_chunk = local_idx == (output_end - output_start - 1)
                grads = torch.autograd.grad(
                    output_vars[:, output_idx : output_idx + 1],
                    prepared_inputs,
                    retain_graph=not is_last_in_chunk,
                    create_graph=False,
                    grad_outputs=torch.ones(
                        output_vars[:, output_idx : output_idx + 1].shape,
                        device=output_vars.device,
                    ),
                )
                chunk_rows.append(torch.cat(grads, dim=1).detach().cpu())
                del grads

            jacob_chunks.append(torch.stack(chunk_rows, dim=1))
            del chunk_rows
            del output_vars

        jacobian = torch.cat(jacob_chunks, dim=1)

    jacobian = jacobian.detach().cpu()

    if convert_to_numpy:
        jacobian = jacobian.numpy()

    return jacobian
