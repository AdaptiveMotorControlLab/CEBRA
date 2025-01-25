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

import torch


def tensors_to_cpu_and_double(vars_):
    cpu_vars = []
    for v in vars_:
        if v.is_cuda:
            v = v.to("cpu")
        cpu_vars.append(v.double())
    return cpu_vars


def tensors_to_cuda(vars_, cuda_device):
    cpu_vars = []
    for v in vars_:
        if not v.is_cuda:
            v = v.to(cuda_device)
        cpu_vars.append(v)
    return cpu_vars


def compute_jacobian(
    model,
    input_vars,
    mode="autograd",
    cuda_device="cuda",
    double_precision=False,
    convert_to_numpy=True,
    hybrid_solver=False,
):

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
