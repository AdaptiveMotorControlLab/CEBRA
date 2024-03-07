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
"""Neural network layers used for building cebra models.

Layers are used in the models defined in :py:mod:`.model`.
"""
import torch
import torch.nn.functional as F
from torch import nn


class _Skip(nn.Module):
    """Add a skip connection to a list of modules

    Args:
        *modules (torch.nn.Module): Modules to add to the bottleneck
        crop (tuple of ints): Number of timesteps to crop around the
            shortcut of the module to match the output with the bottleneck
            layers. This can be typically inferred from the strides/sizes
            of any conv layers within the bottleneck.
    """

    def __init__(self, *modules, crop=(1, 1)):
        super().__init__()
        self.module = nn.Sequential(*modules)
        self.crop = slice(
            crop[0],
            -crop[1] if isinstance(crop[1], int) and crop[1] > 0 else None)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Compute forward pass through the skip connection.

        Implements the operation ``self.module(inp[..., self.crop]) + skip``.

        Args:
            inp: 3D input tensor

        Returns:
            3D output tensor of same dimension as `inp`.
        """
        skip = self.module(inp)
        return inp[..., self.crop] + skip


class Squeeze(nn.Module):
    """Squeeze 3rd dimension of input tensor, pass through otherwise."""

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """Squeeze 3rd dimension of input tensor, pass through otherwise.

        Args:
            inp: 1-3D input tensor

        Returns:
            If the third dimension of the input tensor can be squeezed,
            return the resulting 2D output tensor. If input is 2D or less,
            return the input.
        """
        if inp.dim() > 2:
            return inp.squeeze(2)
        return inp


class _Norm(nn.Module):

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return inp / torch.norm(inp, dim=1, keepdim=True)


class _MeanAndConv(nn.Module):

    def __init__(self, inp, output, kernel, *, stride):
        super().__init__()
        self.downsample = stride
        self.layer = nn.Conv1d(inp, output, kernel, stride=stride)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        connect = self.layer(inp)
        downsampled = F.interpolate(inp, scale_factor=1 / self.downsample)
        return torch.cat([connect, downsampled[..., :connect.size(-1)]], dim=1)
