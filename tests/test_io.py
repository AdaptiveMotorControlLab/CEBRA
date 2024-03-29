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
import _util
import torch
from torch import nn

import cebra.io


class HasDeviceDummy(cebra.io.HasDevice):

    def __init__(self):
        super().__init__()
        self.foo = torch.tensor([42])
        self.bar = torch.tensor([42])
        self.baz = nn.Parameter(torch.tensor([42.0]))


class Container(cebra.io.HasDevice):

    def __init__(self):
        super().__init__()
        self.foo = nn.Parameter(torch.tensor([42.0]))
        self.bar = nn.Linear(3, 3)
        self.baz = HasDeviceDummy()


class MoveToDeviceImplicit(cebra.io.HasDevice):

    def __init__(self):
        super().__init__()
        # sets the device to CPU
        self.cpu = torch.tensor([42], device="cpu")
        self.move_to_cpu = torch.tensor([42], device="cuda")
        assert self.move_to_cpu.device.type == "cpu"


class MoveToDeviceExplicit(cebra.io.HasDevice):

    def __init__(self):
        super().__init__("cuda")
        # sets the device to CPU
        self.move_to_cuda = torch.tensor([42], device="cpu")
        assert self.move_to_cuda.device.type == "cuda"


@_util.requires_cuda
def test_move_to_device_implicit():
    MoveToDeviceImplicit()


@_util.requires_cuda
def test_move_to_device_explicit():
    MoveToDeviceExplicit()


def _assert_device(obj, device):
    if isinstance(obj, nn.Module):
        for p in obj.parameters():
            assert p.device.type == device
    else:
        assert obj.device.type == device


@_util.requires_cuda
def test_has_device():
    dummy = HasDeviceDummy()
    assert dummy.device == "cpu"
    assert dummy._tensors == {"foo", "bar", "baz"}
    assert dummy.foo.device.type == "cpu"
    assert dummy.baz.device.type == "cpu"
    dummy.to("cuda")
    assert dummy.device == "cuda"
    assert dummy.foo.device.type == "cuda"
    assert dummy.baz.device.type == "cuda"


@_util.requires_cuda
def test_has_device_nested():
    container = Container()
    assert container.device == "cpu"
    container.to("cuda")
    assert container.baz.device == "cuda"
    assert container.bar.weight.device.type == "cuda"
    assert container.baz.foo.device.type == "cuda"
    assert container.baz.baz.device.type == "cuda"
    assert container.foo.device.type == "cuda"
    _assert_device(container.bar, "cuda")
