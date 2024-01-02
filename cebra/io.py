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
"""Helper classes and functions for I/O functionality."""

import joblib
import numpy as np
import sklearn.decomposition
import torch
from torch import nn

if torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"


def device() -> str:
    """The preferred compute device.

    Returns:
        ``cuda``, if available, otherwise ``cpu``.
    """
    return _device


class HasDevice:
    """Base class for classes that use CPU/CUDA processing via PyTorch.

    If implementing this class, any derived instanced will track any attribute
    of types ``torch.Tensor`` (this includes ``torch.nn.Parameter``),
    ``torch.nn.Module`` as well as any class subclassing :py:class:`HasDevice`
    itself.

    When calling :py:meth:`to`, all of these attributes will themselves be moved
    to the specified ``device``.

    Any instance of this class will need to be initialized. This can happen explicitly
    through the constructor (by specifying a device during initialization) or by assigning
    the first element, which will assign the device of the first element to the whole
    instance.

    Every following assignment will yield in tensors, parameters, modules and other
    instances being moved to the device of the instance.

    Args:
        device: The device name, typically ``cpu`` or ``cuda``, or combined
            with a device idea, e.g. ``cuda:0``.

    Note:
        Do not subclass this class when some dependency to a compute device
        is already implemented, e.g. via a pytorch ``torch.nn.Module``.
    """

    def __init__(self, device: str = None):
        if device is not None:
            self._init(device)

    def _init(self, device):
        self._tensors = set()
        self._children = set()
        self._modules = set()
        self._set_device(device)

    def _set_device(self, device):
        if device is None:
            return
        if not isinstance(device, str):
            device = device.type
        if device not in ("cpu", "cuda", "mps"):
            if device.startswith("cuda"):
                _, id_ = device.split(":")
                if int(id_) >= torch.cuda.device_count():
                    raise ValueError(device)
            else:
                raise ValueError(device)
        self._device = device

    @property
    def _initialized(self) -> bool:
        """

        ``True`` if the instance was either initialized upon creation via
        passing the ``device`` option to the constructor, or later by setting
        any attribute of the instance. The device of the instance will then
        be set to the first element's device.
        """
        return hasattr(self, "_modules")

    def _assert_initialized(self, device=None):
        if not self._initialized:
            self._init(device=device)
        assert self._initialized
        # if dataclasses.is_dataclass(self):
        # else:
        #    raise RuntimeError(f"{self} not initialized. Did you call super().__init__?")

    @property
    def device(self) -> str:
        """Returns the name of the current compute device."""
        self._assert_initialized()
        return self._device

    def to(self, device: str) -> "HasDevice":
        """Moves the instance to the specified device.

        Args:
            device: The device (`cpu` or `cuda`) to move this instance to.

        Returns:
            the instance itself.
        """
        self._assert_initialized()
        self._set_device(device)
        if device is None:
            return self
        for tensor in self._tensors:
            setattr(self, tensor, getattr(self, tensor))
        for child in self._children:
            getattr(self, child).to(device)
        for module in self._modules:
            getattr(self, module).to(device)
        return self

    def __setattr__(self, property, value):
        if isinstance(value, torch.Tensor):
            self._assert_initialized(device=value.device.type)
            self._tensors.add(property)
            value = value.to(self.device)
            assert value is not None
        elif isinstance(value, HasDevice):
            self._assert_initialized(device=value.device)
            self._children.add(property)
            value = value.to(self.device)
            assert value is not None
        elif isinstance(value, nn.Module):
            param = None
            for param in value.parameters():
                break  # Only get the first parameter
            if param is not None:
                self._assert_initialized(device=param.device.type)
                self._modules.add(property)
                value.to(self.device)
            assert value is not None
        if value is None:
            self._assert_initialized()
            for container in (self._tensors, self._children, self._modules):
                if property in container:
                    print(f"Remove {property}")
                    container.remove(property)
                    break
        if property == "device":
            raise ValueError(
                f"Cannot set the device directly. Instead, call {self}.to(device)."
            )
        super().__setattr__(property, value)


def reduce(data, *, ratio=None, num_components=None):
    """Map the specified data to its principal components

    Specify either an explained variance ``ratio`` between 0 and 1,
    or a number of principle components to use.

    Args:
        ratio: The ratio (needs to be between 0 and 1) of explained variance
            required by the returned number of components. Note that the
            dimension of the output will vary based on the provided input
            data.
        num_components: The number of principal components to return

    Returns:
        An ``(N, d)`` array, where the dimension ``d`` is either limited by
        the specified number of components, or is chosen to explain the specified
        variance in the data.

    """
    if (ratio is None) and (num_components is None):
        raise ValueError(
            "Specify either a threshold on the explained variance, or a maximum"
            "number of principle components")
    pca = sklearn.decomposition.PCA(num_components)
    data = data.reshape(len(data), -1)
    pca.fit(data)
    if ratio is None:
        i = None
    else:
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Ratio must be in (0, 1], but got {ratio}.")
        i = np.where(np.cumsum(pca.explained_variance_ratio_) > ratio)[0][0]
    return pca.transform(data)[:, :i]


class FileKeyValueDataset:
    """Load datasets from HDF, torch, numpy or joblib files.

    The data is directly accessible through attributes of instances of this class.

    Args:
        path: The filepath for loading the data from. Should point to a file
            in a valid file format (hdf, torch, numpy, joblib). Valid extensions
            are ``jl``, ``joblib``, ``h5``, ``hdf``, ``hdf5``, ``pth``, ``pt`` and
            ``npz``.

    Example:

        >>> import cebra.io
        >>> import joblib
        >>> import tempfile
        >>> from pathlib import Path
        >>> tmp_file = Path(tempfile.gettempdir(), 'test.jl')
        >>> _ = joblib.dump({'foo' : 42}, tmp_file)
        >>> data = cebra.io.FileKeyValueDataset(tmp_file)
        >>> data.foo
        42

    """

    def __init__(self, path: str):
        self.path = path
        keys = []
        for key, value in self._iterate_items():
            setattr(self, key, value)
            keys.append(key)
        self.keys = tuple(keys)

    def __repr__(self):
        sizes = []
        for key in self.keys:
            val = getattr(self, key)
            val_str = type(val).__name__
            if isinstance(val, np.ndarray):
                val_str = f"{val_str}(shape={val.shape}, dtype={val.dtype})"
            sizes.append(f"{key} = {val_str}")
        sizes = ",\n  ".join(sizes)
        return f"{type(self).__name__}(keys=(\n  {sizes}\n))"

    def _iterate_items(self):
        extension = self.path.suffix
        if extension in [".jl", ".joblib"]:
            dataset = joblib.load(self.path)
        elif extension in [".h5", ".hdf", ".hdf5"]:
            raise NotImplementedError()
        elif extension in [".pth", ".pt"]:
            dataset = torch.load(self.path)
        elif extension in [".npz"]:
            dataset = np.load(self.path, allow_pickle=True)
        else:
            raise ValueError(f"Invalid file format: {extension} in {self.path}")
        for key in dataset.keys():
            yield key, dataset[key]
