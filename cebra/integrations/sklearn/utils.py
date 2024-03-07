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
import warnings

import numpy.typing as npt
import sklearn.utils.validation as sklearn_utils_validation
import torch

import cebra.helper


def update_old_param(old: dict, new: dict, kwargs: dict, default) -> tuple:
    """Handle deprecated arguments of a function until they are replaced.

    Note:
        If both the deprecated and new arguments are present, then an error is raised,
        else if only the deprecated argument is present, a warning is raised and the
        old argument is used in place of the new one.

    Args:
        old: A dictionary containing the deprecated arguments.
        new: A dictionary containing the new arguments.
        kwargs: A dictionary containing all the arguments.

    Returns:
        The updated ``kwargs`` set of arguments.

    """
    if kwargs[old] is None and kwargs[new] is None:  # none are present
        kwargs[new] = default
    elif kwargs[old] is not None and kwargs[new] is not None:  # both are present
        raise ValueError(
            f"{old} and {new} cannot be assigned simultaneously. Assign only {new}"
        )
    elif kwargs[old] is not None:  # old version is present but not the new one
        warnings.warn(f"{old} is deprecated. Use {new} instead")
        kwargs[new] = kwargs[old]

    return kwargs


def check_input_array(X: npt.NDArray, *, min_samples: int) -> npt.NDArray:
    """Check validity of the input data, using scikit-learn native function.

    Note:
        * Assert that the array is non-empty, 2D and containing only finite values.
        * Assert that the array has at least {min_samples} samples and 1 feature dimension.
        * Assert that the array is not sparse.
        * Check for the dtype of X and convert values to float if needed.

    Args:
        X: Input data array to check.
        min_samples: Minimum of samples in the dataset.

    Returns:
        The converted and validated array.
    """
    return sklearn_utils_validation.check_array(
        X,
        accept_sparse=False,
        accept_large_sparse=False,
        dtype=("float16", "float32", "float64"),
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=True,
        allow_nd=False,
        ensure_min_samples=min_samples,
        ensure_min_features=1,
    )


def check_label_array(y: npt.NDArray, *, min_samples: int):
    """Check validity of the labels, using scikit-learn native function.

    Note:
        * Assert that the array is non-empty and containing only finite values.
        * Assert that the array has at least {min_samples} samples.
        * Assert that the array is not sparse.
        * Check for the dtype of y and convert values to numeric if needed.

    Args:
        y: Labels array to check.
        min_samples: Minimum of samples in the label array.

    Returns:
        The converted and validated labels.
    """
    return sklearn_utils_validation.check_array(
        y,
        accept_sparse=False,
        accept_large_sparse=False,
        dtype="numeric",
        order=None,
        copy=False,
        force_all_finite=True,
        ensure_2d=False,
        allow_nd=False,
        ensure_min_samples=min_samples,
    )


def check_device(device: str) -> str:
    """Select a device depending on the requirement and availabilities.

    Args:
        device: The device to return, if possible.

    Returns:
        Either cuda, cuda:device_id, mps, or cpu depending on {device} and availability in the environment.
    """

    if device == "cuda_if_available":
        if torch.cuda.is_available():
            return "cuda"
        elif cebra.helper._is_mps_availabe(torch):
            return "mps"
        else:
            return "cpu"
    elif device.startswith("cuda:") and len(device) > 5:
        cuda_device_id = device[5:]
        if cuda_device_id.isdigit():
            device_count = torch.cuda.device_count()
            device_id = int(cuda_device_id)
            if device_id < device_count:
                return f"cuda:{device_id}"
            else:
                raise ValueError(
                    f"CUDA device {device_id} is not available. Available device IDs are 0 to {device_count - 1}."
                )
        else:
            raise ValueError(
                f"Invalid CUDA device ID format. Please use 'cuda:device_id' where '{cuda_device_id}' is an integer."
            )
    elif device == "cuda" and torch.cuda.is_available():
        return "cuda:0"
    elif device == "cpu":
        return device
    elif device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                raise ValueError(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                raise ValueError(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )

        return device

    raise ValueError(f"Device needs to be cuda, cpu or mps, but got {device}.")


def check_fitted(model: "cebra.models.Model") -> bool:
    """Check if an estimator is fitted.

    Args:
        model: The model to assess.

    Returns:
        True if fitted.
    """
    return hasattr(model, "n_features_")
