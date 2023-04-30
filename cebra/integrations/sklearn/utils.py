#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import warnings

import numpy.typing as npt
import sklearn.utils.validation as sklearn_utils_validation
import torch


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
        Either cuda or cpu depending on {device} and availability in the environment.
    """
    if device == "cuda_if_available":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif device in ["cuda", "cpu"]:
        return device
    raise ValueError(f"Device needs to be cuda or cpu, but got {device}.")


def check_fitted(model: "cebra.models.Model") -> bool:
    """Check if an estimator is fitted.

    Args:
        model: The model to assess.

    Returns:
        True if fitted.
    """
    return hasattr(model, "n_features_")
