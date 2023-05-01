import warnings
import numpy.typing as npt
import sklearn.utils.validation as sklearn_utils_validation
import torch



        kwargs[new] = default
        raise ValueError(
        )
        warnings.warn(f"{old} is deprecated. Use {new} instead")
        kwargs[new] = kwargs[old]

    return kwargs


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
        return device
    raise ValueError(f"Device needs to be cuda or cpu, but got {device}.")
