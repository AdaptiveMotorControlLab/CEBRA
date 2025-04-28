import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as sklearn_utils_validation
import torch

import cebra
import cebra.integrations.sklearn.utils as sklearn_utils
import cebra.models


#NOTE: Deprecated: transform is now handled in the solver but the original
#      method is kept here for testing.
def cebra_transform_deprecated(cebra_model,
                               X: Union[npt.NDArray, torch.Tensor],
                               session_id: Optional[int] = None) -> npt.NDArray:
    """Transform an input sequence and return the embedding.

    Args:
        cebra_model: The CEBRA model to use for the transform.
        X: A numpy array or torch tensor of size ``time x dimension``.
        session_id: The session ID, an :py:class:`int` between 0 and :py:attr:`num_sessions` for
            multisession, set to ``None`` for single session.

    Returns:
        A :py:func:`numpy.array` of size ``time x output_dimension``.

    Example:

        >>> import cebra
        >>> import numpy as np
        >>> dataset =  np.random.uniform(0, 1, (1000, 30))
        >>> cebra_model = cebra.CEBRA(max_iterations=10)
        >>> cebra_model.fit(dataset)
        CEBRA(max_iterations=10)
        >>> embedding = cebra_model.transform(dataset)

    """
    warnings.warn(
        "The method is deprecated "
        "but kept for testing puroposes."
        "We recommend using `transform` instead.",
        DeprecationWarning,
        stacklevel=2)

    sklearn_utils_validation.check_is_fitted(cebra_model, "n_features_")
    model, offset = cebra_model._select_model(X, session_id)

    # Input validation
    X = sklearn_utils.check_input_array(X, min_samples=len(cebra_model.offset_))
    input_dtype = X.dtype

    with torch.no_grad():
        model.eval()

        if cebra_model.pad_before_transform:
            X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)),
                       mode="edge")
        X = torch.from_numpy(X).float().to(cebra_model.device_)

        if isinstance(model, cebra.models.ConvolutionalModelMixin):
            # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
            X = X.transpose(1, 0).unsqueeze(0)
            output = model(X).cpu().numpy().squeeze(0).transpose(1, 0)
        else:
            # Standard evaluation, (T, C, dt)
            output = model(X).cpu().numpy()

    if input_dtype == "float64":
        return output.astype(input_dtype)

    return output


# NOTE: Deprecated: batched transform can now be performed (more memory efficient)
#       using the transform method of the model, and handling padding is implemented
#       directly in the base Solver. This method is kept for testing purposes.
@torch.no_grad()
def multiobjective_transform_deprecated(solver: "cebra.solvers.Solver",
                                        inputs: torch.Tensor) -> torch.Tensor:
    """Transform the input data using the model.

    Args:
        solver: The solver containing the model and device.
        inputs: The input data to transform.

    Returns:
        The transformed data.
    """

    warnings.warn(
        "The method is deprecated "
        "but kept for testing puroposes."
        "We recommend using `transform` instead.",
        DeprecationWarning,
        stacklevel=2)

    offset = solver.model.get_offset()
    solver.model.eval()
    X = inputs.cpu().numpy()
    X = np.pad(X, ((offset.left, offset.right - 1), (0, 0)), mode="edge")
    X = torch.from_numpy(X).float().to(solver.device)

    if isinstance(solver.model.module, cebra.models.ConvolutionalModelMixin):
        # Fully convolutional evaluation, switch (T, C) -> (1, C, T)
        X = X.transpose(1, 0).unsqueeze(0)
        outputs = solver.model(X)

        # switch back from (1, C, T) -> (T, C)
        if isinstance(outputs, torch.Tensor):
            assert outputs.dim() == 3 and outputs.shape[0] == 1
            outputs = outputs.squeeze(0).transpose(1, 0)
        elif isinstance(outputs, tuple):
            assert all(tensor.dim() == 3 and tensor.shape[0] == 1
                       for tensor in outputs)
            outputs = (output.squeeze(0).transpose(1, 0) for output in outputs)
            outputs = tuple(outputs)
        else:
            raise ValueError("Invalid condition in solver.transform")
    else:
        # Standard evaluation, (T, C, dt)
        outputs = solver.model(X)

    return outputs
