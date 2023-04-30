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
"""Datasets to be used as part of the sklearn framework."""

from typing import Iterable, Optional

import numpy as np
import numpy.typing as npt
import torch

import cebra.data
import cebra.integrations.sklearn.utils as cebra_sklearn_utils
import cebra.models
import cebra.solver


class SklearnDataset(cebra.data.SingleSessionDataset):
    """Dataset for wrapping array-like input/index pairs.

    The dataset is initialized with a data matrix ``X`` and an arbitrary number
    of labels ``y``, which can include continuous and up to one discrete index.
    All input arrays are checked and converted for use with CEBRA.

    Attributes:
        X: array-like of shape ``(N, d)``.
        y: A list of multiple array-like of shape ``(N, k[i])`` for continual
            inputs, including up to one discrete array-like of shape ``(N,)``.
        device: Compute device, can be ``cpu`` or ``cuda``.

    Example:

        >>> import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset
        >>> import numpy as np
        >>> data = np.random.uniform(0, 1, (100, 30))
        >>> index1 = np.random.uniform(0, 10, (100, ))
        >>> index2 = np.random.uniform(0, 10, (100, 2))
        >>> dataset = cebra_sklearn_dataset.SklearnDataset(data, (index1, index2))

    """

    def __init__(self, X: npt.NDArray, y: tuple, device="cpu"):
        super().__init__(device=device)
        self._parse_data(X)
        self._parse_labels(y)

    @property
    def input_dimension(self) -> int:
        """The feature dimension of the data."""
        return self.neural.size(1)

    @property
    def continuous_index(self) -> Optional[torch.Tensor]:
        """The continuous indexes of the dataset."""
        return self._continuous_index

    @property
    def discrete_index(self) -> Optional[torch.Tensor]:
        """The discrete indexes of the dataset."""
        return self._discrete_index

    @property
    def continuous_index_dimensions(self) -> int:
        """The continuous indexes features dimension."""
        if self.continuous_index is None:
            return 0
        return self.continuous_index.size(1)

    @property
    def discrete_index_dimensions(self) -> int:
        """The discrete indexes features dimension."""
        if self.discrete_index is None:
            return 0
        return 1

    @property
    def total_index_dimensions(self) -> int:
        """The total (both continuous and discrete) indexes feature dimensions."""
        return self.continuous_index_dimensions + self.discrete_index_dimensions

    def _check_dimensions(self):
        pass

    def _parse_data(self, X: npt.NDArray):
        """Check input data validity and convert to torch.Tensor

        Args:
            X: The 2D input data array.
        """
        # NOTE(stes) in practice this value should be much higher, but more than
        # one sample is a conservative default here to ensure that sklearn tests
        # passes with the correct error messages.
        X = cebra_sklearn_utils.check_input_array(X, min_samples=2)
        self.neural = torch.from_numpy(X).float().to(self.device)

    def _parse_labels(self, labels: Optional[tuple]):
        """Check labels validity and convert to torch.Tensor

        Args:
            labels: Tuple containing the sets of labels, either continuous or discrete.
        """
        # Check that labels are provided in a tuple format
        if labels is None:
            raise ValueError("Labels cannot be None.")
        if not isinstance(labels, tuple):
            raise TypeError(
                f"Expected a tuple, but got {type(labels)}. "
                f"When passing indices to {type(self).__name__} during "
                f"initialization, make sure that you pass them in "
                f"a tuple rather than a {type(labels)}.")

        continuous_index = []
        discrete_index = []
        for y in labels:
            # Validate the set of index format
            y = cebra_sklearn_utils.check_label_array(y,
                                                      min_samples=len(
                                                          self.neural))
            if y is None:
                raise ValueError("Labels cannot be None.")
            if not isinstance(y, np.ndarray):
                raise ValueError(
                    f"Elements in index need to be numpy arrays, torch tensors, "
                    f"or lists that can be converted to arrays, but got {type(y)}"
                )

            # Define the index as either continuous or discrete indices, depending
            # on the dtype in the index array.
            if y.dtype in (np.float32, np.float64):
                y = torch.from_numpy(y).float()
                if y.dim() == 1:
                    y = y.unsqueeze(1)
                continuous_index.append(y)
            elif y.dtype in (np.int32, np.int64):
                y = torch.from_numpy(y).long().squeeze()
                if y.dim() > 1:
                    raise ValueError(
                        f"All discrete indices need to be one dimensional, got {y.size()}."
                    )
                discrete_index.append(y)

        if len(discrete_index) > 1:
            raise ValueError(f"Only 1D discrete indices are allowed, "
                             f"but got {len(discrete_index)} discrete indices")

        self._continuous_index = None
        self._discrete_index = None
        if len(continuous_index) > 0:
            self._continuous_index = torch.cat(continuous_index,
                                               dim=1).to(self.device)
        if len(discrete_index) > 0:
            (self._discrete_index,) = discrete_index
            self._discrete_index = self._discrete_index.to(self.device)

    def __getitem__(self, index: Iterable) -> npt.NDArray:
        """

        Args:
            index: List of index to return from the data.

        Returns
            [ No.Samples x Neurons x 10 ]
        """
        index = self.expand_index(index).to(self.device)
        return self.neural[index].transpose(2, 1)

    def __len__(self) -> int:
        """Number of samples in the neural data."""
        return len(self.neural)
