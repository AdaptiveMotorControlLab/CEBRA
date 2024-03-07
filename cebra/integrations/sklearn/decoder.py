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
"""Some decoders following the ``scikit-learn`` API."""

import abc
from typing import Generator, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import sklearn
import sklearn.base
import sklearn.neighbors
import torch

import cebra.helper


class Decoder(abc.ABC, sklearn.base.BaseEstimator):
    """Abstract base class for implementing a decoder."""

    @abc.abstractmethod
    def fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        y: Union[npt.NDArray, torch.Tensor],
    ) -> "Decoder":
        """Fit the decoder.

        Args:
            X: The data matrix to decode from.
            y: A 1D array containing the targets.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, X: Union[npt.NDArray, torch.Tensor]) -> npt.NDArray:
        """Predict the data ``X``.

        Args:
            X: The data matrix to predict.
        """
        raise NotImplementedError()

    def score(
            self, X: Union[npt.NDArray, torch.Tensor],
            y: Union[npt.NDArray, torch.Tensor]) -> Tuple[float, float, float]:
        """Returns performances of the decoder instance on the provided data ``X``.

        Args:
            X: A data matrix.
            y: The true targets.

        Returns:
            The R2 score on ``X``.
        """
        if isinstance(y, torch.Tensor):
            y = y.numpy()

        prediction = self.predict(X)
        test_score = sklearn.metrics.r2_score(y, prediction)
        return test_score


class KNNDecoder(Decoder):
    """Decoder implementing the k-nearest neighbors vote.

    Note:
        See `sklearn.neighbors.KNeighborsClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_
        and `sklearn.neighbors.KNeighborsRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html>`_.

    Attributes:
        n_neighbors (int): An integer indicating the K number of neighbors to consider.
        metric (str): The metric to evaluate the KNN decoder's performances.

    Examples:

        >>> from cebra import KNNDecoder
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 2))
        >>> decoder = KNNDecoder()
        >>> decoder.fit(X, y)
        KNNDecoder()
        >>> score = decoder.score(X, y)

    """

    def __init__(self, n_neighbors: int = 3, metric: Optional[str] = "cosine"):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        y: Union[npt.NDArray, torch.Tensor],
    ) -> "KNNDecoder":
        """Fit the KNN decoder.

        Args:
            X: The data matrix.
            y: A 1D array containing the targets.

        Returns:
            ``self``, to allow chaining of operations.
        """

        # Check validity of the target vector
        if len(y) != len(X):
            raise ValueError(
                f"Invalid shape: y and X must have the same number of samples, got y:{len(y)} and X:{len(X)}."
            )

        # Use regression or classification, based on if the targets are continuous or discrete
        if cebra.helper._is_floating(y):
            self.knn = sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=self.n_neighbors, metric=self.metric)
        elif cebra.helper._is_integer(y):
            self.knn = sklearn.neighbors.KNeighborsClassifier(
                n_neighbors=self.n_neighbors, metric=self.metric)
        else:
            raise NotImplementedError(
                f"Invalid type: targets must be either floats or integers, got y:{y.dtype}."
            )

        self.knn.fit(X, y)
        return self

    def predict(
        self, X: Union[npt.NDArray,
                       torch.Tensor]) -> Union[npt.NDArray, torch.Tensor]:
        """Predict the targets for data ``X``.

        Args:
            X: The data matrix.

        Returns:
            A matrix with the prediction for each data sample.
        """
        return self.knn.predict(X)

    def iter_hyperparams() -> Generator[dict, None, None]:
        """Create sets of parameters.

        Note:
            It can be used for parametrized testing.

        Yields:
            A dictionary containing sets of parameters to be used for
            testing.
        """
        for n in np.power(np.arange(1, 6, dtype=int), 2):
            yield dict(n_neighbors=n, metric="cosine")


class L1LinearRegressor(Decoder):
    """A linear model trained with L1 prior as regularizer (aka the Lasso).

    Attributes:
        alpha (float): regularization strength coefficient.

    Note:
        See `sklearn.linear_model.Lasso <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.

    Examples:

        >>> from cebra import L1LinearRegressor
        >>> import numpy as np
        >>> X = np.random.uniform(0, 1, (100, 50))
        >>> y = np.random.uniform(0, 10, (100, 2))
        >>> decoder = L1LinearRegressor()
        >>> decoder.fit(X, y)
        L1LinearRegressor()
        >>> score = decoder.score(X, y)
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(
        self,
        X: Union[npt.NDArray, torch.Tensor],
        y: Union[npt.NDArray, torch.Tensor],
    ) -> "L1LinearRegressor":
        """Fit the Lasso regressor.

        Args:
            X: The data matrix.
            y: A 1D array containing the targets.

        Returns:
            ``self``, to allow chaining of operations.
        """
        # Check the targets validity
        if len(y) != len(X):
            raise ValueError(
                f"Invalid shape: y and X must have the same number of samples, got y:{len(y)} and X:{len(X)}."
            )

        if not (cebra.helper._is_integer(y) or cebra.helper._is_floating(y)):
            raise NotImplementedError(
                f"Invalid type: targets must be numeric, got y:{y.dtype}")

        self.model = sklearn.linear_model.Lasso(alpha=self.alpha)
        self.model.fit(X, y)
        return self

    def predict(
        self, X: Union[npt.NDArray,
                       torch.Tensor]) -> Union[npt.NDArray, torch.Tensor]:
        """Predict the targets for data ``X``.

        Args:
            X: The data matrix.

        Returns:
            A matrix with the prediction for each data sample.
        """
        return self.model.predict(X)

    def iter_hyperparams() -> Generator[dict, None, None]:
        """Create sets of parameters.

        Note:
            It can be used for testing.

        Yields:
            A dictionary containing sets of parameters to be used for
            testing.
        """
        for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
            yield dict(alpha=alpha)
