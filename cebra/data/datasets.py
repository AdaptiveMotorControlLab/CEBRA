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
"""Pre-defined datasets."""

import abc
import collections
import types
from typing import List, Tuple, Union

import literate_dataclasses as dataclasses
import numpy as np
import numpy.typing as npt
import torch
from numpy.typing import NDArray

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex


class TensorDataset(cebra_data.SingleSessionDataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    If dealing with datasets sufficiently small to fit :py:func:`numpy.array` or :py:class:`torch.Tensor`, this
    dataset is sufficient---the sampling auxiliary variable should be specified with a dataloader.
    Based on whether `continuous` and/or `discrete` auxiliary variables are provided, this class
    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
            Array of dtype ``float`` or float Tensor of shape ``(N, D)``, containing neural activity over time.
        continuous:
            Array of dtype ```float`` or float Tensor of shape ``(N, d)``, containing the continuous behavior
            variables over the same time dimension.
        discrete:
            Array of dtype ```int64`` or integer Tensor of shape ``(N, d)``, containing the discrete behavior
            variables over the same time dimension.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> data = torch.randn((100, 30))
        >>> index1 = torch.randn((100, 2))
        >>> index2 = torch.randint(0,5,(100, ))
        >>> dataset = cebra.data.datasets.TensorDataset(data, continuous=index1, discrete=index2)

    """

    def __init__(
        self,
        neural: Union[torch.Tensor, npt.NDArray],
        continuous: Union[torch.Tensor, npt.NDArray] = None,
        discrete: Union[torch.Tensor, npt.NDArray] = None,
        offset: int = 1,
    ):
        super().__init__()
        self.neural = self._to_tensor(neural, torch.FloatTensor).float()
        self.continuous = self._to_tensor(continuous, torch.FloatTensor)
        self.discrete = self._to_tensor(discrete, torch.LongTensor)
        if self.continuous is None and self.discrete is None:
            raise ValueError(
                "You have to pass at least one of the arguments 'continuous' or 'discrete'."
            )
        self.offset = offset

    def _to_tensor(self, array, check_dtype=None):
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if check_dtype is not None:
            if not isinstance(array, check_dtype):
                raise TypeError(f"{type(array)} instead of {check_dtype}.")
        return array

    @property
    def input_dimension(self) -> int:
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        if self.continuous is None:
            raise NotImplementedError()
        return self.continuous

    @property
    def discrete_index(self):
        if self.discrete is None:
            raise NotImplementedError()
        return self.discrete

    def __len__(self):
        return len(self.neural)

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)


class DatasetCollection(cebra_data.MultiSessionDataset):
    """Multi session dataset made up of a list of datasets.

    Args:
        *datasets: Collection of datasets to add to the collection. The order
            will be maintained for indexing.

    Example:

        >>> import cebra.data
        >>> import torch
        >>> session1 = torch.randn((100, 30))
        >>> session2 = torch.randn((100, 50))
        >>> index1 = torch.randn((100, 4))
        >>> index2 = torch.randn((100, 4)) # same index dim as index1
        >>> dataset = cebra.data.DatasetCollection(
        ...               cebra.data.TensorDataset(session1, continuous=index1),
        ...               cebra.data.TensorDataset(session2, continuous=index2))

    """

    def _has_not_none_attribute(self, obj, key):
        """Check if obj.key exists.

        Returns:
            ``True`` if the key exists and is not ``None``. ``False`` if
            either the key does not exist, the attribute has value ``None``
            or the access raises a ``NotImplementedError``.
        """
        try:
            if hasattr(obj, key):
                if getattr(obj, key) is not None:
                    return True
        except NotImplementedError:
            return False
        return False

    def _unpack_dataset_arguments(
        self, datasets: Tuple[cebra_data.SingleSessionDataset]
    ) -> List[cebra_data.SingleSessionDataset]:
        if len(datasets) == 0:
            raise ValueError("Need to supply at least one dataset.")
        elif len(datasets) == 1:
            (dataset_generator,) = datasets
            if isinstance(dataset_generator, types.GeneratorType):
                return list(dataset_generator)
            else:
                raise ValueError(
                    "You need to specify either a single generator, "
                    "or multiple SingleSessionDataset instances.")
        else:
            return datasets

    def __init__(
        self,
        *datasets: cebra_data.SingleSessionDataset,
    ):
        super().__init__()
        self._datasets: List[
            cebra_data.SingleSessionDataset] = self._unpack_dataset_arguments(
                datasets)

        continuous = all(
            self._has_not_none_attribute(session, "continuous_index")
            for session in self.iter_sessions())
        discrete = all(
            self._has_not_none_attribute(session, "discrete_index")
            for session in self.iter_sessions())

        if not (continuous or discrete):
            raise ValueError(
                "The provided datasets need to define either continuous or discrete indices, "
                "or both. Continuous: {continuous}; discrete: {discrete}. "
                "Note that _all_ provided datasets need to define the indexing function of choice."
            )

        if continuous:
            self._cindex = torch.cat(list(
                self._iter_property("continuous_index")),
                                     dim=0)
        else:
            self._cindex = None
        if discrete:
            raise NotImplementedError(
                "Multisession implementation does not support discrete index yet."
            )
        else:
            self._dindex = None

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the dataset."""
        return len(self._datasets)

    @property
    def input_dimension(self):
        return super().input_dimension

    def get_input_dimension(self, session_id: int) -> int:
        """Get the feature dimension of the required session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session input dimension for the requested session id.
        """
        return self.get_session(session_id).input_dimension

    def get_session(self, session_id: int) -> cebra_data.SingleSessionDataset:
        """Get the dataset for the specified session.

        Args:
            session_id: The session ID, an integer between 0 and
                :py:attr:`num_sessions`.

        Returns:
            A single session dataset for the requested session
            id.
        """
        return self._datasets[session_id]

    @property
    def continuous_index(self) -> torch.Tensor:
        return self._cindex

    @property
    def discrete_index(self) -> torch.Tensor:
        return self._dindex

    def _apply(self, func):
        return (func(data) for data in self.iter_sessions())

    def _iter_property(self, attr):
        return (getattr(data, attr) for data in self.iter_sessions())
