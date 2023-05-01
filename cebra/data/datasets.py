"""Pre-defined datasets."""

import abc
import collections
import types
from typing import List, Tuple, Union

import literate_dataclasses as dataclasses
import numpy as np
import torch
from numpy.typing import NDArray

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex


class TensorDataset(cebra_data.SingleSessionDataset):
    """Discrete and/or continuously indexed dataset based on torch/numpy arrays.

    can be used with the discrete, continuous and/or mixed data loader classes.

    Args:
        neural:
        continuous:
            variables over the same time dimension.
        discrete:
            variables over the same time dimension.
    """

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
            will be maintained for indexing.


    """

    def _has_not_none_attribute(self, obj, key):
        """Check if obj.key exists.

        Returns:
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

        if len(datasets) == 0:
            raise ValueError("Need to supply at least one dataset.")
        elif len(datasets) == 1:
            if isinstance(dataset_generator, types.GeneratorType):
                return list(dataset_generator)
            else:
                raise ValueError(
                    "You need to specify either a single generator, "
                    "or multiple SingleSessionDataset instances.")
        else:
            return datasets

        super().__init__()

        continuous = all(
            self._has_not_none_attribute(session, "continuous_index")
        discrete = all(
            self._has_not_none_attribute(session, "discrete_index")

        if not (continuous or discrete):
            raise ValueError(
                "The provided datasets need to define either continuous or discrete indices, "
                "or both. Continuous: {continuous}; discrete: {discrete}. "
                "Note that _all_ provided datasets need to define the indexing function of choice."
            )

        if continuous:
            self._cindex = torch.cat(list(
                                     dim=0)
        else:
            self._cindex = None
        if discrete:
        else:
            self._dindex = None

    @property
        """The number of sessions in the dataset."""
        return len(self._datasets)

    @property

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
        return self._cindex

    @property
        return self._dindex

    def _apply(self, func):

    def _iter_property(self, attr):
