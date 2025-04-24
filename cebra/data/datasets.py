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
"""Pre-defined datasets."""

import types
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
import numpy.typing as npt
import torch

import cebra.data as cebra_data
import cebra.helper as cebra_helper
import cebra.io as cebra_io
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex
from cebra.data.datatypes import Offset

if TYPE_CHECKING:
    from cebra.models import Model


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

    def __init__(self,
                 neural: Union[torch.Tensor, npt.NDArray],
                 continuous: Union[torch.Tensor, npt.NDArray] = None,
                 discrete: Union[torch.Tensor, npt.NDArray] = None,
                 offset: Offset = Offset(0, 1),
                 device: str = "cpu"):
        super().__init__(device=device)
        self.neural = self._to_tensor(neural, check_dtype="float").float()
        self.continuous = self._to_tensor(continuous, check_dtype="float")
        self.discrete = self._to_tensor(discrete, check_dtype="int")
        if self.continuous is None and self.discrete is None:
            raise ValueError(
                "You have to pass at least one of the arguments 'continuous' or 'discrete'."
            )
        self.offset = offset

    def _to_tensor(
            self,
            array: Union[torch.Tensor, npt.NDArray],
            check_dtype: Optional[Literal["int",
                                          "float"]] = None) -> torch.Tensor:
        """Convert :py:func:`numpy.array` to :py:class:`torch.Tensor` if necessary and check the dtype.

        Args:
            array: Array to check.
            check_dtype: If not `None`, list of dtypes to which the values in `array`
                must belong to. Defaults to None.

        Returns:
            The `array` as a :py:class:`torch.Tensor`.
        """
        if array is None:
            return None
        if isinstance(array, np.ndarray):
            array = torch.from_numpy(array)
        if check_dtype is not None:
            if check_dtype not in ["int", "float"]:
                raise ValueError(
                    f"check_dtype must be 'int' or 'float', got {check_dtype}")
            if (check_dtype == "int" and not cebra_helper._is_integer(array)
               ) or (check_dtype == "float" and
                     not cebra_helper._is_floating(array)):
                raise TypeError(
                    f"Array has type {array.dtype} instead of {check_dtype}.")
        if cebra_helper._is_floating(array):
            array = array.float()
        if cebra_helper._is_integer(array):
            # NOTE(stes): Required for standardizing number format on
            # windows machines.
            array = array.long()
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


def _assert_datasets_same_device(
        datasets: List[cebra_data.SingleSessionDataset]) -> str:
    """Checks if the list of datasets are all on the same device.

    Args:
        datasets: List of datasets.

    Returns:
        The device name if all datasets are on the same device.

    Raises:
        ValueError: If datasets are not all on the same device.
    """
    devices = set([dataset.device for dataset in datasets])
    if len(devices) != 1:
        raise ValueError("Datasets are not all on the same device")
    return devices.pop()


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
        self._datasets: List[
            cebra_data.SingleSessionDataset] = self._unpack_dataset_arguments(
                datasets)

        device = _assert_datasets_same_device(self._datasets)
        super().__init__(device=device)

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
            self._dindex = torch.cat(list(
                self._iter_property("discrete_index")),
                                     dim=0)
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


# TODO(stes): This should be a single session dataset?
class DatasetxCEBRA(cebra_io.HasDevice):
    """Dataset class for xCEBRA models.

    This class handles neural data and associated labels for xCEBRA models, providing
    functionality for data loading and batch preparation.

    Attributes:
        neural: Neural data as a torch.Tensor or numpy array
        labels: Labels associated with the data
        offset: Offset for the dataset

    Args:
        neural: Neural data as a torch.Tensor or numpy array
        device: Device to store the data on (default: "cpu")
        **labels: Additional keyword arguments for labels associated with the data
    """

    def __init__(
        self,
        neural: Union[torch.Tensor, npt.NDArray],
        device="cpu",
        **labels,
    ):
        super().__init__(device)
        self.neural = neural
        self.labels = labels
        self.offset = Offset(0, 1)

    @property
    def input_dimension(self) -> int:
        """Get the input dimension of the neural data.

        Returns:
            The number of features in the neural data
        """
        return self.neural.shape[1]

    def __len__(self):
        """Get the length of the dataset.

        Returns:
            Number of samples in the dataset
        """
        return len(self.neural)

    def configure_for(self, model: "Model"):
        """Configure the dataset offset for the provided model.

        Call this function before indexing the dataset. This sets the
        ``offset`` attribute of the dataset.

        Args:
            model: The model to configure the dataset for.
        """
        self.offset = model.get_offset()

    def expand_index(self, index: torch.Tensor) -> torch.Tensor:
        """Expand indices based on the configured offset.

        Args:
            index: A one-dimensional tensor of type long containing indices
                to select from the dataset.

        Returns:
            An expanded index of shape ``(len(index), len(self.offset))`` where
            the elements will be
            ``expanded_index[i,j] = index[i] + j - self.offset.left`` for all ``j``
            in ``range(0, len(self.offset))``.

        Note:
            Requires the :py:attr:`offset` to be set.
        """
        offset = torch.arange(-self.offset.left,
                              self.offset.right,
                              device=index.device)

        index = torch.clamp(index, self.offset.left,
                            len(self) - self.offset.right)

        return index[:, None] + offset[None, :]

    def __getitem__(self, index):
        """Get item(s) from the dataset at the specified index.

        Args:
            index: Index or indices to retrieve

        Returns:
            The neural data at the specified indices, with dimensions transposed
        """
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def load_batch_supervised(self, index: Batch,
                              labels_supervised) -> torch.Tensor:
        """Load a batch for supervised learning.

        Args:
            index: Batch indices for reference data
            labels_supervised: Labels to load for supervised learning

        Returns:
            Batch containing reference data and corresponding labels
        """
        assert index.negative is None
        assert index.positive is None
        labels = [
            self.labels[label].to(self.device) for label in labels_supervised
        ]

        return Batch(
            reference=self[index.reference],
            positive=[label[index.reference] for label in labels],
            negative=None,
        )

    def load_batch_contrastive(self, index: BatchIndex) -> Batch:
        """Load a batch for contrastive learning.

        Args:
            index: BatchIndex containing reference, positive and negative indices

        Returns:
            Batch containing reference, positive and negative samples
        """
        assert isinstance(index.positive, list)
        return Batch(
            reference=self[index.reference],
            positive=[self[idx] for idx in index.positive],
            negative=self[index.negative],
        )
