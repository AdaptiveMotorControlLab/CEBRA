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
import copy
import warnings
from typing import List, Optional, Union

import joblib
import numpy as np
import numpy.typing as npt
import scipy.linalg
import torch

import cebra.data.base as cebra_data_base
import cebra.data.multi_session as cebra_data_multisession
import cebra.data.single_session as cebra_data_singlesession


def get_loader_options(dataset: cebra_data_base.Dataset) -> List[str]:
    """Return all possible dataloaders for the given dataset."""

    loader_options = []
    if isinstance(dataset, cebra_data_singlesession.SingleSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(cebra_data_singlesession.ContinuousDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            loader_options.append(cebra_data_singlesession.DiscreteDataLoader)
        else:
            mixed = False
        if mixed:
            loader_options.append(cebra_data_singlesession.MixedDataLoader)
    elif isinstance(dataset, cebra_data_multisession.MultiSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(
                cebra_data_multisession.ContinuousMultiSessionDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            pass  # not implemented yet
        else:
            mixed = False
        if mixed:
            pass  # not implemented yet
    else:
        raise TypeError(f"Invalid dataset type: {dataset}")
    return loader_options


def _require_numpy_array(array: Union[npt.NDArray, torch.Tensor]):
    if not isinstance(array, np.ndarray):
        if isinstance(array, torch.Tensor):
            array = array.numpy()
        else:
            raise ValueError(
                f"Invalid dtype for the provided embedding, got a {type(array)}."
            )
    elif len(array.shape) == 1:
        array = np.expand_dims(array, axis=1)

    return array


class OrthogonalProcrustesAlignment:
    """Aligns two dataset by solving the orthogonal Procrustes problem.

    Tip:
        In linear algebra, the orthogonal Procrustes problem is a matrix approximation
        problem. Considering two matrices A and B, it consists in finding the orthogonal
        matrix R which most closely maps A to B, so that it minimizes the Frobenius norm of
        ``(A @ R) - B`` subject to ``R.T @ R = I``.
        See :py:func:`scipy.linalg.orthogonal_procrustes` for more information.

    For each dataset, the data and labels to align the data on is provided.

    1. The ``top_k`` indexes of the labels to align (``label``) that are the closest to the labels of the reference dataset (``ref_label``) are selected and used to sample from the dataset to align (``data``).
    2. ``data`` and ``ref_data`` (the reference dataset) are subsampled to the same number of samples ``subsample``.
    3. The orthogonal mapping is computed, using :py:func:`scipy.linalg.orthogonal_procrustes`, on those subsampled datasets.
    4. The resulting orthongonal matrix ``_transform`` can be used to map the original ``data`` to the ``ref_data``.

    Note:
        ``data`` and ``ref_data`` can be of different sample size (axis 0) but **must** have the same number
        of features (axis 1) to be aligned.

    Attributes:
        top_k (int): Number of indexes in the labels of the matrix to align to consider for alignment
            (``label``). The selected indexes consist in the ``top_k`` th indexes the closest to the
            reference labels (``ref_label``).
        subsample (int): Number of samples to subsample the ``data`` and ``ref_data`` from, to solve the orthogonal
            Procrustes problem on.
    """

    def __init__(self, top_k: int = 5, subsample: Optional[int] = None):
        self.subsample = subsample
        self.top_k = top_k

    def _distance(self, label_i: npt.NDArray,
                  label_j: npt.NDArray) -> npt.NDArray:
        """Compute the Euclidean distance between two matrices.

        Args:
            label_i: An array.
            label_j: A second array.

        Returns:
            The Euclidean distance between ``label_i`` and ``label_j``.
        """
        norm_i = (label_i**2).sum(1)
        norm_j = (label_j**2).sum(1)
        diff = np.einsum("nd,md->nm", label_i, label_j)
        diff = norm_i[:, None] + norm_j[None, :] - 2 * diff
        return diff

    def fit(
        self,
        ref_data: Union[npt.NDArray, torch.Tensor],
        data: Union[npt.NDArray, torch.Tensor],
        ref_label: Optional[npt.NDArray] = None,
        label: Optional[npt.NDArray] = None,
    ) -> "OrthogonalProcrustesAlignment":
        """Compute the matrix solution of the orthogonal Procrustes problem.

        The obtained matrix is used to align a dataset to a reference dataset.

        Args:
            ref_data: Reference data matrix on which to align the data.
            data: Data matrix to align on the reference dataset.
            ref_label: Set of indices associated to ``ref_data``.
            label: Set of indices associated to ``data``.

        Returns:
            ``self``, for chaining operations.

        Example:

            >>> import cebra.data.helper
            >>> import numpy as np
            >>> ref_embedding = np.random.uniform(0, 1, (1000, 30))
            >>> aux_embedding = np.random.uniform(0, 1, (800, 30))
            >>> ref_label = np.random.uniform(0, 1, (1000, 1))
            >>> aux_label = np.random.uniform(0, 1, (800, 1))
            >>> orthogonal_procrustes = cebra.data.helper.OrthogonalProcrustesAlignment()
            >>> orthogonal_procrustes = orthogonal_procrustes.fit(ref_data=ref_embedding,
            ...                                                   data=aux_embedding,
            ...                                                   ref_label=ref_label,
            ...                                                   label=aux_label)

        """
        if len(ref_data.shape) == 1:
            ref_data = np.expand_dims(ref_data, axis=1)
        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)

        if ref_data.shape[0] == data.shape[
                0] and ref_label is None and label is None:
            # data are already aligned across timepoints, so we can use them as labels
            n_samples = ref_data.shape[0]
            ref_label = np.expand_dims(np.arange(n_samples), axis=1)
            label = np.expand_dims(np.arange(n_samples), axis=1)
        elif ref_data.shape[0] == data.shape[0] and (ref_label is None or
                                                     label is None):
            raise ValueError(
                f"Missing labels: the data to align are the same shape but you provided only "
                f"one of the sets of labels. Either provide both the reference and alignment "
                f"labels or none.")
        else:
            if ref_label is None or label is None:
                raise ValueError(
                    f"Missing labels: the data to align are not the same shape, "
                    f"provide labels to align the data and reference data.")

            if len(ref_label.shape) == 1:
                ref_label = np.expand_dims(ref_label, axis=1)
            if len(label.shape) == 1:
                label = np.expand_dims(label, axis=1)

        if ref_data.shape[0] < self.top_k:
            raise ValueError(
                f"Invalid reference data, reference data should at least "
                f"contain {self.top_k} samples, got {ref_data.shape[0]}.")

        if ref_data.shape[0] != ref_label.shape[0]:
            raise ValueError(
                f"Mismatched reference data and associated reference labels: "
                f"they should contain the same number of samples, but "
                f"got ref_data:{ref_data.shape[0]} samples and ref_labels:{ref_label.shape[0]} samples."
            )

        if data.shape[0] != label.shape[0]:
            raise ValueError(
                f"Mismatched reference data and associated reference labels: "
                f"they should contain the same number of samples, but "
                f"got ref_data:{data.shape[0]} samples and ref_labels:{label.shape[0]} samples."
            )

        distance = self._distance(label, ref_label)

        # keep indexes of the {self.top_k} labels the closest to the reference labels
        target_idx = np.argsort(distance, axis=1)[:, :self.top_k]

        if data.shape[1] != ref_data.shape[1]:
            raise ValueError(
                "Invalid data: reference data and data should have the same number of features but "
                f"got data: {data.shape[1]} features and ref_data: {ref_data.shape[1]} features."
            )

        # Get the whole data to align and only the selected closest samples
        # from the reference data.
        X = data[:, None].repeat(self.top_k, axis=1).reshape(-1, data.shape[1])
        Y = ref_data[target_idx].reshape(-1, ref_data.shape[1])

        # Augment data and reference data so that same size
        if self.subsample is not None:
            if self.subsample > len(X):
                warnings.warn(
                    f"The number of datapoints in the dataset ({len(X)}) "
                    f"should be larger than the 'subsample' "
                    f"parameter ({self.subsample}). Ignoring subsampling and "
                    f"computing alignment on the full dataset instead, which will "
                    f"give better results.")
            else:
                if self.subsample < 1000:
                    warnings.warn(
                        "This function is experimental when the subsample dimension "
                        "is less than 1000. You can probably use the whole dataset "
                        "for alignment by setting subsample=None.")

                idc = np.random.choice(len(X), self.subsample)
                X = X[idc]
                Y = Y[idc]

        # Compute orthogonal matrix that most closely maps X to Y using the orthogonal Procrustes problem.
        self._transform, _ = scipy.linalg.orthogonal_procrustes(X, Y)

        return self

    def transform(self, data: Union[npt.NDArray, torch.Tensor]) -> npt.NDArray:
        """Transform the data using the matrix solution computed in py:meth:`fit`.

        Args:
            data: The 2D data matrix to align.

        Returns:
            The aligned input matrix.

        Example:

            >>> import cebra.data.helper
            >>> import numpy as np
            >>> ref_embedding = np.random.uniform(0, 1, (1000, 30))
            >>> aux_embedding = np.random.uniform(0, 1, (800, 30))
            >>> ref_label = np.random.uniform(0, 1, (1000, 1))
            >>> aux_label = np.random.uniform(0, 1, (800, 1))
            >>> orthogonal_procrustes = cebra.data.helper.OrthogonalProcrustesAlignment()
            >>> orthogonal_procrustes = orthogonal_procrustes.fit(ref_data=ref_embedding,
            ...                                                   data=aux_embedding,
            ...                                                   ref_label=ref_label,
            ...                                                   label=aux_label)
            >>> aligned_aux_embedding = orthogonal_procrustes.transform(data=aux_embedding)
            >>> assert aligned_aux_embedding.shape == aux_embedding.shape

        """
        if self.transform is None:
            raise ValueError("Call fit() first.")

        if len(data.shape) == 1:
            data = np.expand_dims(data, axis=1)

        return data @ self._transform

    def fit_transform(
        self,
        ref_data: npt.NDArray,
        data: npt.NDArray,
        ref_label: Optional[npt.NDArray] = None,
        label: Optional[npt.NDArray] = None,
    ) -> npt.NDArray:
        """Compute the matrix solution to align a data array to a reference matrix.

        Note:
            Uses a combination of :py:meth:`~.OrthogonalProcrustesAlignment.fit` and :py:meth:`~.OrthogonalProcrustesAlignment.transform`.


        Args:
            ref_data: Reference data matrix on which to align the data.
            data: Data matrix to align on the reference dataset.
            ref_label: Set of indices associated to ``ref_data``.
            label: Set of indices associated to ``data``.

        Returns:
            The ``data`` matrix aligned onto the reference data matrix.

        Example:

            >>> import cebra.data.helper
            >>> import numpy as np
            >>> ref_embedding = np.random.uniform(0, 1, (1000, 30))
            >>> aux_embedding = np.random.uniform(0, 1, (800, 30))
            >>> ref_label = np.random.uniform(0, 1, (1000, 1))
            >>> aux_label = np.random.uniform(0, 1, (800, 1))
            >>> orthogonal_procrustes = cebra.data.helper.OrthogonalProcrustesAlignment(top_k=10,
            ...                                                                         subsample=700)
            >>> aligned_aux_embedding = orthogonal_procrustes.fit_transform(ref_data=ref_embedding,
            ...                                                             data=aux_embedding,
            ...                                                             ref_label=ref_label,
            ...                                                             label=aux_label)
            >>> assert aligned_aux_embedding.shape == aux_embedding.shape

        """
        self.fit(ref_data, data, ref_label, label)
        return self.transform(data)


def ensemble_embeddings(
    embeddings: List[Union[npt.NDArray, torch.Tensor]],
    labels: Optional[List[Union[npt.NDArray, torch.Tensor]]] = None,
    post_norm: bool = False,
    n_jobs: int = 0,
) -> npt.NDArray:
    """Ensemble aligned embeddings together.

    The embeddings contained in ``embeddings`` are aligned onto the same embedding,
    using :py:class:`OrthogonalProcrustesAlignment`. Then, they are averaged and the
    resulting averaged embedding is the ensemble embedding.

    Tip:
        By ensembling embeddings coming from the same dataset but obtained from different models,
        the resulting joint embedding usually shows an increase in performances compared to the individual
        embeddings.

    Note:
        The embeddings in ``embeddings`` must be the same shape, i.e., the same number of samples
        and same number of features (axis 1).

    Args:
        embeddings: List of embeddings to align and ensemble.
        labels: Optional list of indexes associated to the embeddings in ``embeddings`` to align the embeddings on.
            To be ensembled, the embeddings should already be aligned on time, and consequently do not require extra
            labels for alignment.
        post_norm: If True, the resulting joint embedding is normalized (divided by its norm across
            the features - axis 1).
        n_jobs: The maximum number of concurrently running jobs to compute embedding alignment in a parallel manner using
            :py:class:`joblib.Parallel`. Specify ``0`` to iterate naively over the embeddings for ensembling without using
            :py:class:`joblib.Parallel`. Specify ``-1`` to use all cores. Using more than a single core can considerably
            speed up the computation of ensembled embeddings for large datasets, but will also require more memory.

    Returns:
        A :py:func:`numpy.array` corresponding to the joint embedding.

    Example:

        >>> import cebra.data.helper
        >>> import numpy as np
        >>> embedding1 = np.random.uniform(0, 1, (100, 4))
        >>> embedding2 = np.random.uniform(0, 1, (100, 4))
        >>> embedding3 = np.random.uniform(0, 1, (100, 4))
        >>> joint_embedding = cebra.data.helper.ensemble_embeddings(embeddings=[embedding1, embedding2, embedding3])
        >>> assert joint_embedding.shape == embedding1.shape

    """
    if n_jobs is None or not isinstance(n_jobs, int):
        raise ValueError(
            "Invalid n_jobs: it should be an Integer, got {n_jobs}. To not use parallelisation, set n_jobs to 0."
        )

    embeddings = [_require_numpy_array(embedding) for embedding in embeddings]

    # align each embedding from the list to the first one
    ortho_alignment = OrthogonalProcrustesAlignment()
    for embedding in embeddings[1:]:
        if embeddings[0].shape != embedding.shape:
            raise ValueError(
                f"Inconsistent embeddings shapes, one embedding is {embedding.shape}, another {embeddings[0].shape}."
            )

    if labels is None:
        labels = [None for i in range(len(embeddings))]

    if n_jobs == 0:
        joint_embedding = copy.deepcopy(embeddings[0])
        for embedding, label in zip(embeddings[1:], labels[1:]):
            aligned_embedding = ortho_alignment.fit_transform(
                embeddings[0], embedding, labels[0], label)
            joint_embedding += aligned_embedding
        joint_embedding = joint_embedding / len(embeddings)

    else:
        delayed_func = [
            joblib.delayed(ortho_alignment.fit_transform)(embeddings[0],
                                                          embedding, labels[0],
                                                          label)
            for embedding, label in zip(embeddings[1:], labels[1:])
        ]
        parallel_pool = joblib.Parallel(n_jobs=n_jobs)
        res = parallel_pool(delayed_func)

        joint_embedding = (embeddings[0] + sum(res)) / len(embeddings)

    if post_norm:
        joint_embedding = joint_embedding / np.linalg.norm(
            joint_embedding, axis=1, keepdims=True)

    return joint_embedding
