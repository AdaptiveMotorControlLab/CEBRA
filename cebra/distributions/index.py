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
"""Index operations for conditional sampling.

Indexing operations---in contrast to data distributions---exhibit deterministic behavior
by returning an element closest in the dataset to a given query sample. This module contains
helper functions for mixed and continuously indexed datasets (i.e., containing discrete and/or
continuous data).

Discrete data has to come in the format of a single label for each datapoint. Multidimensional
discrete labels should be converted accordingly.
"""

import numpy as np
import torch

import cebra.data
import cebra.distributions.base as cebra_distributions
import cebra.io

_INF = float("inf")


def _is_float_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor)


def _check_is_float_tensor(sender, tensor):
    if not _is_float_tensor(tensor):
        raise ValueError(
            f"{sender} requires a torch.Tensor of floating point type "
            f"(either on cpu or gpu), but got {type(tensor)} with dtype={tensor.dtype}."
        )
    return isinstance(tensor, torch.Tensor) and torch.is_floating_point(tensor)


class DistanceMatrix(cebra.io.HasDevice):
    """Compute shortest distances between dataset samples.

    Args:
        samples: The continuous values that will be used to index
            the dataset and specify the conditional distribution.

    Note:
        This implementation is not particularly efficient on very
        large datasets. For these cases, packages like FAISS offer
        more optimized retrieval functions.

        As a rule of thumb, using this class is suitable for
        datasets for which the dataset can be hosted on GPU memory.
    """

    def __init__(self, samples: torch.Tensor):
        _check_is_float_tensor(self, samples)
        self.index = samples
        self.xTx = self.index.square().sum(1, keepdim=True)

    def __call__(self, query, mask=None):
        """Compute the pairwise distances between index and query.

        Args:
            query: (n, d)
                The query matrix
            mask: (N,)
                A binary mask with same length as the index

        Returns: (M,n)
            Pairwise distances between the m samples with True binary
            mask (default: N) and the given query samples.
        """
        # TODO(stes): slow
        query = query.to(self.device)

        qTq = query.square().sum(1, keepdim=True).T
        if mask is None:
            xTq = torch.einsum("ni,mi->nm", self.index, query)
            xTx = self.xTx
        else:
            xTq = torch.einsum("ni,mi->nm", self.index[mask], query)
            xTx = self.xTx[mask]
        return xTx + qTq - 2 * xTq


class OffsetDistanceMatrix(DistanceMatrix):
    """Compute shortest distances, ignoring samples close to the boundary.

    Compared to the standard ``DistanceMatrix``, this class should be used
    for datasets and learning setups where multiple timesteps are fed into
    the network at once --- the samples close to the time-series boundary
    should be ignored in the sampling process in these cases.

    Args:
        samples: The continuous values that will be used to index
            the dataset and specify the conditional distribution.
        offset: The number of timesteps to ignored at each size of the
            dataset

    TODO:
        * switch offset to `cebra.data.Offset`
    """

    def __init__(self, samples, offset: int = 1):
        super().__init__(samples)
        self.inf = torch.tensor(_INF)
        self.offset = cebra.data.Offset(offset)
        if len(self.offset) < 1:
            raise ValueError(
                f"Choose an offset of at least 1, otherwise use {type(super())}."
            )
        self.offset.mask_array(self.xTx, self.inf)


class ContinuousIndex(cebra_distributions.Index, cebra.io.HasDevice):
    """Naive nearest neighbor search implementation.

    index: tensor(N, d)
        the values used for kNN search
    offset: int or (int,int)
        the time offset in each direction
    """

    def __init__(self, index):
        super().__init__()
        _check_is_float_tensor(self, index)
        self.dist_matrix = DistanceMatrix(index)

    def search(self, query):
        """Return index location closest to query."""
        distance = self.dist_matrix(query)
        return torch.argmin(distance, dim=0)
        # TODO(stes) handle offsets
        # + self.dist_matrix.offset.left


class ConditionalIndex(cebra_distributions.Index):
    """Index a dataset based on both continuous and discrete information.

    In contrast to the standard :py:class:`.base.Index` class, the :py:class:`ConditionalIndex`
    accept both discrete and continuous indexing information.

    This index considers the discrete indexing information first to
    identify possible positive pairs. Then among these candidate samples,
    behaves like an :py:class:`.base.Index` and returns the samples closest in terms of
    the information in the continuous index.

    Args:
        discrete: The discrete indexing information, which should be
            limited to a 1d feature space. If higher dimensional discrete
            vectors are used, they should be first re-formatted to fit
            this structure.
        continuous: The continuous indexing information, which can be a
            vector of arbitrary dimension and will be used to define the
            distance between the samples that share the same discrete
            index.
    """

    def __init__(self, discrete, continuous):
        _check_is_float_tensor(self, continuous)
        if discrete is None:
            raise ValueError(
                "The specified discrete index was set to None. "
                "If this was intended, use Index instead of ConditionalIndex "
                "which does not require to specify discrete indexing "
                "information.")
        if len(discrete) != len(continuous):
            raise ValueError(
                f"Discrete ({len(discrete)} samples) and continuous "
                f"({len(continuous)} samples) need to match in their number "
                "of samples.")
        if len(discrete.shape) > 1:
            raise ValueError(
                f"Discrete indexing information needs to be limited to a 1d "
                f"array/tensor. Multi-dimensional discrete indices should be "
                f"reformatted first.")
            # TODO(stes): Once a helper function exists, the error message should
            #            mention it.

        self.discrete = discrete
        self.continuous = continuous

        self.distance_matrix = DistanceMatrix(self.continuous)

        self.mask_x = {
            int(v): (self.discrete == v) for v in torch.unique(discrete)
        }
        self.mask_idx = {
            int(v): torch.nonzero(mask_x).squeeze()
            for v, mask_x in self.mask_x.items()
        }

    def search(self, continuous, discrete=None):
        """Search closest sample based on continuous and discrete indexing
        information.

        Args:
            continuous:
                Samples from the continuous index
            discrete:
                Optionally matching samples from the discrete index,
                used to pre-select matching indices.
        """
        if continuous.shape[1] != self.continuous.shape[1]:
            raise ValueError(f"Shape of continuous index does not match along "
                             f"the feature dimension. "
                             f"Expected {self.continuous.shape[1]}d, but got "
                             f"{continuous.shape[1]}d.")
        if discrete is None:
            return self.search_naive(continuous, discrete=None)

        # TODO(stes) select based on expected speed advantage
        return self.search_naive(continuous, discrete)

    def __getitem__(self, value):
        # TODO(stes): this function might not be used; consider removing
        #            for removing, tests should pass while this function
        #            returns a deprecation exception; it is unclear why
        #            this adds an overall value to the API.
        continuous, discrete = value
        return self.search(continuous, discrete)

    def search_naive(self, continuous, discrete):
        """Brute force search
        Fast especially for small indices

        Args:
            continuous:
                TODO
            discrete:
                TODO
        """
        # TODO(stes): slow
        continuous = continuous.to(self.device)

        distance = self.distance_matrix(continuous)
        if discrete is not None:
            # TODO(stes): slow
            discrete = discrete.to(self.device)
            mask = torch.eq(self.discrete[:, None], discrete[None, :])
            distance[~mask] = _INF  # TODO: inefficient
        return torch.argmin(distance, dim=0)

    def search_iterative(self, continuous, discrete):
        """Iterative search
        Gets faster especially for >1e6 samples in the index.

        Args:
            continuous:
                TODO
            discrete:
                TODO
        """
        # TODO(stes): slow
        discrete = discrete.to(self.device)
        continuous = continuous.to(self.device)

        ret = torch.zeros_like(discrete)
        for v in torch.unique(discrete):
            mask_x = self.mask_x[int(v)]
            mask_q = discrete == v
            diff = self.distance_matrix(continuous[mask_q], mask=mask_x)
            ret[mask_q] = self.mask_idx[int(v)][torch.argmin(diff, dim=0)]
        return ret


class MultiSessionIndex(cebra_distributions.Index):
    """Index multiple sessions.

    Args:
        indices: Indices for the different sessions. Indices of multi-session
            datasets should have matching feature dimension.
    """

    def __init__(self, *indices):
        self.indices = indices

    def search(self, query):
        """Return closest element in each of the datasets.

        Args:
            query: The query which is applied to each index of
                the dataset.

        Returns:
            A list of indices from each session.
        """
        return [index.search(query) for index in self.indices]
