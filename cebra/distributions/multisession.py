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
"""Continuous variable multi-session sampling."""

import random
from typing import Optional

import numpy as np
import numpy.typing as npt
import torch

import cebra.distributions as cebra_distr
import cebra.io


def _resolve_device(dataset, device: Optional[str] = None) -> str:
    """Device of the sampler: ``device``, else the dataset's, with ``"cuda"`` made explicit."""

    def _normalize_cuda_device(device_name: str) -> str:
        if device_name != "cuda":
            return device_name

        # Prefer the concrete device index from dataset tensors if available.
        index = getattr(dataset, "continuous_index", None)
        if index is None:
            index = getattr(dataset, "discrete_index", None)
        if index is not None and index.device.type == "cuda" and index.device.index is not None:
            return f"cuda:{index.device.index}"

        # Fall back to currently selected CUDA device instead of implicit cuda:0.
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return device_name

    if device is not None:
        return _normalize_cuda_device(str(device))
    if hasattr(dataset, "device"):
        return _normalize_cuda_device(str(dataset.device))
    return _normalize_cuda_device(cebra.io.device())


def _search(data, query):
    if query.ndim == 1:
        query = query[None]
    return np.argmin(abs(data[None, :] - query[:, None]).sum(-1), axis=1)


def _invert_index(idx):
    """Invert an indexing function

    Let the given array define a function v: [N]->[N], then
    the returned array defines its inverse.

    Example:

        >>> import numpy as np
        >>> idx = np.array([2,3,1,0])
        >>> idx_inv = _invert_index(idx)
        >>> print(idx[idx_inv].tolist())
        [0, 1, 2, 3]

    """
    out = np.zeros_like(idx)
    out[idx] = np.arange(len(idx))
    return out


class MultisessionSampler(cebra_distr.PriorDistribution,
                          cebra_distr.ConditionalDistribution):
    """Continuous multi-session sampling.

    Align embeddings across multiple sessions, using a continuous
    index. The transitions between index samples are computed across
    all sessions.

    Note:
        The batch dimension of positive samples are shuffled.
        Before applying the contrastive loss, either the reference samples
        need to be aligned with the positive samples, or vice versa:

        >>> import cebra.distributions.multisession as cebra_distributions_multisession
        >>> import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset
        >>> import cebra.data
        >>> import torch
        >>> from torch import nn
        >>> # Multisession training: one model per dataset (different input dimensions)
        >>> session1 = torch.rand(100, 30)
        >>> session2 = torch.rand(100, 50)
        >>> index1 = torch.rand(100)
        >>> index2 = torch.rand(100)
        >>> num_features = 8
        >>> dataset = cebra.data.DatasetCollection(
        ...               cebra_sklearn_dataset.SklearnDataset(session1, (index1, )),
        ...               cebra_sklearn_dataset.SklearnDataset(session2, (index2, )))
        >>> model = nn.ModuleList([
        ...                cebra.models.init(
        ...                    name="offset1-model",
        ...                    num_neurons=dataset.input_dimension,
        ...                    num_units=32,
        ...                    num_output=num_features,
        ...                ) for dataset in dataset.iter_sessions()]).to("cpu")
        >>> sampler = cebra_distributions_multisession.MultisessionSampler(dataset, time_offset=10)

        >>> # ref and pos samples from all datasets
        >>> ref = sampler.sample_prior(100)
        >>> pos, idx, rev_idx = sampler.sample_conditional(ref)
        >>> ref = torch.LongTensor(ref)
        >>> pos = torch.LongTensor(pos)

        >>> # Then the embedding spaces can be concatenated
        >>> refs, poss = [], []
        >>> for i in range(len(model)):
        ...     refs.append(model[i](dataset._datasets[i][ref[i]]))
        ...     poss.append(model[i](dataset._datasets[i][pos[i]]))
        >>> ref = torch.stack(refs, dim=0)
        >>> pos = torch.stack(poss, dim=0)

        >>> # Now the index can be applied to the stacked features,
        >>> # to align reference to positive samples
        >>> aligned_ref = sampler.mix(ref, idx)
        >>> reference = aligned_ref.view(-1, num_features)
        >>> positive = pos.view(-1, num_features)
        >>> loss = (reference - positive)**2

        >>> # .. or the reverse index, to align positive to reference samples
        >>> aligned_pos = sampler.mix(pos, rev_idx)
        >>> reference = ref.view(-1, num_features)
        >>> positive = aligned_pos.view(-1, num_features)
        >>> loss = (ref - pos)**2

        The reason for this implementation is that ``dataset[i]`` will in
        general have different dimensions (for example, number of neurons),
        per session. In contrast to the reference and positive
        indices, this data cannot be stacked and models need to be applied
        session by session.

    After data processing, the dimensionality of the returned features
    matches. The resulting embeddings can be concatenated, and shuffling
    (across the session axis) can be applied to the reference samples, or
    reversed for the positive samples.

    Note:
        This function does currently not support explicitly selected
        discrete indices. They should be added as dimensions to the
        continuous index. More weight can be added to the discrete
        dimensions by using larger values in one-hot coding.

    TODO:
        * Add a dedicated sampler for mixed multi session sampling.
        * Add better CUDA support and refactor ``numpy`` implementation.
    """

    def __init__(self, dataset, time_offset, device: Optional[str] = None):
        self.dataset = dataset
        self.device = _resolve_device(dataset, device)

        # TODO(stes): implement in pytorch
        self.all_data = self.dataset.continuous_index.cpu().numpy()
        self.session_lengths = np.array(self.dataset.session_lengths)

        self.lengths = np.cumsum(self.session_lengths)
        self.lengths[1:] = self.lengths[:-1]
        self.lengths[0] = 0

        # TODO(stes): unify naming
        time_delta = time_offset

        # NOTE(celia): Keep this on CPU: a previous CUDA torch.cat here could OOM due to
        # large temporary allocations when concatenating across many sessions.
        time_differences = []
        for dataset in self.dataset.iter_sessions():
            continuous_index = dataset.continuous_index
            if continuous_index.device.type != "cpu":
                continuous_index = continuous_index.cpu()
            time_differences.append((continuous_index[time_delta:] -
                                     continuous_index[:-time_delta]).numpy())
        self.time_difference = np.concatenate(time_differences, axis=0)

        self.index = [
            cebra_distr.ContinuousIndex(dataset.continuous_index.float().to(
                self.device)) for dataset in self.dataset.iter_sessions()
        ]

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the index."""
        return len(self.lengths)

    def mix(self, array: npt.NDArray, idx: npt.NDArray):
        """Re-order array elements according to the given index mapping.

        The given array should be of the shape ``(session, batch, ...)`` and the
        indices should have length ``session x batch``, representing a mapping
        between indices.

        The resulting array will be rearranged such that
        ``out.reshape(session*batch, -1)[i] = array.reshape(session*batch, -1)[idx[i]]``

        For the inverse mapping, convert the indices first using ``_invert_index``
        function.

        Args:
            array: A 2D matrix containing samples for each session.
            idx: A list of indexes to re-order ``array`` on.
        """
        n, m = array.shape[:2]
        return array.reshape(n * m, -1)[idx].reshape(array.shape)

    def sample_prior(self, num_samples):
        # TODO(stes) implement empirical/uniform resampling
        ref_idx = np.random.uniform(0, 1, (self.num_sessions, num_samples))
        ref_idx = (ref_idx * self.session_lengths[:, None]).astype(int)
        return ref_idx

    def sample_conditional(self, idx: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional distribution.

        Note:
            * Reference samples are sampled equally between sessions.
            * Queries are computed for each reference as in single-session,
              meaning by adding a random time shift to each reference sample.
            * In order to guarantee the same number of positive samples per
              session, queries are randomly assigned to a session and its
              corresponding positive sample is searched in that session only.
            * As a result, ref/pos pairing is shuffled and can be recovered
              the reverse shuffle operation.

        Args:
            idx: Reference indices, with dimension ``(session, batch)``.

        Returns:
            Positive indices (1st return value), which will be grouped by
            session and *not* match the reference indices.
            In addition, a mapping will be returned to apply the same shuffle operation
            that was applied to assign a query to a session along session/batch dimension
            to the reference indices (2nd return value), or reverse the shuffle operation
            (3rd return value).
            Returned shapes are ``(session, batch), (session, batch), (session, batch)``.

        TODO:
            * re-implement in pytorch for additional speed gains
        """

        shape = idx.shape
        # TODO(stes) unclear why we cannot restrict to 2d overall
        # idx has shape (2, #samples per batch)
        s = idx.shape[:2]
        idx = (idx + self.lengths[:, None]).flatten()

        diff_idx = torch.randint(len(self.time_difference), (len(idx),))
        query = self.all_data[idx] + self.time_difference[diff_idx]

        # shuffle operation to assign each query to a session
        idx = np.random.permutation(len(query))

        # TODO this part fails in Pytorch
        query = query[idx.reshape(s)]
        query = torch.from_numpy(query).to(self.device)

        pos_idx = torch.zeros(shape, device=self.device).long()
        for i in range(self.num_sessions):
            pos_idx[i] = self.index[i].search(query[i])
        pos_idx = pos_idx.cpu().numpy()

        # reverse indices to recover the ref/pos samples matching
        idx_rev = _invert_index(idx)
        return pos_idx, idx, idx_rev

    def __getitem__(self, pos_idx):
        pos_samples = np.zeros(pos_idx.shape[:2] + (self.data.shape[2],))
        for i in range(self.num_sessions):
            pos_samples[i] = self.data[i][pos_idx[i]]
        return pos_samples


class DiscreteMultisessionSampler(cebra_distr.PriorDistribution,
                                  cebra_distr.ConditionalDistribution):
    """Discrete multi-session sampling.

    Discrete indices don't need to be aligned. Positive pairs are found
    by matching the discrete index in randomly assigned sessions.

    After data processing, the dimensionality of the returned features
    matches. The resulting embeddings can be concatenated, and shuffling
    (across the session axis) can be applied to the reference samples, or
    reversed for the positive samples.

    TODO:
        * Add better CUDA support and refactor ``numpy`` implementation.
    """

    def __init__(self, dataset, device: Optional[str] = None):
        self.dataset = dataset
        self.device = _resolve_device(dataset, device)

        # TODO(stes): implement in pytorch
        self.all_data = self.dataset.discrete_index.cpu().numpy()
        self.session_lengths = np.array(self.dataset.session_lengths)

        self.lengths = np.cumsum(self.session_lengths)
        self.lengths[1:] = self.lengths[:-1]
        self.lengths[0] = 0

        self.index = [
            cebra_distr.DiscreteUniform(dataset.discrete_index.int().to(
                self.device)) for dataset in self.dataset.iter_sessions()
        ]

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the index."""
        return len(self.lengths)

    def mix(self, array: np.ndarray, idx: np.ndarray):
        """Re-order array elements according to the given index mapping.

        The given array should be of the shape ``(session, batch, ...)`` and the
        indices should have length ``session x batch``, representing a mapping
        between indices.

        The resulting array will be rearranged such that
        ``out.reshape(session*batch, -1)[i] = array.reshape(session*batch, -1)[idx[i]]``

        For the inverse mapping, convert the indices first using ``_invert_index``
        function.

        Args:
            array: A 2D matrix containing samples for each session.
            idx: A list of indexes to re-order ``array`` on.
        """
        n, m = array.shape[:2]
        return array.reshape(n * m, -1)[idx].reshape(array.shape)

    def sample_prior(self, num_samples):
        # TODO(stes) implement empirical/uniform resampling
        ref_idx = np.random.uniform(0, 1, (self.num_sessions, num_samples))
        ref_idx = (ref_idx * self.session_lengths[:, None]).astype(int)
        return ref_idx

    def sample_conditional(self, idx: torch.Tensor) -> torch.Tensor:
        """Sample from the conditional distribution.

        Note:
            * Reference samples are sampled equally between sessions.
            * In order to guarantee the same number of positive samples per
              session, reference samples are randomly assigned to a session and its
              corresponding positive sample is searched in that session only.
            * As a result, ref/pos pairing is shuffled and can be recovered
              the reverse shuffle operation.

        Args:
            idx: Reference indices, with dimension ``(session, batch)``.

        Returns:
            Positive indices (1st return value), which will be grouped by
            session and *not* match the reference indices.
            In addition, a mapping will be returned to apply the same shuffle operation
            that was applied to assign reference samples to a session along session/batch dimension
            (2nd return value), or reverse the shuffle operation (3rd return value).
            Returned shapes are ``(session, batch), (session, batch), (session, batch)``.

        TODO:
            * re-implement in pytorch for additional speed gains
        """

        shape = idx.shape
        # TODO(stes) unclear why we cannot restrict to 2d overall
        # idx has shape (2, #samples per batch)
        s = idx.shape[:2]
        idx_all = (idx + self.lengths[:, None]).flatten()

        # get discrete indices
        query = self.all_data[idx_all]

        # shuffle operation to assign each index to a session
        idx = np.random.permutation(len(query))

        # TODO this part fails in Pytorch
        # apply shuffle
        query = query[idx.reshape(s)]
        query = torch.from_numpy(query).to(self.device)

        # sample conditional for each assigned session
        pos_idx = torch.zeros(shape, device=self.device).long()
        for i in range(self.num_sessions):
            pos_idx[i] = self.index[i].sample_conditional(query[i])
        pos_idx = pos_idx.cpu().numpy()

        # reverse indices to recover the ref/pos samples matching
        idx_rev = _invert_index(idx)
        return pos_idx, idx, idx_rev

    def __getitem__(self, pos_idx):
        pos_samples = np.zeros(pos_idx.shape[:2] + (self.data.shape[2],))
        for i in range(self.num_sessions):
            pos_samples[i] = self.data[i][pos_idx[i]]
        return pos_samples


class UnifiedSampler(MultisessionSampler):
    """Multi-session sampling, considering them as a single session.

    For each sample of a reference session, every other session contributes the sample with
    the nearest auxiliary variable. The matched samples are concatenated on the neuron axis, so
    a single model is trained on all sessions.

    Nearest-label search:

    - Ties: when several samples are equally near (e.g. the same stimulus shown on several
      trials), ``random_ties=True`` (default) draws one uniformly at random; ``False`` always
      takes the first, so only the first occurrence (e.g. the first trial) is ever used.
    - Repeated labels are handled automatically: if a session has at most half as many
      distinct labels as samples, distances are computed to the distinct labels only, which is
      faster. Otherwise (e.g. measured position or velocity) all samples are searched, as if
      the step did not exist; only a one-time check per session is added. Nothing needs to be
      set.

    Example:
        >>> import cebra.distributions.multisession as cebra_distributions_multisession
        >>> import cebra.integrations.sklearn.dataset as cebra_sklearn_dataset
        >>> import cebra.data
        >>> import torch
        >>> from torch import nn
        >>> # Multisession training: one model per dataset (different input dimensions)
        >>> session1 = torch.rand(100, 30)
        >>> session2 = torch.rand(100, 50)
        >>> index1 = torch.rand(100)
        >>> index2 = torch.rand(100)
        >>> num_features = 8
        >>> dataset = cebra.data.UnifiedDataset(
        ...               cebra_sklearn_dataset.SklearnDataset(session1, (index1, )),
        ...               cebra_sklearn_dataset.SklearnDataset(session2, (index2, )))
        >>> model = cebra.models.init(
        ...                    name="offset1-model",
        ...                    num_neurons=dataset.input_dimension,
        ...                    num_units=32,
        ...                    num_output=num_features,
        ...         ).to("cpu")
        >>> sampler = cebra_distributions_multisession.UnifiedSampler(dataset, time_offset=10)

        >>> # ref and pos samples from all datasets
        >>> ref = sampler.sample_prior(100)
        >>> pos = sampler.sample_conditional(ref)
        >>> ref = torch.LongTensor(ref)
        >>> pos = torch.LongTensor(pos)
        >>> loss = (ref - pos)**2

        >>> # Reproducible tie draws: pass a seed
        >>> seeded = cebra_distributions_multisession.UnifiedSampler(dataset, time_offset=10, seed=0)
        >>> matched = seeded.sample_all_sessions(torch.arange(20), session_id=0)
        >>> matched.shape
        torch.Size([2, 20])

    Note:
        This function does currently not support explicitly selected
        discrete indices. They should be added as dimensions to the
        continuous index. More weight can be added to the discrete
        dimensions by using larger values in one-hot coding.

    """

    #: Upper bound on the number of elements of the ``(session length, chunk)`` distance matrix
    #: held in memory at once during the nearest-label search.
    max_search_elements: int = 2**24

    def __init__(self,
                 dataset,
                 time_offset,
                 device: Optional[str] = None,
                 random_ties: bool = True,
                 tie_rtol: float = 1e-6,
                 seed: Optional[int] = None):
        """
        Args:
            dataset: The :py:class:`~cebra.data.UnifiedDataset` to sample from.
            time_offset: Step ``Δ`` of the label differences ``label[t + Δ] - label[t]`` used to
                shift the positive samples.
            device: Device of the sampled indices. Default: the device of the dataset.
            random_ties: Draw one of several equally near samples at random instead of taking
                the first. Keep ``True`` when labels repeat within a session.
            tie_rtol: Relative tolerance for two distances to count as tied.
            seed: Seed of the tie draws (made on CPU, so identical on any device). ``None``
                uses the global torch RNG.
        """
        super().__init__(dataset, time_offset, device=device)
        self.random_ties = random_ties
        self.tie_rtol = tie_rtol
        self.generator = (None if seed is None else
                          torch.Generator().manual_seed(int(seed)))

    def _unique_labels(self, session_id: int):
        """Distinct labels of a session for :py:meth:`_search`, computed once and cached.

        With repeated labels (e.g. the same stimulus shown on several trials), distances are
        computed to the distinct labels only, which is faster the more often each label repeats.
        If more than half of
        the rows are distinct (e.g. measured position), every row is its own label, so the step
        is skipped where it would not help.

        Returns:
            ``(dist_matrix, row_label, first_row)``: distance matrix over the labels, label id
            of each row, and first row of each label (labels numbered in order of appearance).
        """
        cache = self.__dict__.setdefault("_label_cache", {})
        if session_id not in cache:
            labels = self.index[session_id].dist_matrix.index
            uniq, inverse = torch.unique(labels, dim=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            num = len(uniq)
            rows = torch.arange(len(labels), device=labels.device)
            if num > len(labels) // 2:
                # Few repeated labels: searching the distinct labels saves little and costs
                # more on GPU, so every row is its own label.
                cache[session_id] = (self.index[session_id].dist_matrix, rows,
                                     rows)
                return cache[session_id]
            first = torch.full((num,), len(labels), device=labels.device)
            first = first.scatter_reduce(0, inverse, rows, reduce="amin")
            order = first.argsort()  # new label id -> old label id
            rank = torch.empty_like(order)
            rank[order] = torch.arange(num, device=order.device)
            cache[session_id] = (cebra_distr.index.DistanceMatrix(uniq[order]),
                                 rank[inverse], first[order])
        return cache[session_id]

    def _search(self,
                session_id: int,
                query: torch.Tensor,
                generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Row of session ``session_id`` nearest to each query, ties broken as set by ``random_ties``.

        Without ``random_ties``: the first nearest row (as ``argmin``). With ``random_ties``: one
        uniform number per query, drawn on CPU from ``generator`` (else the sampler's, else the
        global RNG), picks one of the ``k`` tied rows, each with probability ``1/k``. Memory is
        bounded by :py:attr:`max_search_elements`.
        """
        if query.ndim == 1:
            query = query[None]
        num_queries = len(query)
        if num_queries == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        dist_matrix, row_label, first_row = self._unique_labels(session_id)
        num_labels = len(first_row)
        num_rows = len(row_label)
        generator = self.generator if generator is None else generator
        query = query.to(dist_matrix.device)
        max_el = int(self.max_search_elements)

        if not self.random_ties:
            uq, inv = torch.unique(query, dim=0, return_inverse=True)
            inv = inv.reshape(-1)
            chunk = max(1, max_el // max(num_labels, 1))
            best = torch.cat([
                dist_matrix(uq[s:s + chunk]).argmin(dim=0)
                for s in range(0, len(uq), chunk)
            ])
            return first_row[best][inv]

        # One uniform number per original query, drawn up front in query order.
        u = torch.rand(num_queries, generator=generator).to(dist_matrix.device)
        uq, inv = torch.unique(query, dim=0, return_inverse=True)
        inv = inv.reshape(-1)
        out = torch.empty(num_queries,
                          dtype=torch.long,
                          device=dist_matrix.device)
        # distinct queries per block: the (labels, block) distances and the (block, rows)
        # tied-row table both stay below max_search_elements
        block = max(1, max_el // max(num_labels, num_rows, 1))
        stride = num_rows + 1
        for start in range(0, len(uq), block):
            distance = dist_matrix(uq[start:start + block])  # (labels, block)
            dmin = distance.min(dim=0, keepdim=True).values
            tied_label = distance <= dmin + self.tie_rtol * dmin.abs(
            ).clamp_min(1.0)
            # (block, rows): the label of each row is tied; running count along the rows
            tied = tied_label.T[:, row_label]
            running = tied.cumsum(dim=1)
            num_tied = running[:, -1]
            cols = torch.arange(len(running), device=running.device)
            # every row of ``running`` is non-decreasing, so with an offset per row the
            # flattened table is sorted and one searchsorted answers all queries of the block
            flat = (running + cols[:, None] * stride).reshape(-1)
            sel = ((inv >= start) & (inv < start + block)).nonzero().flatten()
            col = inv[sel] - start
            k = num_tied[col]
            rank = (u[sel] * k).floor().long().clamp_max(k - 1)
            # (rank + 1)-th tied row: number of rows whose running count is <= rank
            pos = torch.searchsorted(flat, col * stride + rank, right=True)
            out[sel] = pos - col * num_rows
        return out

    def sample_all_uniform_prior(self, num_samples: int) -> npt.NDArray:
        """Returns uniformly sampled index for all sessions of the dataset.

        Args:
            num_samples: Number of samples to sample in each session.

        Returns:
            ``(N, num_samples)`` with ``N`` the number of sessions. Array of
            samples, uniformly picked for each session.
        """
        return super().sample_prior(num_samples=num_samples)

    def sample_prior(self,
                     num_samples: int,
                     session_id: Optional[int] = None) -> npt.NDArray:
        """Return uniformly sampled indices for all sessions.

        First, the reference indexes in a reference session are uniformly sampled.
        Then the reference indexes for the other sessions are sampled so that their
        corresponding auxiliary variables are close to the reference indexes of the
        reference session.

        Args:
            num_samples: Number of samples to pick.
            session_id: ID of the session to use as the reference session. If ``None``,
                the session is randomly selected.

        Returns:
            A :py:func:`numpy.array` containing the idx of the reference samples for all
            sessions.
        """

        # Randomly pick the reference session
        if session_id is None:
            session_id = random.choice(list(range(self.num_sessions)))

        # Sample prior for all sessions
        idx = self.sample_all_uniform_prior(num_samples=num_samples)
        # Keep the idx for the reference session only
        idx = torch.from_numpy(idx[session_id])

        # Sample the references indexes in other sessions, based on their distance to the
        # reference idx in the reference session.
        return self.sample_all_sessions(idx, session_id).cpu().numpy()

    def _get_query(self,
                   reference_idx: torch.Tensor,
                   session_id: int,
                   aligned: bool = False) -> torch.Tensor:
        """Get the query for the given reference indices.

        Args:
            reference_idx: Reference indices, with dimension ``(num_samples, )``.
            session_id: Session ID of the reference session, whose ``idx`` are present in
                ``reference_idx``.
            aligned: If True, no time difference is added to the query. Otherwise, a
                random time difference is added to the query.
        """

        # self.all_data is numpy-backed, so ensure indices are on host before indexing.
        if torch.is_tensor(reference_idx):
            reference_idx = reference_idx.detach().cpu().numpy()
        cum_idx = reference_idx + self.lengths[session_id]
        if aligned:
            query = self.all_data[cum_idx]
        else:
            diff_idx = torch.randint(len(self.time_difference),
                                     (len(reference_idx),))
            query = self.all_data[cum_idx] + self.time_difference[diff_idx]
        return torch.from_numpy(query).to(self.device)

    def sample_all_sessions(
            self,
            ref_idx: torch.Tensor,
            session_id: int,
            generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Match the reference samples of one session in every other session.

        In each other session, the sample with the nearest auxiliary variable is chosen (see
        the class docstring for ties).

        Args:
            ref_idx: Uniformly sampled ``idx`` for the reference session, ``(num_samples, )``, values
                can be in ``[0, len(session)]``.
            session_id: Session ID of the reference session, whose ``idx`` are present in ``ref_idx``.
            generator: CPU generator for the tie draws (only used with ``random_ties``). ``None``
                uses the sampler generator (see ``seed``), or the global torch RNG if there is none.

        Returns:
            The prior for all sessions, creating a "pseudo-animal", where ``idx`` sampled in different
            sessions correspond to points in the recordings where the auxiliary variables are similar.

        """
        # Get the continuous data corresponding to the idx
        # all_data: (sum(self.session_lengths), )
        # ref_idx: (num_samples, ), values in [O, len(get_session[session_id])]
        # self.lengths: (num_sessions, ), cumsum of the length of each session, providing the first
        # element of a session in self.all_data.
        # cum_ref_idx: (num_samples, ), values of ref_idx, switched to correspond to the indexes in
        # of session_id, in the flatten array self.all_data.
        all_idx = torch.zeros(self.num_sessions,
                              len(ref_idx),
                              device=self.device).long()
        query = self._get_query(
            reference_idx=ref_idx, session_id=session_id,
            aligned=True)  # same query for all + no time diff added

        for i in range(self.num_sessions):
            # except for the session_id provided
            if i == session_id:
                continue
            # NOTE(celia): this is code to get different query for each. Unused for now.
            #query = self._get_query(reference_idx=ref_idx,
            #                        session_id=session_id,
            #                        aligned=False)

            # get the idx of the datapoint that is the closest to the query
            all_idx[i] = self._search(i, query,
                                      generator)  # search in the whole dataset

            # all_idx[i] = self.index[i].search_or_mask(
            #     query, threshold=self.distance_threshold[i])

        all_idx[session_id] = ref_idx
        return all_idx

    def sample_conditional(self, reference_idx: npt.NDArray) -> torch.Tensor:
        """Sample from the conditional distribution.

        Contrary to the :py:class:`MultisessionSampler`, conditional distribution
        is sampled so that the samples match the reference samples. They are sampled
        from the same session as each reference idx only, rather than across all
        sessions.

        Args:
            reference_idx: Reference indices, with dimension ``(session, batch)``.

        Returns:
            Positive indices, which will be grouped by
            session and match the reference indices.
            Returned shape is ``(session, batch)``.

        """

        cond_idx = torch.zeros((reference_idx.shape[0], reference_idx.shape[1]),
                               dtype=torch.int,
                               device=self.device).long()

        for session_id in range(self.num_sessions):
            query = self._get_query(reference_idx=reference_idx[session_id],
                                    session_id=session_id)

            cond_idx[session_id] = self._search(session_id, query)

        return cond_idx.cpu().numpy()
