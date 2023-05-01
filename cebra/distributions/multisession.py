"""Continuous variable multi-session sampling."""

import numpy as np
import torch

import cebra.distributions as cebra_distr
import cebra.io

_device = cebra.io.device()


def _search(data, query):
    if query.ndim == 1:
        query = query[None]
    return np.argmin(abs(data[None, :] - query[:, None]).sum(-1), axis=1)


def _invert_index(idx):
    """Invert an indexing function

    Let the given array define a function v: [N]->[N], then
    the returned array defines its inverse.


        >>> import numpy as np
        >>> idx = np.array([2,3,1,0])
        >>> print(idx[idx_inv].tolist())
    """
    out = np.zeros_like(idx)
    out[idx] = np.arange(len(idx))
    return out


class MultisessionSampler(cebra_distr.PriorDistribution,
                          cebra_distr.ConditionalDistribution):

    Align embeddings across multiple sessions, using a continuous
    index. The transitions between index samples are computed across
    all sessions.

        >>> ref = sampler.sample_prior(100)
        >>> pos, idx, rev_idx = sampler.sample_conditional(ref)

        >>> for i in range(len(model)):
        >>> loss = (ref - pos)**2


    After data processing, the dimensionality of the returned features
    (across the session axis) can be applied to the reference samples, or
    reversed for the positive samples.


    TODO:
        * Add a dedicated sampler for mixed multi session sampling.
        * Add better CUDA support and refactor ``numpy`` implementation.
    """

    def __init__(self, dataset, time_offset):
        self.dataset = dataset

        self.all_data = self.dataset.continuous_index.cpu().numpy()
        self.session_lengths = np.array(self.dataset.session_lengths)

        self.lengths = np.cumsum(self.session_lengths)
        self.lengths[1:] = self.lengths[:-1]
        self.lengths[0] = 0

        time_delta = time_offset
        self.time_difference = (torch.cat(
            [
                dataset.continuous_index[time_delta:] -
                dataset.continuous_index[:-time_delta]
                for dataset in self.dataset.iter_sessions()
            ],
            dim=0,
        ).cpu().numpy())

        self.index = [
            cebra_distr.ContinuousIndex(
                dataset.continuous_index.float().to(_device))
            for dataset in self.dataset.iter_sessions()
        ]

    @property
    def num_sessions(self) -> int:
        """The number of sessions in the index."""
        return len(self.lengths)

    def mix(self, array: np.ndarray, idx: np.ndarray):
        """Re-order array elements according to the given index mapping.

        indices should have length ``session x batch``, representing a mapping
        between indices.

        The resulting array will be rearranged such that
        ``out.reshape(session*batch, -1)[i] = array.reshape(session*batch, -1)[idx[i]]``

        function.

        Args:
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

        Args:
            idx: Reference indices, with dimension ``(session, batch)``.

        Returns:
            Positive indices (1st return value), which will be grouped by
            session and *not* match the reference indices.
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

        idx = np.random.permutation(len(query))

        # TODO this part fails in Pytorch
        query = query[idx.reshape(s)]
        query = torch.from_numpy(query).to(_device)

        pos_idx = torch.zeros(shape, device=_device).long()
        for i in range(self.num_sessions):
            pos_idx[i] = self.index[i].search(query[i])
        pos_idx = pos_idx.cpu().numpy()

        idx_rev = _invert_index(idx)
        return pos_idx, idx, idx_rev

    def __getitem__(self, pos_idx):
        pos_samples = np.zeros(pos_idx.shape[:2] + (self.data.shape[2],))
        for i in range(self.num_sessions):
            pos_samples[i] = self.data[i][pos_idx[i]]
        return pos_samples
