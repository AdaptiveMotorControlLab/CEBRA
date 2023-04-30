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
"""Datasets and loaders for single session training.

All dataloaders should be implemented using ``dataclasses`` for handling
arguments and configuration values and subclass :py:class:`.base.Loader`.
"""

import abc
import collections
from typing import List

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import Batch
from cebra.data.datatypes import BatchIndex

__all__ = [
    "SingleSessionDataset",
    "DiscreteDataLoader",
    "ContinuousDataLoader",
    "MixedDataLoader",
    "HybridDataLoader",
    "FullDataLoader",
]


class SingleSessionDataset(cebra_data.Dataset):
    """A dataset with data from a single experimental session.

    A single experimental session contains a single data matrix with shape
    ``num_timesteps x dimension``, potentially paired with auxiliary information
    of shape ``num_timesteps x aux_dimension``.

    Loaders for single session datasets can be found in
    :py:mod:`cebra.data.single_session`.
    """

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError

    def load_batch(self, index: BatchIndex) -> Batch:
        """Return the data at the specified index location."""
        return Batch(
            positive=self[index.positive],
            negative=self[index.negative],
            reference=self[index.reference],
        )


@dataclasses.dataclass
class DiscreteDataLoader(cebra_data.Loader):
    """Supervised contrastive learning on fully discrete dataset.

    Reference and negative samples will be drawn from a uniform prior
    distribution. Depending on the ``prior`` attribute, the prior
    will uniform over time-steps (setting ``empirical``), or be adjusted
    such that each discrete value in the dataset is uniformly distributed
    (setting ``uniform``).

    The positive samples will have a matching discrete auxiliary variable
    as the reference samples.

    Sampling is implemented in the
    :py:class:`cebra.distributions.discrete.DiscreteUniform`
    and
    :py:class:`cebra.distributions.discrete.DiscreteEmpirical`
    distributions.

    Args:
        See dataclass fields.
    """

    prior: str = dataclasses.field(
        default="empirical",
        doc="""Re-sampling mode for the discrete index.

    The option `empirical` uses label frequencies as they appear in the dataset.
    The option `uniform` re-samples the dataset and adjust the frequencies of less
    common class labels.
    For balanced datasets, it is typically more accurate to stick to the `empirical`
    option.
    """,
    )

    @property
    def index(self):
        """The (discrete) dataset index."""
        return self.dataset.discrete_index

    def __post_init__(self):
        super().__post_init__()
        if self.dataset.discrete_index is None:
            raise ValueError("Dataset does not provide a discrete index.")
        self._init_distribution()

    def _init_distribution(self):
        if self.prior == "uniform":
            self.distribution = cebra.distributions.discrete.DiscreteUniform(
                self.index)
        elif self.prior == "empirical":
            self.distribution = cebra.distributions.discrete.DiscreteEmpirical(
                self.index)
        else:
            raise ValueError(
                f"Invalid choice of prior distribution. Got '{self.prior}', but "
                f"only accept 'uniform' or 'empirical' as potential values.")

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference samples will be sampled from the empirical or uniform prior
        distribution (if uniform, the discrete index values will be used to perform
        histogram normalization).

        The positive samples will be sampled such that their discrete index value
        corresponds to the respective value of the reference samples.

        The negative samples will be sampled from the same distribution as the
        reference examples.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.
        """
        reference_idx = self.distribution.sample_prior(num_samples * 2)
        negative_idx = reference_idx[num_samples:]
        reference_idx = reference_idx[:num_samples]
        reference = self.index[reference_idx]
        positive_idx = self.distribution.sample_conditional(reference)
        return BatchIndex(reference=reference_idx,
                          positive=positive_idx,
                          negative=negative_idx)


@dataclasses.dataclass
class ContinuousDataLoader(cebra_data.Loader):
    """Contrastive learning conditioned on a continuous behavior variable.

    Reference and negative samples will be drawn from a uniform prior
    distribution across all time-steps. The positive sample will be distributed
    around the reference example using either

    * time information (``time``): In this case, a :py:class:`cebra.distributions.continuous.TimeContrastive`
      distribution is used for sampling. Positive pairs will have a fixed ``time_offset``
      from the reference samples' time steps.
    * auxiliary variables, using the empirical distribution of how behavior various across
      ``time_offset`` timesteps (``time_delta``). Sampling for this setting is implemented
      in :py:class:`cebra.distributions.continuous.TimedeltaDistribution`.
    * alternatively, the distribution can be selected to be a Gaussian distribution parametrized
      by a fixed ``delta`` around the reference sample, using the implementation in
      :py:class:`cebra.distributions.continuous.DeltaDistribution`.

    Args:
        See dataclass fields.
    """

    conditional: str = dataclasses.field(
        default="time_delta",
        doc="""Information on how the positive samples should be acquired.
    Setting to ``time_delta`` computes the differences between adjacent samples
    in the dataset, and uses ``reference + diff`` as the query for collecting the
    positive pair. Setting to ``time`` will use adjacent pairs of samples only
    and become equivalent to time contrastive learning.
    """,
    )
    time_offset: int = dataclasses.field(default=10)
    delta: float = dataclasses.field(default=0.1)

    def __post_init__(self):
        # TODO(stes): Based on how to best handle larger scale datasets, copying the tensors
        #            here might be sub-optimal. The final behavior should be determined after
        #            e.g. integrating the FAISS dataloader back in.
        super().__post_init__()
        self._init_distribution()

    def _init_distribution(self):
        if self.conditional == "time":
            self.distribution = cebra.distributions.TimeContrastive(
                time_offset=self.time_offset,
                num_samples=len(self.dataset.neural),
                device=self.device,
            )
        else:
            if self.dataset.continuous_index is None:
                raise ValueError(
                    f"Dataset {self.dataset} does not provide a continuous index."
                )
            if self.conditional == "time_delta":
                self.distribution = cebra.distributions.TimedeltaDistribution(
                    self.dataset.continuous_index,
                    self.time_offset,
                    device=self.device)
            elif self.conditional == "delta":
                self.distribution = cebra.distributions.DeltaDistribution(
                    self.dataset.continuous_index,
                    self.delta,
                    device=self.device)
            else:
                raise ValueError(self.conditional)

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference and negative samples will be sampled uniformly from
        all available time steps.

        The positive samples will be sampled conditional on the reference
        samples according to the specified ``conditional`` distribution.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.
        """
        reference_idx = self.distribution.sample_prior(num_samples * 2)
        negative_idx = reference_idx[num_samples:]
        reference_idx = reference_idx[:num_samples]
        positive_idx = self.distribution.sample_conditional(reference_idx)
        return BatchIndex(reference=reference_idx,
                          positive=positive_idx,
                          negative=negative_idx)


@dataclasses.dataclass
class MixedDataLoader(cebra_data.Loader):
    """Mixed discrete-continuous data loader.

    This data loader combines the functionality of
    :py:class:`DiscreteDataLoader` and :py:class:`ContinuousDataLoader`
    for datasets that provide both a continuous and discrete variables.

    Sampling can be configured in different modes:

    1. Positive pairs always share their discrete variable.
    2. Positive pairs are drawn only based on their conditional,
       not discrete variable.
    """

    conditional: str = dataclasses.field(default="time_delta")
    time_offset: int = dataclasses.field(default=10)

    @property
    def dindex(self):
        # TODO(stes) rename to discrete_index
        return self.dataset.discrete_index

    @property
    def cindex(self):
        # TODO(stes) rename to continuous_index
        return self.dataset.continuous_index

    def __post_init__(self):
        super().__post_init__()
        self.distribution = cebra.distributions.MixedTimeDeltaDistribution(
            discrete=self.dindex,
            continuous=self.cindex,
            time_delta=self.time_offset)

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference and negative samples will be sampled uniformly from
        all available time steps.

        The positive distribution will either share the discrete value of
        the reference samples, and then sampled as in the
        :py:class:`ContinuousDataLoader`, or just sampled based on the
        conditional variable.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.

        Todo:
            - Add the ``empirical`` vs. ``discrete`` sampling modes to this
              class.
            - Sample the negatives with matching discrete variable
        """
        reference_idx = self.distribution.sample_prior(num_samples)
        return BatchIndex(
            reference=reference_idx,
            negative=self.distribution.sample_prior(num_samples),
            positive=self.distribution.sample_conditional(reference_idx),
        )


@dataclasses.dataclass
class HybridDataLoader(cebra_data.Loader):
    """Contrastive learning using both time and behavior information.

    The dataloader combines two training modes implemented in
    :py:class:`ContinuousDataLoader` and combines time and behavior information
    into a joint embedding.

    Args:
        See dataclass fields.
    """

    conditional: str = dataclasses.field(default="time_delta")
    time_offset: int = dataclasses.field(default=10)
    delta: float = dataclasses.field(default=0.1)

    @property
    def index(self):
        """The (continuous) dataset index."""
        if self.dataset.continuous_index is not None:
            return self.dataset.continuous_index
        else:
            raise ValueError("No continuous variable exist")

    def __post_init__(self):
        # TODO(stes): Based on how to best handle larger scale datasets, copying the tensors
        #            here might be sub-optimal. The final behavior should be determined after
        #            e.g. integrating the FAISS dataloader back in.
        super().__post_init__()
        index = self.index.to(self.device)

        if self.conditional != "time_delta":
            raise NotImplementedError(
                f"Hybrid training is currently only implemented using the ``time_delta`` "
                f"continual distribution.")

        self.time_distribution = cebra.distributions.TimeContrastive(
            time_offset=self.time_offset,
            num_samples=len(self.dataset.neural),
            device=self.device,
        )
        self.behavior_distribution = cebra.distributions.TimedeltaDistribution(
            self.dataset.continuous_index, self.time_offset, device=self.device)

    def get_indices(self, num_samples: int) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference and negative samples will be sampled uniformly from
        all available time steps, and a total of ``2*num_samples`` will be
        returned for both.

        For the positive samples, ``num_samples`` are sampled according to the
        behavior conditional distribution, and another ``num_samples`` are
        sampled according to the dime contrastive distribution. The indices
        for the positive samples are concatenated across the first dimension.

        Args:
            num_samples: The number of samples (batch size) of the returned
                :py:class:`cebra.data.datatypes.BatchIndex`.

        Returns:
            Indices for reference, positive and negatives samples.

        Todo:
            Add the ``empirical`` vs. ``discrete`` sampling modes to this
            class.
        """
        reference_idx = self.time_distribution.sample_prior(num_samples * 2)
        negative_idx = reference_idx[num_samples:]
        reference_idx = reference_idx[:num_samples]
        behavior_positive_idx = self.behavior_distribution.sample_conditional(
            reference_idx)
        time_positive_idx = self.time_distribution.sample_conditional(
            reference_idx)
        return BatchIndex(
            reference=reference_idx,
            positive=torch.cat([behavior_positive_idx, time_positive_idx]),
            negative=negative_idx,
        )


@dataclasses.dataclass
class FullDataLoader(ContinuousDataLoader):
    """Data loader for batch gradient descent, loading the whole dataset at once."""

    def __post_init__(self):
        super().__post_init__()
        self.batch_size = None

    @property
    def offset(self):
        return self.dataset.offset

    def get_indices(self, num_samples=None) -> BatchIndex:
        """Samples indices for reference, positive and negative examples.

        The reference indices are all available (valid, according to the
        model's offset) indices in the dataset, in order.

        The negative indices are a permutation of the reference indices.

        The positive indices are sampled as before from the conditional
        distribution, given the reference samples.

        Returns:
            Indices for reference, positive and negatives samples. The
            batch size will be equal to the dataset size, lesser the
            length of the model index.

        Todo:
            Add the ``empirical`` vs. ``discrete`` sampling modes to this
            class.
        """
        assert num_samples is None

        reference_idx = torch.arange(
            self.offset.left,
            len(self.dataset.neural) - len(self.dataset.offset) - 1,
            device=self.device,
        )
        negative_idx = reference_idx[torch.randperm(len(reference_idx))]
        positive_idx = self.distribution.sample_conditional(reference_idx)

        return cebra.data.BatchIndex(reference=reference_idx,
                                     positive=positive_idx,
                                     negative=negative_idx)

    def __iter__(self):
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield index
