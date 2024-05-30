#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
from typing import List

import literate_dataclasses as dataclasses

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import BatchIndex
from cebra.distributions.continuous import Prior


@dataclasses.dataclass
class MultiObjectiveLoader(cebra_data.Loader):
    """Baseclass of RegCL Data Loader. Yields batches of the specified size from the given dataset object.
    """
    dataset: int = dataclasses.field(
        default=None,
        doc="""A dataset instance specifying a ``__getitem__`` function.""",
    )
    num_steps: int = dataclasses.field(default=None)
    batch_size: int = dataclasses.field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.batch_size > len(self.dataset.neural):
            raise ValueError("Batch size can't be larger than data.")
        self.prior = Prior(self.dataset.neural, device=self.device)

    def get_indices(self):
        return NotImplementedError

    def __iter__(self):
        return NotImplementedError

    def add_config(self, config):
        raise NotImplementedError


@dataclasses.dataclass
class SupervisedMultiObjectiveLoader(MultiObjectiveLoader):
    """Supervised RegCL data Loader. Yields batches of the specified size from the given dataset object.
    """
    sampling_mode_supervised: str = dataclasses.field(
        default="ref_shared",
        doc="""Type of sampling performed, re whether reference are shared or not.
                 are shared. Options will be ref_shared, independent.""")

    def __post_init__(self):
        super().__post_init__()
        self.labels = []

    def add_config(self, config):
        self.labels.append(config['label'])

    def get_indices(self, num_samples: int):
        if self.sampling_mode_supervised == "ref_shared":
            reference_idx = self.prior.sample_prior(num_samples)
        else:
            raise ValueError(
                f"Sampling mode {self.sampling_mode_supervised} is not implemented."
            )

        batch_index = BatchIndex(
            reference=reference_idx,
            positive=None,
            negative=None,
        )

        return batch_index

    def __iter__(self):
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield self.dataset.load_batch_supervised(index, self.labels)


@dataclasses.dataclass
class ContrastiveMultiObjectiveLoader(MultiObjectiveLoader):
    """Contrastive RegCL data Loader. Yields batches of the specified size from the given dataset object.
    """

    sampling_mode_contrastive: str = dataclasses.field(
        default="refneg_shared",
        doc=
        """Type of sampling performed, re whether reference and negative samples
            are shared. Options will be ref_shared, neg_shared and refneg_shared"""
    )

    def __post_init__(self):
        super().__post_init__()
        self.distributions = []

    def add_config(self, config):
        kwargs_distribution = config['kwargs']
        if config['distribution'] == "time":
            distribution = cebra.distributions.TimeContrastive(
                time_offset=kwargs_distribution['time_offset'],
                num_samples=len(self.dataset.neural),
                device=self.device,
            )
        elif config['distribution'] == "time_delta":
            distribution = cebra.distributions.TimedeltaDistribution(
                continuous=self.dataset.labels[
                    kwargs_distribution['label_name']],
                time_delta=kwargs_distribution['time_delta'],
                device=self.device)
        elif config['distribution'] == "delta_normal":
            distribution = cebra.distributions.DeltaNormalDistribution(
                continuous=self.dataset.labels[
                    kwargs_distribution['label_name']],
                delta=kwargs_distribution['delta'],
                device=self.device)
        elif config['distribution'] == "delta_vmf":
            distribution = cebra.distributions.DeltaVMFDistribution(
                continuous=self.dataset.labels[
                    kwargs_distribution['label_name']],
                delta=kwargs_distribution['delta'],
                device=self.device)
        else:
            raise NotImplementedError(
                f"Distribution {config['distribution']} is not implemented yet."
            )

        self.distributions.append(distribution)

    def get_indices(self, num_samples: int):
        """Sample and return the specified number of indices."""

        if self.sampling_mode_contrastive == "refneg_shared":
            ref_and_neg = self.prior.sample_prior(num_samples * 2)
            reference_idx = ref_and_neg[:num_samples]
            negative_idx = ref_and_neg[num_samples:]

            positives_idx = []
            for distribution in self.distributions:
                idx = distribution.sample_conditional(reference_idx)
                positives_idx.append(idx)

            batch_index = BatchIndex(
                reference=reference_idx,
                positive=positives_idx,
                negative=negative_idx,
            )
        else:
            raise ValueError(
                f"Sampling mode {self.sampling_mode_contrastive} is not implemented yet."
            )

        return batch_index

    def __iter__(self):
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield self.dataset.load_batch_contrastive(index)
