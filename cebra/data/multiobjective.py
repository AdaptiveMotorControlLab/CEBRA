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

import literate_dataclasses as dataclasses
import numpy as np
import torch

import cebra.data as cebra_data
import cebra.distributions
from cebra.data.datatypes import BatchIndex
from cebra.distributions.continuous import Prior


@dataclasses.dataclass
class MultiObjectiveLoader(cebra_data.Loader):
    """Baseclass of Multiobjective Data Loader. Yields batches of the specified size from the given dataset object.
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
    """Supervised Multiobjective data Loader. Yields batches of the specified size from the given dataset object.
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

    def get_indices(self, num_samples: int, num_negatives: int = None):
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
    """Contrastive Multiobjective data Loader. Yields batches of the specified size from the given dataset object.
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
        elif config["distribution"] == "unified_time_delta":
            distribution = cebra.distributions.UnifiedSampler(
                self.dataset,
                kwargs_distribution['time_delta'],
                device=self.device,
            )
        else:
            raise NotImplementedError(
                f"Distribution {config['distribution']} is not implemented yet."
            )

        self.distributions.append(distribution)

    def get_indices(self, num_samples: int, num_negatives: int = None):
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


@dataclasses.dataclass
class UnifiedContrastiveMultiObjectiveLoader(cebra_data.Loader):
    """Unified Contrastive Multiobjective data Loader for multi-session training.
    
    Uses a single shared model across sessions with per-objective time_delta distributions.
    Reference and negative samples are shared across all objectives and sampled uniformly
    across sessions via UnifiedSampler. Positive samples are drawn per-objective using
    time_delta sampling with random shifts.

    The historical ``MultiObjective`` name is retained; the current unified xCEBRA
    workflow uses a single objective.

    Yields batches of the specified size from the given UnifiedDataset.
    """

    sampling_mode_contrastive: str = dataclasses.field(
        default="refneg_shared",
        doc="""Type of sampling performed, re whether reference and negative samples
            are shared. Only 'refneg_shared' is supported for unified multiobjective."""
    )

    def __post_init__(self):
        super().__post_init__()
        self.distributions = []

    def add_config(self, config):
        """Add the positive-sample distribution of one objective.

        Args:
            config: Dict with ``"distribution"``, which must be ``"unified_time_delta"``, and
                ``"kwargs"``. Note: ``kwargs["time_delta"]`` and ``kwargs["label_name"]`` are
                ignored; the sampler uses the loader's ``time_offset`` and the dataset's
                ``continuous_index``.
        """
        kwargs_distribution = config['kwargs']
        
        if config['distribution'] != "unified_time_delta":
            raise NotImplementedError(
                f"Only 'unified_time_delta' distribution is supported for unified multiobjective, "
                f"got '{config['distribution']}'."
            )
        
        if not hasattr(self.dataset, 'continuous_index'):
            raise ValueError(
                "UnifiedDataset must have 'continuous_index' attribute."
            )
        
        # Not handled: the sampler always uses the loader's ``time_offset``
        # (default 10), not ``kwargs['time_delta']``, and always samples on the dataset's primary
        # ``continuous_index``; per-objective labels (``label_name`` / ``add_labels``) are ignored.
        # Every current launcher passes time_delta=10, so the two agree.
        distribution = cebra.distributions.UnifiedSampler(
            self.dataset,
            self.time_offset,
            device=self.device,
        )
        self.distributions.append(distribution)

    def get_indices(self, num_samples: int):
        """Sample and return the specified number of indices for unified multiobjective.
        
        Reference and negative indices are shared across all objectives and sampled
        uniformly across sessions. Positive indices are drawn per-objective using
        time_delta distributions with random shifts.
        
        Args:
            num_samples: Batch size (number of samples per objective).
            
        Returns:
            A BatchIndex with:
            - reference: shape (num_sessions, num_samples) tensor shared across objectives
            - positive: list of per-objective positive indices, each (num_sessions, num_samples) tensor
            - negative: shape (num_sessions, num_samples) tensor shared across objectives
        """
        if self.sampling_mode_contrastive != "refneg_shared":
            raise ValueError(
                f"Sampling mode {self.sampling_mode_contrastive} is not implemented yet."
            )
        
        # Sample shared reference and negative using the first distribution
        # These come back as numpy arrays from UnifiedSampler
        ref_idx_np = self.distributions[0].sample_prior(self.batch_size)
        neg_idx_np = self.distributions[0].sample_prior(self.batch_size)
        ref_idx = torch.from_numpy(ref_idx_np).to(self.device)
        neg_idx = torch.from_numpy(neg_idx_np).to(self.device)

        # Sample per-objective positives
        positives_idx = []
        for distribution in self.distributions:
            pos_idx_np = distribution.sample_conditional(ref_idx_np)
            pos_idx = torch.from_numpy(pos_idx_np).to(self.device)
            positives_idx.append(pos_idx)

        return BatchIndex(
            reference=ref_idx,
            positive=positives_idx,
            negative=neg_idx,
        )

    def __iter__(self):
        for _ in range(len(self)):
            index = self.get_indices(num_samples=self.batch_size)
            yield self.dataset.load_batch_contrastive(index)
