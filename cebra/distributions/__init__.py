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
"""Distributions and indexing helper functions for training CEBRA models.

This package contains classes for sampling and indexing of datasets.
Typically, the functionality of
classes in this module is guided by the auxiliary variables of CEBRA. A dataset would pass auxiliary
variables to a sampler, and within the sampler the *indices* of reference, negative and positive
samples will be sampled based on the auxiliary information. Custom ways of sampling should therefore
be implemented in this package. Functionality in this package is fully agnostic to the actual signal
to be analysed, and only considers the auxiliary information of a dataset (called "index").


Distributions take data samples and allow to sample or re-sample from the
dataset. Sampling from the prior distribution is done via "sample_prior",
sampling from the conditional distribution via "sample_conditional".

For fast lookups in datasets, indexing classes provide 1-nearest-neighbor
searches with L2 and cosine similarity metrics (recommended on GPU) or using
standard multi-threaded dataloading with FAISS as the backend for retrieving
data.
"""

from cebra.distributions.base import *
from cebra.distributions.continuous import *
from cebra.distributions.discrete import *
from cebra.distributions.index import *
from cebra.distributions.mixed import *
from cebra.distributions.multisession import *

__all__ = [
    "Index",
    "Offset",
    "DistanceMatrix",
    "OffsetDistanceMatrix",
    "ConditionalIndex",
    "MultiSessionIndex",
    "Prior",
    "TimeContrastive",
    "TimedeltaDistribution",
    "MultiSessionTimeDelta",
    "Discrete",
    "DiscreteUniform",
    "DiscreteEmpirical",
    "DeltaNormalDistribution",
    "MultivariateDiscrete",
    "MultisessionSampler",
]
