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
    "MultivariateDiscrete",
    "MultisessionSampler",
]
