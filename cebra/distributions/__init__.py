"""Distributions and indexing helper functions for training CEBRA models.

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

__all__ = [
]
