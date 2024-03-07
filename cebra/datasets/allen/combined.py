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
"""Joint Allen pseudomouse Ca/Neuropixel datasets.

References:
    *Deitch, Daniel, Alon Rubin, and Yaniv Ziv. "Representational drift in the mouse visual cortex." Current biology 31.19 (2021): 4327-4339.
    *de Vries, Saskia EJ, et al. "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." Nature neuroscience 23.1 (2020): 138-151.
    *https://github.com/zivlab/visual_drift
    *http://observatory.brain-map.org/visualcoding
    *https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
    *Siegle, Joshua H., et al. "Survey of spiking in the mouse visual system reveals functional hierarchy." Nature 592.7852 (2021): 86-92.

"""

import glob
import hashlib

import h5py
import joblib
import numpy as np
import pandas as pd
import scipy.io
import torch
from numpy.random import Generator
from numpy.random import PCG64
from sklearn.decomposition import PCA

import cebra.data
from cebra.datasets import parametrize
from cebra.datasets import register
from cebra.datasets.allen import ca_movie
from cebra.datasets.allen import ca_movie_decoding
from cebra.datasets.allen import neuropixel_movie
from cebra.datasets.allen import neuropixel_movie_decoding
from cebra.datasets.allen import NUM_NEURONS
from cebra.datasets.allen import SEEDS
from cebra.datasets.allen import SEEDS_DISJOINT


@parametrize(
    "allen-movie1-ca-neuropixel-10ms-{num_neurons}-{seed}",
    num_neurons=NUM_NEURONS,
    seed=SEEDS,
)
class AllenMovieDataset(cebra.data.DatasetCollection):
    """A joint pseudomouse dataset of 30Hz calcium events and 120 Hz Neuropixels recording during allen Movie1 stimulus.

    It loads instances of AllenCaMovieDataset and AllenNeuropixelMovie120HzDataset for the VISp.

    Args:
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        seed: The random seeds for sampling neurons.

    """

    def __init__(self, num_neurons=1000, seed=111, area="VISp"):
        super().__init__(
            ca_movie.AllenCaMovieDataset(num_neurons, seed, area),
            neuropixel_movie.AllenNeuropixelMovie120HzDataset(
                num_neurons, seed, area),
        )

    def __repr__(self):
        return f"CaNeuropixelDataset"


@parametrize(
    "allen-movie-one-ca-neuropixel-{cortex}-{num_neurons}-{split_flag}-10-{seed}",
    cortex=["VISp", "VISpm", "VISam", "VISrl", "VISal", "VISl"],
    num_neurons=NUM_NEURONS,
    split_flag=["train", "test"],
    seed=SEEDS,
)
class AllenMovieOneCaNPCortexDataset(cebra.data.DatasetCollection):
    """A joint pseudomouse dataset of 30Hz calcium events and 120 Hz Neuropixels recording during allen Movie1 stimulus with train/test split.

    It loads instances of AllenCaMoviesDataset and AllenNeuropixelMovie120HzCorticesDataset.

    Args:
        cortex: The cortical area to sample the neurons from.
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        split_flag: The split to load. Choose between `train` and `test`.
        seed: The random seeds for sampling neurons.

    """

    def __init__(self,
                 num_neurons=1000,
                 seed=111,
                 cortex="VISp",
                 split_flag="train"):
        super().__init__(
            ca_movie_decoding.AllenCaMoviesDataset("one", cortex, num_neurons,
                                                   split_flag, seed, 10),
            neuropixel_movie_decoding.
            AllenNeuropixelMovieDecoding120HzCorticesDataset(
                "one", cortex, num_neurons, split_flag, seed),
        )

    def __repr__(self):
        return f"CaNeuropixelMovieOneCorticesDataset"


@parametrize(
    "allen-movie-one-ca-neuropixel-{cortex}-disjoint-{group}-{num_neurons}-{split_flag}-10-{seed}",
    cortex=["VISp", "VISam", "VISrl", "VISal"],
    num_neurons=NUM_NEURONS,
    split_flag=["train", "test"],
    group=[0, 1],
    seed=SEEDS_DISJOINT,
)
class AllenMovieOneCaNPCortexDisjointDataset(cebra.data.DatasetCollection):
    """A joint pseudomouse dataset of 30Hz calcium events and 120 Hz Neuropixels recording during allen Movie1 stimulus.

    It loads instances of AllenCaMoviesDisjointDataset and AllenNeuropixelMovie120HzCorticesDisjointDataset.

    Args:
        cortex: The cortical area to sample the neurons from.
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        split_flag: The split to load. Choose between `train` and `test`.
        group: The index of the group among disjoint sets of the sampled neurons.
        seed: The random seeds for sampling neurons.

    """

    def __init__(self, group, num_neurons, seed, cortex, split_flag="train"):
        super().__init__(
            ca_movie_decoding.AllenCaMoviesDisjointDataset(
                "one", cortex, group, num_neurons, split_flag, seed, 10),
            neuropixel_movie_decoding.
            AllenNeuropixelMovie120HzCorticesDisjointDataset(
                group, num_neurons, seed, cortex, split_flag),
        )

    def __repr__(self):
        return f"CaNeuropixelMovieOneCorticesDisjointDataset"
