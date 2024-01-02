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
"""Allen pseudomouse Neuropixels decoding dataset with train/test split.

References:
    *https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
    *Siegle, Joshua H., et al. "Survey of spiking in the mouse visual system reveals functional hierarchy." Nature 592.7852 (2021): 86-92.

"""
import glob
import hashlib
import os
import pathlib

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
from cebra.datasets import allen
from cebra.datasets import get_datapath
from cebra.datasets import parametrize
from cebra.datasets import register
from cebra.datasets.allen import ca_movie_decoding
from cebra.datasets.allen import NUM_NEURONS
from cebra.datasets.allen import SEEDS
from cebra.datasets.allen import SEEDS_DISJOINT

_DEFAULT_DATADIR = get_datapath()


@parametrize(
    "allen-movie-{num_movie}-neuropixel-{cortex}-{num_neurons}-{split_flag}-10-{seed}",
    num_movie=["one"],
    cortex=["VISp", "VISpm", "VISam", "VISrl", "VISal", "VISl"],
    num_neurons=NUM_NEURONS,
    split_flag=["train", "test"],
    seed=SEEDS,
)
class AllenNeuropixelMovieDecoding120HzCorticesDataset(
        ca_movie_decoding.AllenCaMoviesDataset,
        cebra.data.SingleSessionDataset):
    """A pseudomouse 120Hz Neuropixels dataset during the allen MOVIE1 stimulus with train/test split.

    A dataset of stacked 120HZ spike counts recorded in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    during the first 10 repeats of the MOVIE1 stimulus in Brain Observatory 1.1 set.
    The units which ISI > 0.5, amplitude < 0.1, presence ratio < 0.95 are excluded.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    The 10th repeat is held out as a test set and the remaining 9 repeats consist a train set.

    Args:
        cortext: The cortical area to sample the neurons from.
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        split_flag: The split to load. Choose between `train` and `test`.
        seed: The random seeds for sampling neurons.

    """

    def __init__(self, num_movie, cortex, num_neurons, split_flag, seed):
        super(ca_movie_decoding.AllenCaMoviesDataset, self).__init__()
        self.num_neurons = num_neurons
        self.seed = seed
        self.split_flag = split_flag
        frame_feature = self._get_video_features(num_movie)
        mice_data = self._get_pseudo_mice(cortex, num_movie)
        pseudo_mice = mice_data["neural"].T
        self.frames_index = mice_data["frames"]
        self.movie_len = int(pseudo_mice.shape[1])
        self.neurons_indices = self._sample_neurons(pseudo_mice)
        self._split(pseudo_mice, frame_feature)

    def _get_pseudo_mice(self, cortex: str, num_movie: str = "one"):
        """Load the pseudomice neuropixels data of the specified cortical area.

        Args:
            cortex: The visual cortical area.
        """
        path = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "allen" / "allen_movie1_neuropixel" / cortex / "neuropixel_pseudomouse_120_filtered.jl"
        data = joblib.load(path)
        return data

    def _split(self, pseudo_mice, frame_feature):
        """Split the dataset into train and test set.

        The first 9 repeats are the train set and the last repeat of the stimulu block is the test set.

        Args:
            pseudo_mice: The pseudomouse neural data.
            frame_feature: The frame feature used as the behavior label.

        """

        if self.split_flag == "train":
            self.neural = (torch.from_numpy(
                pseudo_mice[self.neurons_indices, :int(self.movie_len / 10 *
                                                       9)]).float().T)
            self.index = frame_feature[self.frames_index[:int(self.movie_len /
                                                              10 * 9)]]
            self.frames_index = self.frames_index[:int(self.movie_len / 10 * 9)]
        elif self.split_flag == "test":
            self.neural = (torch.from_numpy(
                pseudo_mice[self.neurons_indices,
                            int(self.movie_len / 10 * 9):]).float().T)
            self.index = frame_feature[self.frames_index[int(self.movie_len /
                                                             10 * 9):]]
            self.frames_index = self.frames_index[int(self.movie_len / 10 * 9):]


@parametrize(
    "allen-movie-one-neuropixel-{cortex}-disjoint-{group}-{num_neurons}-{split_flag}-10-{seed}",
    cortex=["VISp", "VISam", "VISrl", "VISal"],
    num_neurons=[400],
    split_flag=["train", "test"],
    seed=SEEDS_DISJOINT,
    group=[0, 1],
)
class AllenNeuropixelMovie120HzCorticesDisjointDataset(
        AllenNeuropixelMovieDecoding120HzCorticesDataset):
    """A disjoint pseudomouse 120Hz Neuropixels dataset of  during the allen MOVIE1 stimulus with train/test splits.

    A dataset of stacked 120Hz spike counts recorded in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    during the first 10 repeats of the MOVIE1 stimulus in Brain Observatory 1.1 set.
    The units which ISI > 0.5, amplitude < 0.1, presence ratio < 0.95 are excluded.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    The disjoint sets of neurons are configured. For example, for each seed, group1 and group2 (called by `group` parameter) are disjoint to each other.
    The 10th repeat is held-out as a test set and the remaining 9 repeats consists a train set.

    Args:
        cortex: The cortical area to sample the neurons from.
        split_flag: The split to load. Choose between `train` and `test`.
        seed: The random seeds for sampling neurons.
        group: The index of the group among disjoint sets of the sampled neurons.

    """

    def __init__(self,
                 group,
                 num_neurons,
                 seed=111,
                 cortex="VISp",
                 split_flag="train",
                 frame_feature_path=pathlib.Path(_DEFAULT_DATADIR) / "allen" /
                 "features" / "allen_movies" / "vit_base" / "8" /
                 "movie_one_image_stack.npz" / "testfeat.pth"):
        self.split_flag = split_flag
        self.seed = seed
        self.group = group
        self.num_neurons = num_neurons
        data = joblib.load(
            pathlib.Path(_DEFAULT_DATADIR) / "allen" /
            "allen_movie1_neuropixel" / cortex /
            "neuropixel_pseudomouse_120_filtered.jl")
        pseudo_mice = data["neural"].T
        self.neurons_indices = self._sample_neurons(pseudo_mice)
        self.movie_len = pseudo_mice.shape[1]
        frame_feature = torch.load(frame_feature_path)
        self.frames_index = data["frames"]
        self._split(pseudo_mice, frame_feature)

    def _sample_neurons(self, pseudo_mice):
        """Randomly sample disjoint neurons.

        The sampled two groups of 400 neurons are non-overlapping.

        Args:
            pseudo_mice: The pseudomouse dataset.

        """

        sampler = Generator(PCG64(self.seed))
        permuted_neurons = sampler.permutation(pseudo_mice.shape[0])
        return np.array_split(permuted_neurons,
                              2)[self.group][:self.num_neurons]
