"""Allen pseudomouse Neuropixels decoding dataset with train/test split.

References:
    *https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
    *Siegle, Joshua H., et al. "Survey of spiking in the mouse visual system reveals functional hierarchy." Nature 592.7852 (2021): 86-92.

"""
import glob
import hashlib
import os

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


@parametrize(
    "allen-movie-{num_movie}-neuropixel-{cortex}-{num_neurons}-{split_flag}-10-{seed}",
    num_neurons=NUM_NEURONS,
    split_flag=["train", "test"],
class AllenNeuropixelMovieDecoding120HzCorticesDataset(
        ca_movie_decoding.AllenCaMoviesDataset,
        cebra.data.SingleSessionDataset):
    """A pseudomouse 120Hz Neuropixels dataset during the allen MOVIE1 stimulus with train/test split.
    A dataset of stacked 120HZ spike counts recorded in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    The units which ISI > 0.5, amplitude < 0.1, presence ratio < 0.95 are excluded.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    Args:
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
        self.movie_len = int(pseudo_mice.shape[1])
        self.neurons_indices = self._sample_neurons(pseudo_mice)
        self._split(pseudo_mice, frame_feature)

        """Load the pseudomice neuropixels data of the specified cortical area.
        Args:
            cortex: The visual cortical area.
        """

        data = joblib.load(
            get_datapath(
            ))
        return data

    def _split(self, pseudo_mice, frame_feature):

        The first 9 repeats are the train set and the last repeat of the stimulu block is the test set.
        Args:
            pseudo_mice: The pseudomouse neural data.
            frame_feature: The frame feature used as the behavior label.
        """

                pseudo_mice[self.neurons_indices, :int(self.movie_len / 10 *
            self.index = frame_feature[self.frames_index[:int(self.movie_len /
                                                              10 * 9)]]
            self.frames_index = self.frames_index[:int(self.movie_len / 10 * 9)]
            self.index = frame_feature[self.frames_index[int(self.movie_len /
                                                             10 * 9):]]
            self.frames_index = self.frames_index[int(self.movie_len / 10 * 9):]


@parametrize(
    "allen-movie-one-neuropixel-{cortex}-disjoint-{group}-{num_neurons}-{split_flag}-10-{seed}",
    num_neurons=[400],
    split_flag=["train", "test"],
    seed=SEEDS_DISJOINT,
class AllenNeuropixelMovie120HzCorticesDisjointDataset(
        AllenNeuropixelMovieDecoding120HzCorticesDataset):
    """A disjoint pseudomouse 120Hz Neuropixels dataset of  during the allen MOVIE1 stimulus with train/test splits.
    A dataset of stacked 120Hz spike counts recorded in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    The units which ISI > 0.5, amplitude < 0.1, presence ratio < 0.95 are excluded.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    Args:
        seed: The random seeds for sampling neurons.
    """

    def __init__(
        self,
        group,
        num_neurons,
        seed=111,
        frame_feature_path=get_datapath(
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        self.split_flag = split_flag
        self.seed = seed
        self.group = group
        self.num_neurons = num_neurons
        data = joblib.load(
            get_datapath(
            ))
        self.neurons_indices = self._sample_neurons(pseudo_mice)
        self.movie_len = pseudo_mice.shape[1]
        frame_feature = torch.load(frame_feature_path)
        self._split(pseudo_mice, frame_feature)

    def _sample_neurons(self, pseudo_mice):
        The sampled two groups of 400 neurons are non-overlapping.
        Args:
            pseudo_mice: The pseudomouse dataset.
        """

        sampler = Generator(PCG64(self.seed))
        permuted_neurons = sampler.permutation(pseudo_mice.shape[0])
        return np.array_split(permuted_neurons,
                              2)[self.group][:self.num_neurons]
