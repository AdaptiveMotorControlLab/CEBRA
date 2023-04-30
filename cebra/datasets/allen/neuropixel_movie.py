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
"""Allen pseudomouse Neuropixels decoding dataset.

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
from cebra.datasets import get_datapath
from cebra.datasets import parametrize
from cebra.datasets import register
from cebra.datasets.allen import ca_movie
from cebra.datasets.allen import NUM_NEURONS
from cebra.datasets.allen import SEEDS


@parametrize(
    "allen-movie1-neuropixel-{num_neurons}-{seed}-10ms",
    num_neurons=NUM_NEURONS,
    seed=SEEDS,
)
class AllenNeuropixelMovie120HzDataset(ca_movie.AllenCaMovieDataset):
    """A pseudomouse 120Hz Neuropixels dataset during the allen MOVIE1 stimulus.

    A dataset of stacked 120HZ spike counts recorded in the primary visual cortex of multiple mice
    during the first 10 repeats of the MOVIE1 stimulus in Brain Observatory 1.1 set.
    The units which ISI > 0.5, amplitude < 0.1, presence ratio < 0.95 are excluded.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.

    Args:
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        seed: The random seeds for sampling neurons.
        frame_feature_path: The path of the movie frame features.

    """

    def _get_pseudo_mice(self, area="VISp"):
        """Construct pseudomouse neural dataset.

        Stack the excitatory neurons from the multiple mice and construct a psuedomouse neural dataset of the specified visual cortical area.
        The neurons which were recorded in all of the sessions A, B, C are included.

        Args:
            area: The visual cortical area to sample the neurons. Possible options: VISp, VISpm, VISam, VISal, VISl, VISrl.

        """
        self.area = area
        list_recording = joblib.load(
            get_datapath(
                f"allen/allen_movie1_neuropixel/{area}/neuropixel_pseudomouse_120_filtered.jl"
            ))
        pseudo_mice = list_recording["neural"]

        return pseudo_mice.transpose(1, 0)

    def _get_index(self, frame_feature):
        """Return behavior labels.

        Construct the behavior labels with the user-defined frame feature.

        Args:
            frame feature: The video frame feature.

        """

        list_recording = joblib.load(
            get_datapath(
                f"allen/allen_movie1_neuropixel/{self.area}/neuropixel_pseudomouse_120_filtered.jl"
            ))
        frames_index = list_recording["frames"]
        return frame_feature[frames_index]
