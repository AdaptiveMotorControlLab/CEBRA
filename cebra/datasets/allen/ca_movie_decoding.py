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
"""Allen pseudomouse Ca decoding dataset with train/test split.

References:
    *Deitch, Daniel, Alon Rubin, and Yaniv Ziv. "Representational drift in the mouse visual cortex." Current biology 31.19 (2021): 4327-4339.
    *de Vries, Saskia EJ, et al. "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." Nature neuroscience 23.1 (2020): 138-151.
    *https://github.com/zivlab/visual_drift
    *http://observatory.brain-map.org/visualcoding

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
from cebra.datasets import get_datapath
from cebra.datasets import parametrize
from cebra.datasets import register
from cebra.datasets.allen import NUM_NEURONS
from cebra.datasets.allen import SEEDS
from cebra.datasets.allen import SEEDS_DISJOINT

_DEFAULT_DATADIR = get_datapath()


@parametrize(
    "allen-movie-{num_movie}-ca-{cortex}-{num_neurons}-{split_flag}-{test_repeat}-{seed}",
    num_movie=["one"],
    cortex=["VISp", "VISpm", "VISam", "VISrl", "VISal", "VISl"],
    num_neurons=NUM_NEURONS,
    split_flag=["train", "test"],
    test_repeat=[10],
    seed=SEEDS,
)
class AllenCaMoviesDataset(cebra.data.SingleSessionDataset):
    """A pseudomouse 30Hz calcium events dataset during the allen MOVIE1 stimulus with train/test splits.

    A dataset of stacked 30Hz calcium events from the excitatory neurons in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    recorded during the 10 repeats of the MOVIE1 stimulus in session A,B and C. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    The 10th repeat is held-out as a test set and the remaining 9 repeats consists a train set.

    Args:
        cortex: The cortical area to sample the neurons from.
        num_neurons: The number of neurons to sample. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        split_flag: The split to load. Choose between `train` and `test`.
        seed: The random seeds for sampling neurons.
        preload: The path to the preloaded neural data. If `None`, the neural data is constructed from the source. Default value is `None`.

    """

    def __init__(
        self,
        num_movie,
        cortex,
        num_neurons,
        split_flag,
        seed,
        test_repeat,
        preload=None,
    ):
        super().__init__()
        self.num_neurons = num_neurons
        self.seed = seed
        self.split_flag = split_flag
        self.test_repeat = test_repeat
        frame_feature = self._get_video_features(num_movie)
        if preload is None:
            pseudo_mice = self._get_pseudo_mice(cortex, num_movie)
            self.movie_len = int(pseudo_mice.shape[1] / 10)
            self.neurons_indices = self._sample_neurons(pseudo_mice)
            self._split(pseudo_mice, frame_feature)
        else:
            data = joblib.load(preload)
            self.neural = data["neural"]
            if split_flag == "train":
                self.index = frame_feature.repeat(9, 1)
            else:
                self.index = frame_feature.repeat(1, 1)

    def _get_video_features(self, num_movie="one"):
        """Return behavior labels.

        The frame feature used as the behavior labels are returned.

        Args:
            num_movie: The number of the moive used as the stimulus. It is fixed to 'one'.

        """

        frame_feature_path = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "allen" / "features" / "allen_movies" / "vit_base" / "8" / f"movie_{num_movie}_image_stack.npz" / "testfeat.pth"
        frame_feature = torch.load(frame_feature_path)
        return frame_feature

    def _sample_neurons(self, pseudo_mice):
        """Randomly sample the specified number of neurons.

        The random sampling of the neurons specified by the `seed` and `num_neurons`.

        Args:
            pseudo_mice: The pseudomouse data.

        """

        sampler = Generator(PCG64(self.seed))
        neurons_indices = sampler.choice(np.arange(pseudo_mice.shape[0]),
                                         size=self.num_neurons)
        return neurons_indices

    def _split(self, pseudo_mice, frame_feature):
        """Split the dataset into train and test set.
        The first 9 repeats are train set and the last repeat is test set.

        Args:
            pseudo_mice: The pseudomouse neural data.
            frame_feature: The behavior labels.

        """

        if self.split_flag == "train":
            neural = np.delete(
                pseudo_mice[self.neurons_indices],
                np.arange(
                    (self.test_repeat - 1) * self.movie_len,
                    self.test_repeat * self.movie_len,
                ),
                axis=1,
            )
            self.index = frame_feature.repeat(9, 1)
        elif self.split_flag == "test":
            neural = pseudo_mice[
                self.neurons_indices,
                (self.test_repeat - 1) * self.movie_len:self.test_repeat *
                self.movie_len,
            ]
            self.index = frame_feature.repeat(1, 1)
        else:
            raise ValueError("split_flag should be either train or test")

        self.neural = torch.from_numpy(neural.T).float()

    def _get_pseudo_mice(self, area, num_movie):
        """Construct pseudomouse neural dataset.

        Stack the excitatory neurons from the multiple mice and construct a psuedomouse neural dataset of the specified visual cortical area.
        The neurons which were recorded in all of the sessions A, B, C are included.

        Args:
            area: The visual cortical area to sample the neurons. Possible options: VISp, VISpm, VISam, VISal, VISl, VISrl.

        """

        path = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "allen" / "visual_drift" / "data" / "calcium_excitatory" / str(area)
        list_mice = path.glob("*.mat")
        exp_containers = [int(file.stem) for file in list_mice]
        ## Load summary file
        summary = pd.read_csv(
            pathlib.Path(_DEFAULT_DATADIR) / "allen" / "data_summary.csv")
        ## Filter excitatory neurons in V1
        area_filtered = summary[(summary["exp"].isin(exp_containers)) &
                                (summary["target"] == area) &
                                ~(summary["cre_line"].str.contains("SSt")) &
                                ~(summary["cre_line"].str.contains("Pvalb")) &
                                ~(summary["cre_line"].str.contains("Vip"))]

        def _convert_to_nums(string):
            return list(
                map(
                    int,
                    string.replace("\n", "").replace("[",
                                                     "").replace("]",
                                                                 "").split(),
                ))

        ## Pseudo V1
        pseudo_mouse = []
        for exp_container in set(area_filtered["exp"]):
            neurons = summary[summary["exp"] == exp_container]["neurons"]
            sessions = summary[summary["exp"] == exp_container]["session_type"]
            seq_sessions = np.array(list(sessions)).argsort()
            common_neurons = set.intersection(
                set(_convert_to_nums(neurons.iloc[0])),
                set(_convert_to_nums(neurons.iloc[1])),
                set(_convert_to_nums(neurons.iloc[2])),
            )
            indices1 = [
                _convert_to_nums(neurons.iloc[0]).index(x)
                for x in common_neurons
            ]
            indices2 = [
                _convert_to_nums(neurons.iloc[1]).index(x)
                for x in common_neurons
            ]
            indices3 = [
                _convert_to_nums(neurons.iloc[2]).index(x)
                for x in common_neurons
            ]
            indices1.sort()
            indices2.sort()
            indices3.sort()
            indices = [indices1, indices2, indices3]
            matfile = pathlib.Path(
                _DEFAULT_DATADIR
            ) / "allen" / "visual_drift" / "data" / "calcium_excitatory" / str(
                area) / f"{exp_container}.mat"
            traces = scipy.io.loadmat(matfile)
            for n, i in enumerate(seq_sessions):
                session = traces["filtered_traces_days_events"][n, 0][
                    indices[i], :]
                pseudo_mouse.append(session)

        pseudo_mouse = np.concatenate(pseudo_mouse)

        return pseudo_mouse

        pseudo_mouse = np.vstack(
            [get_neural_data(num_movie, mice) for mice in list_mice])

        return pseudo_mouse

    def __len__(self):
        return self.neural.size(0)

    @property
    def continuous_index(self):
        return self.index

    @property
    def input_dimension(self):
        return self.neural.size(1)

    def __getitem__(self, index):
        index = self.expand_index(index)

        return self.neural[index].transpose(2, 1)


@parametrize(
    "allen-movie-{num_movie}-ca-{cortex}-disjoint-{group}-{num_neurons}-{split_flag}-{test_repeat}-{seed}",
    num_movie=["one"],
    cortex=["VISp", "VISpm", "VISam", "VISrl", "VISal", "VISl"],
    num_neurons=[400],
    split_flag=["train", "test"],
    test_repeat=[10],
    seed=SEEDS_DISJOINT,
    group=[0, 1],
)
class AllenCaMoviesDisjointDataset(AllenCaMoviesDataset,
                                   cebra.data.SingleSessionDataset):
    """A disjoint pseudomouse 30Hz calcium events dataset of  during the allen MOVIE1 stimulus with train/test splits.

    A dataset of stacked 30Hz calcium events from the excitatory neurons in the visual cortices (VISp, VISpm, VISam, VISrl, VISal, VISl) of multiple mice
    recorded during the 10 repeats of the MOVIE1 stimulus in session A. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    The disjoint sets of 400 neurons are configured. For example, for each seed, group1 and group2 (called by `group` parameter) are disjoint to each other.
    The 10th repeat is held-out as a test set and the remaining 9 repeats consists a train set.

    Args:
        cortex: The cortical area to sample the neurons from.
        split_flag: The split to load. Choose between `train` and `test`.
        seed: The random seeds for sampling neurons.
        group: The index of the group among disjoint sets of the sampled neurons.

    """

    def __init__(self, num_movie, cortex, group, num_neurons, split_flag, seed,
                 test_repeat):
        super(AllenCaMoviesDataset, self).__init__()
        self.num_neurons = num_neurons
        self.seed = seed
        self.split_flag = split_flag
        self.test_repeat = test_repeat
        self.group = group
        frame_feature = self._get_video_features(num_movie)
        pseudo_mice = self._get_pseudo_mice(cortex, num_movie)
        self.movie_len = int(pseudo_mice.shape[1] / 10)
        self.neurons_indices = self._sample_neurons(pseudo_mice)
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

    def _get_pseudo_mice(self, area, num_movie):
        """Construct pseudomouse neural dataset.

        Stack the excitatory neurons from the multiple mice and construct a psuedomouse neural dataset of the specified visual cortical area.
        The neurons recorded in session A are used.

        Args:
            area: The visual cortical area to sample the neurons. Possible options: VISp, VISpm, VISam, VISal, VISl, VISrl.

        """
        path = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "allen" / "visual_drift" / "data" / "calcium_excitatory" / str(area)
        list_mice = path.glob("*")

        def _get_neural_data(num_movie, mat_file):
            mat = scipy.io.loadmat(mat_file)
            if num_movie == "one":
                mat_index = None
                mat_key = "united_traces_days_events"
            elif num_movie == "two":
                mat_index = (2, 1)
                mat_key = "filtered_traces_days_events"
            elif num_movie == "three":
                mat_index = (0, 1)
                mat_key = "filtered_traces_days_events"
            else:
                raise ValueError("num_movie should be one, two or three")

            if mat_index is not None:
                events = mat[mat_key][mat_index[0], mat_index[1]]
            else:
                events = mat[mat_key][:, :, 0]  ## Take one session only

            return events

        pseudo_mouse = np.vstack(
            [_get_neural_data(num_movie, mice) for mice in list_mice])

        return pseudo_mouse
