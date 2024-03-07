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
"""Allen pseudomouse Ca dataset.

References:
    *Deitch, Daniel, Alon Rubin, and Yaniv Ziv. "Representational drift in the mouse visual cortex." Current biology 31.19 (2021): 4327-4339.
    *de Vries, Saskia EJ, et al. "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." Nature neuroscience 23.1 (2020): 138-151.
    *https://github.com/zivlab/visual_drift
    *http://observatory.brain-map.org/visualcoding

"""

import glob
import hashlib
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

_DEFAULT_DATADIR = get_datapath()


@parametrize("allen-movie1-ca-{num_neurons}-{seed}",
             num_neurons=NUM_NEURONS,
             seed=SEEDS)
class AllenCaMovieDataset(cebra.data.SingleSessionDataset):
    """A pseudomouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.

    A dataset of stacked 30Hz calcium events from the excitatory neurons in the primary visual cortex of multiple mice
    during the 10 repeats of the MOVIE1 stimulus in session A,B and C. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.

    Args:
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons. Choose from 10, 30, 50, 100, 200, 400, 600, 800, 900, 1000.
        seed: The random seeds for sampling neurons.
        frame_feature_path: The path of the movie frame features.
        load: The path to the preloaded neural data. If `None`, the neural data is constructed from the source. Default value is `None`.

    """

    def __init__(
        self,
        num_neurons=10,
        seed=111,
        area="VISp",
        frame_feature_path=pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
        pca=False,
        load=None,
    ):
        super().__init__()

        frame_feature = torch.load(frame_feature_path)

        if load is None:
            pseudo_data = self._get_pseudo_mice(area)
            sampler = Generator(PCG64(seed))
            neurons_indices = sampler.choice(np.arange(pseudo_data.shape[0]),
                                             size=num_neurons)
            if pca:
                sampled_neural = pseudo_data[neurons_indices, :]
                pca_ = PCA()
                neural = pca_.fit_transform(sampled_neural.transpose(1,
                                                                     0))[:, :32]
            else:
                neural = pseudo_data[neurons_indices, :].transpose(1, 0)
            self.neural = torch.from_numpy(neural).float()
        else:
            data = joblib.load(load)
            self.neural = data["neural"]
        self.index = self._get_index(frame_feature)

    def _get_index(self, frame_feature):
        """Return the behavior label.

        Construct the behavior labels with the user-defined frame feature.

        Args:
            frame_feature: The behavior label of each movie frame.

        """

        return frame_feature.repeat(10, 1)

    def _get_pseudo_mice(self, area: str):
        """Construct pseudomouse neural dataset.

        Stack the excitatory neurons from the multiple mice and construct a psuedomouse neural dataset of the specified visual cortical area.
        The neurons which were recorded in all of the sessions A, B, C are included.

        Args:
            area: The visual cortical area to sample the neurons. Possible options: VISp, VISpm, VISam, VISal, VISl, VISrl.

        """

        self.area = area
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


@parametrize("allen-movie1-ca-{num_neurons}-{seed}-preload",
             num_neurons=NUM_NEURONS,
             seed=SEEDS)
class AllenCaMoviePreLoadDataset(AllenCaMovieDataset):
    """A pre-loaded pseudomouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.

    It loads the pre-loaded pseudomouse 30Hz VISp calcium events dataset during the allen MOVIE1 stimulus.

    Args:
        num_neurons: The number of neurons to randomly sample from the stacked pseudomouse neurons.
        seed: The random seeds for sampling neurons.

    """

    def __init__(self, num_neurons, seed):
        preload = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "allen_preload" / f"allen-movie1-ca-{num_neurons}-{seed}.jl"
        if not preload.is_file():
            print("The dataset is not yet preloaded.")
            preload = None
        super().__init__(num_neurons=num_neurons, seed=seed, load=preload)
