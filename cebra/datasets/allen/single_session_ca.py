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
"""Allen single mouse dataset.

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
from cebra.datasets import init
from cebra.datasets import parametrize
from cebra.datasets import register

_DEFAULT_DATADIR = get_datapath()

_SINGLE_SESSION_CA = (
    pathlib.Path(_DEFAULT_DATADIR) / "allen" / "visual_drift" / "data" /
    "calcium_excitatory" / "VISp" / "680156909.mat",
    pathlib.Path(_DEFAULT_DATADIR) / "allen" / "visual_drift" / "data" /
    "calcium_excitatory" / "VISp" / "511510779.mat",
    pathlib.Path(_DEFAULT_DATADIR) / "allen" / "visual_drift" / "data" /
    "calcium_excitatory" / "VISp" / "679702882.mat",
    pathlib.Path(_DEFAULT_DATADIR) / "allen" / "visual_drift" / "data" /
    "calcium_excitatory" / "VISp" / "688678764.mat",
)


@parametrize(
    "allen-movie1-ca-single-session-{session_id}",
    session_id=range(len(_SINGLE_SESSION_CA)),
)
class SingleSessionAllenCa(cebra.data.SingleSessionDataset):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.

    A dataset of a single mouse 30Hz calcium events from the excitatory neurons in the primary visual cortex
    during the 10 repeats of the MOVIE1 stimulus in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.

    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        frame_feature_path: The path of the movie frame features.

    """

    def __init__(
        self,
        session_id: int,
        frame_feature_path: str = pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
        pca: bool = False,
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        if pca:
            pca_ = PCA()
            neural = pca_.fit_transform(
                traces["filtered_traces_days_events"][0,
                                                      0].transpose(1,
                                                                   0))[:, :32]
        else:
            neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)
        self.neural = torch.from_numpy(neural).float()
        frame_feature = torch.load(frame_feature_path)
        self.index = frame_feature.repeat(10, 1)

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
    "allen-movie1-ca-single-session-corrupt-{session_id}",
    session_id=range(len(_SINGLE_SESSION_CA)),
)
class SingleSessionAllenCa(cebra.data.SingleSessionDataset):
    """A corrupted single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.

    A dataset of a single mouse 30Hz calcium events from the excitatory neurons in the primary visual cortex
    during the 10 repeats of the MOVIE1 stimulus in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame, but in randomly shuffled order.

    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        frame_feature_path: The path of the movie frame features.

    """

    def __init__(
        self,
        session_id: int,
        frame_feature_path: str = pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
        pca: bool = False,
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        if pca:
            pca_ = PCA()
            neural = pca_.fit_transform(
                traces["filtered_traces_days_events"][0,
                                                      0].transpose(1,
                                                                   0))[:, :32]
        else:
            neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)

        self.neural = torch.from_numpy(neural).float()
        frame_feature = torch.load(frame_feature_path)
        self.frame_index = np.arange(900)
        rng = np.random.Generator(np.random.PCG64(111))
        rng.shuffle(self.frame_index)
        self.index = frame_feature.repeat(10, 1)[np.tile(self.frame_index, 10)]

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
    "allen-movie1-ca-single-session-time-{session_id}",
    session_id=range(len(_SINGLE_SESSION_CA)),
)
class SingleSessionAllenCaTime(SingleSessionAllenCa):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.

    A dataset of a single mouse 30Hz calcium events from the excitatory neurons in the primary visual cortex
    during the 10 repeats of the MOVIE1 stimulus in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    There is no behavioral variable used as the continuous label.

    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        frame_feature_path: The path of the movie frame features.

    """

    @property
    def continuous_index(self):
        return torch.arange(len(self))


@parametrize(
    "allen-movie1-ca-single-session-decoding-{session_id}-repeat-{repeat_no}-{split_flag}",
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=np.arange(10),
    split_flag=["train", "test"],
)
class SingleSessionAllenCaDecoding(cebra.data.SingleSessionDataset):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus with train/test splits.

    A dataset of a single mouse 30Hz calcium events from the excitatory neurons in the primary visual cortex
    during the 10 repeats of the MOVIE1 stimulus in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame.
    A neural recording during the chosen repeat is used as a test set and the remaining 9 repeats are used as a train set.

    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        repeat_no: The nth repeat to use as the test set. Choose between 0-9.
        split_flag: The `train`/`test` split to load.
        frame_feature_path: The path of the movie frame features.
        pca: If true, 32 principal components of the PCA transformed calcium data are used as neural input. Default value is `False`.

    """

    def __init__(
        self,
        session_id: int,
        repeat_no: int,
        split_flag: str,
        frame_feature_path: str = pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
        pca: bool = False,
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        if pca:
            pca_ = PCA()
            neural = pca_.fit_transform(
                traces["filtered_traces_days_events"][0,
                                                      0].transpose(1,
                                                                   0))[:, :32]
        else:
            neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)

        test_idx = np.arange(900 * repeat_no, 900 * (repeat_no + 1))
        train_idx = np.delete(np.arange(9000), test_idx)
        frame_feature = torch.load(frame_feature_path)
        if split_flag == "train":
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.index = frame_feature.repeat(9, 1)
        elif split_flag == "test":
            self.neural = torch.from_numpy(neural[test_idx]).float()
            self.index = frame_feature

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
    "allen-movie1-ca-single-session-leave2out-{session_id}-repeat-{repeat_no}-{split_flag}",
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=[0, 2, 4, 6, 8],
    split_flag=["train", "valid", "test"],
)
class SingleSessionAllenCaDecodingLeave2Out(cebra.data.SingleSessionDataset):

    def __init__(
        self,
        session_id,
        repeat_no,
        split_flag,
        frame_feature_path=pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)

        valid_idx = np.arange(900 * repeat_no, 900 * (repeat_no + 1))
        test_idx = np.arange(900 * (repeat_no + 1), 900 * (repeat_no + 2))
        train_idx = np.delete(np.arange(9000),
                              np.concatenate([valid_idx, test_idx]))
        frame_feature = torch.load(frame_feature_path)
        if split_flag == "train":
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.index = frame_feature.repeat(9, 1)
        elif split_flag == "valid":
            self.neural = torch.from_numpy(neural[valid_idx]).float()
            self.index = frame_feature
        elif split_flag == "test":
            self.neural = torch.from_numpy(neural[test_idx]).float()
            self.index = frame_feature
        else:
            raise ValueError(split_flag)

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
    "allen-movie1-ca-multi-session-decoding-repeat-{repeat_no}-{split_flag}",
    repeat_no=np.arange(10),
    split_flag=["train", "test"],
)
class MultiSessionAllenCaDecoding(cebra.data.DatasetCollection):

    def __init__(self, repeat_no, split_flag):
        super().__init__(
            *[
                init(
                    f"allen-movie1-ca-single-session-decoding-{session_id}-repeat-{repeat_no}-{split_flag}"
                ) for session_id in range(4)
            ],)


@parametrize(
    "allen-movie1-ca-multi-session-leave2out-repeat-{repeat_no}-{split_flag}",
    repeat_no=[0, 2, 4, 6, 8],
    split_flag=["train", "valid", "test"],
)
class MultiSessionAllenCaLeave2Out(cebra.data.DatasetCollection):

    def __init__(self, repeat_no, split_flag):
        super().__init__(
            *[
                init(
                    f"allen-movie1-ca-single-session-leave2out-{session_id}-repeat-{repeat_no}-{split_flag}"
                ) for session_id in range(4)
            ],)


@parametrize(
    "allen-movie1-ca-single-session-decoding-corrupt-{session_id}-repeat-{repeat_no}-{split_flag}",
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=[9],
    split_flag=["train", "test"],
)
class SingleSessionAllenCaDecoding(cebra.data.SingleSessionDataset):
    """A corrupted single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus with train/test splits.

    A dataset of a single mouse 30Hz calcium events from the excitatory neurons
    in the primary visual cortex during the 10 repeats of the MOVIE1 stimulus
    in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    The continuous labels corresponding to a DINO embedding of each stimulus frame,
    but in randomly shuffled order.
    A neural recording during the chosen repeat is used as a test set and the
    remaining 9 repeats are used as a train set.

    Args:
        session_id: The integer value to pick a session among 4 sessions with the
            largest number of recorded neruons. Choose between 0-3.
        repeat_no: The nth repeat to use as the test set. Choose between 0-9.
        split_flag: The `train`/`test` split to load.
        frame_feature_path: The path of the movie frame features.

    """

    def __init__(
        self,
        session_id: int,
        repeat_no: int,
        split_flag: str,
        frame_feature_path: str = pathlib.Path(_DEFAULT_DATADIR) / "allen" /
        "features" / "allen_movies" / "vit_base" / "8" /
        "movie_one_image_stack.npz" / "testfeat.pth",
        pca: bool = False,
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        if pca:
            pca_ = PCA()
            neural = pca_.fit_transform(
                traces["filtered_traces_days_events"][0,
                                                      0].transpose(1,
                                                                   0))[:, :32]
        else:
            neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)

        test_idx = np.arange(900 * repeat_no, 900 * (repeat_no + 1))
        train_idx = np.delete(np.arange(9000), test_idx)
        frame_feature = torch.load(frame_feature_path)
        rng = np.random.Generator(np.random.PCG64(111))
        if split_flag == "train":
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.frame_index = np.arange(900)
            rng.shuffle(self.frame_index)
            self.index = frame_feature.repeat(9,
                                              1)[np.tile(self.frame_index, 9)]
        elif split_flag == "test":
            self.neural = torch.from_numpy(neural[test_idx]).float()
            self.index = frame_feature
            self.frame_index = np.arange(900)
            rng.shuffle(self.frame_index)
            self.index = frame_feature[self.frame_index]

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
