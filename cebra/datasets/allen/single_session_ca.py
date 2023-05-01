"""Allen single mouse dataset.

References:
    *Deitch, Daniel, Alon Rubin, and Yaniv Ziv. "Representational drift in the mouse visual cortex." Current biology 31.19 (2021): 4327-4339.
    *de Vries, Saskia EJ, et al. "A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex." Nature neuroscience 23.1 (2020): 138-151.
    *https://github.com/zivlab/visual_drift
    *http://observatory.brain-map.org/visualcoding

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
from cebra.datasets import init
from cebra.datasets import parametrize
from cebra.datasets import register

_SINGLE_SESSION_CA = (
    get_datapath(
    get_datapath(
    get_datapath(
    get_datapath(


class SingleSessionAllenCa(cebra.data.SingleSessionDataset):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.
    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        frame_feature_path: The path of the movie frame features.
    """

    def __init__(
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        ),
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


class SingleSessionAllenCa(cebra.data.SingleSessionDataset):
    """A corrupted single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.
    during the 10 repeats of the MOVIE1 stimulus in session type A. The preprocessed data from *Deitch et al. (2021) are used.
    Args:
        frame_feature_path: The path of the movie frame features.

    """

    def __init__(
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        ),
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


class SingleSessionAllenCaTime(SingleSessionAllenCa):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus.
    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        frame_feature_path: The path of the movie frame features.

    """

    @property
    def continuous_index(self):
        return torch.arange(len(self))


@parametrize(
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=np.arange(10),
class SingleSessionAllenCaDecoding(cebra.data.SingleSessionDataset):
    """A single mouse 30Hz calcium events dataset during the allen MOVIE1 stimulus with train/test splits.
    A neural recording during the chosen repeat is used as a test set and the remaining 9 repeats are used as a train set.
    Args:
        session_id: The integer value to pick a session among 4 sessions with the largest number of recorded neruons. Choose between 0-3.
        repeat_no: The nth repeat to use as the test set. Choose between 0-9.
        split_flag: The `train`/`test` split to load.
        frame_feature_path: The path of the movie frame features.

    """

    def __init__(
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        ),
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
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.index = frame_feature.repeat(9, 1)
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
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=[0, 2, 4, 6, 8],
class SingleSessionAllenCaDecodingLeave2Out(cebra.data.SingleSessionDataset):

    def __init__(
        self,
        session_id,
        repeat_no,
        split_flag,
        frame_feature_path=get_datapath(
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        ),
    ):
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
        neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)

        valid_idx = np.arange(900 * repeat_no, 900 * (repeat_no + 1))
        test_idx = np.arange(900 * (repeat_no + 1), 900 * (repeat_no + 2))
        train_idx = np.delete(np.arange(9000),
                              np.concatenate([valid_idx, test_idx]))
        frame_feature = torch.load(frame_feature_path)
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.index = frame_feature.repeat(9, 1)
            self.neural = torch.from_numpy(neural[valid_idx]).float()
            self.index = frame_feature
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
    repeat_no=np.arange(10),
class MultiSessionAllenCaDecoding(cebra.data.DatasetCollection):

    def __init__(self, repeat_no, split_flag):


@parametrize(
    repeat_no=[0, 2, 4, 6, 8],
class MultiSessionAllenCaLeave2Out(cebra.data.DatasetCollection):

    def __init__(self, repeat_no, split_flag):


@parametrize(
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=[9],
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
            "allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        ),
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
            self.neural = torch.from_numpy(neural[train_idx]).float()
            self.frame_index = np.arange(900)
            rng.shuffle(self.frame_index)
            self.index = frame_feature.repeat(9,
                                              1)[np.tile(self.frame_index, 9)]
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
