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
from cebra.datasets import init
from cebra.datasets import parametrize
from cebra.datasets import register

_SINGLE_SESSION_CA = (


class SingleSessionAllenCa(cebra.data.SingleSessionDataset):

    def __init__(
        self.path = _SINGLE_SESSION_CA[session_id]
        traces = scipy.io.loadmat(self.path)
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

class SingleSessionAllenCaTime(SingleSessionAllenCa):

    @property
    def continuous_index(self):
        return torch.arange(len(self))


@parametrize(
    session_id=range(len(_SINGLE_SESSION_CA)),
    repeat_no=[0, 2, 4, 6, 8],
class SingleSessionAllenCaDecodingLeave2Out(cebra.data.SingleSessionDataset):

    def __init__(
        self,
        session_id,
        repeat_no,
        split_flag,
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
