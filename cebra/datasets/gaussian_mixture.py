from typing import Tuple

import joblib as jl
import literate_dataclasses as dataclasses
import numpy as np
import sklearn
import torch

import cebra.data
import cebra.io
from cebra.datasets import get_datapath
from cebra.datasets import parametrize
from cebra.datasets import register


@parametrize(
class ContinuousGaussianMixtureDataset(cebra.data.SingleSessionDataset):

    Args:
        noise: The applied noise distribution applied.
    """

        super().__init__()
        self.noise = noise
        data = jl.load(

    @property
    def input_dimension(self):
        return self.neural.size(1)

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def split(self, split, train_ratio=0.8, valid_ratio=1):
        tot_len = len(self.neural)
        train_idx = np.arange(tot_len)[:int(tot_len * train_ratio)]
        valid_idx = np.arange(tot_len)[int(tot_len * train_ratio):]

            self.neural = self.neural[train_idx]
            self.index = self.index[train_idx]

            self.neural = self.neural[valid_idx]
            self.index = self.index[valid_idx]

            pass

    def __len__(self):
        return len(self.neural)

    def __repr__(self):
        return f"ContinuousGaussianMixtureDataset(noise: {self.noise}, shape: {self.neural.shape})"
