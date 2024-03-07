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


@register("continuous-gaussian-mixture")
@parametrize(
    "continuous-gaussian-mixture-{noise}",
    noise=["poisson", "gaussian", "laplace", "uniform", "refractory_poisson"],
)
class ContinuousGaussianMixtureDataset(cebra.data.SingleSessionDataset):
    """A dataset of synthetically generated continuous labels and the corresponding 2D latents
    and 100D noisy observations.

    Args:
        noise: The applied noise distribution applied.
    """

    def __init__(self, noise: str = "poisson"):
        super().__init__()
        self.noise = noise
        data = jl.load(
            get_datapath(f"synthetic/continuous_label_{self.noise}.jl"))
        self.latent = data["z"]
        self.index = torch.from_numpy(data["u"]).float()
        self.neural = torch.from_numpy(data["x"]).float()

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

        if split == "train":
            self.neural = self.neural[train_idx]
            self.index = self.index[train_idx]

        elif split == "valid":
            self.neural = self.neural[valid_idx]
            self.index = self.index[valid_idx]

        elif split == "all":
            pass

    def __len__(self):
        return len(self.neural)

    def __repr__(self):
        return f"ContinuousGaussianMixtureDataset(noise: {self.noise}, shape: {self.neural.shape})"
