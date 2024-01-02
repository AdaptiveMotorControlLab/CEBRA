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
"""Rat hippocampus dataset

References:
    * Grosmark, A.D., and Buzsáki, G. (2016). Diversity in neural firing dynamics supports both rigid and learned
       hippocampal sequences. Science 351, 1440–1443.
    * Chen, Z., Grosmark, A.D., Penagos, H., and Wilson, M.A. (2016). Uncovering representations of sleep-associated
       hippocampal ensemble spike activity. Sci. Rep. 6, 32193.
    * Grosmark, A.D., Long J. and Buzsáki, G. (2016); Recordings from hippocampal area CA1, PRE, during and POST
      novel spatial learning. CRCNS.org. http://dx.doi.org/10.6080/K0862DC5

"""

import hashlib
import os

import joblib
import numpy as np
import scipy.io
import sklearn.model_selection
import sklearn.neighbors
import torch

import cebra.data
from cebra.datasets import get_datapath
from cebra.datasets import init
from cebra.datasets import parametrize
from cebra.datasets import register

_DEFAULT_DATADIR = get_datapath()

rat_dataset_urls = {
    "achilles": {
        "url":
            "https://figshare.com/ndownloader/files/40849463?private_link=9f91576cbbcc8b0d8828",
        "checksum":
            "c52f9b55cbc23c66d57f3842214058b8"
    },
    "buddy": {
        "url":
            "https://figshare.com/ndownloader/files/40849460?private_link=9f91576cbbcc8b0d8828",
        "checksum":
            "36341322907708c466871bf04bc133c2"
    },
    "cicero": {
        "url":
            "https://figshare.com/ndownloader/files/40849457?private_link=9f91576cbbcc8b0d8828",
        "checksum":
            "a83b02dbdc884fdd7e53df362499d42f"
    },
    "gatsby": {
        "url":
            "https://figshare.com/ndownloader/files/40849454?private_link=9f91576cbbcc8b0d8828",
        "checksum":
            "2b889da48178b3155011c12555342813"
    }
}


@register("rat-hippocampus-single")
@parametrize(
    "rat-hippocampus-single-{name}",
    name=["achilles", "buddy", "cicero", "gatsby"],
)
class SingleRatDataset(cebra.data.SingleSessionDataset):
    """A single rat hippocampus tetrode recording while the rat navigates on a linear track.

    Neural data is spike counts binned into 25ms time window and the continuous behavior label is position and the running direction (left, right) of a rat.
    The behavior label is structured as 3D array consists of position, right, and left.

    Args:
        name: The name of the rat to use. Choose among 'achilles', 'buddy', 'cicero' and 'gatsby'.

    """

    def __init__(self, name="achilles", root=_DEFAULT_DATADIR, download=True):
        location = os.path.join(root, "rat_hippocampus")
        file_path = os.path.join(location, f"{name}.jl")

        super().__init__(download=download,
                         data_url=rat_dataset_urls[name]["url"],
                         data_checksum=rat_dataset_urls[name]["checksum"],
                         location=location,
                         file_name=f"{name}.jl")

        data = joblib.load(file_path)
        self.neural = torch.from_numpy(data["spikes"]).float()
        self.index = torch.from_numpy(data["position"]).float()
        self.name = name

    @property
    def input_dimension(self):
        return self.neural.size(1)

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def __len__(self):
        return len(self.neural)

    def __repr__(self):
        return f"RatDataset(name: {self.name}, shape: {self.neural.shape})"

    def decode(self, x_train, y_train, x_test, y_test):
        """kNN decoding function.

        Perform a kNN decoding for n_neighbors = 1,4,9,26,25 with the given train set and test set.

        Args:
            x_train: The train set data
            y_train: The train set label
            x_test: The test set data
            y_test: The test set label

        """

        nn = np.power(np.linspace(1, 10, 6, dtype=int), 2)
        metric = {}
        for n in nn:
            knn = sklearn.neighbors.KNeighborsRegressor(n_neighbors=n)
            knn.fit(x_train, y_train)
            pred = knn.predict(x_test)
            err = np.median(abs(pred[:, 0] - y_test[:, 0]))
            score = knn.score(x_test, y_test)
            metric[f"n{n}_err"] = err
            metric[f"n{n}_r2"] = score
        return metric


@register("rat-hippocampus-3fold-trial-split")
@parametrize(
    "rat-hippocampus-{name}-3fold-trial-split-{split_no}",
    name=["achilles", "buddy", "cicero", "gatsby"],
    split_no=[0, 1, 2],
)
class SingleRatTrialSplitDataset(SingleRatDataset):
    """A single rat hippocampus tetrode recording while the rat navigates on a linear track with 3-fold splits.

    Neural data is spike counts binned into 25ms time window and the behavior is position and the running direction (left, right) of a rat.
    The behavior label is structured as 3D array consists of position, right, and left.
    The neural and behavior recordings are parsed into trials (a round trip from one end of the track) and the trials are split into a train, valid and test set with k=3 nested cross validation.

    Args:
        name: The name of a rat to use. Choose among 'achilles', 'buddy', 'cicero' and 'gatsby'.
        split_no: The `k` for k-fold split. Choose among 0, 1, 2.
        split: The split to use. Choose among 'train', 'valid', 'test', 'all', and 'wo_test'(all trials except test split).

    """

    def __init__(self,
                 name="achilles",
                 split_no=0,
                 split=None,
                 root=_DEFAULT_DATADIR):
        super().__init__(name=name, root=root)
        self.split_no = split_no
        self.split_name = split
        if split is not None:
            self._split(split)

    def _split(self, split, **kwargs):
        """Split the dataset into 3-fold nested cross validation scheme.

        The recordings are parsed into trials and split into a train, valid, test set with 3-fold nested cross validation scheme.

        Args:
            split: The split to use. Choose among 'train', 'valid', 'test', 'all', and 'wo_test'(all trials except test split).

        """

        direction_change_idx = np.where(self.index[1:, 1] != self.index[:-1,
                                                                        1])[0]
        trial_change_idx = np.append(
            np.insert(direction_change_idx[1::2], 0, 0), len(self.index))
        total_trials_num = len(trial_change_idx) - 1

        outer_folds = np.array_split(
            np.arange(total_trials_num),
            3)  ## Divide data into 3 equal trial-sized array
        inner_folds = sklearn.model_selection.KFold(n_splits=3,
                                                    random_state=None,
                                                    shuffle=False)
        ## in each outer fold array, make train, valid, test split

        train_trials = []
        valid_trials = []
        test_trials = []

        for out_fold in outer_folds:
            train_trial, val_test_trial = list(
                inner_folds.split(out_fold))[self.split_no]
            test_trial, valid_trial = np.array_split(val_test_trial, 2)
            train_trials.extend(np.array(out_fold)[train_trial])
            valid_trials.extend(np.array(out_fold)[valid_trial])
            test_trials.extend(np.array(out_fold)[test_trial])

        if split == "train":
            trials = train_trials
        elif split == "valid":
            trials = valid_trials
        elif split == "test":
            trials = test_trials
        elif split == "all":
            trials = np.arange(total_trials_num)
        elif split == "wo_test":
            trials = np.concatenate([train_trials, valid_trials])
        else:
            raise ValueError(
                f"'{split}' is not a valid split. Use 'train', 'valid' or 'test'"
            )
        self.selected_indices = tuple(
            slice(trial_change_idx[i], trial_change_idx[i + 1]) for i in trials)
        self.neural = torch.cat([
            self.neural[trial_change_idx[i]:trial_change_idx[i + 1]]
            for i in trials
        ])
        self.index = torch.cat([
            self.index[trial_change_idx[i]:trial_change_idx[i + 1]]
            for i in trials
        ])

        cumulated_len = np.cumsum(
            [trial_change_idx[i + 1] - trial_change_idx[i] for i in trials])
        self.concat_idx = cumulated_len[:-1][np.array(trials[:-1]) +
                                             1 != trials[1:]]


@parametrize(
    "rat-hippocampus-{name}-corrupt-{seed}",
    name=["achilles", "buddy", "cicero", "gatsby"],
    seed=np.arange(1000),
)
class SingleRatCorruptDataset(SingleRatDataset):
    """A single rat hippocampus tetrode recording while the rat navigates on a linear track with a shuffled behavior label.

    Neural data is spike counts binned into 25ms time window and the behavior is position and the running direction (left, right) of a rat.
    The behavior label is structured as 3D array consists of position, right, and left and it is shuffled in random orders.

    Args:
        name: The name of the rat to use. Choose among 'achilles', 'buddy', 'cicero' and 'gatsby'.
        seed: The random seed to set the shuffling.

    """

    def __init__(self, name, seed, root=_DEFAULT_DATADIR):
        super().__init__(name=name, root=root)
        rng = np.random.Generator(np.random.PCG64(seed))
        shuffled_index = np.arange(len(self.index))
        rng.shuffle(shuffled_index)
        self.index = self.index[shuffled_index]


@register("rat-hippocampus-multisubjects-3fold-trial-split")
@parametrize("rat-hippocampus-multisubjects-3fold-trial-split-{split_no}",
             split_no=[0, 1, 2])
class MultipleRatsTrialSplitDataset(cebra.data.DatasetCollection):
    """4 rats hippocampus tetrode recording while the rat navigates on a linear track with 3-fold splits.

    Neural and behavior recordings of 4 rats.
    For each rat, neural data is spike counts binned into 25ms time window and the behavior is position and the running direction (left, right) of a rat.
    The behavior label is structured as 3D array consists of position, right, and left.
    Neural and behavior recordings of each rat are parsed into trials (a round trip from one end of the track) and the trials are split into a train, valid and test set with k=3 nested cross validation.

    Args:
        split_no: The `k` for k-fold split. Choose among 0, 1, and 2.
        split: The split to use. Choose among 'train', 'valid', 'test', 'all', and 'wo_test'(all trials except test split).

    """

    def __init__(self, split_no=0, split=None):
        super().__init__(
            *[
                init(f"rat-hippocampus-{name}-3fold-trial-split-{split_no}",
                     split=split)
                for name in ["achilles", "buddy", "cicero", "gatsby"]
            ],)
        self.names = [dataset.name for dataset in self._datasets]
        self.shapes = [dataset.neural.shape for dataset in self._datasets]
        self._split = split

    def __repr__(self):
        return (
            f"MultipleRatsTrialSplitDataset(name: {self.names}, shape: {self.shapes})"
        )

    def split(self, split):
        assert split == self._split
