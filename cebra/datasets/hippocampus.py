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
from cebra.datasets import register

_DEFAULT_DATADIR = get_datapath()


@parametrize(
    "rat-hippocampus-single-{name}",
)
class SingleRatDataset(cebra.data.SingleSessionDataset):
    """

    def __init__(self, name="achilles", root=_DEFAULT_DATADIR):
        super().__init__()
        path = os.path.join(root, f"rat_hippocampus/{name}.jl")
        data = joblib.load(path)
        self.name = name

    @property
    def input_dimension(self):
        return self.neural.size(1)

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):

    def __len__(self):
        return len(self.neural)

    def __repr__(self):
        return f"RatDataset(name: {self.name}, shape: {self.neural.shape})"


@register("rat-hippocampus-3fold-trial-split")

    def __init__(self,
                 name="achilles",
                 split_no=0,
                 split=None,
                 root=_DEFAULT_DATADIR):
        super().__init__(name=name, root=root)
        self.split_name = split
        if split is not None:
        Args:
        direction_change_idx = np.where(
            self.index[1:, 1] != self.index[:-1, 1])[0]
        trial_change_idx = np.append(
            np.insert(direction_change_idx[1::2], 0, 0), len(self.index))
        total_trials_num = len(trial_change_idx) - 1

        outer_folds = np.array_split(
            np.arange(total_trials_num),
            3)  ## Divide data into 3 equal trial-sized array
        inner_folds = sklearn.model_selection.KFold(n_splits=3,
                                                    random_state=None,
                                                    shuffle=False)

        train_trials = []
        valid_trials = []
        test_trials = []
            train_trial, val_test_trial = list(
                inner_folds.split(out_fold))[self.split_no]
            test_trial, valid_trial = np.array_split(val_test_trial, 2)

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
class SingleRatCorruptDataset(SingleRatDataset):
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

    def __init__(self, split_no=0, split=None):
        super().__init__(
            *[
                init(f"rat-hippocampus-{name}-3fold-trial-split-{split_no}",
                     split=split)
                for name in ["achilles", "buddy", "cicero", "gatsby"]
            ],
            continuous=True,
            discrete=False,
        )
        self.names = [dataset.name for dataset in self._datasets]
        self.shapes = [dataset.neural.shape for dataset in self._datasets]
        self._split = split

    def __repr__(self):

    def split(self, split):
        assert split == self._split
