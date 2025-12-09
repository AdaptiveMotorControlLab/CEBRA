import os

import joblib as jl
import numpy as np
import scipy.stats
import sklearn.cluster
import torch

import cebra.data
import cebra.io
from cebra.datasets import get_datapath
from cebra.datasets import parametrize

_DEFAULT_DATADIR = get_datapath()


class NLBDataset(cebra.data.SingleSessionDataset):

    def __init__(self, bin_width, split="train"):
        super().__init__()
        self.bin_width = int(bin_width)
        self.split_tag = split

    def define_dataset(self, bin_width, smoothing, split):
        if bin_width == str(5):
            name_bin = ""
        else:
            name_bin = f"_{bin_width}"

        if smoothing != 0:
            smth = f"_smth{smoothing}"
        else:
            smth = ""

        if split == "valid":
            self.split = "eval"
            file_name = "train"
        elif split == "train":
            self.split = "train"
            file_name = "train"
        elif split == "test":
            self.split = "eval"
            file_name = "test"
        elif split == "train+valid":
            self.split = "train"
            file_name = "full-train"

        return name_bin, smth, file_name

    def __getitem__(self, index):
        index = self.expand_index_in_trial(index,
                                           trial_ids=self.trial_ids,
                                           trial_borders=self.trial_borders)
        return self.neural[index].transpose(2, 1)

    def __len__(self):
        return len(self.neural)

    @property
    def input_dimension(self):
        return self.neural.size(1)

    @property
    def continuous_index(self):
        if self.split_tag != "test":
            return self.index
        else:
            return None

    @property
    def discrete_index(self):
        if self.split_tag != "test":
            return self.trial_type
        else:
            return None


@parametrize("mc-maze-{size}-{bin_width}",
             size=["standard", "small", "medium", "large"],
             bin_width=["20", "5"])
class MCMazeDataset(NLBDataset):

    def __init__(self,
                 size,
                 bin_width,
                 smoothing=50,
                 split="train",
                 zscored_index=False):
        super().__init__(bin_width=bin_width, split=split)
        if size == "standard" or size == "":
            name_size = ""
        else:
            name_size = f"_{size}"

        self.name_size = name_size
        name_bin, smth, file_name = self.define_dataset(bin_width=bin_width,
                                                        smoothing=smoothing,
                                                        split=split)

        dataset_name = f"mc_maze{name_size}_{file_name}{name_bin}{smth}.jl"
        data_dict = jl.load(os.path.join(_DEFAULT_DATADIR, dataset_name))

        self.neural_trial = torch.from_numpy(
            data_dict[f"{self.split}_neural_heldin"]).float()
        self.neural = self.neural_trial.reshape(-1, self.neural_trial.shape[-1])
        self.trial_len = self.neural_trial.shape[1]
        self.num_trials = self.neural_trial.shape[0]
        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

        if self.split_tag != "test":
            self.neural_trial_heldout = torch.from_numpy(
                data_dict[f"{self.split}_neural_heldout"]).float()
            self.neural_heldout = self.neural_trial_heldout.reshape(
                -1, self.neural_trial_heldout.shape[-1])
            self.index_trial = torch.from_numpy(
                data_dict[f"{self.split}_behavior"]).float()
            self.index = self.index_trial.reshape(-1,
                                                  self.index_trial.shape[-1])
            if zscored_index:
                self.index = scipy.stats.zscore(self.index)

            self.index = torch.from_numpy(self.index).float()

    def __repr__(self):
        return f"MC-Maze{self.name_size}"


@parametrize("area2-bump-{bin_width}", bin_width=["20", "5"])
class AREA2Dataset(NLBDataset):

    def __init__(self,
                 bin_width,
                 smoothing=40,
                 split="train",
                 zscored_index=False):
        super().__init__(bin_width=bin_width, split=split)
        name_bin, smth, file_name = self.define_dataset(bin_width=bin_width,
                                                        smoothing=smoothing,
                                                        split=split)

        dataset_name = f"area2_bump_{file_name}{name_bin}{smth}.jl"
        data_dict = jl.load(os.path.join(_DEFAULT_DATADIR, dataset_name))

        self.neural_trial = torch.from_numpy(
            data_dict[f"{self.split}_neural_heldin"]).float()
        self.neural = self.neural_trial.reshape(-1, self.neural_trial.shape[-1])
        self.trial_len = self.neural_trial.shape[1]
        self.num_trials = self.neural_trial.shape[0]
        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])
        self.timepoint_labels = torch.from_numpy(
            np.linspace(0, 10,
                        self.trial_len).reshape(1, -1).repeat(self.num_trials,
                                                              axis=0).flatten())

        if self.split_tag != "test":
            self.neural_trial_heldout = torch.from_numpy(
                data_dict[f"{self.split}_neural_heldout"]).float()
            self.neural_heldout = self.neural_trial_heldout.reshape(
                -1, self.neural_trial_heldout.shape[-1])
            self.index_trial = torch.from_numpy(
                data_dict[f"{self.split}_behavior"]).float()
            self.pos_trial = self.index_trial.cumsum(axis=1)
            kmeans = sklearn.cluster.KMeans(9)
            X = self.pos_trial[:, -10:].mean(1)
            kmeans.fit(X)
            trial_labels = kmeans.predict(X)
            passive_idx = np.linalg.norm(kmeans.cluster_centers_,
                                         axis=1).argmin()
            active_trials = trial_labels != passive_idx
            self.trial_type = torch.from_numpy(
                active_trials.reshape(1, -1).repeat(self.trial_len,
                                                    axis=1).flatten()).long()
            self.index = self.index_trial.reshape(-1,
                                                  self.index_trial.shape[-1])
            if zscored_index:
                self.index = scipy.stats.zscore(self.index)

            self.index = torch.from_numpy(self.index).float()

    def __repr__(self):
        return f"Area2-Bump"


@parametrize("mc-rtt-{bin_width}", bin_width=["20", "5"])
class MCRTTDataset(NLBDataset):

    def __init__(self,
                 bin_width,
                 smoothing=40,
                 split="train",
                 zscored_index=False):
        super().__init__(bin_width=bin_width, split=split)
        name_bin, smth, file_name = self.define_dataset(bin_width=bin_width,
                                                        smoothing=smoothing,
                                                        split=split)

        dataset_name = f"mc_rtt_{file_name}{name_bin}{smth}.jl"
        data_dict = jl.load(os.path.join(_DEFAULT_DATADIR, dataset_name))

        self.neural_trial = torch.from_numpy(
            data_dict[f"{self.split}_neural_heldin"]).float()
        self.neural = self.neural_trial.reshape(-1, self.neural_trial.shape[-1])
        self.trial_len = self.neural_trial.shape[1]
        self.num_trials = self.neural_trial.shape[0]
        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

        if self.split_tag != "test":
            self.neural_trial_heldout = torch.from_numpy(
                data_dict[f"{self.split}_neural_heldout"]).float()
            self.neural_heldout = self.neural_trial_heldout.reshape(
                -1, self.neural_trial_heldout.shape[-1])
            self.index_trial = torch.from_numpy(
                data_dict[f"{self.split}_behavior"]).float()
            self.index = self.index_trial.reshape(-1,
                                                  self.index_trial.shape[-1])
            if zscored_index:
                self.index = scipy.stats.zscore(self.index)

            self.index = torch.from_numpy(self.index).float()

        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

    def __repr__(self):
        return f"MC-RTT"
