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
import sklearn.model_selection
import cebra.datasets
from . import pi_vae

class SyntheticDataset:
    def __init__(self, path="/data/synthetic_data/sim_100d_poisson_cont_label.npz", offset_left=0, offset_right=1):
        data = joblib.load(path)
        self.neural = data["x"]
        self.index = data["u"]
        self.latent = data['z']
        self.offset_left = offset_left
        self.offset_right = offset_right
        self.idx = np.arange(len(self.neural))
        
        
    def split(self, flag):
        tot_len = len(self.neural)
        train_idx=np.arange(tot_len)[:int(tot_len*0.8)]
        valid_idx = np.arange(tot_len)[int(tot_len*0.8):]
        test_idx =np.arange(tot_len)[int(tot_len*0.8):]
        
        if flag == 'train':
            self.neural = self.neural[train_idx]
            self.index = self.index[train_idx]
            self.idx = train_idx
        
        elif flag == 'valid':
            self.neural = self.neural[valid_idx]
            self.index = self.index[valid_idx]
            self.idx = valid_idx
        elif flag == 'all':
            pass
        
              
    @property
    def n_neurons(self):
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        index = self.expand_index(index)
        return self.neural[index].squeeze()

    def expand_index(self, index):
        offset = np.arange(-self.offset_left, self.offset_right)
        index = np.clip(index, self.offset_left, len(self) - self.offset_right)

        return index[:, None] + offset[None, :]

    def __len__(self):
        return len(self.neural)

            


class SingleRatDataset:
    """Dataset for a single rat hippocampus recording on linear track.

    Neural data in ime window size of 10 and the behavior is position and running dirction of a rat.
    Both position and running dirction are considered as continuous variables.
    Neural data is of shape ``[timepoints * neurons]`` and behavior is of shape ``[timepoints x 3(position, right, left)]``.
    """

    def __init__(self, name="achilles", offset_left=5, offset_right=5):
        path = f"/data/rat_hippocampus/{name}.jl"
        data = joblib.load(path)
        self.neural = data["spikes"]
        self.index = data["position"]
        self.name = name
        self.offset_left = offset_left
        self.offset_right = offset_right
        self.idx = np.arange(len(self.neural))

    def split(self, flag):
        tot_len = len(self.neural)
        train_test_split = int(len(self.neural) * 0.8)
        train_valid_split = int(train_test_split * 0.1)
        if flag == "test":
            self.neural = self.neural[train_test_split + train_valid_split:]
            self.index = self.index[train_test_split + train_valid_split:]
            self.idx = np.arange(tot_len)[train_test_split + train_valid_split:]
        elif flag == "valid":
            self.neural = self.neural[train_test_split:train_valid_split +
                                      train_test_split]
            self.index = self.index[train_test_split:train_valid_split +
                                    train_test_split]
            self.idx = np.arange(tot_len)[train_test_split:train_valid_split +
                                          train_test_split]
        elif flag == "train":
            self.neural = self.neural[:train_test_split]
            self.index = self.index[:train_test_split]
            self.idx = np.arange(tot_len)[:train_test_split]

    @property
    def input_dimesion(self):
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].squeeze()

    def expand_index(self, index):
        offset = np.arange(-self.offset_left, self.offset_right)
        index = np.clip(index, self.offset_left, len(self) - self.offset_right)

        return index[:, None] + offset[None, :]

    def __len__(self):
        return len(self.neural)

class SingleRatCVSplitDataset:
    """Dataset for a single rat hippocampus recording on linear track.

    Neural data in ime window size of 10 and the behavior is position and running dirction of a rat.
    Both position and running dirction are considered as continuous variables.
    Neural data is of shape ``[timepoints * neurons]`` and behavior is of shape ``[timepoints x 3(position, right, left)]``.
    """

    def __init__(self, name="achilles", split_no = 0, offset_left=5, offset_right=5):
        path = f"/data/rat_hippocampus/{name}.jl"
        data = joblib.load(path)
        self.neural = data["spikes"]
        self.index = data["position"]
        self.name = name
        self.offset_left = offset_left
        self.offset_right = offset_right
        self.split_no = split_no
        self.idx = np.arange(len(self.neural))

    def split(self, split):
        """Split dataset into 3-fold nested train/valid/split, specified by `split`.
        
        
        Args:
            split: One of ``train``, ``valid``, ``test``, ``all`` to specify which split should be used.
        """
        direction_change_idx = np.where(self.index[1:,1] != self.index[:-1,1])[0]
        trial_change_idx = np.append(np.insert(direction_change_idx[1::2], 0,0), len(self.index))
        total_trials_num = len(trial_change_idx)-1
        
        outer_folds = np.array_split(np.arange(total_trials_num), 3) ## Divide data into 3 equal trial-sized array
        inner_folds = sklearn.model_selection.KFold(n_splits=3, random_state=None, shuffle=False) 
        ## in each outer fold array, make train, valid, test split
        
        train_trials=[]
        valid_trials=[]
        test_trials=[]

        for out_fold in outer_folds:
            train_trial, val_test_trial=list(inner_folds.split(out_fold))[self.split_no]
            test_trial, valid_trial=np.array_split(val_test_trial, 2)
            train_trials.extend(np.array(out_fold)[train_trial])
            valid_trials.extend(np.array(out_fold)[valid_trial])
            test_trials.extend(np.array(out_fold)[test_trial])
        
        if split == 'train':
            trials = train_trials
        elif split == 'valid':
            trials = valid_trials
        elif split == 'test':
            trials = test_trials
        elif split == 'all':
            trials = np.arange(total_trials_num)
        else:
            raise ValueError(
                f"'{split}' is not a valid split. Use 'train', 'valid' or 'test'"
            )
        self.neural = np.concatenate([self.neural[trial_change_idx[i]:trial_change_idx[i+1]] for i in trials])
        self.index = np.concatenate([self.index[trial_change_idx[i]:trial_change_idx[i+1]] for i in trials])
        
        cumulated_len=np.cumsum([trial_change_idx[i+1]- trial_change_idx[i] for i in trials])
        self.concat_idx = cumulated_len[:-1][np.array(trials[:-1])+1 != trials[1:]]

    @property
    def n_neurons(self):
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].squeeze()

    def expand_index(self, index):
        offset = np.arange(-self.offset_left, self.offset_right)
        index = np.clip(index, self.offset_left, len(self) - self.offset_right)

        return index[:, None] + offset[None, :]

    def __len__(self):
        return len(self.neural)

class SingleSessionAllenCaDataset():

    def __init__(self, path, offset_left=5, offset_right=5):
        self.path = path
        self.offset_left = offset_left
        self.offset_right = offset_right
        traces = scipy.io.loadmat(self.path)
        neural = traces["filtered_traces_days_events"][0, 0].transpose(1, 0)
        self.neural = torch.from_numpy(neural).float()

        frame_feature = torch.load(
            "/data/allen/features/allen_movies/vit_base/8/movie_one_image_stack.npz/testfeat.pth"
        )
        self.index = frame_feature.repeat(10, 1)

    def split(self, flag):
        tot_len = len(self.neural)
        train_test_split = int(len(self.neural) * 0.9)
        if flag == "test":
            self.neural = self.neural[train_test_split:]
            self.index = self.index[train_test_split:]
            self.idx = np.arange(tot_len)[train_test_split:]
        elif flag == "train":
            self.neural = self.neural[:train_test_split]
            self.index = self.index[:train_test_split]
            self.idx = np.arange(tot_len)[:train_test_split]
        elif flag == 'all':
            self.idx = np.arange(tot_len)
    @property
    def n_neurons(self):
        return self.neural.shape[1]

    @property
    def continuous_index(self):
        return self.index

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].squeeze()

    def expand_index(self, index):
        offset = np.arange(-self.offset_left, self.offset_right)
        index = np.clip(index, self.offset_left, len(self) - self.offset_right)

        return index[:, None] + offset[None, :]

    def __len__(self):
        return len(self.neural)

class MonkeyReachingDataset():

    def __init__(
            self,
            path="/data/s1_reaching/sub-Han_desc-train_behavior+ecephys.nwb",
            session='active',
            label = 'target',
            split='all',
            offset_left=5,
            offset_right=5):
        if not session == 'active-passive':
            self.data = cebra.datasets.monkey_reaching._load_data(
                path, session, split)
        else:
            self.data = cebra.datasets.monkey_reaching._load_data(
                path, 'all', split)
        self.data_path = path
        self.session = session
        self.offset_left = offset_left
        self.offset_right = offset_right
        self.trial_len = int(self.data['trial_len'])
        self.num_trials = int(self.data['num_trials'])
        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]

        self.neural = self.data['spikes']
        self.pos = self.data['pos']
        
        
        if label == 'pos':
            self.index = self.pos
        
        elif label == 'target':
        
            if self.session != 'active-passive':
                self.index = np.concatenate(
                    [[t] * self.trial_len for t in self.data['movement_dir']])
            else:
                self.index = np.concatenate(
                    [[t] * self.trial_len for t in self.data['movement_dir_actpas']])

        
    @property
    def n_neurons(self):
        return self.neural.shape[1]

    def __len__(self):
        return self.neural.shape[0]

    @property
    def discrete_index(self):
        return self.index
    
    @property
    def continuous_index(self):
        return self.pos

    def __getitem__(self, index):
        index = self.expand_index_in_trial(index,
                                           trial_ids=self.trial_ids,
                                           trial_borders=self.trial_borders)
        return self.neural[index]

    def expand_index_in_trial(self, index, trial_ids, trial_borders):
        """
        offset = np.arange(-self.offset_left, self.offset_right)
        index = np.array([
            np.clip(i, trial_borders[trial_ids[i]] + self.offset_left,
                    trial_borders[trial_ids[i] + 1] - self.offset_right)
            for i in index
        ])
        return index[:, None] + offset[None, :]

