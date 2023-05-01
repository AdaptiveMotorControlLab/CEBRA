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
from cebra.datasets import parametrize
from cebra.datasets import register


class AllenCaMovieDataset(cebra.data.SingleSessionDataset):
    """

    def __init__(
        super().__init__()

        frame_feature = torch.load(frame_feature_path)

        if load is None:
            neurons_indices = sampler.choice(np.arange(pseudo_data.shape[0]),
                                             size=num_neurons)
            self.neural = torch.from_numpy(neural).float()
        else:
            data = joblib.load(load)
            self.neural = data["neural"]

    def _get_pseudo_mice(self, area: str):
        list_mice = glob.glob(
        exp_containers = [
            int(mice.split(f"{area}/")[1].replace(".mat", ""))
            for mice in list_mice
        ]
        ## Load summary file
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
