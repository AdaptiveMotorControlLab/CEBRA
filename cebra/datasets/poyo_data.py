import pathlib
from enum import Enum
from typing import Union

import joblib as jl
import pandas as pd

import cebra.data
import cebra.io
from cebra.datasets import get_datapath

_DEFAULT_DATADIR = get_datapath()


class Task(Enum):
    CO = 0
    RT = 1


class PoyoDataset(cebra.data.SingleSessionDataset):

    def __init__(self, file: Union[pathlib.Path, str], drop_outliers: bool):
        super().__init__()

        data_dict = jl.load(file)
        data_df = pd.DataFrame.from_dict(data_dict)

        # e.g. filename: perich_sub-C_ses-CO-20131003_samplNone_smthNone
        if "perich" in str(file):
            file_list = str(file).split("-")
            if file_list[2] in Task.__members__:
                data_df["task"] = Task[file_list[2]].value
            else:
                raise NotImplementedError("Only CO and RT tasks are supported.")
        else:
            raise NotImplementedError("Only perich dataset is supported.")

        if drop_outliers:
            data_df = data_df[data_df[("subtask_index", "subtask_idx")] != 5]

        self.neural = data_df["Spikes"].values
        self.index = data_df["hand_vel"].values
        self.trial_id = data_df["trial_id"].values
        self.subtask_index = data_df[("subtask_index", "subtask_idx")].values

        self.task = data_df["task"].values[0]

    def __len__(self):
        return len(self.neural)

    @property
    def input_dimension(self):
        return self.neural.size(1)

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)
