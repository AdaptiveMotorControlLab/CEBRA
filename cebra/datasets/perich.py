import pathlib
from enum import Enum
from typing import Union

import joblib as jl
import numpy as np
import pandas as pd
from torch_brain.data import Dataset  # Migrated from poyo to torch_brain

import cebra.data
import cebra.io
from cebra.datasets import get_datapath

_DEFAULT_DATADIR = get_datapath()


class Task(Enum):
    CO = 0
    RT = 1


class PerichDataset(cebra.data.SingleSessionDataset):

    def __init__(self,
                 file: Union[pathlib.Path, str],
                 drop_outliers: bool = True):
        super().__init__()

        dataset = self.load_dataset(path=file)

        # Extract recording_id from path for data retrieval
        path_obj = pathlib.Path(file)
        if path_obj.suffix == '.h5':
            recording_id = f"{path_obj.parent.name}/{path_obj.stem}"
        else:
            raise ValueError("File path must point to an h5 file")

        # Get the full recording data using brain_torch API
        # brain_torch.Dataset.get_recording_data() returns full data without slicing
        data = dataset.get_recording_data(recording_id)

        self.trial_id = self.get_trial_numbers(data)
        trial_mask = (self.trial_id != -1)

        self.subtask_index = data.cursor.subtask_index
        if drop_outliers:
            subtask_mask = self.subtask_index != 5
        else:
            subtask_mask = np.ones_like(self.subtask_index, dtype=bool)

        self.neural, self.index = self.get_neural_and_vel_data(data)
        self.neural = self.neural[trial_mask & subtask_mask]
        self.index = self.index[trial_mask & subtask_mask]

        self.session = pathlib.Path(data.session).name
        self.train_mask = data.cursor.train_mask[trial_mask & subtask_mask]
        self.test_mask = data.cursor.test_mask[trial_mask & subtask_mask]
        self.valid_mask = data.cursor.valid_mask[trial_mask & subtask_mask]

        self.trial_id = self.trial_id[trial_mask & subtask_mask]
        self.subtask_index = self.subtask_index[trial_mask & subtask_mask]

    def __len__(self):
        return len(self.neural)

    @property
    def input_dimension(self):
        return self.neural.size(1)

    def __getitem__(self, index):
        """Return [ No.Samples x Neurons x 10 ]"""
        index = self.expand_index(index)
        return self.neural[index].transpose(2, 1)

    def load_dataset(self, path: str, split: str = "train") -> Dataset:
        """Load dataset from processed torch_brain format.
        
        Args:
            path: Path to the h5 file or parent directory
            split: Which split to use (used for sampling intervals)
            
        Returns:
            Dataset object containing the recording
        """
        # Extract the recording_id from the path
        # torch_brain expects: root/brainset/session.h5
        # Format: recording_id = "brainset/session"

        path_obj = pathlib.Path(path)
        if path_obj.suffix == '.h5':
            # If path is a file, extract parent (brainset) and stem (session)
            root = str(path_obj.parent.parent)
            recording_id = f"{path_obj.parent.name}/{path_obj.stem}"
        else:
            # If path is a directory, use it as root
            root = str(path_obj)
            recording_id = None  # Will need to be specified differently

        return Dataset(root=root,
                       recording_id=recording_id,
                       split=split,
                       keep_files_open=True)

    def get_neural_and_vel_data(self, data):
        spike_times = data.spikes.timestamps
        unit_indices = data.spikes.unit_index

        n_neurons = np.max(data.spikes.unit_index) + 1
        time_bins = data.cursor.timestamps
        total_time = len(time_bins)

        psth = np.zeros((n_neurons, total_time))
        for i, spike_time in enumerate(spike_times):
            unit_index = unit_indices[i]
            time_bin = np.digitize(spike_time, time_bins) - 1
            if 0 <= time_bin < total_time:
                psth[unit_index, time_bin] += 1

        vel = data.cursor.vel
        return psth.T, vel

    def get_trial_numbers(self, data):
        cursor_timestamps = data.cursor.timestamps
        trial_starts = data.trials.start
        trial_ends = data.trials.end

        trial_numbers = np.full(cursor_timestamps.shape, -1)

        for trial_num, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            in_trial = (cursor_timestamps >= start) & (cursor_timestamps <= end)
            trial_numbers[in_trial] = trial_num

        return trial_numbers
