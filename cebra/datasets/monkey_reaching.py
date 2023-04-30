"""Ephys neural and behavior data used for the monkey reaching experiment. 

References:
    * Chowdhury, Raeed H., Joshua I. Glaser, and Lee E. Miller. "Area 2 of primary somatosensory cortex encodes kinematics of the whole arm." Elife 9 (2020).
    * Chowdhury, Raeed; Miller, Lee (2022) Area2 Bump: macaque somatosensory area 2 spiking activity during reaching with perturbations (Version 0.220113.0359) [Data set]. `DANDI archive <https://doi.org/10.48324/dandi.000127/0.220113.0359>`_
    * Pei, Felix, et al. "Neural Latents Benchmark'21: Evaluating latent variable models of neural population activity." arXiv preprint arXiv:2109.04463 (2021).

"""

import hashlib
import os
import pickle as pk

import joblib as jl
import numpy as np
import scipy.io
import torch

try:
    from nlb_tools.nwb_interface import NWBDataset
except ImportError:
    import warnings
    warnings.warn(
        ("Could not import the nlb_tools package required for data loading "
         "of cebra.datasets.monkey_reaching. Dataset will not be available. "
         "If required, you can install the dataset by running "
         "pip install git+https://github.com/neurallatents/nlb_tools."))

import cebra.data
from cebra.datasets import get_datapath
from cebra.datasets import register


    Ephys and behavior recording from -100ms and 500ms from the movement onset in 1ms bin size.
    The behavior labels include trial types, target directions and the x,y hand positions.
    Args:
        session: The session type to load among 'active', 'passive' and 'all'.
    """

    def _get_info(trial_info, data):
        passive = []
        direction = []
        direction_actpas = []
        for index, trial in trial_info.iterrows():
                passive.append(True)
            else:
                passive.append(False)

        return {
        }

    dataset = NWBDataset(path, split_heldout=False)



    else:

        session_mask = ~dataset.trial_info.ctr_hold_bump
        session_mask = dataset.trial_info.ctr_hold_bump
        session_mask = True
    else:

    mask = split_mask & session_mask

                                     align_range=(-100, 500),
                                     ignored_trials=~mask)
    trial_info = dataset.trial_info[mask]

        data_dic = _get_info(trial_info[:mask.sum() // 2],
                             trials[:len(trials) // 2])
        data_dic = _get_info(trial_info[mask.sum() // 2:],
                             trials[len(trials) // 2:])
    else:
        data_dic = _get_info(trial_info, trials)

    return data_dic


class Area2BumpDataset(cebra.data.SingleSessionDataset):
    """Base dataclass to generate monkey reaching datasets.

    Args:
    """

        super().__init__()
        self.path = path
        self.session = session
        else:
            self.load_session = session
        self._post_load()

    def split(self, split):

        Args:
            split: The split. It can be either `all`, `train`, `valid`, `test`.
        """

        self.data = jl.load(
            os.path.join(self.path, f"{self.load_session}_{split}.jl"))
        self._post_load()

    def _post_load(self):
        """Read and assign neural and behavior recording into the class attributes."""

        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

        self.passive = torch.from_numpy(
                           ])).long()
        self.target = torch.from_numpy(
            np.concatenate([
            ])).long()
        self.target_actpas = torch.from_numpy(
            np.concatenate([
            ])).long()
        self.trial_indices = torch.from_numpy(self.trial_indices).float()

    @property
    def input_dimension(self):
        return self.neural.size(1)

    def __len__(self):
        return len(self.neural)

    @property
    def discrete_index(self):
        return self.passive

    @property
    def continuous_index(self):
        return self.pos

    def __repr__(self):

    def __getitem__(self, index):
        index = self.expand_index_in_trial(index,
                                           trial_ids=self.trial_ids,
                                           trial_borders=self.trial_borders)
        return self.neural[index].transpose(2, 1)


class Area2BumpShuffledDataset(Area2BumpDataset):
    """Base dataclass to generate shuffled monkey reaching datasets.

    Args:
    """

    def _post_load(self):
        rng = np.random.Generator(np.random.PCG64(1))

        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

        shuffle_index = np.arange(len(self.neural))
        rng.shuffle(shuffle_index)

        self.passive = torch.from_numpy(
                           ])).long()
        self.passive_shuffled = self.passive[shuffle_index]
        self.pos_shuffled = self.pos[shuffle_index]
        self.target = torch.from_numpy(
            np.concatenate([
            ])).long()
        self.target_actpas = torch.from_numpy(
            np.concatenate([
            ])).long()
        self.target_shuffled = self.target[shuffle_index]
        self.trial_indices = torch.from_numpy(self.trial_indices).float()


def _create_area2_dataset():
    """Register the monkey reaching datasets of different trial types, behavior labels.
    The trial types are 'active', 'passive', 'all' and 'active-passive'.
    """


        class Dataset(Area2BumpDataset):
            """Monkey reaching dataset with hand position labels.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                    return self.passive
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos

        class Dataset(Area2BumpDataset):
            """Monkey reaching dataset with target direction labels.
            The dataset loads discrete target direction (0-7) as behavior labels.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                    return self.target_actpas
                else:
                    return self.target

            @property
            def continuous_index(self):
                return None

        class Dataset(Area2BumpDataset):
            """Monkey reaching dataset with hand position labels and discrete target labels.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                return self.target

            @property
            def continuous_index(self):
                return self.pos


_create_area2_dataset()


def _create_area2_shuffled_dataset():
    """Register the shuffled monkey reaching datasets of different trial types, behavior labels.
    The trial types are 'active' and 'active-passive'.
    """


        class Dataset(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled trial type.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                    return self.passive_shuffled
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos

        class Dataset(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled hand position.
            The dataset loads continuous x,y hand position in randomly shuffled order.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                    return self.passive
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos_shuffled

        class Dataset(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled hand position.
            Args:
            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                    return self.passive
                else:
                    return self.target_shuffled

            @property
            def continuous_index(self):
                return None


_create_area2_shuffled_dataset()
