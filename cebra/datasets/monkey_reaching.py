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
"""Ephys neural and behavior data used for the monkey reaching experiment.

References:
    * Chowdhury, Raeed H., Joshua I. Glaser, and Lee E. Miller.
      "Area 2 of primary somatosensory cortex encodes kinematics of the whole arm."
      Elife 9 (2020).
    * Chowdhury, Raeed; Miller, Lee (2022)
      Area2 Bump: macaque somatosensory area 2 spiking activity during reaching
      with perturbations (Version 0.220113.0359) [Data set].
      `DANDI archive <https://doi.org/10.48324/dandi.000127/0.220113.0359>`_
    * Pei, Felix, et al.
      "Neural Latents Benchmark'21: Evaluating latent variable models of neural
      population activity." arXiv preprint arXiv:2109.04463 (2021).
"""

import pathlib
from typing import Union

import joblib as jl
import numpy as np
import torch

import cebra.data
from cebra.datasets import get_datapath
from cebra.datasets import register

_DEFAULT_DATADIR = get_datapath()


def _load_data(
    path: Union[str, pathlib.Path] = None,
    session: str = "active",
    split: str = "train",
):
    """Load and preprocess neural and behavior data of monkey reaching task from NWBDataset.

    Ephys and behavior recording from -100ms and 500ms from the movement onset in 1ms bin size.
    Neural recording is smoothened with Gaussian kernel with 40ms std.
    The behavior labels include trial types, target directions and the x,y hand positions.

    Args:
        path: The path to the nwb file.
        session: The session type to load among 'active', 'passive' and 'all'.
        split: The split to load among 'train', 'valid', 'test' and 'all'.

    """

    if path is None:
        path = pathlib.Path(
            _DEFAULT_DATADIR
        ) / "s1_reaching" / "sub-Han_desc-train_behavior+ecephys.nwb"
    else:
        path = pathlib.Path(path)

    try:
        from nlb_tools.nwb_interface import NWBDataset
    except ImportError:
        raise ImportError(
            "Could not import the nlb_tools package required for data loading "
            "the raw reaching datasets in NWB format. "
            "If required, you can install the dataset by running "
            "pip install nlb_tools."
            # NOTE(stes): Install nlb_tools manually for now, also see
            # note in setup.cfg
            # or installing cebra with the [datasets] "
            #"dependencies: pip install 'cebra[datasets]'")
        )

    def _get_info(trial_info, data):
        passive = []
        direction = []
        direction_actpas = []
        for index, trial in trial_info.iterrows():
            if trial["ctr_hold_bump"]:
                passive.append(True)
                direction.append(int(trial["bump_dir"] / 45))
                direction_actpas.append(int(trial["bump_dir"] / 45) + 8)
            else:
                passive.append(False)
                direction.append(int(trial["cond_dir"] / 45))
                direction_actpas.append(int(trial["cond_dir"] / 45))
            spikes = data["spikes_smth_40"].to_numpy()
            velocity = data["hand_vel"].to_numpy()
            position = data["hand_pos"].to_numpy()

        return {
            "spikes": spikes,
            "vel": velocity,
            "pos": position,
            "passive": np.array(passive),
            "movement_dir": np.array(direction),
            "movement_dir_actpas": np.array(direction_actpas),
            "num_trials": len(trial_info),
            "trial_len": int(len(spikes) / len(trial_info)),
        }

    dataset = NWBDataset(path, split_heldout=False)
    dataset.smooth_spk(40, name="smth_40")

    if split == "train":
        split_mask = dataset.trial_info.split == "train"

    elif split == "all":
        split_mask = dataset.trial_info.split != "none"

    elif split == "valid" or split == "test":
        split_mask = dataset.trial_info.split == "val"
    else:
        raise ValueError("--split argument should be train, valid, test or all")

    if session == "active":
        session_mask = ~dataset.trial_info.ctr_hold_bump
    elif session == "passive":
        session_mask = dataset.trial_info.ctr_hold_bump
    elif session == "all":
        session_mask = True
    else:
        raise ValueError("--session argument should be active, passive or all")

    mask = split_mask & session_mask

    trials = dataset.make_trial_data(align_field="move_onset_time",
                                     align_range=(-100, 500),
                                     ignored_trials=~mask)
    trial_info = dataset.trial_info[mask]

    if split == "valid":
        data_dic = _get_info(trial_info[:mask.sum() // 2],
                             trials[:len(trials) // 2])
    elif split == "test":
        data_dic = _get_info(trial_info[mask.sum() // 2:],
                             trials[len(trials) // 2:])
    else:
        data_dic = _get_info(trial_info, trials)

    return data_dic


monkey_reaching_urls = {
    "all_all.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/all_all.jl.gz",
        "checksum":
            "dea556301fa4fafa86e28cf8621cab5a",
        "gzipped_checksum":
            "399abc6e9ef0b23a0d6d057c6f508939"
    },
    "all_train.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/all_train.jl.gz",
        "checksum":
            "e280e4cd86969e6fd8bfd3a8f402b2fe",
        "gzipped_checksum":
            "eb52c8641fe83ae2a278b372ddec5f69"
    },
    "all_test.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/all_test.jl.gz",
        "checksum":
            "25d3ff2c15014db8b8bf2543482ae881",
        "gzipped_checksum":
            "7688245cf15e0b92503af943ce9f66aa"
    },
    "all_valid.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/all_valid.jl.gz",
        "checksum":
            "8cd25169d31f83ae01b03f7b1b939723",
        "gzipped_checksum":
            "b169fc008b4d092fe2a1b7e006cd17a7"
    },
    "active_all.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/active_all.jl.gz",
        "checksum":
            "c626acea5062122f5a68ef18d3e45e51",
        "gzipped_checksum":
            "b7b86e2ae00bb71341de8fc352dae097"
    },
    "active_train.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/active_train.jl.gz",
        "checksum":
            "72a48056691078eee22c36c1992b1d37",
        "gzipped_checksum":
            "56687c633efcbff6c56bbcfa35597565"
    },
    "active_test.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/active_test.jl.gz",
        "checksum":
            "35b7e060008a8722c536584c4748f2ea",
        "gzipped_checksum":
            "2057ef1846908a69486a61895d1198e8"
    },
    "active_valid.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/active_valid.jl.gz",
        "checksum":
            "dd58eb1e589361b4132f34b22af56b79",
        "gzipped_checksum":
            "60b8e418f234877351fe36f1efc169ad"
    },
    "passive_all.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/passive_all.jl.gz",
        "checksum":
            "bbb1bc9d8eec583a46f6673470fc98ad",
        "gzipped_checksum":
            "afb257efa0cac3ccd69ec80478d63691"
    },
    "passive_train.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/passive_train.jl.gz",
        "checksum":
            "f22e05a69f70e18ba823a0a89162a45c",
        "gzipped_checksum":
            "24d98d7d41a52591f838c41fe83dc2c6"
    },
    "passive_test.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/passive_test.jl.gz",
        "checksum":
            "42453ae3e4fd27d82d297f78c13cd6b7",
        "gzipped_checksum":
            "f1ff4e9b7c4a0d7fa9dcd271893f57ab"
    },
    "passive_valid.jl": {
        "url":
            "https://cebra.fra1.digitaloceanspaces.com/data/monkey_reaching_preload_smth_40/passive_valid.jl.gz",
        "checksum":
            "2dcc10c27631b95a075eaa2d2297bb4a",
        "gzipped_checksum":
            "311fcb6a3e86022f12d78828f7bd29d5"
    }
}


@register("area2-bump")
class Area2BumpDataset(cebra.data.SingleSessionDataset):
    """Base dataclass to generate monkey reaching datasets.

    Ephys and behavior recording from -100ms and 500ms from the movement
    onset in 1ms bin size.
    Neural recording is smoothened with Gaussian kernel with 40ms std.
    The behavior labels can include trial types, target directions and the
    x,y hand positions.
    After initialization of the dataset, split method can splits the data
    into 'train', 'valid' and 'test' split.

    Args:
        path: The path to the directory where the preloaded data is.
        session: The trial type. Choose between 'active', 'passive',
            'all', 'active-passive'.

    """

    def __init__(self,
                 path: str = get_datapath("monkey_reaching_preload_smth_40/"),
                 session: str = "active",
                 download=True):
        super().__init__()
        self.path = path
        self.download = download
        self.session = session
        if session == "active-passive":
            self.load_session = "all"
        else:
            self.load_session = session

        super().__init__(
            download=self.download,
            data_url=monkey_reaching_urls[f"{self.load_session}_all.jl"]["url"],
            data_checksum=monkey_reaching_urls[f"{self.load_session}_all.jl"]
            ["checksum"],
            gzipped_checksum=monkey_reaching_urls[f"{self.load_session}_all.jl"]
            .get("gzipped_checksum"),
            location=self.path,
            file_name=f"{self.load_session}_all.jl",
        )

        self.data = jl.load(
            pathlib.Path(self.path) / f"{self.load_session}_all.jl")
        self._post_load()

    def split(self, split):
        """Split the dataset.

        The train trials are the same as one defined in Neural Latent
        Benchmark (NLB) Dataset.
        The half of the valid trials defined in NLBDataset is used as
        the valid set and the other half is used as the test set.

        Args:
            split: The split. It can be either `all`, `train`, `valid`, `test`.

        """

        super().__init__(
            download=self.download,
            data_url=monkey_reaching_urls[f"{self.load_session}_{split}.jl"]
            ["url"],
            data_checksum=monkey_reaching_urls[
                f"{self.load_session}_{split}.jl"]["checksum"],
            gzipped_checksum=monkey_reaching_urls[
                f"{self.load_session}_{split}.jl"].get("gzipped_checksum"),
            location=self.path,
            file_name=f"{self.load_session}_{split}.jl",
        )
        self.data = jl.load(
            pathlib.Path(self.path) / f"{self.load_session}_{split}.jl")
        self._post_load()

    def _post_load(self):
        """Read and assign neural and behavior recording into the class attributes."""

        self.trial_len = int(self.data["trial_len"])
        self.num_trials = int(self.data["num_trials"])
        self.neural = torch.from_numpy(self.data["spikes"]).float()
        self.trial_ids = np.concatenate(
            [[n] * self.trial_len for n in range(self.num_trials)])
        self.trial_borders = [
            self.trial_len * i for i in range(self.num_trials + 1)
        ]
        self.trial_indices = np.concatenate(
            [np.arange(self.trial_len) for n in range(self.num_trials)])

        self.passive = torch.from_numpy(
            np.concatenate([[t] * self.trial_len for t in self.data["passive"]
                           ])).long()
        self.pos = torch.from_numpy(self.data["pos"]).float()
        self.vel = torch.from_numpy(self.data["vel"]).float()
        self.target = torch.from_numpy(
            np.concatenate([
                [t] * self.trial_len for t in self.data["movement_dir"]
            ])).long()
        self.target_actpas = torch.from_numpy(
            np.concatenate([
                [t] * self.trial_len for t in self.data["movement_dir_actpas"]
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
        return f"MonkeyArea2BumpDataset(name: Discrete active/passive & " \
               f"continuous hand position, shape: {self.neural.shape})"

    def __getitem__(self, index):
        index = self.expand_index_in_trial(index,
                                           trial_ids=self.trial_ids,
                                           trial_borders=self.trial_borders)
        return self.neural[index].transpose(2, 1)


@register("area2-bump-shuffled")
class Area2BumpShuffledDataset(Area2BumpDataset):
    """Base dataclass to generate shuffled monkey reaching datasets.

    Ephys and behavior recording from -100ms and 500ms from the movement
    onset in 1ms bin size.
    Neural recording is smoothened with Gaussian kernel with 40ms std.
    The shuffled behavior labels can include trial types, target directions
    and the x,y hand positions.

    After initialization of the dataset, split method can splits the data
    into 'train', 'valid' and 'test' split.

    Args:
        path: The path to the directory where the preloaded data is.
        session: The trial type. Choose between 'active', 'passive', 'all',
            'active-passive'.

    """

    def _post_load(self):
        rng = np.random.Generator(np.random.PCG64(1))

        self.trial_len = int(self.data["trial_len"])
        self.num_trials = int(self.data["num_trials"])
        self.neural = torch.from_numpy(self.data["spikes"]).float()
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
            np.concatenate([[t] * self.trial_len for t in self.data["passive"]
                           ])).long()
        self.passive_shuffled = self.passive[shuffle_index]
        self.pos = torch.from_numpy(self.data["pos"]).float()
        self.pos_shuffled = self.pos[shuffle_index]
        self.target = torch.from_numpy(
            np.concatenate([
                [t] * self.trial_len for t in self.data["movement_dir"]
            ])).long()
        self.target_actpas = torch.from_numpy(
            np.concatenate([
                [t] * self.trial_len for t in self.data["movement_dir_actpas"]
            ])).long()
        self.target_shuffled = self.target[shuffle_index]
        self.trial_indices = torch.from_numpy(self.trial_indices).float()


def _create_area2_dataset():
    """Register the monkey reaching datasets of different trial types, behavior labels.

    The trial types are 'active', 'passive', 'all' and 'active-passive'.
    The 'active-passive' type distinguishes movement direction between active, passive
    (0-7 for active and 8-15 for passive) and 'all' does not (0-7).

    """

    PATH = pathlib.Path(_DEFAULT_DATADIR) / "monkey_reaching_preload_smth_40"
    for session_type in ["active", "passive", "active-passive", "all"]:

        @register(f"area2-bump-pos-{session_type}")
        class DatasetV1(Area2BumpDataset):
            """Monkey reaching dataset with hand position labels.

            The dataset loads continuous x,y hand position as behavior labels.
            For the 'active-passive' trial type, it additionally loads discrete binary
            label of active(0)/passive(1).

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                    'active-passive'.

            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                if self.session == "active-passive":
                    return self.passive
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos

        @register(f"area2-bump-target-{session_type}")
        class DatasetV2(Area2BumpDataset):
            """Monkey reaching dataset with target direction labels.

            The dataset loads discrete target direction (0-7) as behavior labels.

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                    'active-passive'.

            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                if self.session == "active-passive":
                    return self.target_actpas
                else:
                    return self.target

            @property
            def continuous_index(self):
                return None

        @register(f"area2-bump-posdir-{session_type}")
        class DatasetV3(Area2BumpDataset):
            """Monkey reaching dataset with hand position labels and discrete target labels.

            The dataset loads continuous x,y hand position and discrete target labels (0-7)
            as behavior labels.
            For active-passive type, the discrete target labels 0-7 for active and 8-16 for
            passive are loaded.

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                'active-passive'.

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
    The behavior labels are randomly shuffled and the trial types are shuffled
    in case of 'shuffled-trial' datasets.

    """

    PATH = pathlib.Path(_DEFAULT_DATADIR) / "monkey_reaching_preload_smth_40"
    for session_type in ["active", "active-passive"]:

        @register(f"area2-bump-pos-{session_type}-shuffled-trial")
        class DatasetV4(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled trial type.

            The dataset loads the discrete binary trial type label active(0)/passive(1)
            in randomly shuffled order.

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                    'active-passive'.

            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                if self.session == "active-passive":
                    return self.passive_shuffled
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos

        @register(f"area2-bump-pos-{session_type}-shuffled-position")
        class DatasetV5(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled hand position.

            The dataset loads continuous x,y hand position in randomly shuffled order.
            For the 'active-passive' trial type, it additionally loads discrete binary label
            of active(0)/passive(1).

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                    'active-passive'.

            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                if self.session == "active-passive":
                    return self.passive
                else:
                    return None

            @property
            def continuous_index(self):
                return self.pos_shuffled

        @register(f"area2-bump-target-{session_type}-shuffled")
        class DatasetV6(Area2BumpShuffledDataset):
            """Monkey reaching dataset with the shuffled hand position.

            The dataset loads discrete target direction (0-7 for active and 0-15 for active-passive)
            in randomly shuffled order.

            Args:
                path: The path to the directory where the preloaded data is.
                session: The trial type. Choose between 'active', 'passive', 'all',
                    'active-passive'.

            """

            def __init__(self, path=PATH, session=session_type):
                super().__init__(path=path, session=session)

            @property
            def discrete_index(self):
                if self.session == "active-passive":
                    return self.passive
                else:
                    return self.target_shuffled

            @property
            def continuous_index(self):
                return None


_create_area2_shuffled_dataset()
