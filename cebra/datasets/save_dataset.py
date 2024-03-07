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
import argparse
import os

import joblib as jl

import cebra.datasets
from cebra.datasets import get_datapath
from cebra.datasets import monkey_reaching


def save_allen_decoding_dataset(savepath=get_datapath("allen_preload/")):
    """Load and save allen decoding dataset for Ca.
    Load and save all neural and behavioral data relevant for allen decoding dataset to reduce data loading time for the experiments using the shared data.
    It saves Ca data for the neuron numbers (10-1000), 5 different seeds for sampling the neurons and the train/test splits.
    Args:
        savepath: The directory to save the loaded data.
    """
    for n in [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]:
        for seed in [111, 222, 333, 444, 555]:
            for split_flag in ["train", "test"]:
                for modality in ["ca"]:
                    dataname = (
                        f"allen-movie1-{modality}-decoding-{n}-{split_flag}-{seed}"
                    )
                    data = cebra.datasets.init(dataname)
                    print(f"Initiated {dataname}")
                    jl.dump({"neural": data.neural},
                            f"{savepath}/{dataname}.jl")
                    print(f"{savepath}/{dataname}.jl")


def save_allen_dataset(savepath=get_datapth("allen_preload/")):
    """Load and save complete allen dataset for Ca.
    Load and save all neural and behavioral data relevant for allen decoding dataset to reduce data loading time for the experiments using the shared data.
    It saves Ca data for the neuron numbers (10-1000), 5 different seeds for sampling the neurons.
    Args:
        savepath: The directory to save the loaded data.
    """
    for n in [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]:
        for seed in [111, 222, 333, 444, 555]:
            for modality in ["neuropixel"]:
                dataname = f"allen-movie1-{modality}-{n}-{seed}"
                data = cebra.datasets.init(dataname)
                print(f"Initiated {dataname}")
                jl.dump({"neural": data.neural}, f"{savepath}/{dataname}.jl")
                print(f"{savepath}/{dataname}.jl")


def save_monkey_dataset(
        savepath: str = get_datapath("monkey_reaching_preload_smth_40/"),):
    """Load and save monkey reaching dataset.
    Load and save all neural and behavioral data relevant for monkey reaching dataset to reduce data loading time for the experiments using the shared data.
    It saves for all possible trials types ('active', 'passive', 'all') and the splits ('all', 'train', 'valid', 'test').
    Args:
        savepath: The directory to save the loaded data.
    """

    for session in ["active", "passive", "all"]:
        for split in ["all", "train", "valid", "test"]:
            data = monkey_reaching._load_data(session=session, split=split)
            dataname = f"{session}_{split}.jl"
            print(os.path.join(savepath, dataname))
            jl.dump(data, os.path.join(savepath, dataname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", default=get_datapath(), type=str)
    args = parser.parse_args()
    save_allen_decoding_dataset(os.path.join(args.savepath, "allen_preload"))
    save_allen_dataset(os.path.join(args.savepath, "allen_preload"))
    save_monkey_dataset(os.path.join(args.savepath, "monkey_reaching_preload"))
