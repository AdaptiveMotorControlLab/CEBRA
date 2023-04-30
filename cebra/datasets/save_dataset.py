import argparse
import os

import joblib as jl

import cebra.datasets
from cebra.datasets import get_datapath
from cebra.datasets import monkey_reaching


    """Load and save allen decoding dataset for Ca.
    Load and save all neural and behavioral data relevant for allen decoding dataset to reduce data loading time for the experiments using the shared data.
    It saves Ca data for the neuron numbers (10-1000), 5 different seeds for sampling the neurons and the train/test splits.
    Args:
        savepath: The directory to save the loaded data.
    """
    for n in [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]:
        for seed in [111, 222, 333, 444, 555]:
                    data = cebra.datasets.init(dataname)


    """Load and save complete allen dataset for Ca.
    Load and save all neural and behavioral data relevant for allen decoding dataset to reduce data loading time for the experiments using the shared data.
    It saves Ca data for the neuron numbers (10-1000), 5 different seeds for sampling the neurons.
    Args:
        savepath: The directory to save the loaded data.
    """
    for n in [10, 30, 50, 100, 200, 400, 600, 800, 900, 1000]:
        for seed in [111, 222, 333, 444, 555]:
                data = cebra.datasets.init(dataname)


def save_monkey_dataset(
    """Load and save monkey reaching dataset.
    Load and save all neural and behavioral data relevant for monkey reaching dataset to reduce data loading time for the experiments using the shared data.
    It saves for all possible trials types ('active', 'passive', 'all') and the splits ('all', 'train', 'valid', 'test').
    Args:
        savepath: The directory to save the loaded data.
    """

            data = monkey_reaching._load_data(session=session, split=split)
            print(os.path.join(savepath, dataname))
            jl.dump(data, os.path.join(savepath, dataname))


    parser = argparse.ArgumentParser()
    args = parser.parse_args()
