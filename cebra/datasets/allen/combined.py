
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
from cebra.datasets.allen import ca_movie
from cebra.datasets.allen import ca_movie_decoding
from cebra.datasets.allen import neuropixel_movie
from cebra.datasets.allen import neuropixel_movie_decoding


        )

    def __repr__(self):
