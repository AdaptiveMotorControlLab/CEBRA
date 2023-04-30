"""Datasets from the Allen Database

TODO(stes): Add additional context and information about the datasets.
"""

NUM_NEURONS = (10, 30, 50, 100, 200, 400, 600, 800, 900, 1000)
SEEDS = (111, 222, 333, 444, 555)
SEEDS_DISJOINT = (111, 222, 333)
from cebra.datasets.allen.ca_movie_decoding import *
from cebra.datasets.allen.ca_movie import *
from cebra.datasets.allen.combined import *
from cebra.datasets.allen.neuropixel_movie_decoding import *
from cebra.datasets.allen.neuropixel_movie import *
from cebra.datasets.allen.single_session_ca import *
