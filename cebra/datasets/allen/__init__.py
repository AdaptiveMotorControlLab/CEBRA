#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
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
