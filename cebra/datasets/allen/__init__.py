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
