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
"""Data loaders use distributions and indices to make samples available for training.

This package contains all helper functions and classes for defining and loading datasets
in the various usage modes of CEBRA, e.g. single- and multi-session datasets.
It is non-specific to a particular dataset (see :py:mod:`cebra.datasets` for actual dataset
implementations). However, the base classes for all datasets are defined here, as well as helper
functions to interact with datasets.

CEBRA supports different dataset types out-of-the box:

- :py:class:`cebra.data.single_session.SingleSessionDataset` is the abstract base class for a single session dataset. Single session datasets
  have the same feature dimension across the samples (e.g., neural data) and all context
  variables (e.g. behavior, stimuli, etc.).
- :py:class:`cebra.data.multi_session.MultiSessionDataset` is the abstract base class for a multi session dataset.
  Multi session datasets contain of multiple single session datasets. Crucially, the dimensionality of the
  auxiliary variable dimension needs to match across the sessions, which allows alignment of multiple sessions.
  The dimensionality of the signal variable can vary arbitrarily between sessions.

Note that the actual implementation of datasets (e.g. for benchmarking) is done in the :py:mod:`cebra.datasets`
package.

"""

# NOTE(stes): intentional ordering of imports to avoid circular imports
#             these imports will not be reordered by isort (see .isort.cfg)
from cebra.data.base import *
from cebra.data.datatypes import *

from cebra.data.single_session import *
from cebra.data.multi_session import *

from cebra.data.datasets import *

from cebra.data.helper import *
