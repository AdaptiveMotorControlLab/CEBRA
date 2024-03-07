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
"""Pre-defined neural network model architectures

This package contains everything related to implementing data encoders and the loss functions
applied to the feature spaces. :py:mod:`cebra.models.criterions` contains the implementations of
InfoNCE and other contrastive losses. All additions regarding how data is encoded and losses are
computed should be added to this package.

"""

import cebra.registry

cebra.registry.add_helper_functions(__name__)

from cebra.models.model import *
from cebra.models.multiobjective import *
from cebra.models.layers import *
from cebra.models.criterions import *

cebra.registry.add_docstring(__name__)
