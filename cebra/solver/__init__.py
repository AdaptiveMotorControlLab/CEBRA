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
"""Variants of CEBRA solvers for single- and multi-session training.

This package contains wrappers around training loops. If you want to customize how
different encoder models are used to transform the reference, positive and negative samples,
how the loss functions are applied to the data, or adapt specifics on how results are logged,
extending the classes in this package is the right way to go.

The module contains the base :py:class:`cebra.solver.base.Solver` class along with multiple variants to deal with
single- and multi-session datasets.
"""

import cebra.registry

cebra.registry.add_helper_functions(__name__)

# pylint: disable=wrong-import-position
from cebra.solver.base import *
from cebra.solver.multi_session import *
from cebra.solver.single_session import *
from cebra.solver.supervised import *

cebra.registry.add_docstring(__name__)
