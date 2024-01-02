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
"""Scikit-Learn interface for CEBRA

The implementation follows the guide on `Developing scikit-learn estimators`_.

Note:
    The main class of this module, :py:class:`CEBRA`, is also available under the
    top-level package name as :py:class:`cebra.CEBRA` and automatically imported
    with :py:mod:`cebra`.

.. _Developing scikit-learn estimators:
    https://scikit-learn.org/stable/developers/develop.html
"""

from cebra.integrations.sklearn import cebra
from cebra.integrations.sklearn import dataset
from cebra.integrations.sklearn import decoder
from cebra.integrations.sklearn import metrics
from cebra.integrations.sklearn import utils
