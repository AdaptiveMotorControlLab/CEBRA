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
"""Attribution methods for CEBRA.

This module was added in v0.6.0 and contains attribution methods described and benchmarked
in [Schneider2025]_.


.. [Schneider2025] Schneider, S., González Laiz, R., Filippova, A., Frey, M., & Mathis, M. W. (2025).
    Time-series attribution maps with regularized contrastive learning.
    The 28th International Conference on Artificial Intelligence and Statistics.
    https://openreview.net/forum?id=aGrCXoTB4P
"""
import cebra.registry

cebra.registry.add_helper_functions(__name__)

from cebra.attribution.attribution_models import *
from cebra.attribution.jacobian_attribution import *
