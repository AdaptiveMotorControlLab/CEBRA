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
