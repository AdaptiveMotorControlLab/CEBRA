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
