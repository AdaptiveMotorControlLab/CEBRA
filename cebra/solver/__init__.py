"""Variants of CEBRA solvers for single- and multi-session training.

single- and multi-session datasets.
"""

import cebra.registry

cebra.registry.add_helper_functions(__name__)

from cebra.solver.base import *
from cebra.solver.multi_session import *
from cebra.solver.single_session import *
from cebra.solver.supervised import *

cebra.registry.add_docstring(__name__)
