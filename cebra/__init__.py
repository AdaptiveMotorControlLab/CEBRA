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
"""CEBRA is a library for estimating Consistent Embeddings of high-dimensional Recordings
using Auxiliary variables. It contains self-supervised learning algorithms implemented in
PyTorch, and has support for a variety of different datasets common in biology and neuroscience.
"""

is_sklearn_available = False
try:
    # TODO(stes): More common integrations people care about (e.g. PyTorch lightning)
    # could be added here.
    from cebra.integrations.sklearn.cebra import CEBRA
    from cebra.integrations.sklearn.decoder import KNNDecoder
    from cebra.integrations.sklearn.decoder import L1LinearRegressor

    is_sklearn_available = True
except ImportError as e:
    # silently fail for now
    pass

is_matplotlib_available = False
try:
    from cebra.integrations.matplotlib import *

    is_matplotlib_available = True
except ImportError as e:
    # silently fail for now
    pass

is_plotly_available = False
try:
    from cebra.integrations.plotly import *

    is_plotly_available = True
except ImportError as e:
    # silently fail for now
    pass

from cebra.data.load import load as load_data

is_load_deeplabcut_available = False
try:
    from cebra.integrations.deeplabcut import load_deeplabcut
    is_load_deeplabcut_available = True
except (ImportError, NameError):
    pass

import cebra.integrations.sklearn as sklearn

__version__ = "0.4.0"
__all__ = ["CEBRA"]
__allow_lazy_imports = False
__lazy_imports = {}


def allow_lazy_imports():
    """Enables lazy imports of all submodules and packages of cebra.

    If called, references to ``cebra.<module_name>`` will be automatically
    lazily imported when first called in the code, and not raise a warning.
    """
    __allow_lazy_imports = True


def __getattr__(key):
    """Lazy import of cebra submodules and -packages.

    Once :py:mod:`cebra` is imported, it is possible to lazy import

    """
    if key == "CEBRA":
        from cebra.integrations.sklearn.cebra import CEBRA

        return CEBRA
    elif key == "KNNDecoder":
        from cebra.integrations.sklearn.decoder import KNNDecoder

        return KNNDecoder
    elif key == "L1LinearRegressor":
        from cebra.integrations.sklearn.decoder import L1LinearRegressor

        return L1LinearRegressor
    elif not key.startswith("_"):
        import importlib
        import warnings

        if key not in __lazy_imports:
            # NOTE(celia): condition needed when testing the string examples
            # so that the function doesn't try to import the testing packages
            # (pytest plugins, SetUpModule and TearDownModule) as cebra.{key}.
            # We just make sure that pytest is installed.
            if any(name in key.lower()
                   for name in ["pytest", "setup", "module"]):
                import pytest

                return importlib.import_module(pytest)
            __lazy_imports[key] = importlib.import_module(f"{__name__}.{key}")
            if not __allow_lazy_imports:
                warnings.warn(
                    f"Your code triggered a lazy import of {__name__}.{key}. "
                    f"While this will (likely) work, it is recommended to "
                    f"add an explicit import statement to you code instead. "
                    f"To disable this warning, you can run "
                    f"``cebra.allow_lazy_imports()``.")
        return __lazy_imports[key]
    raise AttributeError(f"module 'cebra' has no attribute '{key}'. "
                         f"Did you import cebra.{key}?")
