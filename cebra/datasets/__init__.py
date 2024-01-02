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
"""Pre-defined demo and benchmark datasets.

This package contains actual implementations of datasets. If you want to add a commonly used (and
public dataset) to CEBRA, this is the right package to do it. Datasets here can be loaded e.g. for testing,
reproducing reference results and benchmarking. When contributing to this package, you should ensure
that the data is publicly available under a suitable license.

"""

import os
import pathlib
from typing import Union
import cebra.registry

cebra.registry.add_helper_functions(__name__)
__DATADIR = os.environ.get("CEBRA_DATADIR", "data")


def get_data_root() -> str:
    """Return the data directory

    See :py:func:`set_datapath` for altering the database during
    runtime, and set the ``CEBRA_DATADIR`` variable in your shell
    environment to modify the datapath.
    """
    return __DATADIR


def set_datapath(path: str = None, update_environ: bool = True):
    """Updates the root data path of the system.

    By default, the function also updates the corresponding environment
    variable.

    Args:
        path: The new datapath
        update_environ: Whether to update the environment variable
            ``CEBRA_DATADIR`` with the new path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The specified path {path} does not exist.")
    if os.path.isfile(path):
        raise FileExistsError(
            f"The specified path {path} is a file, not a directory.")
    __DATADIR = path
    os.environ["CEBRA_DATADIR"] = __DATADIR


def get_datapath(path: str = None) -> str:
    """Convert a relative to a system-dependent absolute data path.

    The data directory is given by ``{root}/{path}``. The root
    directory can be specified through the environment variable
    ``CEBRA_DATADIR``.

    Args:
        path: The path as a ``str`` or ``pathlib.Path`` object.
            The path is relative to the system data directory.

    Returns:
        The absolute path to the dataset.
    """
    if path is None:
        return get_data_root()
    path = str(path)
    return os.path.join(get_data_root(), path)


# pylint: disable=wrong-import-position
import warnings
from cebra.datasets.demo import *

try:
    from cebra.datasets.allen import *
    from cebra.datasets.gaussian_mixture import *
    from cebra.datasets.hippocampus import *
    from cebra.datasets.monkey_reaching import *
    from cebra.datasets.synthetic_data import *
except ModuleNotFoundError as e:
    import warnings

    warnings.warn(f"Could not initialize one or more datasets: {e}. "
                  f"For using the datasets, consider installing the "
                  f"[datasets] extension via pip.")

cebra.registry.add_docstring(__name__)
