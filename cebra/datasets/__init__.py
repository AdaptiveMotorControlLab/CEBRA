
import os
import pathlib
from typing import Union
import cebra.registry

cebra.registry.add_helper_functions(__name__)


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


def get_datapath(path: str = None) -> str:
    """Convert a relative to a system-dependent absolute data path.

    The data directory is given by ``{root}/{path}``. The root
    directory can be specified through the environment variable

    Args:
            The path is relative to the system data directory.

    Returns:
        The absolute path to the dataset.
    """
    if path is None:
        return get_data_root()
    path = str(path)
    return os.path.join(get_data_root(), path)


import warnings
from cebra.datasets.demo import *
try:
    from cebra.datasets.allen import *
    from cebra.datasets.gaussian_mixture import *
    from cebra.datasets.hippocampus import *
    from cebra.datasets.monkey_reaching import *
except ModuleNotFoundError as e:
    import warnings
    warnings.warn(f"Could not initialize one or more datasets: {e}. "
                  f"For using the datasets, consider installing the "
                  f"[datasets] extension via pip.")

cebra.registry.add_docstring(__name__)
