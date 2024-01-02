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
"""A simple API for loading various data formats used with CEBRA.

Availability of different data formats depends on the installed
dependencies. If a dependency is not installed, an attempt to load
a file of that format will throw an error with further installation
instructions.

Currently available formats:

- HDF5 via ``h5py``
- Pickle files via ``pickle``
- Joblib files via ``joblib``
- Various dataframe formats via ``pandas``.
- Matlab files via ``scipy.io.loadmat``
- DeepLabCut (single animal) files via ``deeplabcut``
"""

import abc
import pathlib
import warnings
from typing import IO, List, Optional, Union

import numpy as np
import numpy.typing as npt
import torch

_IS_H5PY_AVAILABLE = True
try:
    import h5py
except ModuleNotFoundError:
    _IS_H5PY_AVAILABLE = False
    warnings.warn(
        "h5py module was not found, be sure it is installed in your env.",
        ImportWarning)

_IS_PANDAS_AVAILABLE = True
try:
    import pandas as pd
except ModuleNotFoundError:
    _IS_PANDAS_AVAILABLE = False
    warnings.warn(
        "pandas module was not found, be sure it is installed in your env.",
        ImportWarning,
    )

_IS_JOBLIB_AVAILABLE = True
try:
    import joblib as jl
except ModuleNotFoundError:
    _IS_JOBLIB_AVAILABLE = False
    warnings.warn(
        "joblib module was not found, be sure it is installed in your env.",
        ImportWarning,
    )

_IS_PICKLE_AVAILABLE = True
try:
    import pickle
except ModuleNotFoundError:
    _IS_PICKLE_AVAILABLE = False
    warnings.warn(
        "pickle module was not found, be sure it is installed in your env.",
        ImportWarning,
    )

_IS_SCIPY_AVAILABLE = True
try:
    import scipy.io
except ModuleNotFoundError:
    _IS_SCIPY_AVAILABLE = False
    warnings.warn(
        "scipy module was not found, be sure it is installed in your env.",
        ImportWarning,
    )

_IS_DLC_INTEGRATION_AVAILABLE = True
try:
    if _IS_PANDAS_AVAILABLE:
        from cebra.integrations.deeplabcut import load_deeplabcut
except ModuleNotFoundError:
    _IS_DLC_INTEGRATION_AVAILABLE = False
    warnings.warn(
        "DLC integration module was not found, be sure it is installed in your env.",
        ImportWarning,
    )


def _module_not_found_error(module_name):
    return ModuleNotFoundError(
        f"Could not load {module_name}. You can manually install {module_name} "
        "or install the [datasets] dependency in cebra: "
        "pip install 'cebra[datasets]'")


class _BaseLoader(abc.ABC):
    """Base loader."""

    @abc.abstractmethod
    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        """Load data from file, at key.

        Args:
            file: The path to the given file to load, in a supported format.
            key: The key referencing the data of interest in the file, if the file has a dictionary-like structure.
            columns: The part of the data to keep in the output 2D-array. For now, it corresponds to the columns of
                a DataFrame to keep if the data selected is a DataFrame.

        Returns:
            The loaded data.
        """
        raise NotImplementedError()


class _NumpyLoader(_BaseLoader):
    """Loader for numpy files.

    Supports ``.npy``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        try:
            loaded_array = np.load(file)
        except ValueError:
            loaded_array = np.load(file, allow_pickle=True)
        return loaded_array


class _NumpyZipLoader(_NumpyLoader):
    """Loader for zipped numpy files.

    Supports ``.npz``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        loaded_zip = _NumpyLoader.load(file, key)
        if key is not None:
            if key in loaded_zip.files:
                if (isinstance(loaded_zip[key], np.ndarray) and
                        "str" not in loaded_zip[key].dtype.name
                   ):  # check that key is valid
                    loaded_array = loaded_zip[key]
                else:
                    raise AttributeError(
                        f"key={key} does not correspond to a valid np.array field of your .npz file."
                    )
            else:
                raise AttributeError(
                    f"key={key} is not a field of your .npz file.")
        else:  # take the first array of the dict
            found_array = False
            for key in list(loaded_zip.keys()):
                if "str" not in loaded_zip[key].dtype.name:
                    loaded_array = loaded_zip[key]
                    found_array = True
                    break
            if not found_array:
                raise AttributeError("No valid array was found in your file.")
        return loaded_array


class _H5pyLoader(_BaseLoader):
    """Loader for HDF5 files.

    Supports ``.h5``, ``.h``, ``.hdf``, ``.hdf5`` as well as the .h5 output files from DLC.
    """

    def load(
        filename: Union[str, pathlib.Path],
        key: Optional[Union[str, int]],
        columns: Optional[list],
    ) -> npt.NDArray:
        if _IS_H5PY_AVAILABLE:
            with h5py.File(filename, "r") as loaded_h5_file:
                # get keys in the file
                df_keys, array_keys = _H5pyLoader._get_keys(loaded_h5_file, key)

                # if the file contains pd.DataFrame and
                # the required key is not an array in the file
                if df_keys and key not in array_keys:
                    # if the pd.DataFrame is a DLC prediction output
                    if key is None and _H5pyLoader._is_dlc_df(
                            filename, df_keys):
                        if _IS_DLC_INTEGRATION_AVAILABLE:
                            loaded_array = load_deeplabcut(filepath=filename,
                                                           keypoints=columns)
                        else:
                            raise ModuleNotFoundError(
                                "DLC integration could not be loaded. "
                                "Most likely, this is because you do not have all "
                                "integrations dependencies installed. Try installing "
                                "cebra with the [integrations] and [datasets] dependency to fix this "
                                "error. You might need to re-start your environment "
                                "after installing: "
                                "pip install 'cebra[integrations,datasets]'.")
                    # if the provided key is valid
                    elif key in df_keys:
                        loaded_array = _PandasLoader.load_from_h5(
                            filename, key, columns)
                    # if no key provided but only one data structure in the file
                    elif len(df_keys) == 1 and not array_keys:
                        loaded_array = _PandasLoader.load_from_h5(
                            filename, df_keys[0], columns)
                    else:
                        raise AttributeError(
                            f"Data in a pd.DataFrame format, please provide the corresponding key, "
                            f"we expect a value from {df_keys}.")
                # if there is no pd.DataFrame or the key corresponds to a numpy.array
                # it will either take the data structure associated with ``key`` or take
                # the first numpy.array found in the data structure
                elif array_keys:
                    loaded_array = loaded_h5_file.get(array_keys[0])[:]
                else:
                    raise AttributeError(
                        "No valid data structure was found in your file.")
        else:
            raise _module_not_found_error("h5py")
        return loaded_array

    @staticmethod
    def _is_df(h5_file: IO[bytes], key: str) -> bool:
        """Check if the ``.h5`` file ``h5_file`` has a format valid to contain a :py:class:`pandas.DataFrame`.

        Args:
            h5_file: A ``.h5`` file to get the keys from.
            key: The key to fetch if it is contained in ``h5_file``.

        Returns:
            True if ``h5_file`` has a valid format to contain a :py:class:`pandas.DataFrame`.
        """
        return isinstance(h5_file.get(key), h5py._hl.group.Group)

    @staticmethod
    def _is_dlc_df(h5_file: IO[bytes], df_keys: List[str]) -> bool:
        """Check if the ``.h5`` file ``h5_file`` has a :py:class:`pandas.DataFrame` table format corresponding
        to the DLC output prediction file format.

        Args:
            h5_file: A ``.h5`` file to check the format.

        Returns:
            True if ``h5_file`` has a valid format to be a DLC output prediction file format.
        """
        try:
            if ["_i_table", "table"] in df_keys:
                df = pd.read_hdf(h5_file, key="table")
            else:
                df = pd.read_hdf(h5_file, key=df_keys[0])
        except KeyError:
            df = pd.read_hdf(h5_file)
        return all(value in df.columns.names
                   for value in ["scorer", "bodyparts", "coords"])

    @staticmethod
    def _is_2D_array(h5_file: IO[bytes], key: str) -> bool:
        """Check if the ``.h5`` file ``h5_file`` has a format valid to contain a :py:func:`numpy.array`.

        Args:
            h5_file: A ``.h5`` file to get the keys from.
            key: The key to fetch if it is contained in ``h5_file``.

        Returns:
            True if ``h5_file`` has a valid format to contain a :py:func:`numpy.array`.
        """
        return (h5_file.get(key).dtype != "object" and
                "u" not in h5_file.get(key).dtype.name)

    @staticmethod
    def _get_keys(h5_file: IO[bytes], key: Optional[str]) -> List[str]:
        """Returns either all the keys contained in the ``.h5`` file ``h5_file`` or ``key`` if contained in ``h5_file``.

        Args:
            h5_file: A ``.h5`` file to get the keys from.
            key: The key to return if it is contained in ``h5_file``.

        Returns:
            A list containing all keys or the key provided as parameter or is empty if the provided key is
            not contained in the :py:class:`pandas.DataFrame`.
        """
        df_keys = []
        array_keys = []
        if key is None:
            for k in list(h5_file.keys()):
                if _H5pyLoader._is_df(h5_file, k):
                    df_keys.append(k)
                elif _H5pyLoader._is_2D_array(h5_file, k):
                    array_keys.append(k)
        elif _H5pyLoader._is_df(h5_file, key):
            df_keys.append(key)
        elif _H5pyLoader._is_2D_array(h5_file, key):
            array_keys.append(key)
        else:
            raise AttributeError(
                f"key={key} does not correspond to a valid np.array or pd.DataFrame field of your .h5 file."
                f"Make sure that your key is valid, got {key}")
        return df_keys, array_keys


class _PandasLoader(_BaseLoader):
    """Loader for files containing :py:class:`pandas.DataFrame`."""

    @staticmethod
    def load_from_h5(file: Union[pathlib.Path, str], key: str,
                     columns: Optional[List[str]]) -> npt.NDArray:
        """Load data stored in a :py:class:`pandas.DataFrame` from the ``.h5`` ``file`` provided.

        Args:
            file: The file to fetch in.
            key: The key associated to the data of interest in the ``file``.
            columns: The columns of the :py:class:`pandas.DataFrame` to keep as the data of interest.

        Note:
            If ``columns`` are provided, the :py:class:pandas.DataFrame` should contain a unique level of
            column indexes.

        Returns:
            A :py:func:`numpy.array` containing the data of interest extracted from the :py:class:`pandas.DataFrame`.
        """
        df = pd.read_hdf(file, key=key)
        if columns is None:
            loaded_array = df.values
        elif isinstance(columns, list) and df.columns.nlevels == 1:
            # Check the column validity
            for c in columns:
                if c not in list(df.columns):
                    raise AttributeError(
                        f"{c} is not a valid column of the dataframe contained in your file, expected values from {list(df.columns)}."
                    )
            loaded_array = df[columns].values
        else:
            raise AttributeError(
                f"DataFrame loading is only handled for dataframes with a single column index, got {df.columns.nlevels} columns indexes."
            )
        return loaded_array


class _PyTorchLoader(_BaseLoader):
    """Loader for PyTorch files.

    Supports ``.pt``, ``.p``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        torch_file = torch.load(file)
        if torch.is_tensor(torch_file):
            loaded_array = torch_file.numpy()
        elif type(torch_file) is dict:
            if key is not None:
                if key in torch_file.keys() and torch.is_tensor(
                        torch_file.get(key)[:]):  # check that key is valid
                    loaded_array = torch_file.get(key)[:].numpy()
                else:
                    raise AttributeError(
                        f"key={key} does not correspond to a valid field of your .pt file."
                    )
            else:
                found_array = False
                for key in list(torch_file.keys()):
                    if torch.is_tensor(torch_file.get(key)[:]):
                        loaded_array = torch_file.get(key)[:].numpy()
                        found_array = True
                        break
                if not found_array:
                    raise AttributeError(
                        "No valid array was found in your file.")
        else:
            raise AttributeError(
                f"{file} contains a model. Be sure to provide a tensor or a dict."
            )
        return loaded_array


class _CsvLoader(_BaseLoader):
    """Loader for CSV files.

    Supports ``.csv``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        if _IS_PANDAS_AVAILABLE:
            try:
                loaded_array = pd.read_csv(file, sep=",", header=None).values
            except pd.errors.EmptyDataError:
                raise AttributeError(".csv file is empty.")
        else:
            raise _module_not_found_error("pandas")
        return loaded_array


class _ExcelLoader(_BaseLoader):
    """Loader for Excel files.

    Supports ``.xls``, ``.xlsx``, ``.xlsm``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        if _IS_PANDAS_AVAILABLE:
            loaded_dict = pd.read_excel(file, header=None,
                                        sheet_name=None)  # dict of df
            if (len(list(loaded_dict.keys())) == 1 and
                    "Sheet" in loaded_dict.keys() and
                    loaded_dict["Sheet"].empty):
                raise AttributeError(".xls file is empty.")

            if key is not None:
                if key in loaded_dict.keys():
                    loaded_array = loaded_dict[key].values
                else:
                    raise AttributeError(
                        f"key={key} is not a valid name for a sheet of your excel file."
                    )
            else:  # take the first sheet of the excel file, we know not empty already
                for key in loaded_dict.keys():
                    loaded_array = loaded_dict[key].values
                    break
        else:
            raise _module_not_found_error("pandas")
        return loaded_array

    # def prepare_engine(extension: str):
    #     if extension in [".xls", ".xlsx", ".xlsm"]:
    #         engine = "xlrd"
    #     elif extension == ".xlsb":
    #         engine = "pyxlsb"
    #     elif extension in [".odf", ".ods", ".odt"]:
    #         engine = "odf"
    #     else:
    #         raise AttributeError(
    #             f"{extension} endfile is not a valid extension for excel files."
    #         )
    #     return engine


class _JoblibLoader(_BaseLoader):
    """Loader for JobLib files.

    Supports ``.jl``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        if _IS_JOBLIB_AVAILABLE:
            loaded_data = jl.load(file, mmap_mode="r")

            # NOTE: data will be stored in a np.array upon loading,
            # if originally np.array, then it is directly it
            # else if dict, it will be of format [dict()]
            # so need to do loaded_data[0] to access the dict
            if type(loaded_data) is dict:  # it is in a dict format: [dict]
                if key is not None:
                    if key in loaded_data.keys():
                        if (isinstance(loaded_data[key], np.ndarray) and
                                "str" not in loaded_data[key].dtype.name
                           ):  # check that key is valid
                            loaded_array = loaded_data[key]
                        else:
                            raise AttributeError(
                                f"key={key} does not correspond to a valid field in the .jl file."
                            )
                    else:
                        raise AttributeError(
                            f"key={key} is not a field in your .jl file.")
                else:
                    found_array = False
                    for key in loaded_data.keys():
                        if "str" not in str(type(loaded_data[key])):
                            loaded_array = loaded_data[key]
                            found_array = True
                            break
                    if not found_array:
                        raise AttributeError(
                            "No valid array was found in your file.")
            elif type(np.array(loaded_data.tolist)) is np.ndarray:
                loaded_array = loaded_data
            else:
                # TODO: make it more robust to other types of data in the .jl file
                raise NotImplementedError(
                    f"{type(loaded_data)} is not handled for .jl files.")
        else:
            raise _module_not_found_error("joblib")
        return loaded_array


class _PickleLoader(_BaseLoader):
    """Loader for pickle files.

    Supports ``.pkl``, ``.p``.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        if _IS_PICKLE_AVAILABLE:
            with open(file, "rb") as pk:
                loaded_data = pickle.load(pk)
            if isinstance(loaded_data, np.ndarray):
                loaded_array = loaded_data
            elif type(loaded_data) is dict:
                if key is not None:
                    if (key in loaded_data.keys() and type(loaded_data[key])
                            is np.ndarray):  # check that key is valid
                        loaded_array = loaded_data[key]
                    else:
                        raise AttributeError(
                            f"key={key} does not correspond to a valid np.array field of the dict stored in your .pk file"
                        )
                else:
                    found_array = False
                    for key in loaded_data.keys():
                        if type(loaded_data[key]) is np.ndarray:
                            loaded_array = loaded_data[key]
                            found_array = True
                            break
                    if not found_array:
                        raise AttributeError(
                            "No valid array was found in your file.")

            else:
                # TODO: make it more robust to other types of data in the .pk file
                raise NotImplementedError(
                    f"{type(loaded_data)} is not handled for .pk files.")
        else:
            raise _module_not_found_error("pickle")
        return loaded_array


class _MatfileLoader(_BaseLoader):
    """Loader for MAT-files.

    Supports ``.mat``. Newer and older versions.
    """

    def load(
        file: Union[str, pathlib.Path],
        key: Optional[Union[str, int]] = None,
        columns: Optional[list] = None,
    ) -> npt.NDArray:
        try:
            loaded_mat = scipy.io.loadmat(file)
            if key is not None:
                if (key in loaded_mat.keys() and
                        type(loaded_mat[key]) is np.ndarray and
                        "str" not in loaded_mat[key].dtype.name):
                    loaded_array = loaded_mat[key]
                else:
                    raise AttributeError(
                        f"key={key} does not correspond to a valid np.array field of the dict stored in your .mat file"
                    )
            else:
                found_array = False
                for key in loaded_mat.keys():
                    if (type(loaded_mat[key]) is np.ndarray and
                            "str" not in loaded_mat[key].dtype.name):
                        loaded_array = loaded_mat[key]
                        found_array = True
                        break
                if not found_array:
                    raise AttributeError(
                        "No valid array was found in your file.")
        except NotImplementedError:
            if _IS_H5PY_AVAILABLE:
                loaded_array = _H5pyLoader.load(file, key)
            else:
                raise _module_not_found_error("h5py")
        return loaded_array


# loaders associating every handled file ending to corresponding class
__loaders = {
    ".npy": _NumpyLoader,
    ".npz": _NumpyZipLoader,
    ".h5": _H5pyLoader,
    ".h": _H5pyLoader,
    ".hdf": _H5pyLoader,
    ".hdf5": _H5pyLoader,
    ".pt": _PyTorchLoader,
    ".pth": _PyTorchLoader,
    ".csv": _CsvLoader,
    ".xls": _ExcelLoader,
    ".xlsx": _ExcelLoader,
    ".xlsm": _ExcelLoader,
    # ".xlsb": _ExcelLoader,
    # ".odf": _ExcelLoader,
    # ".ods": _ExcelLoader,
    # ".odt": _ExcelLoader,
    ".jl": _JoblibLoader,
    ".pkl": _PickleLoader,
    ".p": _PickleLoader,
    ".mat": _MatfileLoader,
}


def load(
    file: Union[str, pathlib.Path],
    key: Optional[Union[str, int]] = None,
    columns: Optional[list] = None,
) -> npt.NDArray:
    """Load a dataset from the given file.

    The following file types are supported:
        - Numpy files: npy, npz;
        - HDF5 files: h5, hdf, hdf5, including h5 generated through DLC;
        - PyTorch files: pt, p;
        - csv files;
        - Excel files: xls, xlsx, xlsm;
        - Joblib files: jl;
        - Pickle files: p, pkl;
        - MAT-files: mat.

    The assumptions on your data are as following:
        - it contains at least one data structure (e.g. a numpy array, a torch.Tensor, etc.);
        - it can be directly in the form of a collection (e.g. a dictionary);
        - if the file contains a collection, the user can provide a key to refer to the data value they want to access;
        - if no key is provided, the first data structure found upon iteration of the collection will be loaded;
        - if a key is provided, it needs to correspond to an existing item of the collection;
        - if a key is provided, the data value accessed needs to be a data structure;
        - the function loads data for only one data structure, even if the file contains more. The function can be called again with the corresponding key to get the other ones.

    Args:
        file: The path to the given file to load, in a supported format.
        key: The key referencing the data of interest in the file, if the file has a dictionary-like structure.
        columns: The part of the data to keep in the output 2D-array. For now, it corresponds to the columns of
            a DataFrame to keep if the data selected is a DataFrame.


    Returns:
        The loaded data.

    Example:

        >>> import cebra
        >>> import cebra.helper as cebra_helper
        >>> import numpy as np
        >>> # Create the files to load the data from
        >>> # Create a .npz file
        >>> X = np.random.normal(0,1,(100,3))
        >>> y = np.random.normal(0,1,(100,4))
        >>> np.savez("data", neural = X, trial = y)
        >>> # Create a .h5 file
        >>> url = "https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.h5?raw=true"
        >>> dlc_file = cebra_helper.download_file_from_url(url) # an .h5 example file
        >>> # Load data
        >>> X = cebra.load_data(file="data.npz", key="neural")
        >>> y_trial_id = cebra.load_data(file="data.npz", key="trial")
        >>> y_behavior = cebra.load_data(file=dlc_file, columns=["Hand", "Tongue"])

    """
    file_ending = pathlib.Path(file).suffix
    loader = _get_loader(file_ending)
    data = loader.load(file, key=key, columns=columns)
    return data


def _get_loader(file_ending: str) -> _BaseLoader:
    """Get corresponding class based on handled file ending.

    Args:
        file_ending: The file ending of the handled file.

    Raises:
        OSError: The file ending is not supported.

    Returns:
        The loader class corresponding to the handle file.
    """
    if file_ending not in __loaders.keys() or file_ending == "":
        raise OSError(f"File ending {file_ending} not supported.")
    return __loaders[file_ending]
