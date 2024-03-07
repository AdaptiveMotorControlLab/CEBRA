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
import itertools
import pathlib
import pickle
import platform
import tempfile
import unittest
from unittest.mock import patch

import h5py
import hdf5storage
import joblib as jl
import numpy as np
import openpyxl
import pandas as pd
import pytest
import scipy.io
import torch

import cebra.data.load as cebra_load

__test_functions = []
__test_functions_error = []
__test_functions_module_not_found = []


def _skip_hdf5storage(*args, **kwargs):
    pytest.skip(reason=(
        "Likely upstream issue with hdf5storage. "
        "For details, see https://github.com/stes/neural_cl/issues/417"))(
            *args, **kwargs)


def test_imports():
    import cebra

    assert hasattr(cebra, "load_data")


def register(*file_endings, requires=()):
    # for each file format
    def _register(f):
        # f is the filename
        # TODO: test loading for file without extension
        # __test_functions.append(f)
        # with all possible extensions
        __test_functions.extend([(lambda filename, dtype: f(
            filename + "." + file_ending, dtype=dtype), file_ending)
                                 for file_ending in file_endings])
        if len(requires) > 0:
            __test_functions_module_not_found.extend([
                (requires, lambda filename: filename + "." + file_ending, lambda
                 filename, dtype: f(filename + "." + file_ending, dtype=dtype))
                for file_ending in file_endings
            ])
        return f

    return _register


def register_error(*file_endings):
    # for each file format
    def _register(f):
        # f is the filename
        # TODO: test loading for file without extension
        # __test_functions_error.append(f)
        # with all possible extensions
        __test_functions_error.extend([
            lambda filename: f(filename + "." + file_ending)
            for file_ending in file_endings
        ])
        return f

    return _register


##### .NPY #####
@register("npy")
def generate_numpy(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.save(filename, A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("npy")
def generate_numpy_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.save(filename, A)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


# def test_load_numpy_pickle():
#     assert False


##### . NPZ #####
@register("npz")
def generate_numpy_confounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("npz")
def generate_numpy_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("npz")
def generate_numpy_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(filename, key="array")
    return A, loaded_A


@register("npz")
def generate_numpy_second(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, other_data="test", array=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("npz")
def generate_numpy_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    _ = cebra_load.load(filename, key="wrong_array")


@register_error("npz")
def generate_numpy_invalid_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    _ = cebra_load.load(filename, key="other_data")


@register_error("npz")
def generate_numpy_no_array(filename, dtype):
    np.savez(filename, array="test_1", other_data="test_2")
    _ = cebra_load.load(filename)


#### .H5 #####
# TODO: test raise ModuleFoundError for h5py


@register("h5", "hdf", "hdf5", "h", requires=("h5py",))
def generate_h5(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_confounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_second(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data="test")
        hf.create_dataset("dataset_2", data=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    _ = cebra_load.load(filename, key="dataset_3")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_invalid_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_no_array(filename, dtype):
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data="test_1")
        hf.create_dataset("dataset_2", data="test_2")
    _ = cebra_load.load(filename)


@register("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_columns(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_col = A[:, :2]
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A", columns=["a", "b"])
    return A_col, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_multi_dataframe(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_B = pd.DataFrame(np.array(B), columns=["c", "d", "e"])
    df_A.to_hdf(filename, "df_A")
    df_B.to_hdf(filename, "df_B")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_single_dataframe_no_key(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(dtype)
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_multi_dataframe_no_key(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(dtype)
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(dtype)
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_B = pd.DataFrame(np.array(B), columns=["c", "d", "e"])
    df_A.to_hdf(filename, "df_A")
    df_B.to_hdf(filename, "df_B")
    _ = cebra_load.load(filename)


@register("h5", "hdf", "hdf5", "h")
def generate_h5_multicol_dataframe(filename, dtype):
    animals = ["mouse1", "mouse2"]
    keypoints = ["a", "b", "c"]
    data = [[[2, 4, 5], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    A = np.array(data).reshape(2, len(keypoints) * len(animals)).astype(dtype)
    df_A = pd.DataFrame(A,
                        columns=pd.MultiIndex.from_product([animals,
                                                            keypoints]))
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_invalid_key(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(dtype)
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    _ = cebra_load.load(filename, key="df_B")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_invalid_column(filename, dtype):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(dtype)
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    _ = cebra_load.load(filename, key="df_A", columns=["d", "b"])


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_multicol_dataframe_columns(filename, dtype):
    animals = ["mouse1", "mouse2"]
    keypoints = ["a", "b", "c"]
    data = [[[2, 4, 5], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    A = np.array(data).reshape(2, len(keypoints) * len(animals)).astype(dtype)
    df_A = pd.DataFrame(A,
                        columns=pd.MultiIndex.from_product([animals,
                                                            keypoints]))
    df_A.to_hdf(filename, "df_A")
    _ = cebra_load.load(filename, key="df_A", columns=["a", "b"])


#### .PT ####
# TODO: test pytorch model is not loaded
# def test_load_torch_model():
#     assert False


@register("pt", "pth")
def generate_torch(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_tensor = torch.tensor(A)
    torch.save(A_tensor, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pt", "pth")
def generate_torch_cofounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pt", "pth")
def generate_torch_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("pt", "pth")
def generate_torch_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register_error("pt", "pth")
def generate_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    _ = cebra_load.load(filename, key="C")


#### .CSV ####
@register("csv", requires=("pandas",))
def generate_csv(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    pd.DataFrame(A).to_csv(filename, header=False, index=False, sep=",")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("csv")
def generate_csv_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    pd.DataFrame(A).to_csv(filename, header=False, index=False, sep=",")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register_error("csv")
def generate_csv_empty_file(filename, dtype):
    with open(filename, "w") as creating_new_csv_file:
        pass
    _ = cebra_load.load(filename)


#### EXCEL ####
@register("xls", "xlsx", "xlsm", requires=("pandas", "pd"))
# TODO(celia): add the following extension:  "xlsb", "odf", "ods", "odt",
# issue to create the files
def generate_excel(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_df = pd.DataFrame(A)
    A_df.to_excel(filename, index=False, header=False)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_cofounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(filename, key="sheet_1")
    return A, loaded_A


@register_error("xls", "xlsx", "xlsm")
def generate_excel_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500, dtype=dtype).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    _ = cebra_load.load(filename, key="sheet_3")


@register_error("xls", "xlsx", "xlsm")
def generate_excel_empty_file(filename, dtype):
    workbook = openpyxl.Workbook()
    workbook.save(filename)
    _ = cebra_load.load(filename)


#### .JL ####
@register("jl")
def generate_joblib(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump(A, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_cofounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    print(filename)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("jl")
def generate_joblib_second(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump({"B": "test", "A": A}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register_error("jl")
def generate_joblib_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    _ = cebra_load.load(filename, key="C")


@register_error("jl")
def generate_joblib_invalid_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    _ = cebra_load.load(filename, key="B")


@register_error("jl")
def generate_joblib_no_array(filename, dtype):
    jl.dump({"A": "test_1", "B": "test_2"}, filename)
    _ = cebra_load.load(filename)


#### .PKL ####
@register("pkl", "p")
def generate_pickle(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump(A, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_cofounder(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_path(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_second(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"B": "test", "A": A}, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("pkl", "p")
def generate_pickle_wrong_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    _ = cebra_load.load(filename, key="C")


@register_error("pkl", "p")
def generate_pickle_invalid_key(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    _ = cebra_load.load(filename, key="B")


@register_error("pkl", "p")
def generate_pickle_no_array(filename, dtype):
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": "test_1", "B": "test_2"}, f)
    _ = cebra_load.load(filename)


#### DLC ####
# @register()
# def generate_load_dlc(filename):
#     assert False


#### .MAT ####
@register("mat")
def generate_mat_old(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("mat")
def generate_mat_old_path(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("mat")
def generate_mat_old_key(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("mat")
def generate_mat_old_second(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    print(dtype)
    A_dict = {"label": "test", "dataset_1": A}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("mat")
def generate_mat_old_wrong_key(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("mat")
def generate_mat_old_invalid_key(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename, key="label")


@register_error("mat")
def generate_mat_old_no_array(filename, dtype):
    """Older matplotlib arrays have their own format."""
    A_dict = {"dataset_1": "test_1", "dataset_2": "test_2"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename)


@register("mat")
def generate_mat_new(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("mat")
def generate_mat_new_path(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("mat")
def generate_mat_new_key(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("mat")
def generate_mat_new_second(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"label": "test", "dataset_1": A.T}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("mat")
def generate_mat_new_wrong_key(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("mat")
def generate_mat_new_invalid_key(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000, dtype=dtype).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename, key="label")


@register_error("mat")
def generate_mat_new_no_array(filename, dtype):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A_dict = {"dataset_1": "test_1", "label": "test_2"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename)


test_functions = list(itertools.product(__test_functions, [np.int32, np.int64]))
test_functions = [(*t, x) for t, x in test_functions]


@pytest.mark.parametrize("save_data,file_ending,dtype", test_functions)
def test_load(save_data, file_ending, dtype):
    with tempfile.NamedTemporaryFile() as tf:
        filename = tf.name  # name, without extension

    if file_ending in ("csv", "xls", "xlsx", "xlsm"):
        if dtype == np.int32:
            pytest.skip(
                "Skipping test. For CSV, XLS, XLSX, and XLM file formats, "
                "the integer loading data type is always int64, regardless of the "
                "data type it was saved with. This can lead to compatibility issues, "
                "especially on Windows. To ensure accurate testing, we only perform "
                "tests with int64 data type for these formats, and we skip the test "
                "cases involving int32.")

    # create data, save it, load it
    saved_array, loaded_array = save_data(filename, dtype=dtype)

    assert isinstance(loaded_array, np.ndarray)
    assert loaded_array.dtype == saved_array.dtype
    assert loaded_array.shape == saved_array.shape
    assert np.allclose(saved_array, loaded_array)


@pytest.mark.parametrize("save_data", __test_functions_error)
def test_load_error(save_data):
    with tempfile.NamedTemporaryFile() as tf:
        filename = tf.name  # name, without extension

    with pytest.raises((AttributeError, TypeError)):
        save_data(filename)


test_functions_module_not_found = list(
    itertools.product(__test_functions_module_not_found, [np.int32, np.int64]))
test_functions_module_not_found = [
    (*t, x) for t, x in test_functions_module_not_found
]


@pytest.mark.parametrize("module_names,get_path,save_data,dtype",
                         test_functions_module_not_found)
def test_module_not_installed(module_names, get_path, save_data, dtype):

    assert len(module_names) > 0
    assert isinstance(module_names, tuple)

    with tempfile.NamedTemporaryFile() as tf:
        filename = tf.name

    saved_array, loaded_array = save_data(filename, dtype=dtype)
    assert np.allclose(saved_array, loaded_array)

    # TODO(stes): Sketch for a test --- needs additional work.
    # with patch.dict('sys.modules', {module: None for module in module_names}):
    #    path = get_path(filename)
    #    with pytest.raises(ModuleNotFoundError, match="cebra[datasets]"):
    #        cebra.data.load.load(path)
