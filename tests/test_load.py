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
import pathlib
import pickle
import tempfile

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


def _skip_hdf5storage(*args, **kwargs):
    pytest.skip(reason=(
        "Likely upstream issue with hdf5storage. "
        "For details, see https://github.com/stes/neural_cl/issues/417"))(
            *args, **kwargs)


def test_imports():
    import cebra

    assert hasattr(cebra, "load_data")


def register(*file_endings):
    # for each file format
    def _register(f):
        # f is the filename
        # TODO: test loading for file without extension
        # __test_functions.append(f)
        # with all possible extensions
        __test_functions.extend([
            lambda filename: f(filename + "." + file_ending)
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
def generate_numpy(filename):
    A = np.arange(1000).reshape(10, 100)
    np.save(filename, A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("npy")
def generate_numpy_path(filename):
    A = np.arange(1000).reshape(10, 100)
    np.save(filename, A)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


# def test_load_numpy_pickle():
#     assert False


##### . NPZ #####
@register("npz")
def generate_numpy_confounder(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("npz")
def generate_numpy_path(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("npz")
def generate_numpy_key(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    loaded_A = cebra_load.load(filename, key="array")
    return A, loaded_A


@register("npz")
def generate_numpy_second(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, other_data="test", array=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("npz")
def generate_numpy_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    _ = cebra_load.load(filename, key="wrong_array")


@register_error("npz")
def generate_numpy_invalid_key(filename):
    A = np.arange(1000).reshape(10, 100)
    np.savez(filename, array=A, other_data="test")
    _ = cebra_load.load(filename, key="other_data")


@register_error("npz")
def generate_numpy_no_array(filename):
    np.savez(filename, array="test_1", other_data="test_2")
    _ = cebra_load.load(filename)


#### .H5 #####
# TODO: test raise ModuleFoundError for h5py


@register("h5", "hdf", "hdf5", "h")
def generate_h5(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_confounder(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_path(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_second(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data="test")
        hf.create_dataset("dataset_2", data=A)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    _ = cebra_load.load(filename, key="dataset_3")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_invalid_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data=A)
        hf.create_dataset("dataset_2", data="test")
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_no_array(filename):
    with h5py.File(filename, "w") as hf:
        hf.create_dataset("dataset_1", data="test_1")
        hf.create_dataset("dataset_2", data="test_2")
    _ = cebra_load.load(filename)


@register("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_columns(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    A_col = A[:, :2]
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A", columns=["a", "b"])
    return A_col, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_multi_dataframe(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_B = pd.DataFrame(np.array(B), columns=["c", "d", "e"])
    df_A.to_hdf(filename, "df_A")
    df_B.to_hdf(filename, "df_B")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register("h5", "hdf", "hdf5", "h")
def generate_h5_single_dataframe_no_key(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_multi_dataframe_no_key(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_B = pd.DataFrame(np.array(B), columns=["c", "d", "e"])
    df_A.to_hdf(filename, "df_A")
    df_B.to_hdf(filename, "df_B")
    _ = cebra_load.load(filename)


@register("h5", "hdf", "hdf5", "h")
def generate_h5_multicol_dataframe(filename):
    animals = ["mouse1", "mouse2"]
    keypoints = ["a", "b", "c"]
    data = [[[2, 4, 5], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    A = np.array(data).reshape(2, len(keypoints) * len(animals))
    df_A = pd.DataFrame(A,
                        columns=pd.MultiIndex.from_product([animals,
                                                            keypoints]))
    df_A.to_hdf(filename, "df_A")
    loaded_A = cebra_load.load(filename, key="df_A")
    return A, loaded_A


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_invalid_key(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    _ = cebra_load.load(filename, key="df_B")


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_dataframe_invalid_column(filename):
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    df_A = pd.DataFrame(np.array(A), columns=["a", "b", "c"])
    df_A.to_hdf(filename, "df_A")
    _ = cebra_load.load(filename, key="df_A", columns=["d", "b"])


@register_error("h5", "hdf", "hdf5", "h")
def generate_h5_multicol_dataframe_columns(filename):
    animals = ["mouse1", "mouse2"]
    keypoints = ["a", "b", "c"]
    data = [[[2, 4, 5], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]]
    A = np.array(data).reshape(2, len(keypoints) * len(animals))
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
def generate_torch(filename):
    A = np.arange(1000).reshape(10, 100)
    A_tensor = torch.tensor(A)
    torch.save(A_tensor, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pt", "pth")
def generate_torch_cofounder(filename):
    A = np.arange(1000).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pt", "pth")
def generate_torch_path(filename):
    A = np.arange(1000).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("pt", "pth")
def generate_torch_key(filename):
    A = np.arange(1000).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register_error("pt", "pth")
def generate_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    A_tensor = torch.tensor(A)
    B = np.arange(500).reshape(10, 50)
    B_tensor = torch.tensor(B)
    torch.save({"A": A_tensor, "B": B_tensor}, filename)
    _ = cebra_load.load(filename, key="C")


#### .CSV ####
@register("csv")
def generate_csv(filename):
    A = np.arange(1000).reshape(10, 100)
    pd.DataFrame(A).to_csv(filename, header=False, index=False, sep=",")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("csv")
def generate_csv_path(filename):
    A = np.arange(1000).reshape(10, 100)
    pd.DataFrame(A).to_csv(filename, header=False, index=False, sep=",")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register_error("csv")
def generate_csv_empty_file(filename):
    with open(filename, "w") as creating_new_csv_file:
        pass
    _ = cebra_load.load(filename)


#### EXCEL ####
@register("xls", "xlsx", "xlsm")
# TODO(celia): add the following extension:  "xlsb", "odf", "ods", "odt",
# issue to create the files
def generate_excel(filename):
    A = np.arange(1000).reshape(10, 100)
    A_df = pd.DataFrame(A)
    A_df.to_excel(filename, index=False, header=False)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_cofounder(filename):
    A = np.arange(1000).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_path(filename):
    A = np.arange(1000).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("xls", "xlsx", "xlsm")
def generate_excel_key(filename):
    A = np.arange(1000).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    loaded_A = cebra_load.load(filename, key="sheet_1")
    return A, loaded_A


@register_error("xls", "xlsx", "xlsm")
def generate_excel_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    A_df = pd.DataFrame(A)
    B = np.arange(500).reshape(10, 50)
    B_df = pd.DataFrame(B)
    with pd.ExcelWriter(filename) as writer:
        A_df.to_excel(writer, index=False, header=False, sheet_name="sheet_1")
        B_df.to_excel(writer, index=False, header=False, sheet_name="sheet_2")
    _ = cebra_load.load(filename, key="sheet_3")


@register_error("xls", "xlsx", "xlsm")
def generate_excel_empty_file(filename):
    workbook = openpyxl.Workbook()
    workbook.save(filename)
    _ = cebra_load.load(filename)


#### .JL ####
@register("jl")
def generate_joblib(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump(A, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_cofounder(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_path(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("jl")
def generate_joblib_second(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"B": "test", "A": A}, filename)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("jl")
def generate_joblib_key(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register_error("jl")
def generate_joblib_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    _ = cebra_load.load(filename, key="C")


@register_error("jl")
def generate_joblib_invalid_key(filename):
    A = np.arange(1000).reshape(10, 100)
    jl.dump({"A": A, "B": "test"}, filename)
    _ = cebra_load.load(filename, key="B")


@register_error("jl")
def generate_joblib_no_array(filename):
    jl.dump({"A": "test_1", "B": "test_2"}, filename)
    _ = cebra_load.load(filename)


#### .PKL ####
@register("pkl", "p")
def generate_pickle(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump(A, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_cofounder(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_path(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    loaded_A = cebra_load.load(filename, key="A")
    return A, loaded_A


@register("pkl", "p")
def generate_pickle_second(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"B": "test", "A": A}, f)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("pkl", "p")
def generate_pickle_wrong_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    _ = cebra_load.load(filename, key="C")


@register_error("pkl", "p")
def generate_pickle_invalid_key(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": A, "B": "test"}, f)
    _ = cebra_load.load(filename, key="B")


@register_error("pkl", "p")
def generate_pickle_no_array(filename):
    A = np.arange(1000).reshape(10, 100)
    with open(filename, "wb") as f:
        pickle.dump({"A": "test_1", "B": "test_2"}, f)
    _ = cebra_load.load(filename)


#### DLC ####
# @register()
# def generate_load_dlc(filename):
#     assert False


#### .MAT ####
@register("mat")
def generate_mat_old(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("mat")
def generate_mat_old_path(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("mat")
def generate_mat_old_key(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("mat")
def generate_mat_old_second(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"label": "test", "dataset_1": A}
    scipy.io.savemat(filename, A_dict)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("mat")
def generate_mat_old_wrong_key(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("mat")
def generate_mat_old_invalid_key(filename):
    """Older matplotlib arrays have their own format."""
    A = np.arange(1000).reshape(10, 100)
    A_dict = {"dataset_1": A, "label": "test"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename, key="label")


@register_error("mat")
def generate_mat_old_no_array(filename):
    """Older matplotlib arrays have their own format."""
    A_dict = {"dataset_1": "test_1", "dataset_2": "test_2"}
    scipy.io.savemat(filename, A_dict)
    _ = cebra_load.load(filename)


@register("mat")
def generate_mat_new(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register("mat")
def generate_mat_new_path(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(pathlib.Path(filename))
    return A, loaded_A


@register("mat")
def generate_mat_new_key(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename, key="dataset_1")
    return A, loaded_A


@register("mat")
def generate_mat_new_second(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"label": "test", "dataset_1": A.T}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    loaded_A = cebra_load.load(filename)
    return A, loaded_A


@register_error("mat")
def generate_mat_new_wrong_key(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename, key="dataset_2")


@register_error("mat")
def generate_mat_new_invalid_key(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A = np.arange(1000).reshape(10, 100)
    # A transposed as matrices are transposed to be stored in HDF5
    A_dict = {"dataset_1": A.T, "label": "test"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename, key="label")


@register_error("mat")
def generate_mat_new_no_array(filename):
    """Newer matplotlib formats are just h5 files"""
    _skip_hdf5storage()
    A_dict = {"dataset_1": "test_1", "label": "test_2"}
    hdf5storage.write(A_dict, ".", filename, matlab_compatible=True)
    _ = cebra_load.load(filename)


@pytest.mark.parametrize("save_data", __test_functions)
def test_load(save_data):
    with tempfile.NamedTemporaryFile() as tf:
        filename = tf.name  # name, without extension

    # create data, save it, load it
    saved_array, loaded_array = save_data(filename)

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
