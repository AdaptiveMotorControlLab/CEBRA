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
import tempfile

import numpy as np
import pandas as pd
import pytest

import cebra.helper
import cebra.integrations.deeplabcut as cebra_dlc
from cebra import CEBRA
from cebra import load_data

ANNOTATED_DLC_URL = "https://github.com/DeepLabCut/DeepLabCut/blob/main/examples/Reaching-Mackenzie-2018-08-30/labeled-data/reachingvideo1/CollectedData_Mackenzie.h5?raw=true"
MULTISESSION_PRED_DLC_URL = (
    "https://github.com/DeepLabCut/UnitTestData/raw/main/data.zip")

MULTISESSION_PRED_KEYPOINTS = ["head", "tail"]
ANNOTATED_KEYPOINTS = ["Hand", "Tongue"]


def test_imports():
    import cebra

    assert hasattr(cebra, "load_deeplabcut")


def _load_dlc_dataframe(filename):
    try:
        df = pd.read_hdf(filename, "df_with_missing")
    except KeyError:
        df = pd.read_hdf(filename)
    return df


def _get_annotated_data(url, keypoints):
    return (cebra.helper.download_file_from_url(url), keypoints)


def _add_likelihood_columns(df, scorer):
    """Add the likelihood column manually"""
    dfs = []
    cols = df.columns.get_level_values("bodyparts").unique()
    for col in cols:
        tmp = df.loc[:, df.columns.get_level_values("bodyparts") == col].copy(
            deep=True)
        tmp.loc[:, (scorer, col,
                    "likelihood")] = np.random.random(size=(tmp.shape[0],))
        dfs.append(tmp)
    return pd.concat(dfs, axis=1)


def _get_predicted_data(url, keypoints):
    annotated_filename, _ = _get_annotated_data(url, keypoints)

    df = _load_dlc_dataframe(annotated_filename)
    scorer = df.columns.get_level_values("scorer")[0]

    with tempfile.NamedTemporaryFile() as tf:
        pred_filename = tf.name + ".h5"

    new_df = _add_likelihood_columns(df, scorer)
    new_df.to_hdf(pred_filename, format="table", key="df_with_missing")
    return (pred_filename, keypoints)


def _get_dlc_files():
    return [
        _get_annotated_data(ANNOTATED_DLC_URL, ANNOTATED_KEYPOINTS),
        _get_predicted_data(ANNOTATED_DLC_URL, ANNOTATED_KEYPOINTS),
    ]


def read_data(filename):
    df = _load_dlc_dataframe(filename)
    bodyparts = df.columns.get_level_values("bodyparts").unique().to_list()
    scorer = df.columns.get_level_values("scorer")[0]
    if "likelihood" in df.columns.get_level_values("coords").unique().to_list():
        df = df.drop("likelihood", axis=1, level=2)
    return df, bodyparts, scorer


### load full dlc file
@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_full_dlc(filename, keypoints):
    df, _, _ = read_data(filename)

    saved_array = df.values
    loaded_array = cebra_dlc.load_deeplabcut(filename)
    assert isinstance(loaded_array, np.ndarray)
    assert loaded_array.dtype == saved_array.dtype
    assert not np.isnan(loaded_array).any()
    assert saved_array.shape[1] == loaded_array.shape[1]


@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_data_full_dlc(filename, keypoints):
    df, _, _ = read_data(filename)

    saved_array = df.values
    loaded_array = load_data(filename)
    assert isinstance(loaded_array, np.ndarray)
    assert loaded_array.dtype == saved_array.dtype
    assert not np.isnan(loaded_array).any()
    assert saved_array.shape[1] == loaded_array.shape[1]


### load some columns/keypoints
@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_core_dlc(filename, keypoints):
    df, _, _ = read_data(filename)
    df = df.iloc[:, df.columns.get_level_values("bodyparts").isin(keypoints)]

    saved_array = df.values
    loaded_array = cebra_dlc.load_deeplabcut(filename, keypoints=keypoints)

    assert isinstance(loaded_array, np.ndarray)
    assert loaded_array.dtype == saved_array.dtype
    assert not np.isnan(loaded_array).any()
    assert saved_array.shape[1] == loaded_array.shape[1]


@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_data_core_dlc(filename, keypoints):
    df, _, _ = read_data(filename)
    df = df.iloc[:, df.columns.get_level_values("bodyparts").isin(keypoints)]

    saved_array = df.values
    loaded_array = load_data(filename, columns=keypoints)

    assert isinstance(loaded_array, np.ndarray)
    assert loaded_array.dtype == saved_array.dtype
    assert not np.isnan(loaded_array).any()
    assert saved_array.shape[1] == loaded_array.shape[1]


### invalid columns/keypoints
@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_dlc_invalid_keypoints(filename, keypoints):
    with pytest.raises(AttributeError):
        _ = cebra_dlc.load_deeplabcut(filename, keypoints=["Hand", "Finger2"])


@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_data_dlc_invalid_keypoints(filename, keypoints):
    with pytest.raises(AttributeError):
        _ = load_data(filename, columns=["Hand", "Finger2"])


### multi-animals
def test_multianimal_dlc_file():
    filename = cebra.helper.download_file_from_zip_url(
        url=MULTISESSION_PRED_DLC_URL)
    with pytest.raises(NotImplementedError, match="Multi-animals.*"):
        _ = cebra_dlc.load_deeplabcut(filename)


def test_multianimal_data_dlc_file():
    filename = cebra.helper.download_file_from_zip_url(
        url=MULTISESSION_PRED_DLC_URL)
    with pytest.raises(NotImplementedError, match="Multi-animals.*"):
        _ = load_data(filename)


### load dlc file integration test
@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_dlc_integration(filename, keypoints):
    loaded_array = cebra_dlc.load_deeplabcut(filename)

    model = CEBRA(max_iterations=10, output_dimension=3)
    model.fit(loaded_array)

    embedding = model.transform(loaded_array)
    assert isinstance(loaded_array, np.ndarray)
    assert embedding.shape == (loaded_array.shape[0], 3)


@pytest.mark.parametrize("filename, keypoints", _get_dlc_files())
def test_load_data_dlc_integration(filename, keypoints):
    loaded_array = load_data(filename)

    model = CEBRA(max_iterations=10, output_dimension=3)
    model.fit(loaded_array)

    embedding = model.transform(loaded_array)
    assert isinstance(loaded_array, np.ndarray)
    assert embedding.shape == (loaded_array.shape[0], 3)
