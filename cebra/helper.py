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
"""Collection of helper functions that did not fit into own modules."""

import io
import pathlib
import tempfile
import urllib
import zipfile
from typing import List

import requests

import cebra.data


def get_loader_options(dataset: cebra.data.Dataset) -> List[str]:
    """Return all possible dataloaders for the given dataset."""

    loader_options = []
    if isinstance(dataset, cebra.data.SingleSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(cebra.data.ContinuousDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            loader_options.append(cebra.data.DiscreteDataLoader)
        else:
            mixed = False
        if mixed:
            loader_options.append(cebra.data.MixedDataLoader)
    elif isinstance(dataset, cebra.data.MultiSessionDataset):
        mixed = True
        if dataset.continuous_index is not None:
            loader_options.append(cebra.data.ContinuousMultiSessionDataLoader)
        else:
            mixed = False
        if dataset.discrete_index is not None:
            pass  # not implemented yet
        else:
            mixed = False
        if mixed:
            pass  # not implemented yet
    else:
        raise TypeError(f"Invalid dataset type: {dataset}")
    return loader_options


def download_file_from_url(url: str) -> str:
    """Download a fole from ``url``.

    Args:
        url: Url to fetch for the file.

    Returns:
        The path to the downloaded file.
    """
    with tempfile.NamedTemporaryFile() as tf:
        filename = tf.name + ".h5"
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return filename


def download_file_from_zip_url(url, file="montblanc_tracks.h5"):
    """Directly extract files without writing the archive to disk."""
    with tempfile.TemporaryDirectory() as tf:
        foldername = tf

    resp = urllib.request.urlopen(url)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        for member in zf.infolist():
            try:
                zf.extract(member, path=foldername)
            except zipfile.error:
                pass
    return pathlib.Path(foldername) / "data" / file
