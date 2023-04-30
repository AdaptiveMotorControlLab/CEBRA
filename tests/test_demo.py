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
import glob
import re
import sys

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

_FILENAMES = glob.glob("demo_notebooks/Demo_hippocampus.ipynb")


def _decrease_max_iterations(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"].split("\n")
            for i in range(len(lines)):
                lines[i] = re.sub(r"max_iterations[ ]*=[ 0-9]+",
                                  "max_iterations=5", lines[i])
            cell["source"] = "\n".join(lines)
    return nb


def _change_file_path(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"].split("\n")
            for i in range(len(lines)):
                lines[i] = re.sub("CURRENT_DIR[ ]*,", "'demo_notebooks',",
                                  lines[i])
                print(lines[i])
            cell["source"] = "\n".join(lines)
    return nb


@pytest.mark.requires_dataset
@pytest.mark.parametrize("filename", _FILENAMES)
def test_demo_notebook(filename):
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
    nb = _decrease_max_iterations(nb)
    nb = _change_file_path(nb)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {}})
