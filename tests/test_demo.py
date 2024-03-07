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
import glob
import re
import sys

import pytest

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
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
    nb = _decrease_max_iterations(nb)
    nb = _change_file_path(nb)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {}})
