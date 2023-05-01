import glob
import re
import sys

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor



def _decrease_max_iterations(nb):
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            lines = cell["source"].split("\n")
            for i in range(len(lines)):
                lines[i] = re.sub(r"max_iterations[ ]*=[ 0-9]+",
                                  "max_iterations=5", lines[i])
            cell["source"] = "\n".join(lines)
    return nb


                lines[i] = re.sub("CURRENT_DIR[ ]*,", "'demo_notebooks',",
                                  lines[i])

@pytest.mark.requires_dataset
@pytest.mark.parametrize("filename", _FILENAMES)
def test_demo_notebook(filename):
    with open(filename) as f:
        nb = nbformat.read(f, as_version=4)
    nb = _decrease_max_iterations(nb)
