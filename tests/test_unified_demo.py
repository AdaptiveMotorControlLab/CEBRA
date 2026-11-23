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
import types

import numpy as np
import pytest

from cebra.datasets.allen import unified_demo


def test_split_sessions():
    neural = np.arange(2 * 5 * 7).reshape(2, 5, 7)
    parts = unified_demo.split_sessions(neural, [3, 4])
    assert [p.shape for p in parts] == [(2, 5, 3), (2, 5, 4)]
    np.testing.assert_array_equal(np.concatenate(parts, axis=-1), neural)
    with pytest.raises(ValueError):
        unified_demo.split_sessions(neural, [3, 3])


def test_wrapped_window_indices():
    offset = types.SimpleNamespace(left=2, right=3)
    idx = unified_demo.wrapped_window_indices(0, 4, 10, offset)
    np.testing.assert_array_equal(idx, [8, 9, 0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        unified_demo.wrapped_window_indices(4, 4, 10, offset)


def test_default_cache_in_data_dir():
    assert unified_demo.DEFAULT_CACHE.name == "allen_demo_cache.npz"
    assert unified_demo.DEFAULT_CACHE.parent.name == "allen"
