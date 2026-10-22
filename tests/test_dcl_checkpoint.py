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
"""DCL checkpoints preserve dynamics as well as encoder parameters."""

import numpy as np
import pytest
import torch

import cebra


@pytest.mark.parametrize("architecture", ["linear", "orthogonal-linear"])
@pytest.mark.parametrize("batch_size", [8, None])
def test_existing_dcl_fit_save_load_and_resume(tmp_path, architecture,
                                               batch_size):
    from cebra.integrations.sklearn.cebra import DCL
    torch.manual_seed(4)
    data = np.random.default_rng(4).normal(size=(100, 5)).astype("float32")
    model = DCL(device="cpu",
                dynamics_model_architecture=architecture,
                output_dimension=3,
                batch_size=batch_size,
                batch_size_negatives=11,
                max_iterations=2,
                distance="euclidean")
    model.fit(data)
    path = tmp_path / "flow.pt"
    model.save(path, backend="sklearn")
    loaded = cebra.CEBRA.load(path, backend="sklearn")
    np.testing.assert_allclose(model.transform(data), loaded.transform(data))
    state = torch.randn(7, 3)
    torch.testing.assert_close(model.dynamics_model_(state),
                               loaded.dynamics_model_(state))
    before = loaded.dynamics_model_(state).detach().clone()
    loaded.partial_fit(data)
    assert torch.isfinite(loaded.dynamics_model_(state)).all()
    assert not torch.equal(before, loaded.dynamics_model_(state))
    legacy = model.solver_.state_dict()
    del legacy["dynamics_model"]
    with pytest.raises(KeyError, match="missing dynamics_model"):
        loaded.solver_.load_state_dict(legacy)
    with pytest.warns(UserWarning, match="Legacy DCL"):
        loaded.solver_.load_state_dict(legacy, strict=False)
