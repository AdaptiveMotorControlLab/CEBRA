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
"""Public DYSCO estimator lifecycle and standard CEBRA reconstruction."""

import numpy as np
import pytest
import torch
from sklearn.base import clone

import cebra


@pytest.mark.parametrize("candidate_mode", ["in_batch", "independent"])
@pytest.mark.parametrize("potential", [None, "kde", "mlp"])
@pytest.mark.parametrize("controlled", [False, True])
def test_dysco_estimator_round_trip_and_exact_resume(tmp_path, candidate_mode,
                                                     potential, controlled):
    rng = np.random.default_rng(4)
    trials = [rng.normal(size=(30, 5)).astype("float32") for _ in range(4)]
    controls = [rng.normal(size=(30, 1)).astype("float32") for _ in trials
               ] if controlled else None
    model = cebra.DYSCO(output_dimension=2,
                        num_hidden_units=8,
                        rollout_steps=3,
                        batch_size=3,
                        batch_size_negatives=5,
                        max_iterations=2,
                        potential=potential,
                        candidate_mode=candidate_mode,
                        device="cpu")
    torch_state = torch.random.get_rng_state().clone()
    model.fit(trials, alignment_ids=[0, 0, 1, 1], controls=controls)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert clone(model).get_params() == model.get_params()
    path = tmp_path / "dysco.pt"
    model.save(path)
    restored = cebra.CEBRA.load(path, map_location="cpu", weights_only=True)
    assert isinstance(restored, cebra.DYSCO)
    assert restored.get_params() == model.get_params()
    assert restored.solver_.reference_model[0] is restored.model_
    assert restored.solver_.reference_model[1] is restored.dynamics_model_
    encoded = model.transform(trials[0])
    np.testing.assert_allclose(encoded,
                               restored.transform(trials[0]),
                               atol=1e-7)
    u = None if controls is None else controls[0]
    np.testing.assert_allclose(model.vector_field(encoded, u),
                               restored.vector_field(encoded, u))
    segment = None if u is None else u[:4]
    np.testing.assert_allclose(model.rollout(encoded[0], 3, segment),
                               restored.rollout(encoded[0], 3, segment))
    model.partial_fit(trials, alignment_ids=[0, 0, 1, 1], controls=controls)
    restored.partial_fit(trials, alignment_ids=[0, 0, 1, 1], controls=controls)
    np.testing.assert_allclose(model.transform(trials[0]),
                               restored.transform(trials[0]),
                               atol=1e-7)
    for first, second in zip(model.solver_.criterion.parameters(),
                             restored.solver_.criterion.parameters()):
        torch.testing.assert_close(first, second)
    torch.testing.assert_close(model.dynamics_model_.field.coefficients,
                               restored.dynamics_model_.field.coefficients)
    restored.to(torch.device("cpu"))
    assert restored.solver_.criterion._sampling_state.device.type == "cpu"


def test_dysco_clone_forward_only_and_explicit_metadata():
    trials = [
        np.random.default_rng(i).normal(size=(25, 3)).astype("float32")
        for i in range(2)
    ]
    model = cebra.DYSCO(num_views=1,
                        bidirectional=False,
                        potential=None,
                        max_iterations=1,
                        batch_size=2,
                        rollout_steps=2,
                        device="cpu")
    with pytest.raises(ValueError, match="alignment_ids"):
        model.fit(trials)
    embeddings = model.fit_transform(trials, alignment_ids=[0, 1])
    assert [x.shape for x in embeddings] == [(25, 3), (25, 3)]
    assert model.partial_fit(trials, alignment_ids=[0, 1]) is model
    with pytest.raises(ValueError, match="dimensions"):
        model.partial_fit([np.zeros((25, 4))] * 2, alignment_ids=[0, 1])
    with pytest.raises(ValueError, match="direction"):
        model.rollout(np.zeros(3), 2, direction="invalid")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_dysco_accelerator_and_cpu_checkpoint(tmp_path):
    trials = [
        np.random.default_rng(i).normal(size=(30, 3)).astype("float32")
        for i in range(2)
    ]
    model = cebra.DYSCO(device="cuda",
                        potential="mlp",
                        num_views=1,
                        max_iterations=1,
                        batch_size=2,
                        rollout_steps=2)
    model.fit(trials, alignment_ids=[0, 1])
    path = tmp_path / "accelerator.pt"
    model.save(path)
    restored = cebra.DYSCO.load(path, map_location="cpu", weights_only=True)
    np.testing.assert_allclose(model.transform(trials[0]),
                               restored.transform(trials[0]),
                               atol=1e-5)
    model.to("cpu")
    assert all(
        p.device.type == "cpu" for p in model.solver_.criterion.parameters())
    assert model.solver_.criterion._sampling_state.device.type == "cpu"
