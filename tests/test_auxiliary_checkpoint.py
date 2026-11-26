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
"""Auxiliary solver checkpoints preserve reference models and DCL dynamics."""

import copy
import itertools

import numpy as np
import pytest
import torch

import cebra
import cebra.models
import cebra.solver


def _make_auxiliary_solver(multisession):

    def make_encoder():
        return torch.nn.Sequential(torch.nn.Linear(5, 3),
                                   torch.nn.BatchNorm1d(3))

    if multisession:
        model = torch.nn.ModuleList([make_encoder(), make_encoder()])
        reference_model = torch.nn.ModuleList([make_encoder(), make_encoder()])
        solver_class = cebra.solver.MultiSessionAuxVariableSolver
    else:
        model = make_encoder()
        reference_model = make_encoder()
        solver_class = cebra.solver.SingleSessionAuxVariableSolver
    criterion = cebra.models.InfoNCE()
    optimizer = torch.optim.Adam(
        itertools.chain(model.parameters(), reference_model.parameters(),
                        criterion.parameters()))
    kwargs = {} if multisession else {"reference_model": reference_model}
    solver = solver_class(model=model,
                          criterion=criterion,
                          optimizer=optimizer,
                          tqdm_on=False,
                          **kwargs)
    if multisession:
        solver.reference_model = reference_model
    return solver


@pytest.mark.parametrize("multisession", [False, True])
def test_auxiliary_reference_round_trip(multisession):
    torch.manual_seed(4)
    solver = _make_auxiliary_solver(multisession)
    restored = _make_auxiliary_solver(multisession)
    inputs = torch.randn(8, 5)
    models = list(solver.model) if multisession else [solver.model]
    references = (list(solver.reference_model)
                  if multisession else [solver.reference_model])
    for reference in references:
        reference(inputs)  # Update the reference encoder's running statistics.
    solver.model.eval()
    solver.reference_model.eval()
    restored.model.eval()
    restored.reference_model.eval()
    restored_references = (list(restored.reference_model)
                           if multisession else [restored.reference_model])
    for reference, other in zip(references, restored_references):
        assert not torch.allclose(reference(inputs), other(inputs))

    restored.load_state_dict(copy.deepcopy(solver.state_dict()))
    restored_models = list(restored.model) if multisession else [restored.model]
    for model, other in zip(models, restored_models):
        torch.testing.assert_close(model(inputs), other(inputs))
    for reference, other in zip(references, restored_references):
        torch.testing.assert_close(reference(inputs), other(inputs))
        for name, value in reference.named_buffers():
            torch.testing.assert_close(value, dict(other.named_buffers())[name])


@pytest.mark.parametrize("multisession", [False, True])
def test_auxiliary_legacy_checkpoint(multisession):
    solver = _make_auxiliary_solver(multisession)
    restored = _make_auxiliary_solver(multisession)
    legacy = copy.deepcopy(solver.state_dict())
    del legacy["reference_model"]
    before = copy.deepcopy(restored.reference_model.state_dict())
    with pytest.raises(KeyError, match="missing reference_model"):
        restored.load_state_dict(legacy)
    with pytest.warns(UserWarning, match="Legacy auxiliary"):
        restored.load_state_dict(legacy, strict=False)
    for name, value in before.items():
        torch.testing.assert_close(value,
                                   restored.reference_model.state_dict()[name])
    for name, value in solver.model.state_dict().items():
        torch.testing.assert_close(value, restored.model.state_dict()[name])


@pytest.mark.parametrize("strict", [False, True])
def test_auxiliary_reference_parameter_strictness(strict):
    solver = _make_auxiliary_solver(multisession=False)
    checkpoint = copy.deepcopy(solver.state_dict())
    del checkpoint["reference_model"]["0.weight"]
    if strict:
        with pytest.raises(RuntimeError, match="Missing key.*0.weight"):
            solver.load_state_dict(checkpoint)
    else:
        solver.load_state_dict(checkpoint, strict=False)


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
    assert loaded.solver_.reference_model[0] is loaded.solver_.model
    assert loaded.solver_.reference_model[1] is loaded.dynamics_model_
    np.testing.assert_allclose(model.transform(data), loaded.transform(data))
    state = torch.randn(7, 3)
    torch.testing.assert_close(model.dynamics_model_(state),
                               loaded.dynamics_model_(state))
    before = loaded.dynamics_model_(state).detach().clone()
    loaded.partial_fit(data)
    assert torch.isfinite(loaded.dynamics_model_(state)).all()
    assert not torch.equal(before, loaded.dynamics_model_(state))
    legacy = model.solver_.state_dict()
    del legacy["reference_model"]
    before = loaded.dynamics_model_(state).detach().clone()
    with pytest.raises(KeyError, match="missing reference_model"):
        loaded.solver_.load_state_dict(legacy)
    with pytest.warns(UserWarning, match="Legacy auxiliary"):
        loaded.solver_.load_state_dict(legacy, strict=False)
    torch.testing.assert_close(loaded.dynamics_model_(state), before)
    np.testing.assert_allclose(model.transform(data), loaded.transform(data))
