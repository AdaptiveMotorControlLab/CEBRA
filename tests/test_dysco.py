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
"""Numerical parity and inherited lifecycle for trajectory contrastive learning."""

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import cebra.data
import cebra.models
import cebra.solver
from cebra.dynamics.polynomial import PolynomialFlow
from cebra.models.dysco import DYSCOInfoNCE
from cebra.models.dysco import KDEPotential
from cebra.models.dysco import MLPPotential


class QuadraticPotential(nn.Module):

    def __init__(self):
        super().__init__()
        self.coefficient = nn.Parameter(torch.tensor(0.15, dtype=torch.float64))

    def forward(self, query, support=None):
        return self.coefficient * query.square().sum(-1, keepdim=True)


@pytest.mark.parametrize("potential_name", ["none", "quadratic", "kde"])
def test_submission_value_and_gradient_parity(potential_name):
    record = json.loads(
        (Path(__file__).parent / "fixtures/dysco_submission.json").read_text())
    case = next(c for c in record["cases"] if c["potential"] == potential_name)
    tensors = [
        torch.tensor(case[key], dtype=torch.float64, requires_grad=True)
        for key in ("encoded", "forward", "backward")
    ]
    potential = None if potential_name == "none" else QuadraticPotential(
    ) if potential_name == "quadratic" else KDEPotential(epsilon=0.35)
    criterion = DYSCOInfoNCE(sample_interval=0.2,
                             temperature=0.7,
                             negative_temperature=1.1,
                             num_positive=2,
                             num_negatives=6,
                             potential=potential).double()
    encoded, forward, backward = tensors
    predictions = torch.stack((forward, backward))
    candidates = torch.stack(tensors).flatten(1, 2)
    indices = {
        key: torch.tensor(value) for key, value in case["indices"].items()
    }
    loss, _, _ = criterion(encoded, predictions, candidates, indices=indices)
    torch.testing.assert_close(loss,
                               torch.tensor(case["loss"], dtype=torch.float64),
                               rtol=1e-10,
                               atol=1e-10)
    variables = tensors + ([potential.coefficient]
                           if potential_name == "quadratic" else [])
    gradients = torch.autograd.grad(loss, variables)
    for value, expected in zip(gradients, case["gradients"]):
        torch.testing.assert_close(value,
                                   torch.tensor(expected, dtype=torch.float64),
                                   rtol=1e-9,
                                   atol=1e-10)


def test_submission_event_sampling_matches_frozen_indices():
    record = json.loads(
        (Path(__file__).parent / "fixtures/dysco_submission.json").read_text())
    case = record["cases"][0]
    encoded = torch.tensor(case["encoded"])
    criterion = DYSCOInfoNCE(num_positive=2, num_negatives=6, seed=17)
    state = torch.random.get_rng_state().clone()
    actual = criterion.sample_indices(encoded, 4)
    assert torch.equal(state, torch.random.get_rng_state())
    for key, expected in case["indices"].items():
        torch.testing.assert_close(actual[key], torch.tensor(expected))


@pytest.mark.parametrize("bidirectional", [True, False])
def test_criterion_finite_difference_gradients(bidirectional):
    torch.manual_seed(5)
    encoded = torch.randn(2, 1, 3, 2, dtype=torch.float64, requires_grad=True)
    predictions = torch.randn(2 if bidirectional else 1,
                              2,
                              1,
                              3,
                              2,
                              dtype=torch.float64,
                              requires_grad=True)
    candidates = torch.randn(3 if bidirectional else 2,
                             2,
                             3,
                             2,
                             dtype=torch.float64,
                             requires_grad=True)
    criterion = DYSCOInfoNCE(sample_interval=0.4,
                             potential=KDEPotential(0.8),
                             bidirectional=bidirectional).double()
    indices = criterion.sample_indices(encoded, 2)
    # Support is explicitly detached; finite differences hold it fixed by using
    # a potential with externally frozen support during this audit.
    support = encoded.detach().reshape(-1, 2)[indices["support"]].clone()

    class FrozenKDE(KDEPotential):

        def forward(self, query, support=None):
            return super().forward(query, self.support)

    potential = FrozenKDE(0.8)
    potential.support = support
    criterion.potential = potential
    assert torch.autograd.gradcheck(
        lambda x, y, z: criterion(x, y, z, indices=indices)[0],
        (encoded, predictions, candidates),
        atol=1e-5)


@pytest.mark.parametrize("architecture", ["offset1-model", "offset10-model"])
@pytest.mark.parametrize("candidate_mode", ["in_batch", "independent"])
@pytest.mark.parametrize("controlled", [True, False])
def test_solver_inherited_training_and_checkpoint(architecture, candidate_mode,
                                                  controlled):
    torch.manual_seed(3)
    dataset = cebra.data.AlignedTrialDataset(
        [torch.randn(40, 5) for _ in range(4)], [0, 0, 1, 1],
        0.02,
        controls=[torch.randn(40, 1) for _ in range(4)] if controlled else None)
    model = cebra.models.init(architecture,
                              num_neurons=5,
                              num_units=8,
                              num_output=2)
    dataset.configure_for(model)
    loader = cebra.data.AlignedTrialLoader(dataset=dataset,
                                           num_steps=2,
                                           batch_size=3,
                                           batch_size_negatives=5,
                                           rollout_steps=3,
                                           candidate_mode=candidate_mode)
    flow = PolynomialFlow(2, control_dim=int(controlled))
    criterion = DYSCOInfoNCE(candidate_mode=candidate_mode,
                             num_negatives=5,
                             potential=MLPPotential(2, 8))
    parameters = list(model.parameters()) + list(flow.parameters()) + list(
        criterion.parameters())
    solver = cebra.solver.init("aligned-trial-dysco",
                               model=model,
                               dynamics_model=flow,
                               criterion=criterion,
                               optimizer=torch.optim.Adam(parameters, lr=1e-3),
                               tqdm_on=False)
    assert {id(p) for p in solver.parameters()} == {id(p) for p in parameters}
    assert len(list(solver.parameters())) == len(parameters)
    before = [
        p.detach().clone()
        for p in (next(model.parameters()), flow.field.coefficients,
                  next(criterion.potential.parameters()))
    ]
    solver.fit(loader)
    assert len(solver.history) == 2
    for previous, current in zip(
            before, (next(model.parameters()), flow.field.coefficients,
                     next(criterion.potential.parameters()))):
        assert torch.isfinite(current).all()
        assert not torch.equal(previous, current)
    restored = copy.deepcopy(solver)
    checkpoint = copy.deepcopy(solver.state_dict())
    restored.load_state_dict(checkpoint)
    assert restored.reference_model[0] is restored.model
    assert restored.reference_model[1] is restored.dynamics_model
    batch = next(iter(loader))
    torch.testing.assert_close(torch.tensor(solver.step(batch)["total"]),
                               torch.tensor(restored.step(batch)["total"]))
    for first, second in zip(solver.reference_model.parameters(),
                             restored.reference_model.parameters()):
        torch.testing.assert_close(first, second)
    assert solver.transform(torch.randn(20, 5)).shape == (20, 2)


def test_controlled_bidirectional_endpoint_consistency():
    model = cebra.models.init("offset1-model",
                              num_neurons=2,
                              num_units=8,
                              num_output=2)
    flow = PolynomialFlow(2, degree=1, control_dim=1, dt=0.1)
    with torch.no_grad():
        flow.field.coefficients.zero_()
        flow.field.coefficients[0, flow.field.terms.index((2,))] = 1
        flow.field.coefficients[1, flow.field.terms.index(())] = 0.2
    criterion = DYSCOInfoNCE(sample_interval=0.1)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(flow.parameters()))
    solver = cebra.solver.init("aligned-trial-dysco",
                               model=model,
                               dynamics_model=flow,
                               criterion=criterion,
                               optimizer=optimizer,
                               tqdm_on=False)
    times = torch.arange(4) * 0.1
    encoded = torch.stack((0.5 * times.square(), 0.2 * times), dim=-1)[None,
                                                                       None]
    encoded = encoded + torch.tensor([0., 1.])[:, None, None, None]
    controls = times[None, None, :, None].expand(2, 1, -1, -1)
    forward, backward = solver._trajectories(encoded, controls)
    torch.testing.assert_close(forward, encoded)
    torch.testing.assert_close(backward, encoded)
