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
"""Numerical and lifecycle tests for the DYSCO dynamics foundation."""

import math

import numpy as np
import pytest
import torch

import cebra
import cebra.dynamics
from cebra.dynamics.polynomial import PolynomialField
from cebra.dynamics.polynomial import PolynomialFlow
from cebra.dynamics.polynomial import RK4Flow


def test_polynomial_field_recovers_lorenz():
    field = PolynomialField(3).double()
    # 1, x, y, z, xx, xy, xz, yy, yz, zz
    with torch.no_grad():
        field.coefficients.zero_()
        field.coefficients[0, 1:3] = torch.tensor([-10., 10.])
        field.coefficients[1, [1, 2, 6]] = torch.tensor([28., -1., -1.],
                                                        dtype=torch.float64)
        field.coefficients[2, [3, 5]] = torch.tensor([-8 / 3, 1.],
                                                     dtype=torch.float64)
    state = torch.randn(2, 4, 7, 3, dtype=torch.float64)
    x, y, z = state.unbind(-1)
    expected = torch.stack((10 * (y - x), x * (28 - z) - y, x * y - 8 * z / 3),
                           -1)
    torch.testing.assert_close(field(state), expected)


def test_polynomial_features_match_sklearn():
    from sklearn.preprocessing import PolynomialFeatures
    state = torch.randn(12, 3, dtype=torch.float64)
    control = torch.randn(12, 2, dtype=torch.float64)
    field = PolynomialField(3, degree=3, control_dim=2).double()
    expected = PolynomialFeatures(3).fit_transform(
        torch.cat((state, control), -1).numpy())
    np.testing.assert_allclose(field.features(state, control).numpy(), expected)


def test_rk4_fourth_order_convergence_and_backward():
    field = PolynomialField(1, degree=1, include_bias=False).double()
    with torch.no_grad():
        field.coefficients.fill_(1)
    initial = torch.ones(2, 1, dtype=torch.float64)
    flow = RK4Flow(field, dt=0.25)
    coarse = flow.rollout(initial, 4)[..., -1, :]
    fine = flow.rollout(initial, 8, dt=0.125)[..., -1, :]
    ratio = ((coarse - math.e).abs() / (fine - math.e).abs()).mean()
    assert 13 < ratio < 18
    recovered = flow.rollout(fine, 8, dt=-0.125)[..., -1, :]
    torch.testing.assert_close(recovered, initial, atol=1e-6, rtol=0)
    assert flow.rollout(initial, 4).shape == (2, 5, 1)


def test_rotational_flow_matches_analytic_orbit():
    field = PolynomialField(2, degree=1, include_bias=False).double()
    with torch.no_grad():
        field.coefficients.copy_(torch.tensor([[0., -2.], [2., 0.]]))
    flow = RK4Flow(field, dt=0.05, substeps=2)
    time = torch.arange(41, dtype=torch.float64) * 0.05
    expected = torch.stack((torch.cos(2 * time), torch.sin(2 * time)), -1)
    actual = flow.rollout(torch.tensor([1., 0.], dtype=torch.float64), 40)
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=0)


def test_control_interpolation_and_gradients():
    field = PolynomialField(1, degree=1, control_dim=1,
                            include_bias=False).double()
    with torch.no_grad():
        field.coefficients.copy_(torch.tensor([[0., 1.]]))
    flow = RK4Flow(field, dt=1, substeps=3)
    initial = torch.zeros(2, 1, dtype=torch.float64, requires_grad=True)
    controls = torch.tensor([[[0.], [1.], [2.]]],
                            dtype=torch.float64).repeat(2, 1, 1)
    controls.requires_grad_()
    actual = flow.rollout(initial, 2, controls)
    torch.testing.assert_close(actual[0, :, 0],
                               torch.tensor([0., 0.5, 2.], dtype=torch.float64))
    actual[..., -1, :].sum().backward()
    assert torch.isfinite(field.coefficients.grad).all()
    assert torch.count_nonzero(controls.grad) == controls.numel()
    torch.testing.assert_close(initial.grad, torch.ones_like(initial))
    recovered = flow.rollout(actual[..., -1, :], 2, controls.flip(-2), dt=-1)
    torch.testing.assert_close(recovered[..., -1, :], initial)


def test_flow_passes_autograd_gradcheck():
    torch.manual_seed(3)
    field = PolynomialField(2).double()
    flow = RK4Flow(field, dt=0.1, substeps=2)
    state = torch.randn(3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(flow, (state,))


@pytest.mark.parametrize("kwargs", [
    dict(latent_dim=0),
    dict(latent_dim=True),
    dict(latent_dim=2, degree=0),
    dict(latent_dim=2, control_dim=-1)
])
def test_invalid_field_configuration(kwargs):
    with pytest.raises(ValueError):
        PolynomialField(**kwargs)


@pytest.mark.parametrize("dt", [0, -0.1, float("nan"), float("inf")])
def test_invalid_flow_interval(dt):
    with pytest.raises(ValueError):
        PolynomialFlow(2, dt=dt)


def test_controls_and_time_nodes_are_required():
    flow = PolynomialFlow(2, control_dim=1)
    state = torch.zeros(3, 2)
    with pytest.raises(ValueError, match="Controls"):
        flow(state)
    with pytest.raises(ValueError, match="time node"):
        flow.rollout(state, 3, torch.zeros(3, 3, 1))
    with pytest.raises(ValueError, match="autonomous"):
        PolynomialFlow(2).rollout(state, 3, torch.zeros(3, 4, 1))
    with pytest.raises(ValueError, match="nonzero"):
        flow.rollout(state, 1, dt=0)


def test_registry_state_dict_roundtrip():
    original = cebra.dynamics.init("polynomial-flow", latent_dim=3).double()
    restored = cebra.dynamics.init("polynomial-flow", latent_dim=3).double()
    restored.load_state_dict(original.state_dict())
    state = torch.randn(4, 3, dtype=torch.float64)
    torch.testing.assert_close(original(state), restored(state))


@pytest.mark.parametrize("architecture", ["polynomial-flow"])
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
    del legacy["reference_model"]
    with pytest.raises(KeyError, match="missing reference_model"):
        loaded.solver_.load_state_dict(legacy)
    with pytest.warns(UserWarning, match="Legacy auxiliary"):
        loaded.solver_.load_state_dict(legacy, strict=False)
