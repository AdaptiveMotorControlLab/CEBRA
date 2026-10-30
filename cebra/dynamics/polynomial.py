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
"""Polynomial vector fields and differentiable, fixed-step flow maps.

The registry exposes a flow map for use with the existing DCL solver. The
underlying vector field is also available for continuous-time diagnostics.
"""

import itertools
import keyword
import math
from typing import Optional

import torch
from torch import nn

from cebra.dynamics import register


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class PolynomialField(nn.Module):
    """A trainable polynomial vector field with optional observed controls.

    Monomials use increasing total degree and lexicographic combinations of
    input indices. For two states, degree two and a bias the order is
    ``1, z0, z1, z0*z0, z0*z1, z1*z1``. Controls follow states in the input
    vector and participate in the same polynomial library.

    Args:
        latent_dim: Number of state coordinates.
        degree: Maximum total polynomial degree.
        control_dim: Number of observed control coordinates.
        include_bias: Include the constant monomial.
    """

    def __init__(self,
                 latent_dim: int,
                 degree: int = 2,
                 control_dim: int = 0,
                 include_bias: bool = True):
        super().__init__()
        _positive_integer(latent_dim, "latent_dim")
        _positive_integer(degree, "degree")
        if isinstance(
                control_dim,
                bool) or not isinstance(control_dim, int) or control_dim < 0:
            raise ValueError("control_dim must be a nonnegative integer.")
        if not isinstance(include_bias, bool):
            raise ValueError("include_bias must be a boolean.")
        self.latent_dim = latent_dim
        self.degree = degree
        self.control_dim = control_dim
        self.include_bias = include_bias
        self.terms = tuple(
            term for order in range(0 if include_bias else 1, degree + 1)
            for term in itertools.combinations_with_replacement(
                range(latent_dim + control_dim), order))
        self.coefficients = nn.Parameter(
            torch.empty(latent_dim, len(self.terms)))
        nn.init.normal_(self.coefficients, std=1e-3)

    def features(self,
                 state: torch.Tensor,
                 control: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate monomials, preserving all leading state dimensions.

        Args:
            state: State tensor with last dimension ``latent_dim``.
            control: Controls with matching leading dimensions and final size
                ``control_dim``; required exactly when ``control_dim > 0``.

        Returns:
            Monomials with shape ``(*state.shape[:-1], len(terms))``.
        """
        if state.ndim < 1 or state.shape[-1] != self.latent_dim:
            raise ValueError("The final state dimension must equal latent_dim.")
        if not state.is_floating_point():
            raise ValueError("State must have a floating-point dtype.")
        if self.control_dim:
            if control is None or control.shape != (*state.shape[:-1],
                                                    self.control_dim):
                raise ValueError(
                    "Controls must match the state batch shape and control_dim."
                )
            if control.dtype != state.dtype or control.device != state.device:
                raise ValueError(
                    "State and controls must have the same dtype and device.")
            inputs = torch.cat((state, control), dim=-1)
        else:
            if control is not None:
                raise ValueError("Controls supplied to an autonomous field.")
            inputs = state
        return torch.stack([
            inputs[..., list(term)].prod(
                dim=-1) if term else torch.ones_like(inputs[..., 0])
            for term in self.terms
        ],
                           dim=-1)

    def forward(self,
                state: torch.Tensor,
                control: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Evaluate the vector field at each state and observed control."""
        return self.features(state, control) @ self.coefficients.T

    def equations(self,
                  precision=6,
                  threshold=0.0,
                  variable_names=None,
                  control_names=None):
        """Return one readable differential equation per latent coordinate.

        Args:
            precision: Positive number of significant coefficient digits.
            threshold: Omit coefficients with absolute value at most this
                nonnegative value. This changes only the displayed equations.
            variable_names: Unique Python identifiers for states; defaults to
                ``z0``, ``z1``, etc.
            control_names: Unique Python identifiers for observed controls;
                defaults to ``u0``, ``u1``, etc. Must differ from state names.

        Returns:
            List of strings such as ``dz0/dt = -10*z0 + 10*z1``. Coefficients
            describe the learned latent coordinates and original time units.
            No coordinate alignment, sparsification or refitting is applied.
        """
        _positive_integer(precision, "precision")
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and nonnegative.")

        def names_or_default(names, count, prefix):
            if names is None:
                return [f"{prefix}{i}" for i in range(count)]
            if isinstance(names, str):
                raise ValueError("Names must be a sequence of identifiers.")
            names = list(names)
            if len(names) != count or any(
                    not isinstance(name, str) or not name.isidentifier() or
                    keyword.iskeyword(name) for name in names):
                raise ValueError(
                    "Supply one valid identifier per state or control coordinate."
                )
            return names

        states = names_or_default(variable_names, self.latent_dim, "z")
        names = states + names_or_default(control_names, self.control_dim, "u")
        if len(set(names)) != len(names):
            raise ValueError("State and control names must be unique.")
        monomials = []
        for term in self.terms:
            factors = []
            for index, repeated in itertools.groupby(term):
                power = len(list(repeated))
                factors.append(names[index] if power ==
                               1 else f"{names[index]}**{power}")
            monomials.append("*".join(factors))
        equations = []
        for name, row in zip(states, self.coefficients.detach().cpu().tolist()):
            terms = []
            for coefficient, monomial in zip(row, monomials):
                if not math.isfinite(coefficient):
                    raise ValueError("Cannot display nonfinite coefficients.")
                if abs(coefficient) <= threshold:
                    continue
                value = f"{abs(coefficient):.{precision}g}"
                term = f"{value}*{monomial}" if monomial else value
                sign = "-" if coefficient < 0 else "+"
                terms.append((sign, term))
            if terms:
                sign, term = terms[0]
                rhs = ("-" if sign == "-" else "") + term
                rhs += "".join(f" {sign} {term}" for sign, term in terms[1:])
            else:
                rhs = "0"
            equations.append(f"d{name}/dt = {rhs}")
        return equations


class RK4Flow(nn.Module):
    """A differentiable flow map for a state/control vector field.

    Args:
        field: Module accepting ``field(state, control)``. Autonomous calls
            receive ``None`` for control.
        dt: Positive observation interval, in the field's time units.
        substeps: Number of RK4 steps per observation interval.

    ``forward`` advances by one observation interval. ``rollout`` returns
    every observation including the initial state, and supports signed time
    steps for backward integration. Provided controls are linearly
    interpolated between observation nodes at every RK4 stage.
    """

    def __init__(self, field: nn.Module, dt: float = 0.02, substeps: int = 1):
        super().__init__()
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        _positive_integer(substeps, "substeps")
        self.field = field
        self.dt = float(dt)
        self.substeps = substeps

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Advance an autonomous state by one configured observation interval."""
        return self.rollout(state, 1)[..., -1, :]

    def rollout(self,
                state: torch.Tensor,
                num_steps: int,
                controls: Optional[torch.Tensor] = None,
                dt: Optional[float] = None) -> torch.Tensor:
        """Integrate at evenly spaced observation nodes.

        Args:
            state: Initial states, shape ``(*batch, latent_dim)``.
            num_steps: Positive number of observation intervals.
            controls: Shape ``(*batch, num_steps + 1, control_dim)``. For a
                negative ``dt``, supply nodes in backward chronological order.
            dt: Signed interval override; defaults to the configured interval.

        Returns:
            States with shape ``(*batch, num_steps + 1, latent_dim)``.
        """
        _positive_integer(num_steps, "num_steps")
        step = self.dt if dt is None else float(dt)
        if not math.isfinite(step) or step == 0:
            raise ValueError(
                "The integration interval must be finite and nonzero.")
        if state.ndim < 1 or not state.is_floating_point():
            raise ValueError(
                "State must be a floating-point tensor with a feature axis.")
        if controls is not None:
            if controls.ndim != state.ndim + 1 or controls.shape[:-2] != state.shape[:-1] or controls.shape[
                    -2] != num_steps + 1:
                raise ValueError(
                    "Controls must match state batch dimensions and include every time node."
                )
        h = step / self.substeps
        trajectory = [state]
        for index in range(num_steps):
            for substep in range(self.substeps):
                if controls is None:
                    u0 = um = u1 = None
                else:
                    start, end = controls[..., index, :], controls[...,
                                                                   index + 1, :]
                    delta = end - start
                    u0 = start + (substep / self.substeps) * delta
                    um = start + ((substep + 0.5) / self.substeps) * delta
                    u1 = start + ((substep + 1) / self.substeps) * delta
                k1 = self.field(state, u0)
                k2 = self.field(state + h * k1 / 2, um)
                k3 = self.field(state + h * k2 / 2, um)
                k4 = self.field(state + h * k3, u1)
                state = state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            trajectory.append(state)
        return torch.stack(trajectory, dim=-2)


@register("polynomial-flow")
class PolynomialFlow(RK4Flow):
    """Polynomial ODE flow compatible with DCL's dynamics-model registry.

    Args:
        latent_dim: Number of latent coordinates.
        degree: Maximum total polynomial degree.
        dt: Positive interval for one application of the flow map.
        substeps: Number of RK4 steps per interval.
        control_dim: Number of observed controls for explicit rollouts.
        include_bias: Include a constant term in the field library.

    The default autonomous map accepts ``(batch, latent_dim)`` states, as
    required by the DCL solver. Controlled rollouts use ``rollout`` directly;
    DCL's current data loader does not supply control segments.
    """

    def __init__(self,
                 latent_dim: int,
                 degree: int = 2,
                 dt: float = 0.02,
                 substeps: int = 1,
                 control_dim: int = 0,
                 include_bias: bool = True):
        super().__init__(PolynomialField(latent_dim, degree, control_dim,
                                         include_bias),
                         dt=dt,
                         substeps=substeps)
