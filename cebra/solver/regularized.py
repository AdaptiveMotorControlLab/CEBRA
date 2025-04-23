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
"""Regularized contrastive learning."""

from typing import Dict, Optional

import literate_dataclasses as dataclasses
import torch

import cebra
import cebra.data
import cebra.models
from cebra.solver import register
from cebra.solver.single_session import SingleSessionSolver


@register("regularized-solver")
@dataclasses.dataclass
class RegularizedSolver(SingleSessionSolver):
    """Optimize a model using Jacobian Regularizer."""

    _variant_name = "regularized-solver"
    log: Dict = dataclasses.field(default_factory=lambda: ({
        "pos": [],
        "neg": [],
        "loss": [],
        "loss_reg": [],
        "temperature": [],
        "reg": [],
        "reg_lambda": [],
    }))

    lambda_JR: Optional[float] = None

    def __post_init__(self):
        super().__post_init__()
        #TODO: rn we are using the full jacobian. Can be optimized later if needed.
        self.jac_regularizer = cebra.models.JacobianReg(n=-1)

    def step(self, batch: cebra.data.Batch) -> dict:
        """Perform a single gradient update using the jacobian regularizaiton!.

        Args:
            batch: The input samples

        Returns:
            Dictionary containing training metrics.
        """

        self.optimizer.zero_grad()
        batch.reference.requires_grad = True
        prediction = self._inference(batch)
        R = self.jac_regularizer(batch.reference, prediction.reference)

        loss, align, uniform = self.criterion(prediction.reference,
                                              prediction.positive,
                                              prediction.negative)
        loss_reg = loss + self.lambda_JR * R

        loss_reg.backward()
        self.optimizer.step()
        self.history.append(loss.item())
        stats = dict(pos=align.item(),
                     neg=uniform.item(),
                     loss=loss.item(),
                     loss_reg=loss_reg.item(),
                     reg=R.item(),
                     temperature=self.criterion.temperature,
                     reg_lambda=(self.lambda_JR * R).item())

        for key, value in stats.items():
            self.log[key].append(value)
        return stats


def _prepare_inputs(inputs):
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.from_numpy(inputs)
    inputs.requires_grad_(True)
    return inputs


def _prepare_model(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model
