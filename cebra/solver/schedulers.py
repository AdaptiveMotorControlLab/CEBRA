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
import abc
import dataclasses
from typing import List

import cebra.registry

cebra.registry.add_helper_functions(__name__)

__all__ = ["Scheduler", "ConstantScheduler", "LinearScheduler", "LinearRampUp"]


@dataclasses.dataclass
class Scheduler(abc.ABC):

    def __post_init__(self):
        pass

    @abc.abstractmethod
    def get_weights(self):
        pass


@register("constant-weight")
@dataclasses.dataclass
class ConstantScheduler(Scheduler):
    initial_weights: List[float]

    def __post_init__(self):
        super().__post_init__()

    def get_weights(self):
        weights = self.initial_weights
        if len(weights) == 0:
            weights = None
        return weights


@register("linear-scheduler")
@dataclasses.dataclass
class LinearScheduler(Scheduler):
    n_splits: int
    step_to_switch_on_reg: int
    step_to_switch_off_reg: int
    start_weight: float
    end_weight: float
    stay_constant_after_switch_off: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.step_to_switch_off_reg > self.step_to_switch_on_reg

    def get_weights(self, step):
        if self.step_to_switch_on_reg is not None:
            if step >= self.step_to_switch_on_reg and step <= self.step_to_switch_off_reg:
                interpolation_factor = min(
                    1.0, (step - self.step_to_switch_on_reg) /
                    (self.step_to_switch_off_reg - self.step_to_switch_on_reg))
                weight = self.start_weight + (
                    self.end_weight - self.start_weight) * interpolation_factor
                weights = [weight] * self.n_splits
            elif self.stay_constant_after_switch_off and step > self.step_to_switch_off_reg:
                weight = self.end_weight
                weights = [weight] * self.n_splits
            else:
                weights = None

            return weights


@register("linear-ramp-up")
@dataclasses.dataclass
class LinearRampUp(LinearScheduler):

    def __post_init__(self):
        super().__post_init__()
        self.stay_constant_after_switch_off = True
