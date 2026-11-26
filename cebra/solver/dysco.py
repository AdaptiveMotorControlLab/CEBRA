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
"""Aligned-trial dynamics learning with the common CEBRA solver lifecycle."""

import torch

import cebra.data
from cebra.models.dysco import DYSCOInfoNCE
from cebra.solver import register
from cebra.solver.single_session import SingleSessionDCLSolver


@register("aligned-trial-dysco")
class DYSCOSolver(SingleSessionDCLSolver):
    """Encode aligned windows and return trajectories to DYSCOInfoNCE.

    Optimization, logging, validation, transform, device handling, and reference
    model persistence are inherited from the existing single-session solvers.
    The criterion owns its potential and event-sampling state.
    """

    def _encode_windows(self, observations):
        offset = len(self.model.get_offset())
        windows = observations.unfold(-2, offset, 1)
        shape = windows.shape[:-2]
        encoded = self.model(windows.reshape(-1, windows.shape[-2], offset))
        if encoded.ndim == 3 and encoded.shape[-1] == 1:
            encoded = encoded.squeeze(-1)
        if encoded.ndim != 2:
            raise ValueError(
                "The encoder must return one embedding per receptive window.")
        return encoded.reshape(*shape, encoded.shape[-1])

    def parameters(self, session_id=None):
        """Iterate over the encoder, dynamics and criterion parameters."""
        self._check_is_session_id_valid(session_id)
        yield from self.reference_model.parameters()
        yield from self.criterion.parameters()

    def _trajectories(self, encoded, controls):
        steps = encoded.shape[-2] - 1
        predictions = [
            self.dynamics_model.rollout(encoded[..., 0, :],
                                        steps,
                                        controls=controls)
        ]
        if self.criterion.bidirectional:
            backward_controls = None if controls is None else controls.flip(-2)
            backward = self.dynamics_model.rollout(encoded[..., -1, :],
                                                   steps,
                                                   controls=backward_controls,
                                                   dt=-self.dynamics_model.dt)
            predictions.append(backward.flip(-2))
        return torch.stack(predictions)

    def _inference(self, batch):
        if not isinstance(self.criterion, DYSCOInfoNCE):
            raise ValueError("DYSCOSolver requires DYSCOInfoNCE.")
        batch.to(self.device)
        if self.criterion.sample_interval != self.dynamics_model.dt:
            raise ValueError(
                "Criterion and dynamics sampling intervals must match.")
        encoded = self._encode_windows(batch.observations)
        predictions = self._trajectories(encoded, batch.controls)
        if self.criterion.candidate_mode == "in_batch":
            candidates = torch.cat((encoded[None], predictions),
                                   dim=0).flatten(1, 2)
        else:
            if batch.negative.shape[0] == 0:
                raise ValueError(
                    "Independent candidates require negative windows.")
            negative = self._encode_windows(batch.negative)
            negative_predictions = self._trajectories(negative,
                                                      batch.negative_controls)
            candidates = torch.cat((negative[None], negative_predictions),
                                   dim=0)
        return cebra.data.Batch(encoded, predictions, candidates)

    def _set_fitted_params(self, loader):
        if loader.dataset.sample_interval != self.dynamics_model.dt:
            raise ValueError(
                "Dataset and dynamics sampling intervals must match.")
        if loader.candidate_mode != self.criterion.candidate_mode:
            raise ValueError("Loader and criterion candidate modes must match.")
        super()._set_fitted_params(loader)
