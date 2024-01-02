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
"""Solver implementations for multi-session datasetes."""

import abc
import os
from collections.abc import Iterable
from typing import List, Optional

import literate_dataclasses as dataclasses
import torch

import cebra
import cebra.data
import cebra.models
import cebra.solver.base as abc_
from cebra.solver import register
from cebra.solver.util import Meter


@register("multi-session")
class MultiSessionSolver(abc_.Solver):
    """Multi session training, contrasting pairs of neural data."""

    _variant_name = "multi-session"

    def _mix(self, array: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        shape = array.shape
        n, m = shape[:2]
        mixed = array.reshape(n * m, -1)[idx]
        return mixed.reshape(shape)

    def _single_model_inference(self, batch: cebra.data.Batch,
                                model: torch.nn.Module) -> cebra.data.Batch:
        """Given a single batch of input examples, computes the feature representation/embedding.

        Args:
            batch: The input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.
            model: The model to use for inference.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.
        """
        batch.to(self.device)
        ref = torch.stack([model(batch.reference)], dim=0)
        pos = torch.stack([model(batch.positive)], dim=0)
        neg = torch.stack([model(batch.negative)], dim=0)

        pos = self._mix(pos, batch.index_reversed)

        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )

    def _inference(self, batches: List[cebra.data.Batch]) -> cebra.data.Batch:
        """Given batches of input examples, computes the feature representations/embeddings.

        Args:
            batches: A list of input data, not necessarily aligned across the batch
                dimension. This means that ``batch.index`` specifies the map
                between reference/positive samples, if not equal ``None``.

        Returns:
            Processed batch of data. While the input data might not be aligned
            across the sample dimensions, the output data should be aligned and
            ``batch.index`` should be set to ``None``.

        """
        refs = []
        poss = []
        negs = []

        for batch, model in zip(batches, self.model):
            batch.to(self.device)
            refs.append(model(batch.reference))
            poss.append(model(batch.positive))
            negs.append(model(batch.negative))
        ref = torch.stack(refs, dim=0)
        pos = torch.stack(poss, dim=0)
        neg = torch.stack(negs, dim=0)

        pos = self._mix(pos, batches[0].index_reversed)

        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )

    def validation(self, loader, session_id: Optional[int] = None):
        """Compute score of the model on data.

        Note:
            Overrides :py:meth:`cebra.solver.base.Solver.validation` in :py:class:`cebra.solver.base.Solver`.
        Args:
            loader: Data loader, which is an iterator over :py:class:`cebra.data.datatypes.Batch` instances.
                Each batch contains reference, positive and negative input samples.
            session_id: The session ID, an integer between 0 and the number of sessions in the
                multisession model, set to None for single session.

        Returns:
            Loss averaged over iterations on data batch.
        """

        assert session_id is not None

        iterator = self._get_loader(loader)  # loader is single session
        total_loss = Meter()
        self.model[session_id].eval()
        for _, batch in iterator:
            prediction = self._single_model_inference(batch,
                                                      self.model[session_id])
            loss, _, _ = self.criterion(prediction.reference,
                                        prediction.positive,
                                        prediction.negative)
            total_loss.add(loss.item())
        return total_loss.average


@register("multi-session-aux")
class MultiSessionAuxVariableSolver(abc_.Solver):
    """Multi session training, contrasting neural data against behavior."""

    _variant_name = "multi-session-aux"
    reference_model: torch.nn.Module

    def _inference(self, batches):
        refs = []
        poss = []
        negs = []
        for batch, model in zip(batches, self.model):
            batch.to(self.device)
            refs.append(model(batch.reference.cuda()))
            poss.append(model(batch.positive.cuda()))
            negs.append(model(batch.negative.cuda()))

        ref = torch.stack(refs, dim=0)
        pos = self._mix(torch.stack(poss, dim=0))
        neg = torch.stack(negs, dim=0)
        num_features = neg.shape[2]

        return cebra.data.Batch(
            reference=ref.view(-1, num_features),
            positive=pos.view(-1, num_features),
            negative=neg.view(-1, num_features),
        )
