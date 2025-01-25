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
from typing import Tuple

import torch
from torch import nn

from cebra.data.datatypes import Batch


class MultiCriterions(nn.Module):

    def __init__(self, losses, mode):
        super(MultiCriterions, self).__init__()
        self.criterions = nn.ModuleList()
        self.slices = []

        for loss_info in losses:
            slice_indices = loss_info['indices']

            if mode == "supervised":
                loss = loss_info['supervised_loss']
            elif mode == "contrastive":
                loss = loss_info['contrastive_loss']
            else:
                raise NotImplementedError

            loss_name = loss['name']
            loss_kwargs = loss.get('kwargs', {})

            if loss_name.startswith("nn"):
                name = loss_name.split(".")[-1]
                criterion = getattr(torch.nn, name, None)
            else:
                import cebra.models
                criterion = getattr(cebra.models.criterions, loss_name, None)

            if criterion is None:
                raise ValueError(f"Loss {loss_name} not found.")
            else:
                criterion = criterion(**loss_kwargs)

            self.criterions.append(criterion)
            self.slices.append(slice(*slice_indices))
            assert len(self.criterions) == len(self.slices)

    def forward(self, predictions: Tuple[Batch]):

        losses = []

        for criterion, prediction in zip(self.criterions, predictions):

            if prediction.negative is None:
                # supervised
                #reference: data, positive: label
                loss = criterion(prediction.reference, prediction.positive)
            else:
                #contrastive
                loss, pos, neg = criterion(prediction.reference,
                                           prediction.positive,
                                           prediction.negative)

            losses.append(loss)

        assert len(self.criterions) == len(predictions) == len(losses)
        return losses
