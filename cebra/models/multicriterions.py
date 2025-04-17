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
"""Support for training CEBRA with multiple criteria.

.. note::
   This module was introduced in CEBRA 0.6.0.

"""
from typing import Tuple

import torch
from torch import nn

from cebra.data.datatypes import Batch


class MultiCriterions(nn.Module):
    """A module for handling multiple loss functions with different criteria.

    This module allows combining multiple loss functions, each operating on specific
    slices of the input data. It supports both supervised and contrastive learning modes.

    Args:
        losses: A list of dictionaries containing loss configurations. Each dictionary should have:
            - 'indices': Tuple of (start, end) indices for the data slice
            - 'supervised_loss': Dict with loss config for supervised mode
            - 'contrastive_loss': Dict with loss config for contrastive mode
            Loss configs should contain:
            - 'name': Name of the loss function
            - 'kwargs': Optional parameters for the loss function
        mode: Either "supervised" or "contrastive" to specify the training mode

    The loss functions can be from torch.nn or custom implementations from cebra.models.criterions.
    Each criterion is applied to its corresponding slice of the input data during forward pass.

    Example:
        >>> import torch
        >>> from cebra.data.datatypes import Batch
        >>> # Define loss configurations for a hybrid model with both contrastive and supervised losses
        >>> losses = [
        ...     {
        ...         'indices': (0, 10),  # First 10 dimensions
        ...         'contrastive_loss': {
        ...             'name': 'InfoNCE',  # Using CEBRA's InfoNCE loss
        ...             'kwargs': {'temperature': 1.0}
        ...         },
        ...         'supervised_loss': {
        ...             'name': 'nn.MSELoss',  # Using PyTorch's MSE loss
        ...             'kwargs': {}
        ...         }
        ...     },
        ...     {
        ...         'indices': (10, 20),  # Next 10 dimensions
        ...         'contrastive_loss': {
        ...             'name': 'InfoNCE',  # Using CEBRA's InfoNCE loss
        ...             'kwargs': {'temperature': 0.5}
        ...         },
        ...         'supervised_loss': {
        ...             'name': 'nn.L1Loss',  # Using PyTorch's L1 loss
        ...             'kwargs': {}
        ...         }
        ...     }
        ... ]
        >>> # Create sample predictions (2 batches of 32 samples each with 10 features)
        >>> ref1 = torch.randn(32, 10)
        >>> pos1 = torch.randn(32, 10)
        >>> neg1 = torch.randn(32, 10)
        >>> ref2 = torch.randn(32, 10)
        >>> pos2 = torch.randn(32, 10)
        >>> neg2 = torch.randn(32, 10)
        >>> predictions = (
        ...     Batch(reference=ref1, positive=pos1, negative=neg1),
        ...     Batch(reference=ref2, positive=pos2, negative=neg2)
        ... )
        >>> # Create multi-criterion module in contrastive mode
        >>> multi_loss = MultiCriterions(losses, mode="contrastive")
        >>> # Forward pass with multiple predictions
        >>> losses = multi_loss(predictions)  # Returns list of loss values
        >>> assert len(losses) == 2  # One loss per criterion
    """

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
