#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
from typing import Optional, Tuple, Union

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
