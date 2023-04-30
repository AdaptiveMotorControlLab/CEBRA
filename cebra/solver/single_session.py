"""Single session solvers embed a single pair of time series."""

import abc
import copy
import os
from collections.abc import Iterable
from typing import List

import literate_dataclasses as dataclasses
import torch

import cebra
import cebra.data
import cebra.models
import cebra.solver.base as abc_
from cebra.solver import register


@register("single-session")
class SingleSessionSolver(abc_.Solver):
    """Single session training with a symmetric encoder.
    This solver assumes that reference, positive and negative samples
    are processed by the same features encoder.
    """

    _variant_name = "single-session"

        batch.to(self.device)
        ref = self.model(batch.reference)
        pos = self.model(batch.positive)
        neg = self.model(batch.negative)
        return cebra.data.Batch(ref, pos, neg)

    def get_embedding(self, data: torch.Tensor) -> torch.Tensor:
        """Return the embedding of the given input data.


        This function does *not* perform checks for correctness of the
        input.

        Args:
            data: The input data tensor of shape `batch_time x dims x time`

        Returns:
            The output data tensor of shape `batch_time x features`.

        TODO:
            - Check if implementing checks in this function would downgrade
              speed during training/inference.
        """
        return self.model(data)[0].T


@register("single-session-aux")
@dataclasses.dataclass
class SingleSessionAuxVariableSolver(abc_.Solver):
    """

    _variant_name = "single-session-aux"
    reference_model: torch.nn.Module = None

    def __post_init__(self):
        super().__post_init__()
        if self.reference_model is None:
            # NOTE(stes): This should work, according to this thread
            # https://discuss.pytorch.org/t/can-i-deepcopy-a-model/52192/19
            # and create a true copy of the model.
            self.reference_model = copy.deepcopy(self.model)
            self.reference_model.to(self.model.device)

    def _inference(self, batch):
        batch.to(self.device)
        ref = self.reference_model(batch.reference)
        pos = self.model(batch.positive)
        neg = self.model(batch.negative)
        return cebra.data.Batch(ref, pos, neg)


@register("single-session-hybrid")
@dataclasses.dataclass
class SingleSessionHybridSolver(abc_.MultiobjectiveSolver):
    """Single session training, contrasting neural data against behavior."""

    _variant_name = "single-session-hybrid"

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        batch.to(self.device)
        behavior_ref = self.model(batch.reference)[0]
        behavior_pos = self.model(batch.positive[:int(len(batch.positive) //
                                                      2)])[0]
        behavior_neg = self.model(batch.negative)[0]
        time_pos = self.model(batch.positive[int(len(batch.positive) // 2):])[1]
        time_ref = self.model(batch.reference)[1]
        time_neg = self.model(batch.negative)[1]
        return cebra.data.Batch(behavior_ref, behavior_pos,
                                behavior_neg), cebra.data.Batch(
                                    time_ref, time_pos, time_neg)


@register("single-session-full")
@dataclasses.dataclass
class BatchSingleSessionSolver(SingleSessionSolver):
    """Optimize a model with batch gradient descent.

    Usage of this solver requires a sufficient amount of GPU memory. Using this solver
    is equivalent to using a single session solver with batch size set to dataset size,
    but requires less computation.
    """

    def fit(self, loader, *args, **kwargs):
        """TODO"""
        self.offset = loader.dataset.offset
        self.neural = loader.dataset.neural.T[None]
        if isinstance(self.model, cebra.models.ConvolutionalModelMixin):
            if self.offset is None:
                raise ValueError("Configure dataset, no offset found.")
            self._mode = "convolutional"
        else:
            self._mode = "fully_connected"
        super().fit(loader, *args, **kwargs)

    def get_embedding(self, data):
        """Compute the embedding of a full input dataset.

        For convolutional models that implement
        :py:class:`cebra.models.model.ConvolutionalModelMixin`),
        the embedding is computed via
        :py:meth:`.SingleSessionSolver.get_embedding`.

        For all other models, it is assumed that the data has shape
        ``(1, dim, time)`` and is transformed into ``(time, dim)``
        format.

        Args:
            data: The input data

        Returns:
            The output embedding of shape ``(time, dimension)``

        See Also:
            * :py:class:`cebra.models.model.ConvolutionalModelMixin`)
            * :py:meth:`.SingleSessionSolver.get_embedding`
        """
        if self._mode == "convolutional":
            return super().get_embedding(data)
        else:
            # data has shape (1, d, T)
            # output needs to be (T, d)
            return self.model(data[0].T)

    def _inference(self, batch: cebra.data.Batch) -> cebra.data.Batch:
        outputs = self.get_embedding(self.neural)
        idc = batch.positive - self.offset.left >= len(outputs)
        batch.positive[idc] = batch.reference[idc]

