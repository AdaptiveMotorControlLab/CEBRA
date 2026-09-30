"""Reference implementations of previously hardcoded offset models before parametrization.

These models are used to verify that the parametrized versions produce identical outputs.
"""

import torch
from torch import nn

import cebra.data
import cebra.data.datatypes
import cebra.models.layers as cebra_layers
from cebra.models.model import _OffsetModel
from cebra.models.model import ConvolutionalModelMixin


class Offset10ModelReference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 10 sample receptive field (offset10-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, num_layers=3),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(5, 5)


class Offset5ModelReference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 5 sample receptive field (offset5-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            cebra_layers._Skip(nn.Conv1d(num_units, num_units, 3), nn.GELU()),
            nn.Conv1d(num_units, num_output, 2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(2, 3)


class Offset15ModelReference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 15 sample receptive field (offset15-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, num_layers=6),
            nn.Conv1d(num_units, num_output, 2),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(7, 8)


class Offset20ModelReference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 20 sample receptive field (offset20-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, num_layers=8),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(10, 10)


class Offset36Reference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 36 sample receptive field (offset36-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, num_layers=16),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(18, 18)


class Offset40Reference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 40 sample receptive field (offset40-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, 18),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(20, 20)


class Offset50Reference(_OffsetModel, ConvolutionalModelMixin):
    """Reference: CEBRA model with a 50 sample receptive field (offset50-model)."""

    def __init__(self, num_neurons, num_units, num_output, normalize=True):
        if num_units < 1:
            raise ValueError(
                f"Hidden dimension needs to be at least 1, but got {num_units}."
            )
        super().__init__(
            nn.Conv1d(num_neurons, num_units, 2),
            nn.GELU(),
            *self._make_layers(num_units, 23),
            nn.Conv1d(num_units, num_output, 3),
            num_input=num_neurons,
            num_output=num_output,
            normalize=normalize,
        )

    def get_offset(self) -> cebra.data.datatypes.Offset:
        return cebra.data.Offset(25, 25)
