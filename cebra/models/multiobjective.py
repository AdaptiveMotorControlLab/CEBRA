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
"""Wrappers for using models with multiobjective solvers.

.. note::

    Experimental as of Nov 06, 2022.
"""

from typing import Tuple

import torch
from torch import nn

import cebra.models


class _Norm(nn.Module):
    """Normalize the input tensor across its first dimension.

    TODO:
        * Move this class to ``cebra.models.layers``.
    """

    def forward(self, inp):
        """Normalize the input tensor across its first dimension."""
        return inp / torch.norm(inp, dim=1, keepdim=True)


class MultiobjectiveModel(nn.Module):
    """Wrapper around contrastive learning models to all training with multiple objectives

    Multi-objective training splits the last layer's feature representation into multiple
    chunks, which are then used for individual training objectives.

    Args:
        module: The module to wrap
        dimensions: A tuple of dimension values to extract from the model's feature embedding.
        renormalize: If True, the individual feature slices will be re-normalized before
            getting returned---this option only makes sense in conjunction with a loss based
            on the cosine distance or dot product.
        output_mode: A mode as defined in ``MultiobjectiveModel.Mode``. Overlapping means that
            when ``dimensions`` are set to `(x0, x1, ...)``, features will be extracted from
            ``0:x0, 0:x1, ...``. When mode is set to separate, features are extracted from
            ``x0:x1, x1:x2, ...``.
        append_last_dimension: Defaults to True, and will allow to omit the last dimension in
            the ``dimensions`` argument (which should be equal to the output dimension) of the
            given model.

    TODO:
        - Update nn.Module type annotation for ``module`` to cebra.models.Model
    """

    class Mode:
        """Mode for slicing and potentially normalizing the output embedding.

        The options are:

        - ``OVERLAPPING``: When ``dimensions`` are set to `(x0, x1, ...)``, features will be
          extracted from ``0:x0, 0:x1, ...``.
        - ``SEPARATE``: Features are extracted from ``x0:x1, x1:x2, ...``

        """

        OVERLAPPING = "overlapping"
        SEPARATE = "separate"
        _ALL = {OVERLAPPING, SEPARATE}

        def is_valid(self, mode):
            """Check if a given string representation is valid.

            Args:
                mode: String representation of the mode.

            Returns:
                ``True`` for a valid representation, ``False`` otherwise.
            """
            return mode in _ALL

    def __init__(
        self,
        module: nn.Module,
        dimensions: Tuple[int],
        renormalize: bool = False,
        output_mode: str = "overlapping",
        append_last_dimension: bool = False,
    ):
        super().__init__()

        if not isinstance(module, cebra.models.Model):
            raise ValueError("Can only wrap models that are subclassing the "
                             "cebra.models.Model abstract base class. "
                             f"Got a model of type {type(module)}.")

        self.module = module
        self.renormalize = renormalize
        self.output_mode = output_mode

        self._norm = _Norm()
        self._compute_slices(dimensions, append_last_dimension)

    @property
    def get_offset(self):
        """See :py:meth:`cebra.models.model.Model.get_offset`."""
        return self.module.get_offset

    @property
    def num_output(self):
        """See :py:attr:`cebra.models.model.Model.num_output`."""
        return self.module.num_output

    def _compute_slices(self, dimensions, append_last_dimension):

        def _valid_dimensions(dimensions):
            return max(dimensions) == self.num_output

        if append_last_dimension:
            if _valid_dimensions(dimensions):
                raise ValueError(
                    f"append_last_dimension should only be used if extra values are "
                    f"available. Last requested dimensionality is already {dimensions[-1]}."
                )
            dimensions += (self.num_output,)
        if not _valid_dimensions(dimensions):
            raise ValueError(
                f"Max of given dimensions needs to match the number of outputs "
                f"in the encoder network. Got {dimensions} and expected a "
                f"maximum value of {self.num_output}.")

        if self.output_mode == self.Mode.OVERLAPPING:
            self.feature_ranges = tuple(
                slice(0, dimension) for dimension in dimensions)
        elif self.output_mode == self.Mode.SEPARATE:
            from_dimension = (0,) + dimensions
            self.feature_ranges = tuple(
                slice(i, j) for i, j in zip(from_dimension, dimensions))
        else:
            raise ValueError(
                f"Unknown mode: '{self.output_mode}', use one of {self.Mode._ALL}."
            )

    def forward(self, inputs):
        """Compute multiple embeddings for a single signal input.

        Args:
            inputs: The input tensor

        Returns:
            A tuple of tensors which are sliced according to `self.feature_ranges`
            if `renormalize` is set to true, each of the tensors will be normalized
            across the first (feature) dimension.

        TODO:
            - Cover this function with unit tests
        """
        output = self.module(inputs)
        outputs = (
            output[:, slice_features] for slice_features in self.feature_ranges)
        if self.renormalize:
            outputs = (self._norm(output) for output in outputs)
        return tuple(outputs)
