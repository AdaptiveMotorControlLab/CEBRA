#
# (c) All rights reserved. ECOLE POLYTECHNIQUE FÉDÉRALE DE LAUSANNE,
# Switzerland, Laboratory of Prof. Mackenzie W. Mathis (UPMWMATHIS) and
# original authors: Steffen Schneider, Jin H Lee, Mackenzie W Mathis. 2023.
#
# Source code:
# https://github.com/AdaptiveMotorControlLab/CEBRA
#
# Please see LICENSE.md for the full license document:
# https://github.com/AdaptiveMotorControlLab/CEBRA/LICENSE.md
#
import collections
from typing import Tuple

import torch

__all__ = ["Batch", "BatchIndex", "Offset"]

# Batch = collections.namedtuple(
#    'batch', ['reference', 'positive', 'negative', 'index', 'index_reversed'],
#    defaults=(None, None))


class Batch:
    """A batch of reference, positive, negative samples and an optional index.

    Attributes:
        reference: The reference samples, typically sampled from the prior
            distribution
        positive: The positive samples, typically sampled from the positive
            conditional distribution depending on the reference samples
        negative: The negative samples, typically sampled from the negative
            conditional distribution depending (but often independent) from
            the reference samples
        index: TODO(stes), see docs for multisession training distributions
        index_reversed: TODO(stes), see docs for multisession training distributions
    """

    __slots__ = ["reference", "positive", "negative", "index", "index_reversed"]

    def __init__(self,
                 reference,
                 positive,
                 negative,
                 index=None,
                 index_reversed=None):
        self.reference = reference
        self.positive = positive
        self.negative = negative
        self.index = index
        self.index_reversed = index_reversed

    def to(self, device):
        """Move all batch elements to the GPU."""
        self.reference = self.reference.to(device)
        self.positive = self.positive.to(device)
        self.negative = self.negative.to(device)
        # TODO(stes): Unclear if the indices should also be best represented by
        # torch.Tensors vs. np.ndarrays---this should probably be updated once
        # the GPU implementation of the multi-session sampler is fully ready.
        # if self.index is not None:
        #    self.index = self.index.to(device)
        # if self.index_reversed is not None:
        #    self.index_reversed = self.index_reversed.to(device)


BatchIndex = collections.namedtuple(
    "BatchIndex",
    ["reference", "positive", "negative", "index", "index_reversed"],
    defaults=(None, None),
)


class Offset:
    """Number of samples left and right from an index.

    When indexing datasets, some operations require input of multiple neighbouring samples
    across the time dimension. ``Offset`` represents a simple pair of left and right
    offsets with respect to a index. It provides the range of samples to consider around the current index for
    sampling across the time dimension.

    The provided offsets are positive :py:class:`int`, so that the ``left`` offset corresponds
    to the number of samples to consider previous to the index while the ``right`` offset is strictly positive and
    corresponds to the the index itself and the number of samples to consider following the index.

    Note:
        By convention, the right bound should always be **strictly positive** as it is including the current index itself.
        Hence, for instance, to only consider the current element, you will have to provide (0,1) at :py:class:`Offset` initialization.

    """

    __slots__ = ["left", "right"]

    def __init__(self, *offset):
        if len(offset) == 1:
            (offset,) = offset
            self.left = offset
            self.right = offset
        elif len(offset) == 2:
            self.left, self.right = offset
        else:
            raise ValueError(
                f"Invalid number of elements to bound the Offset, expect 1 or 2 elements, got {len(offset)}."
            )
        self._check_offset_positive()

    def _check_offset_positive(self):
        for offset in [self.right, self.left]:
            if offset < 0:
                raise ValueError(
                    f"Invalid Offset bounds, expect value superior or equal to 0, got {offset}."
                )

        if self.right == 0:
            raise ValueError(
                f"Invalid right bound. By convention, the right bound includes the current index. It should be at least set to 1, "
                f"got {self.right}")

    @property
    def _right(self):
        return None if self.right == 0 else -self.right

    @property
    def left_slice(self):
        """Slice from array start to left border."""
        return slice(0, self.left)

    @property
    def right_slice(self):
        """Slice from right border to array end."""
        return slice(self._right, None)

    @property
    def valid_slice(self):
        """Slice between the two borders."""
        return slice(self.left, self._right)

    def __len__(self):
        return self.left + self.right

    def mask_array(self, array, value):
        array[self.left_slice] = value
        array[self.right_slice] = value
        return array

    def __repr__(self):
        return f"Offset(left = {self.left}, right = {self.right}, length = {len(self)})"
