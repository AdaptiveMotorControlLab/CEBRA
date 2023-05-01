import collections

import torch

__all__ = ["Batch", "BatchIndex", "Offset"]

#    'batch', ['reference', 'positive', 'negative', 'index', 'index_reversed'],
#    defaults=(None, None))


    """A batch of reference, positive, negative samples and an optional index.

    Attributes:
        reference: The reference samples, typically sampled from the prior
            distribution
        positive: The positive samples, typically sampled from the positive
            conditional distribution depending on the reference samples
        negative: The negative samples, typically sampled from the negative
            the reference samples
        index: TODO(stes), see docs for multisession training distributions
        index_reversed: TODO(stes), see docs for multisession training distributions
    """


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
        #    self.index = self.index.to(device)
        #    self.index_reversed = self.index_reversed.to(device)


BatchIndex = collections.namedtuple(


class Offset:
    """Number of samples left and right from an index.

    When indexing datasets, some operations require input of multiple neighbouring samples
    """


    def __init__(self, *offset):
        if len(offset) == 1:
            (offset,) = offset
            self.left = offset
            self.right = offset
            self.left, self.right = offset

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
