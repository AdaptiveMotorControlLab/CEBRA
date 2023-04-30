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
"""Demo datasets for testing CEBRA.

Note that none of the datasets will yield useful embeddings.
They are exclusively meant for unit tests and benchmarking
purposes.
"""

import torch

import cebra.data
import cebra.io
from cebra.datasets import register

_DEFAULT_NUM_TIMEPOINTS = 100000


class DemoDataset(cebra.data.SingleSessionDataset):

    def __init__(self, num_timepoints=_DEFAULT_NUM_TIMEPOINTS, num_neural=4):
        super().__init__()
        self.neural = torch.randn(num_timepoints, num_neural).float()
        self.offset = cebra.data.Offset(5, 5)

    @property
    def input_dimension(self):
        return self.neural.size(1)

    def __len__(self):
        return len(self.neural)

    @property
    def rf(self):
        return 10

    def __getitem__(self, index):
        assert index.dim() == 1, (index.dim(), index.shape)
        index = self.expand_index(index)
        assert (index >= 0).all()
        assert (index < len(self)).all()
        return self.neural[index].transpose(2, 1)


@register("demo-discrete")
class DemoDatasetDiscrete(DemoDataset):
    """Demo dataset for testing."""

    def __init__(self, num_timepoints=_DEFAULT_NUM_TIMEPOINTS, num_neural=4):
        super().__init__(num_timepoints, num_neural)
        self.index = torch.randint(0, 10, (num_timepoints,)).long()

    @property
    def discrete_index(self):
        return self.index


@register("demo-continuous")
class DemoDatasetContinuous(DemoDataset):
    """Demo dataset for testing."""

    def __init__(self,
                 num_timepoints=_DEFAULT_NUM_TIMEPOINTS,
                 num_neural=4,
                 num_behavior=3):
        super().__init__(num_timepoints, num_neural)
        self.index = torch.randn(num_timepoints, num_behavior).float()

    @property
    def continuous_index(self):
        return self.index


@register("demo-mixed")
class DemoDatasetMixed(DemoDataset):
    """Demo dataset for testing."""

    def __init__(self,
                 num_timepoints=_DEFAULT_NUM_TIMEPOINTS,
                 num_neural=4,
                 num_behavior=3):
        super().__init__(num_timepoints, num_neural)
        self.dindex = torch.randint(0, 10, (num_timepoints,)).long()
        self.cindex = torch.randn((num_timepoints, num_behavior)).float()

    @property
    def continuous_index(self):
        return self.cindex

    @property
    def discrete_index(self):
        return self.dindex


# TODO(stes) remove this from the demo datasets until multi-session training
# with discrete indices is implemented in the sklearn API.
# @register("demo-discrete-multisession")
class MultiDiscrete(cebra.data.DatasetCollection):
    """Demo dataset for testing."""

    def __init__(self, nums_neural=[3, 4, 5]):
        super().__init__(*[
            DemoDatasetDiscrete(_DEFAULT_NUM_TIMEPOINTS, num_neural)
            for num_neural in nums_neural
        ])


@register("demo-continuous-multisession")
class MultiContinuous(cebra.data.DatasetCollection):

    def __init__(
        self,
        nums_neural=[3, 4, 5],
        num_behavior=5,
        num_timepoints=_DEFAULT_NUM_TIMEPOINTS,
    ):
        super().__init__(*[
            DemoDatasetContinuous(num_timepoints, num_neural, num_behavior)
            for num_neural in nums_neural
        ])


# TODO(stes) remove this from the demo datasets until multi-session training
# with mixed indices is implemented in the sklearn API.
# @register("demo-mixed-multisession")
class MultiMixed(cebra.data.DatasetCollection):

    def __init__(self, nums_neural=[3, 4, 5], num_behavior=5):
        super().__init__(*[
            DemoDatasetMixed(_DEFAULT_NUM_TIMEPOINTS, num_neural, num_behavior)
            for num_neural in nums_neural
        ])
