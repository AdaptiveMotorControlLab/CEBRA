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
import warnings

import pytest
import torch

import cebra
from cebra.data import ContrastiveMultiObjectiveLoader
from cebra.data import DatasetxCEBRA
from cebra.solver import MultiObjectiveConfig


@pytest.fixture
def config():
    neurons = torch.randn(100, 5)
    behavior1 = torch.randn(100, 2)
    behavior2 = torch.randn(100, 1)
    data = DatasetxCEBRA(neurons, behavior1=behavior1, behavior2=behavior2)
    loader = ContrastiveMultiObjectiveLoader(dataset=data,
                                             num_steps=1,
                                             batch_size=24)
    return MultiObjectiveConfig(loader)


def test_imports():
    pass


def test_add_data(config):
    config.set_slice(0, 10)
    config.set_loss('loss_name', param1='value1')
    config.set_distribution('distribution_name', param2='value2')
    config.push()

    assert len(config.total_info) == 1
    assert config.total_info[0]['slice'] == (0, 10)
    assert config.total_info[0]['losses'] == {
        "name": 'loss_name',
        "kwargs": {
            'param1': 'value1'
        }
    }
    assert config.total_info[0]['distributions'] == {
        "name": 'distribution_name',
        "kwargs": {
            'param2': 'value2'
        }
    }


def test_overwriting_key_warning(config):
    with warnings.catch_warnings(record=True) as w:
        config.set_slice(0, 10)
        config.set_slice(10, 20)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Configuration key already exists" in str(w[-1].message)


def test_missing_slice_error(config):
    with pytest.raises(RuntimeError, match="Slice configuration is missing"):
        config.set_loss('loss_name', param1='value1')
        config.set_distribution('distribution_name', param2='value2')
        config.push()


def test_missing_distributions_error(config):
    with pytest.raises(RuntimeError,
                       match="Distributions configuration is missing"):
        config.set_slice(0, 10)
        config.set_loss('loss_name', param1='value1')
        config.push()


def test_missing_losses_error(config):
    with pytest.raises(RuntimeError, match="Losses configuration is missing"):
        config.set_slice(0, 10)
        config.set_distribution('distribution_name', param2='value2')
        config.push()


def test_finalize(config):
    config.set_slice(0, 6)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time", time_offset=1)
    config.push()

    config.set_slice(3, 6)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time_delta", time_delta=3, label_name="behavior2")
    config.push()

    config.finalize()

    assert len(config.losses) == 2
    assert config.losses[0]['indices'] == (0, 6)
    assert config.losses[1]['indices'] == (3, 6)

    assert len(config.feature_ranges) == 2
    assert config.feature_ranges[0] == slice(0, 6)
    assert config.feature_ranges[1] == slice(3, 6)

    assert len(config.loader.distributions) == 2
    assert isinstance(config.loader.distributions[0],
                      cebra.distributions.continuous.TimeContrastive)
    assert config.loader.distributions[0].time_offset == 1

    assert isinstance(config.loader.distributions[1],
                      cebra.distributions.continuous.TimedeltaDistribution)
    assert config.loader.distributions[1].time_delta == 3


def test_non_unique_feature_ranges_error(config):
    config.set_slice(0, 10)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time", time_offset=1)
    config.push()

    config.set_slice(0, 10)
    config.set_loss("FixedEuclideanInfoNCE", temperature=1.)
    config.set_distribution("time_delta", time_delta=3, label_name="behavior2")
    config.push()

    with pytest.raises(RuntimeError, match="Feature ranges are not unique"):
        config.finalize()
