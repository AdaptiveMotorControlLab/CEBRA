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
import collections.abc as collections_abc

import pytest
import sklearn.utils.estimator_checks
import torch


def requires_cuda(function):
    """Skip tests that require CUDA access if not GPU is available."""
    if torch.cuda.is_available():
        return function
    else:
        decorator = pytest.mark.skip(reason="CUDA not available.")
        return decorator(function)


def _pytest_params(fast_arguments, slow_arguments):
    pytest_params = []
    for fast_arg in fast_arguments:
        if isinstance(fast_arg,
                      collections_abc.Iterable):  # if more than one argument
            pytest_params.append(pytest.param(*fast_arg,
                                              marks=pytest.mark.fast))
        else:  # if only one argument
            pytest_params.append(pytest.param(fast_arg, marks=pytest.mark.fast))

    for slow_arg in slow_arguments:
        if isinstance(slow_arg, collections_abc.Iterable):
            pytest_params.append(pytest.param(*slow_arg,
                                              marks=pytest.mark.slow))
        else:
            pytest_params.append(pytest.param(slow_arg, marks=pytest.mark.slow))

    return pytest_params


def parametrize_slow(arg_names, fast_arguments, slow_arguments):
    return pytest.mark.parametrize(
        arg_names,
        _pytest_params(fast_arguments, slow_arguments),
    )


def parametrize_with_checks_slow(fast_arguments, slow_arguments):
    fast_params = [
        list(
            sklearn.utils.estimator_checks.check_estimator(
                fast_arg, generate_only=True))[0] for fast_arg in fast_arguments
    ]
    slow_params = [
        list(
            sklearn.utils.estimator_checks.check_estimator(
                slow_arg, generate_only=True))[0] for slow_arg in slow_arguments
    ]
    return parametrize_slow("estimator,check", fast_params, slow_params)
