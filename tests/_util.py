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
import inspect

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


def parametrize_with_checks_slow(fast_arguments, slow_arguments, generate_only=True):
    """Parametrize tests with sklearn estimator checks, supporting fast/slow test modes.
    
    Args:
        fast_arguments: List of estimators to use for fast tests.
        slow_arguments: List of estimators to use for slow tests.
        generate_only: If True, only generate tests without running them (default: True).
                      This parameter is only used with sklearn < 1.5. In newer versions,
                      tests are always generated (not run immediately).
    
    Returns:
        A pytest parametrize decorator configured with fast and slow test parameters.
    """
    # Check if check_estimator supports generate_only parameter (sklearn < 1.5)
    check_estimator_sig = inspect.signature(sklearn.utils.estimator_checks.check_estimator)
    supports_generate_only = 'generate_only' in check_estimator_sig.parameters
    
    if supports_generate_only:
        # Old sklearn API (<= 1.4.x): use check_estimator with generate_only=True
        fast_params = [
            list(
                sklearn.utils.estimator_checks.check_estimator(
                    fast_arg, generate_only=generate_only))[0] for fast_arg in fast_arguments
        ]
        slow_params = [
            list(
                sklearn.utils.estimator_checks.check_estimator(
                    slow_arg, generate_only=generate_only))[0] for slow_arg in slow_arguments
        ]
    else:
        # New sklearn API (>= 1.5): use parametrize_with_checks to get test params
        # For each estimator, get the first check
        fast_params = []
        for fast_arg in fast_arguments:
            decorator = sklearn.utils.estimator_checks.parametrize_with_checks([fast_arg])
            # Extract the generator from the decorator and get first item
            gen = decorator.mark.args[1]
            fast_params.append(next(gen))
            
        slow_params = []
        for slow_arg in slow_arguments:
            decorator = sklearn.utils.estimator_checks.parametrize_with_checks([slow_arg])
            # Extract the generator from the decorator and get first item
            gen = decorator.mark.args[1]
            slow_params.append(next(gen))
    
    return parametrize_slow("estimator,check", fast_params, slow_params)


def parametrize_device(func):
    _devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
    return pytest.mark.parametrize("device", _devices)(func)
