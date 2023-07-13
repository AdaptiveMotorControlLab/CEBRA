"""Configuration options for pytest

See Also:
    * https://docs.pytest.org/en/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
"""

import pytest


def pytest_addoption(parser):
    """Define customized pytest flags.

    Examples:
        >>> pytest tests/test_sklearn.py --runfast
    """
    parser.addoption("--runfast",
                     action="store_true",
                     default=False,
                     help="don't run slow tests")
    parser.addoption("--runbenchmark",
                     action="store_true",
                     default=False,
                     help="run benchmark test")


def pytest_configure(config):
    """Describe customized markers."""
    config.addinivalue_line("markers", "slow: run tests with slow arguments")
    config.addinivalue_line("markers", "fast: run tests with fast arguments")
    config.addinivalue_line("markers", "benchmark: run benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Select tests to skip based on current flag.

    By default, slow arguments are used and fast ones are skipped.

    If runfast flag is provided, tests are run with the arguments marked
    as fast, arguments marked as slow are be skipped.

    """
    if config.getoption("--runfast"):
        # --runfast given in cli: skip slow tests
        skip_slow = pytest.mark.skip(
            reason="test marked as slowing a --runfast mode")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    else:
        skip_fast = pytest.mark.skip(
            reason="test marked as fast, run only in --runfast mode")
        for item in items:
            if "fast" in item.keywords:
                item.add_marker(skip_fast)

    if not config.getoption("--runbenchmark"):
        # --runbenchmark: run benchmark testing
        skip_benchmark = pytest.mark.skip(reason="skip benchmark testing")
        for item in items:
            if "benchmark" in item.keywords:
                item.add_marker(skip_benchmark)
