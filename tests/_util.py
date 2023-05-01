import pytest
import torch


def requires_cuda(function):
    """Skip tests that require CUDA access if not GPU is available."""
    if torch.cuda.is_available():
        return function
    else:
        decorator = pytest.mark.skip(reason="CUDA not available.")
        return decorator(function)
