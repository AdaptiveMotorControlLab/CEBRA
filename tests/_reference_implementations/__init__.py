"""Reference implementations for testing consistency and backward compatibility.

This package contains reference implementations of previously deprecated or
parametrized model components, used for testing consistency and backward compatibility
in the test suite.
"""

from .deprecated_transforms import (
    cebra_transform_deprecated,
    multiobjective_transform_deprecated,
)
from .reference_offset_models import (
    Offset5ModelReference,
    Offset10ModelReference,
    Offset15ModelReference,
    Offset20ModelReference,
    Offset36Reference,
    Offset40Reference,
    Offset50Reference,
)

__all__ = [
    "cebra_transform_deprecated",
    "multiobjective_transform_deprecated",
    "Offset5ModelReference",
    "Offset10ModelReference",
    "Offset15ModelReference",
    "Offset20ModelReference",
    "Offset36Reference",
    "Offset40Reference",
    "Offset50Reference",
]
