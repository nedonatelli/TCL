"""
Core utilities and constants for the Tracker Component Library.

This module provides foundational functionality used throughout the library:
- Physical and mathematical constants
- Input validation utilities
- Array manipulation helpers compatible with MATLAB conventions
"""

from pytcl.core.array_utils import (
    column_vector,
    row_vector,
    wrap_to_2pi,
    wrap_to_pi,
    wrap_to_range,
)
from pytcl.core.constants import (
    EARTH_FLATTENING,
    EARTH_ROTATION_RATE,
    EARTH_SEMI_MAJOR_AXIS,
    GRAVITATIONAL_CONSTANT,
    SPEED_OF_LIGHT,
    WGS84,
    PhysicalConstants,
)
from pytcl.core.validation import (
    ArraySpec,
    ScalarSpec,
    ValidationError,
    check_compatible_shapes,
    ensure_2d,
    ensure_column_vector,
    ensure_positive_definite,
    ensure_row_vector,
    ensure_square_matrix,
    ensure_symmetric,
    validate_array,
    validate_inputs,
    validate_same_shape,
)

__all__ = [
    # Constants
    "SPEED_OF_LIGHT",
    "GRAVITATIONAL_CONSTANT",
    "EARTH_SEMI_MAJOR_AXIS",
    "EARTH_FLATTENING",
    "EARTH_ROTATION_RATE",
    "WGS84",
    "PhysicalConstants",
    # Validation
    "ValidationError",
    "validate_array",
    "validate_inputs",
    "validate_same_shape",
    "check_compatible_shapes",
    "ArraySpec",
    "ScalarSpec",
    "ensure_2d",
    "ensure_column_vector",
    "ensure_row_vector",
    "ensure_square_matrix",
    "ensure_symmetric",
    "ensure_positive_definite",
    # Array utilities
    "wrap_to_pi",
    "wrap_to_2pi",
    "wrap_to_range",
    "column_vector",
    "row_vector",
]
