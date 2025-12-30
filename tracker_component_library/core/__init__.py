"""
Core utilities and constants for the Tracker Component Library.

This module provides foundational functionality used throughout the library:
- Physical and mathematical constants
- Input validation utilities
- Array manipulation helpers compatible with MATLAB conventions
"""

from tracker_component_library.core.constants import (
    SPEED_OF_LIGHT,
    GRAVITATIONAL_CONSTANT,
    EARTH_SEMI_MAJOR_AXIS,
    EARTH_FLATTENING,
    EARTH_ROTATION_RATE,
    WGS84,
    PhysicalConstants,
)
from tracker_component_library.core.validation import (
    validate_array,
    ensure_2d,
    ensure_column_vector,
    ensure_row_vector,
)
from tracker_component_library.core.array_utils import (
    wrap_to_pi,
    wrap_to_2pi,
    wrap_to_range,
    column_vector,
    row_vector,
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
    "validate_array",
    "ensure_2d",
    "ensure_column_vector",
    "ensure_row_vector",
    # Array utilities
    "wrap_to_pi",
    "wrap_to_2pi",
    "wrap_to_range",
    "column_vector",
    "row_vector",
]
