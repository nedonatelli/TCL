"""
Core utilities and constants for the Tracker Component Library.

This module provides foundational functionality used throughout the library:
- Physical and mathematical constants
- Input validation utilities
- Array manipulation helpers compatible with MATLAB conventions
- Custom exception hierarchy for consistent error handling
- Optional dependency management
- Module maturity classification system
"""

from pytcl.core.array_utils import (
    block_diag,
    column_vector,
    is_positive_definite,
    is_positive_semidefinite,
    make_readonly,
    meshgrid_ij,
    nearest_positive_definite,
    normalize_vector,
    outer_product,
    repmat,
    row_vector,
    safe_cholesky,
    skew_symmetric,
    unskew,
    unvec,
    vec,
    wrap_to_2pi,
    wrap_to_360,
    wrap_to_pi,
    wrap_to_pm180,
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
from pytcl.core.exceptions import (
    ComputationError,
    ConfigurationError,
    ConvergenceError,
    DataError,
    DependencyError,
    DimensionError,
    EmptyContainerError,
    FormatError,
    MethodError,
    NumericalError,
    ParameterError,
    ParseError,
    RangeError,
    SingularMatrixError,
    StateError,
    TCLError,
    UninitializedError,
    ValidationError,
)
from pytcl.core.maturity import (
    MaturityLevel,
    get_maturity,
    get_maturity_summary,
    get_modules_by_maturity,
    is_production_ready,
    is_stable,
)
from pytcl.core.optional_deps import (
    LazyModule,
    check_dependencies,
    import_optional,
    is_available,
    requires,
)
from pytcl.core.paths import (
    ensure_data_dir,
    get_data_dir,
)
from pytcl.core.validation import (
    ArraySpec,
    ScalarSpec,
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
    # Exceptions (base)
    "TCLError",
    # Exceptions (validation)
    "ValidationError",
    "DimensionError",
    "ParameterError",
    "RangeError",
    # Exceptions (computation)
    "ComputationError",
    "ConvergenceError",
    "NumericalError",
    "SingularMatrixError",
    # Exceptions (state)
    "StateError",
    "UninitializedError",
    "EmptyContainerError",
    # Exceptions (configuration)
    "ConfigurationError",
    "MethodError",
    "DependencyError",
    # Exceptions (data)
    "DataError",
    "FormatError",
    "ParseError",
    # Validation utilities
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
    # Array utilities. array_utils is a public module of general-purpose
    # helpers and every public name in it is re-exported here. The previous
    # selection had no rationale: wrap_to_pi/wrap_to_2pi/wrap_to_range were
    # exported while wrap_to_360/wrap_to_pm180 were not, and column_vector and
    # row_vector while vec and unvec were not.
    "block_diag",
    "column_vector",
    "is_positive_definite",
    "is_positive_semidefinite",
    "make_readonly",
    "meshgrid_ij",
    "nearest_positive_definite",
    "normalize_vector",
    "outer_product",
    "repmat",
    "row_vector",
    "safe_cholesky",
    "skew_symmetric",
    "unskew",
    "unvec",
    "vec",
    "wrap_to_2pi",
    "wrap_to_360",
    "wrap_to_pi",
    "wrap_to_pm180",
    "wrap_to_range",
    # Optional dependencies
    "is_available",
    "import_optional",
    "requires",
    "check_dependencies",
    "LazyModule",
    # Data directory
    "get_data_dir",
    "ensure_data_dir",
    # Maturity classification
    "MaturityLevel",
    "get_maturity",
    "get_modules_by_maturity",
    "get_maturity_summary",
    "is_stable",
    "is_production_ready",
]
