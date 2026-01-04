"""
Custom exception hierarchy for the Tracker Component Library.

This module provides a structured exception hierarchy for consistent error
handling across the library. Custom exceptions enable more specific error
catching, better error messages, and improved debugging.

Exception Hierarchy
-------------------
TCLError (base)
├── ValidationError (input validation failures)
│   ├── DimensionError (array shape/dimension mismatches)
│   ├── ParameterError (invalid parameter values)
│   └── RangeError (out-of-range values)
├── ComputationError (numerical computation failures)
│   ├── ConvergenceError (iterative algorithm non-convergence)
│   ├── NumericalError (numerical stability issues)
│   └── SingularMatrixError (singular/non-invertible matrix)
├── StateError (object state violations)
│   ├── UninitializedError (object not initialized)
│   └── EmptyContainerError (container has no elements)
├── ConfigurationError (configuration/setup issues)
│   ├── MethodError (invalid method/algorithm selection)
│   └── DependencyError (missing optional dependency)
└── DataError (data format/structure issues)
    ├── FormatError (invalid data format)
    └── ParseError (data parsing failures)

Examples
--------
Catching specific exception types:

>>> from pytcl.core.exceptions import ConvergenceError, ParameterError
>>> try:
...     result = solve_kepler(M=1.5, e=1.5)  # Invalid eccentricity
... except ParameterError as e:
...     print(f"Invalid parameter: {e}")

Catching all TCL errors:

>>> from pytcl.core.exceptions import TCLError
>>> try:
...     result = compute_orbit(...)
... except TCLError as e:
...     print(f"TCL error: {e}")
"""

from typing import Any, Optional, Sequence


class TCLError(Exception):
    """
    Base exception for all Tracker Component Library errors.

    All custom exceptions in the library inherit from this class,
    allowing users to catch all TCL-specific errors with a single
    except clause.

    Parameters
    ----------
    message : str
        Human-readable error message.
    details : dict, optional
        Additional context about the error (parameter values, etc.).

    Examples
    --------
    >>> raise TCLError("Something went wrong", details={"value": 42})
    """

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# =============================================================================
# Validation Errors
# =============================================================================


class ValidationError(TCLError, ValueError):
    """
    Base exception for input validation failures.

    Raised when function inputs fail validation checks. This extends
    both TCLError (for TCL-specific catching) and ValueError (for
    compatibility with code expecting standard Python exceptions).

    Parameters
    ----------
    message : str
        Description of the validation failure.
    parameter : str, optional
        Name of the parameter that failed validation.
    expected : str, optional
        Description of what was expected.
    actual : Any, optional
        The actual value that was provided.

    Examples
    --------
    >>> raise ValidationError(
    ...     "Invalid matrix dimensions",
    ...     parameter="P",
    ...     expected="3x3 matrix",
    ...     actual="2x4 array"
    ... )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[Any] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if parameter is not None:
            details["parameter"] = parameter
        if expected is not None:
            details["expected"] = expected
        if actual is not None:
            details["actual"] = actual
        details.update(kwargs)
        super().__init__(message, details)
        self.parameter = parameter
        self.expected = expected
        self.actual = actual


class DimensionError(ValidationError):
    """
    Exception for array dimension and shape mismatches.

    Raised when array dimensions or shapes don't match requirements
    or aren't compatible with each other.

    Parameters
    ----------
    message : str
        Description of the dimension mismatch.
    expected_shape : tuple, optional
        Expected shape.
    actual_shape : tuple, optional
        Actual shape.
    parameter : str, optional
        Name of the parameter.

    Examples
    --------
    >>> raise DimensionError(
    ...     "Covariance matrix must be 3x3",
    ...     expected_shape=(3, 3),
    ...     actual_shape=(2, 4),
    ...     parameter="P"
    ... )
    """

    def __init__(
        self,
        message: str,
        expected_shape: Optional[tuple[int, ...]] = None,
        actual_shape: Optional[tuple[int, ...]] = None,
        parameter: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if expected_shape is not None:
            details["expected_shape"] = expected_shape
        if actual_shape is not None:
            details["actual_shape"] = actual_shape
        super().__init__(message, parameter=parameter, **details, **kwargs)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class ParameterError(ValidationError):
    """
    Exception for invalid parameter values.

    Raised when a parameter value violates constraints (type, value
    range, allowed values, etc.).

    Parameters
    ----------
    message : str
        Description of the invalid parameter.
    parameter : str, optional
        Name of the invalid parameter.
    value : Any, optional
        The invalid value.
    constraint : str, optional
        Description of the constraint that was violated.

    Examples
    --------
    >>> raise ParameterError(
    ...     "Variance must be positive",
    ...     parameter="variance",
    ...     value=-1.0,
    ...     constraint="must be > 0"
    ... )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if value is not None:
            details["value"] = value
        if constraint is not None:
            details["constraint"] = constraint
        super().__init__(message, parameter=parameter, **details, **kwargs)
        self.value = value
        self.constraint = constraint


class RangeError(ValidationError):
    """
    Exception for out-of-range values.

    Raised when a numeric value falls outside an allowed range.

    Parameters
    ----------
    message : str
        Description of the range violation.
    parameter : str, optional
        Name of the parameter.
    value : float, optional
        The out-of-range value.
    min_value : float, optional
        Minimum allowed value.
    max_value : float, optional
        Maximum allowed value.

    Examples
    --------
    >>> raise RangeError(
    ...     "Eccentricity must be in [0, 1) for elliptic orbits",
    ...     parameter="e",
    ...     value=1.5,
    ...     min_value=0.0,
    ...     max_value=1.0
    ... )
    """

    def __init__(
        self,
        message: str,
        parameter: Optional[str] = None,
        value: Optional[float] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if value is not None:
            details["value"] = value
        if min_value is not None:
            details["min"] = min_value
        if max_value is not None:
            details["max"] = max_value
        super().__init__(message, parameter=parameter, **details, **kwargs)
        self.value = value
        self.min_value = min_value
        self.max_value = max_value


# =============================================================================
# Computation Errors
# =============================================================================


class ComputationError(TCLError, RuntimeError):
    """
    Base exception for numerical computation failures.

    Raised when a numerical algorithm fails to produce a valid result.
    This extends RuntimeError for compatibility with code that catches
    standard computation errors.

    Parameters
    ----------
    message : str
        Description of the computation failure.
    algorithm : str, optional
        Name of the algorithm that failed.

    Examples
    --------
    >>> raise ComputationError(
    ...     "Failed to compute eigenvalues",
    ...     algorithm="numpy.linalg.eig"
    ... )
    """

    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if algorithm is not None:
            details["algorithm"] = algorithm
        details.update(kwargs)
        super().__init__(message, details)
        self.algorithm = algorithm


class ConvergenceError(ComputationError):
    """
    Exception for iterative algorithm convergence failures.

    Raised when an iterative algorithm fails to converge within
    the maximum number of iterations.

    Parameters
    ----------
    message : str
        Description of the convergence failure.
    algorithm : str, optional
        Name of the algorithm.
    iterations : int, optional
        Number of iterations performed.
    max_iterations : int, optional
        Maximum iterations allowed.
    residual : float, optional
        Final residual or error value.
    tolerance : float, optional
        Convergence tolerance.

    Examples
    --------
    >>> raise ConvergenceError(
    ...     "Kepler's equation did not converge",
    ...     algorithm="Newton-Raphson",
    ...     iterations=100,
    ...     max_iterations=100,
    ...     residual=1e-5,
    ...     tolerance=1e-12
    ... )
    """

    def __init__(
        self,
        message: str,
        algorithm: Optional[str] = None,
        iterations: Optional[int] = None,
        max_iterations: Optional[int] = None,
        residual: Optional[float] = None,
        tolerance: Optional[float] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if iterations is not None:
            details["iterations"] = iterations
        if max_iterations is not None:
            details["max_iterations"] = max_iterations
        if residual is not None:
            details["residual"] = residual
        if tolerance is not None:
            details["tolerance"] = tolerance
        super().__init__(message, algorithm=algorithm, **details, **kwargs)
        self.iterations = iterations
        self.max_iterations = max_iterations
        self.residual = residual
        self.tolerance = tolerance


class NumericalError(ComputationError):
    """
    Exception for numerical stability issues.

    Raised when numerical operations fail due to precision issues,
    overflow, underflow, or ill-conditioned computations.

    Parameters
    ----------
    message : str
        Description of the numerical issue.
    operation : str, optional
        The operation that failed.
    condition_number : float, optional
        Matrix condition number (if applicable).

    Examples
    --------
    >>> raise NumericalError(
    ...     "Matrix is ill-conditioned",
    ...     operation="matrix inversion",
    ...     condition_number=1e16
    ... )
    """

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        condition_number: Optional[float] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if operation is not None:
            details["operation"] = operation
        if condition_number is not None:
            details["condition_number"] = condition_number
        super().__init__(message, **details, **kwargs)
        self.operation = operation
        self.condition_number = condition_number


class SingularMatrixError(ComputationError):
    """
    Exception for singular or non-invertible matrix operations.

    Raised when a matrix operation requires an invertible matrix
    but the matrix is singular or nearly singular.

    Parameters
    ----------
    message : str
        Description of the singularity issue.
    matrix_name : str, optional
        Name of the singular matrix.
    determinant : float, optional
        Determinant value (near zero).

    Examples
    --------
    >>> raise SingularMatrixError(
    ...     "Covariance matrix is singular",
    ...     matrix_name="P",
    ...     determinant=1e-20
    ... )
    """

    def __init__(
        self,
        message: str,
        matrix_name: Optional[str] = None,
        determinant: Optional[float] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if matrix_name is not None:
            details["matrix"] = matrix_name
        if determinant is not None:
            details["determinant"] = determinant
        super().__init__(message, **details, **kwargs)
        self.matrix_name = matrix_name
        self.determinant = determinant


# =============================================================================
# State Errors
# =============================================================================


class StateError(TCLError):
    """
    Base exception for object state violations.

    Raised when an operation is attempted on an object that is not
    in a valid state for that operation.

    Parameters
    ----------
    message : str
        Description of the state violation.
    object_type : str, optional
        Type of the object.
    current_state : str, optional
        Description of the current state.
    required_state : str, optional
        Description of the required state.

    Examples
    --------
    >>> raise StateError(
    ...     "Cannot update without prediction",
    ...     object_type="KalmanFilter",
    ...     current_state="uninitialized",
    ...     required_state="predicted"
    ... )
    """

    def __init__(
        self,
        message: str,
        object_type: Optional[str] = None,
        current_state: Optional[str] = None,
        required_state: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if object_type is not None:
            details["object_type"] = object_type
        if current_state is not None:
            details["current_state"] = current_state
        if required_state is not None:
            details["required_state"] = required_state
        details.update(kwargs)
        super().__init__(message, details)
        self.object_type = object_type
        self.current_state = current_state
        self.required_state = required_state


class UninitializedError(StateError):
    """
    Exception for uninitialized object access.

    Raised when an operation requires an initialized object but
    the object hasn't been properly initialized.

    Parameters
    ----------
    message : str
        Description of the issue.
    object_type : str, optional
        Type of the uninitialized object.
    required_initialization : str, optional
        What initialization is required.

    Examples
    --------
    >>> raise UninitializedError(
    ...     "Tracker not initialized",
    ...     object_type="SingleTargetTracker",
    ...     required_initialization="call initialize() first"
    ... )
    """

    def __init__(
        self,
        message: str,
        object_type: Optional[str] = None,
        required_initialization: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            message,
            object_type=object_type,
            current_state="uninitialized",
            required_state=required_initialization,
            **kwargs,
        )


class EmptyContainerError(StateError):
    """
    Exception for empty container operations.

    Raised when an operation requires a non-empty container but
    the container has no elements.

    Parameters
    ----------
    message : str
        Description of the issue.
    container_type : str, optional
        Type of the empty container.
    operation : str, optional
        The operation that requires non-empty container.

    Examples
    --------
    >>> raise EmptyContainerError(
    ...     "Cannot query empty RTree",
    ...     container_type="RTree",
    ...     operation="nearest neighbor query"
    ... )
    """

    def __init__(
        self,
        message: str,
        container_type: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if operation is not None:
            details["operation"] = operation
        super().__init__(
            message,
            object_type=container_type,
            current_state="empty",
            **details,
            **kwargs,
        )


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(TCLError):
    """
    Base exception for configuration and setup issues.

    Raised when there are problems with algorithm configuration,
    method selection, or dependency availability.

    Parameters
    ----------
    message : str
        Description of the configuration issue.

    Examples
    --------
    >>> raise ConfigurationError("Invalid filter configuration")
    """

    pass


class MethodError(ConfigurationError, ValueError):
    """
    Exception for invalid method or algorithm selection.

    Raised when an unknown or unsupported method/algorithm is specified.

    Parameters
    ----------
    message : str
        Description of the issue.
    method : str, optional
        The invalid method name.
    valid_methods : sequence of str, optional
        List of valid method names.

    Examples
    --------
    >>> raise MethodError(
    ...     "Unknown assignment method",
    ...     method="invalid_method",
    ...     valid_methods=["hungarian", "auction", "greedy"]
    ... )
    """

    def __init__(
        self,
        message: str,
        method: Optional[str] = None,
        valid_methods: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if method is not None:
            details["method"] = method
        if valid_methods is not None:
            details["valid_methods"] = list(valid_methods)
        super().__init__(message, details)
        self.method = method
        self.valid_methods = valid_methods


class DependencyError(ConfigurationError, ImportError):
    """
    Exception for missing optional dependencies.

    Raised when an optional dependency is required but not installed.

    Parameters
    ----------
    message : str
        Description of the missing dependency.
    package : str, optional
        Name of the missing package.
    feature : str, optional
        Feature that requires the dependency.
    install_command : str, optional
        Command to install the dependency.

    Examples
    --------
    >>> raise DependencyError(
    ...     "plotly is required for interactive plotting",
    ...     package="plotly",
    ...     feature="3D visualization",
    ...     install_command="pip install plotly"
    ... )
    """

    def __init__(
        self,
        message: str,
        package: Optional[str] = None,
        feature: Optional[str] = None,
        install_command: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if package is not None:
            details["package"] = package
        if feature is not None:
            details["feature"] = feature
        if install_command is not None:
            details["install"] = install_command
        super().__init__(message, details)
        self.package = package
        self.feature = feature
        self.install_command = install_command


# =============================================================================
# Data Errors
# =============================================================================


class DataError(TCLError):
    """
    Base exception for data format and structure issues.

    Raised when input data has format or structural problems.

    Parameters
    ----------
    message : str
        Description of the data issue.

    Examples
    --------
    >>> raise DataError("Invalid input data format")
    """

    pass


class FormatError(DataError, ValueError):
    """
    Exception for invalid data format.

    Raised when data doesn't conform to the expected format.

    Parameters
    ----------
    message : str
        Description of the format issue.
    expected_format : str, optional
        Description of the expected format.
    actual_format : str, optional
        Description of the actual format.

    Examples
    --------
    >>> raise FormatError(
    ...     "Invalid TLE format",
    ...     expected_format="69 characters per line",
    ...     actual_format="line 1 has 65 characters"
    ... )
    """

    def __init__(
        self,
        message: str,
        expected_format: Optional[str] = None,
        actual_format: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if expected_format is not None:
            details["expected"] = expected_format
        if actual_format is not None:
            details["actual"] = actual_format
        super().__init__(message, details)
        self.expected_format = expected_format
        self.actual_format = actual_format


class ParseError(DataError, ValueError):
    """
    Exception for data parsing failures.

    Raised when data cannot be parsed or interpreted.

    Parameters
    ----------
    message : str
        Description of the parsing failure.
    data_type : str, optional
        Type of data being parsed.
    position : int or tuple, optional
        Position where parsing failed.
    reason : str, optional
        Reason for the parsing failure.

    Examples
    --------
    >>> raise ParseError(
    ...     "Failed to parse TLE checksum",
    ...     data_type="TLE",
    ...     position=68,
    ...     reason="invalid checksum digit"
    ... )
    """

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        position: Optional[int | tuple[int, ...]] = None,
        reason: Optional[str] = None,
        **kwargs: Any,
    ):
        details: dict[str, Any] = {}
        if data_type is not None:
            details["data_type"] = data_type
        if position is not None:
            details["position"] = position
        if reason is not None:
            details["reason"] = reason
        super().__init__(message, details)
        self.data_type = data_type
        self.position = position
        self.reason = reason


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Base exception
    "TCLError",
    # Validation errors
    "ValidationError",
    "DimensionError",
    "ParameterError",
    "RangeError",
    # Computation errors
    "ComputationError",
    "ConvergenceError",
    "NumericalError",
    "SingularMatrixError",
    # State errors
    "StateError",
    "UninitializedError",
    "EmptyContainerError",
    # Configuration errors
    "ConfigurationError",
    "MethodError",
    "DependencyError",
    # Data errors
    "DataError",
    "FormatError",
    "ParseError",
]
