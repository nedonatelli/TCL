"""
Input validation utilities for the Tracker Component Library.

This module provides decorators and functions for validating input arrays,
ensuring consistent behavior across the library and providing helpful error
messages when inputs don't meet requirements.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Literal, Sequence, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Type variable for generic function signatures
F = TypeVar("F", bound=Callable[..., Any])


class ValidationError(ValueError):
    """Exception raised when input validation fails."""

    pass


def validate_array(
    arr: ArrayLike,
    name: str = "array",
    *,
    dtype: type | np.dtype[Any] | None = None,
    ndim: int | tuple[int, ...] | None = None,
    shape: tuple[int | None, ...] | None = None,
    min_ndim: int | None = None,
    max_ndim: int | None = None,
    finite: bool = False,
    non_negative: bool = False,
    positive: bool = False,
    allow_empty: bool = True,
) -> NDArray[Any]:
    """
    Validate and convert an array-like input to a NumPy array.

    Parameters
    ----------
    arr : array_like
        Input to validate and convert.
    name : str, optional
        Name of the parameter (for error messages). Default is "array".
    dtype : type or np.dtype, optional
        If provided, ensure the array has this dtype (or can be safely cast).
    ndim : int or tuple of int, optional
        If provided, ensure the array has exactly this number of dimensions.
        Can be a tuple to allow multiple valid dimensionalities.
    shape : tuple, optional
        If provided, validate the shape. Use None for dimensions that can be any size.
        Example: (3, None) requires first dimension to be 3, second can be any size.
    min_ndim : int, optional
        Minimum number of dimensions required.
    max_ndim : int, optional
        Maximum number of dimensions allowed.
    finite : bool, optional
        If True, ensure all elements are finite (no inf or nan). Default is False.
    non_negative : bool, optional
        If True, ensure all elements are >= 0. Default is False.
    positive : bool, optional
        If True, ensure all elements are > 0. Default is False.
    allow_empty : bool, optional
        If False, raise an error for empty arrays. Default is True.

    Returns
    -------
    NDArray
        Validated NumPy array.

    Raises
    ------
    ValidationError
        If the input fails any validation check.

    Examples
    --------
    >>> validate_array([1, 2, 3], "position", ndim=1, finite=True)
    array([1, 2, 3])

    >>> validate_array([[1, 2], [3, 4]], "matrix", shape=(2, 2))
    array([[1, 2],
           [3, 4]])
    """
    # Convert to array
    try:
        result = np.asarray(arr)
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Cannot convert {name} to array: {e}") from e

    # Apply dtype if specified
    if dtype is not None:
        try:
            result = result.astype(dtype, copy=False)
        except (ValueError, TypeError) as e:
            raise ValidationError(f"Cannot convert {name} to dtype {dtype}: {e}") from e

    # Check if empty
    if not allow_empty and result.size == 0:
        raise ValidationError(f"{name} cannot be empty")

    # Check ndim
    if ndim is not None:
        valid_ndims = (ndim,) if isinstance(ndim, int) else ndim
        if result.ndim not in valid_ndims:
            raise ValidationError(
                f"{name} must have {ndim} dimension(s), got {result.ndim}"
            )

    if min_ndim is not None and result.ndim < min_ndim:
        raise ValidationError(
            f"{name} must have at least {min_ndim} dimension(s), got {result.ndim}"
        )

    if max_ndim is not None and result.ndim > max_ndim:
        raise ValidationError(
            f"{name} must have at most {max_ndim} dimension(s), got {result.ndim}"
        )

    # Check shape
    if shape is not None:
        if len(shape) != result.ndim:
            raise ValidationError(
                f"{name} must have {len(shape)} dimensions, got {result.ndim}"
            )
        for i, (expected, actual) in enumerate(zip(shape, result.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(
                    f"{name} dimension {i} must be {expected}, got {actual}"
                )

    # Check finite
    if finite and not np.all(np.isfinite(result)):
        raise ValidationError(f"{name} must contain only finite values")

    # Check non-negative
    if non_negative and np.any(result < 0):
        raise ValidationError(f"{name} must contain only non-negative values")

    # Check positive
    if positive and np.any(result <= 0):
        raise ValidationError(f"{name} must contain only positive values")

    return result


def ensure_2d(
    arr: ArrayLike,
    name: str = "array",
    axis: Literal["row", "column", "auto"] = "auto",
) -> NDArray[Any]:
    """
    Ensure an array is 2D, promoting 1D arrays as needed.

    Parameters
    ----------
    arr : array_like
        Input array.
    name : str, optional
        Name of the parameter (for error messages).
    axis : {'row', 'column', 'auto'}, optional
        How to promote 1D arrays:
        - 'row': Make 1D array a row vector (1, n)
        - 'column': Make 1D array a column vector (n, 1)
        - 'auto': Preserve as-is for 2D, use 'column' for 1D

    Returns
    -------
    NDArray
        2D array.

    Examples
    --------
    >>> ensure_2d([1, 2, 3], axis='column')
    array([[1],
           [2],
           [3]])

    >>> ensure_2d([1, 2, 3], axis='row')
    array([[1, 2, 3]])
    """
    result = validate_array(arr, name, min_ndim=1, max_ndim=2)

    if result.ndim == 1:
        if axis == "row":
            result = result.reshape(1, -1)
        elif axis == "column" or axis == "auto":
            result = result.reshape(-1, 1)

    return result


def ensure_column_vector(arr: ArrayLike, name: str = "vector") -> NDArray[Any]:
    """
    Ensure input is a column vector (n, 1).

    Parameters
    ----------
    arr : array_like
        Input array, must be 1D or a column vector.
    name : str, optional
        Name of the parameter (for error messages).

    Returns
    -------
    NDArray
        Column vector with shape (n, 1).

    Examples
    --------
    >>> ensure_column_vector([1, 2, 3])
    array([[1],
           [2],
           [3]])
    """
    result = validate_array(arr, name, min_ndim=1, max_ndim=2)

    if result.ndim == 1:
        return result.reshape(-1, 1)
    elif result.ndim == 2:
        if result.shape[1] != 1:
            raise ValidationError(
                f"{name} must be a column vector (n, 1), got shape {result.shape}"
            )
        return result
    else:
        raise ValidationError(f"{name} must be 1D or 2D, got {result.ndim}D")


def ensure_row_vector(arr: ArrayLike, name: str = "vector") -> NDArray[Any]:
    """
    Ensure input is a row vector (1, n).

    Parameters
    ----------
    arr : array_like
        Input array, must be 1D or a row vector.
    name : str, optional
        Name of the parameter (for error messages).

    Returns
    -------
    NDArray
        Row vector with shape (1, n).

    Examples
    --------
    >>> ensure_row_vector([1, 2, 3])
    array([[1, 2, 3]])
    """
    result = validate_array(arr, name, min_ndim=1, max_ndim=2)

    if result.ndim == 1:
        return result.reshape(1, -1)
    elif result.ndim == 2:
        if result.shape[0] != 1:
            raise ValidationError(
                f"{name} must be a row vector (1, n), got shape {result.shape}"
            )
        return result
    else:
        raise ValidationError(f"{name} must be 1D or 2D, got {result.ndim}D")


def ensure_square_matrix(arr: ArrayLike, name: str = "matrix") -> NDArray[Any]:
    """
    Ensure input is a square matrix.

    Parameters
    ----------
    arr : array_like
        Input array.
    name : str, optional
        Name of the parameter (for error messages).

    Returns
    -------
    NDArray
        Square matrix.

    Raises
    ------
    ValidationError
        If input is not a 2D square array.
    """
    result = validate_array(arr, name, ndim=2)

    if result.shape[0] != result.shape[1]:
        raise ValidationError(f"{name} must be square, got shape {result.shape}")

    return result


def ensure_symmetric(
    arr: ArrayLike,
    name: str = "matrix",
    rtol: float = 1e-10,
    atol: float = 1e-10,
) -> NDArray[Any]:
    """
    Ensure input is a symmetric matrix.

    Parameters
    ----------
    arr : array_like
        Input array.
    name : str, optional
        Name of the parameter (for error messages).
    rtol : float, optional
        Relative tolerance for symmetry check. Default is 1e-10.
    atol : float, optional
        Absolute tolerance for symmetry check. Default is 1e-10.

    Returns
    -------
    NDArray
        Symmetric matrix (symmetrized if nearly symmetric).

    Raises
    ------
    ValidationError
        If input is not symmetric within tolerance.
    """
    result = ensure_square_matrix(arr, name)

    if not np.allclose(result, result.T, rtol=rtol, atol=atol):
        raise ValidationError(f"{name} must be symmetric")

    # Enforce exact symmetry
    return (result + result.T) / 2


def ensure_positive_definite(
    arr: ArrayLike,
    name: str = "matrix",
    rtol: float = 1e-10,
) -> NDArray[Any]:
    """
    Ensure input is a positive definite matrix.

    Parameters
    ----------
    arr : array_like
        Input array.
    name : str, optional
        Name of the parameter (for error messages).
    rtol : float, optional
        Relative tolerance for eigenvalue check. Default is 1e-10.

    Returns
    -------
    NDArray
        Positive definite matrix.

    Raises
    ------
    ValidationError
        If input is not positive definite.
    """
    result = ensure_symmetric(arr, name)

    try:
        eigenvalues = np.linalg.eigvalsh(result)
    except np.linalg.LinAlgError as e:
        raise ValidationError(f"Could not compute eigenvalues of {name}: {e}") from e

    min_eigenvalue = np.min(eigenvalues)
    threshold = -rtol * np.max(np.abs(eigenvalues))

    if min_eigenvalue < threshold:
        raise ValidationError(
            f"{name} must be positive definite, "
            f"minimum eigenvalue is {min_eigenvalue:.2e}"
        )

    return result


def validate_same_shape(*arrays: ArrayLike, names: Sequence[str] | None = None) -> None:
    """
    Validate that all input arrays have the same shape.

    Parameters
    ----------
    *arrays : array_like
        Arrays to compare.
    names : sequence of str, optional
        Names for error messages. If not provided, uses "array_0", "array_1", etc.

    Raises
    ------
    ValidationError
        If arrays have different shapes.
    """
    if len(arrays) < 2:
        return

    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]

    shapes = [np.asarray(arr).shape for arr in arrays]

    if not all(s == shapes[0] for s in shapes):
        shape_strs = [f"{name}: {shape}" for name, shape in zip(names, shapes)]
        raise ValidationError(
            f"Arrays must have the same shape. Got: {', '.join(shape_strs)}"
        )


def validated_array_input(
    param_name: str,
    *,
    dtype: type | np.dtype[Any] | None = None,
    ndim: int | tuple[int, ...] | None = None,
    shape: tuple[int | None, ...] | None = None,
    finite: bool = False,
) -> Callable[[F], F]:
    """
    Decorator factory for validating a specific array parameter.

    Parameters
    ----------
    param_name : str
        Name of the parameter to validate.
    dtype : type or np.dtype, optional
        Required dtype.
    ndim : int or tuple of int, optional
        Required number of dimensions.
    shape : tuple, optional
        Required shape (None for any size in a dimension).
    finite : bool, optional
        If True, require all finite values.

    Returns
    -------
    Callable
        Decorator that validates the specified parameter.

    Examples
    --------
    >>> @validated_array_input("x", ndim=1, finite=True)
    ... def my_func(x, y=1):
    ...     return np.sum(x) + y
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            import inspect

            sig = inspect.signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            if param_name in bound.arguments:
                bound.arguments[param_name] = validate_array(
                    bound.arguments[param_name],
                    param_name,
                    dtype=dtype,
                    ndim=ndim,
                    shape=shape,
                    finite=finite,
                )

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


class ArraySpec:
    """
    Specification for array validation in @validate_inputs decorator.

    Parameters
    ----------
    dtype : type or np.dtype, optional
        Required dtype.
    ndim : int or tuple of int, optional
        Required dimensionality.
    shape : tuple, optional
        Required shape (None for any size).
    min_ndim : int, optional
        Minimum dimensions required.
    max_ndim : int, optional
        Maximum dimensions allowed.
    finite : bool, optional
        Require all finite values.
    non_negative : bool, optional
        Require all values >= 0.
    positive : bool, optional
        Require all values > 0.
    allow_empty : bool, optional
        Allow empty arrays. Default True.
    square : bool, optional
        Require square matrix.
    symmetric : bool, optional
        Require symmetric matrix.
    positive_definite : bool, optional
        Require positive definite matrix.

    Examples
    --------
    >>> spec = ArraySpec(ndim=2, finite=True, square=True)
    >>> @validate_inputs(matrix=spec)
    ... def process_matrix(matrix):
    ...     return np.linalg.inv(matrix)
    """

    def __init__(
        self,
        *,
        dtype: type | np.dtype[Any] | None = None,
        ndim: int | tuple[int, ...] | None = None,
        shape: tuple[int | None, ...] | None = None,
        min_ndim: int | None = None,
        max_ndim: int | None = None,
        finite: bool = False,
        non_negative: bool = False,
        positive: bool = False,
        allow_empty: bool = True,
        square: bool = False,
        symmetric: bool = False,
        positive_definite: bool = False,
    ):
        self.dtype = dtype
        self.ndim = ndim
        self.shape = shape
        self.min_ndim = min_ndim
        self.max_ndim = max_ndim
        self.finite = finite
        self.non_negative = non_negative
        self.positive = positive
        self.allow_empty = allow_empty
        self.square = square
        self.symmetric = symmetric
        self.positive_definite = positive_definite

    def validate(self, arr: ArrayLike, name: str) -> NDArray[Any]:
        """Validate an array against this specification."""
        result = validate_array(
            arr,
            name,
            dtype=self.dtype,
            ndim=self.ndim,
            shape=self.shape,
            min_ndim=self.min_ndim,
            max_ndim=self.max_ndim,
            finite=self.finite,
            non_negative=self.non_negative,
            positive=self.positive,
            allow_empty=self.allow_empty,
        )

        if self.positive_definite:
            result = ensure_positive_definite(result, name)
        elif self.symmetric:
            result = ensure_symmetric(result, name)
        elif self.square:
            result = ensure_square_matrix(result, name)

        return result


class ScalarSpec:
    """
    Specification for scalar validation in @validate_inputs decorator.

    Parameters
    ----------
    dtype : type, optional
        Required type (int, float, etc.).
    min_value : float, optional
        Minimum allowed value (inclusive).
    max_value : float, optional
        Maximum allowed value (inclusive).
    finite : bool, optional
        Require finite value.
    positive : bool, optional
        Require value > 0.
    non_negative : bool, optional
        Require value >= 0.

    Examples
    --------
    >>> spec = ScalarSpec(dtype=int, min_value=1, max_value=10)
    >>> @validate_inputs(k=spec)
    ... def get_k_nearest(k, data):
    ...     return data[:k]
    """

    def __init__(
        self,
        *,
        dtype: type | None = None,
        min_value: float | None = None,
        max_value: float | None = None,
        finite: bool = False,
        positive: bool = False,
        non_negative: bool = False,
    ):
        self.dtype = dtype
        self.min_value = min_value
        self.max_value = max_value
        self.finite = finite
        self.positive = positive
        self.non_negative = non_negative

    def validate(self, value: Any, name: str) -> Any:
        """Validate a scalar value against this specification."""
        # Type check
        if self.dtype is not None:
            if not isinstance(value, self.dtype):
                try:
                    value = self.dtype(value)
                except (ValueError, TypeError) as e:
                    raise ValidationError(
                        f"{name} must be {self.dtype.__name__}, got {type(value).__name__}"
                    ) from e

        # Convert to float for numeric checks
        try:
            num_value = float(value)
        except (ValueError, TypeError):
            if any(
                [
                    self.finite,
                    self.positive,
                    self.non_negative,
                    self.min_value is not None,
                    self.max_value is not None,
                ]
            ):
                raise ValidationError(
                    f"{name} must be numeric for range validation"
                ) from None
            return value

        # Finite check
        if self.finite and not np.isfinite(num_value):
            raise ValidationError(f"{name} must be finite, got {value}")

        # Positive check
        if self.positive and num_value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")

        # Non-negative check
        if self.non_negative and num_value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")

        # Range checks
        if self.min_value is not None and num_value < self.min_value:
            raise ValidationError(f"{name} must be >= {self.min_value}, got {value}")

        if self.max_value is not None and num_value > self.max_value:
            raise ValidationError(f"{name} must be <= {self.max_value}, got {value}")

        return value


def validate_inputs(
    **param_specs: ArraySpec | ScalarSpec | dict[str, Any],
) -> Callable[[F], F]:
    """
    Decorator for validating multiple function parameters.

    This decorator enables declarative input validation using specification
    objects (ArraySpec, ScalarSpec) or dictionaries of validation options.

    Parameters
    ----------
    **param_specs : ArraySpec | ScalarSpec | dict
        Keyword arguments mapping parameter names to validation specs.
        Each spec can be:
        - ArraySpec: For array validation
        - ScalarSpec: For scalar validation
        - dict: Options passed to ArraySpec (for convenience)

    Returns
    -------
    Callable
        Decorated function with input validation.

    Examples
    --------
    >>> @validate_inputs(
    ...     x=ArraySpec(ndim=2, finite=True),
    ...     P=ArraySpec(ndim=2, positive_definite=True),
    ...     k=ScalarSpec(dtype=int, min_value=1),
    ... )
    ... def kalman_update(x, P, z, H, R, k=1):
    ...     # x and P are guaranteed valid here
    ...     pass

    Using dict shorthand:

    >>> @validate_inputs(
    ...     state={"ndim": 1, "finite": True},
    ...     covariance={"ndim": 2, "positive_definite": True},
    ... )
    ... def predict(state, covariance, dt):
    ...     pass

    Notes
    -----
    Validation happens in the order parameters are defined in the decorator.
    If any validation fails, a ValidationError is raised with a descriptive
    message identifying the parameter and the constraint violated.

    See Also
    --------
    ArraySpec : Specification class for array validation.
    ScalarSpec : Specification class for scalar validation.
    validate_array : Lower-level array validation function.
    """

    def decorator(func: F) -> F:
        import inspect

        # Pre-fetch signature for efficiency
        sig = inspect.signature(func)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            for param_name, spec in param_specs.items():
                if param_name not in bound.arguments:
                    continue

                value = bound.arguments[param_name]

                # Convert dict to ArraySpec
                if isinstance(spec, dict):
                    spec = ArraySpec(**spec)

                # Validate using spec
                if isinstance(spec, (ArraySpec, ScalarSpec)):
                    bound.arguments[param_name] = spec.validate(value, param_name)
                else:
                    raise TypeError(
                        f"Invalid spec type for {param_name}: {type(spec)}. "
                        "Use ArraySpec, ScalarSpec, or dict."
                    )

            return func(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def check_compatible_shapes(
    *shapes: tuple[int, ...],
    names: Sequence[str] | None = None,
    dimension: int | None = None,
) -> None:
    """
    Check that array shapes are compatible for operations.

    Parameters
    ----------
    *shapes : tuple of int
        Shapes to check for compatibility.
    names : sequence of str, optional
        Names for error messages.
    dimension : int, optional
        If provided, only check compatibility along this dimension.

    Raises
    ------
    ValidationError
        If shapes are not compatible.

    Examples
    --------
    >>> check_compatible_shapes((3, 4), (4, 5), names=["A", "B"], dimension=0)
    # Raises: A has 3 rows but B has 4 rows

    >>> check_compatible_shapes((3, 4), (4, 5), names=["A", "B"])
    # Passes (inner dimensions compatible for matrix multiply)
    """
    if len(shapes) < 2:
        return

    if names is None:
        names = [f"array_{i}" for i in range(len(shapes))]

    if dimension is not None:
        # Check specific dimension
        dims = [s[dimension] if len(s) > dimension else None for s in shapes]
        valid_dims = [d for d in dims if d is not None]
        if valid_dims and not all(d == valid_dims[0] for d in valid_dims):
            dim_strs = [f"{n}={d}" for n, d in zip(names, dims) if d is not None]
            raise ValidationError(
                f"Arrays have incompatible sizes along dimension {dimension}: "
                f"{', '.join(dim_strs)}"
            )
