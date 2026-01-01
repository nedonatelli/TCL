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
    dtype: type | np.dtype | None = None,
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
            raise ValidationError(f"{name} must have {ndim} dimension(s), got {result.ndim}")

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
            raise ValidationError(f"{name} must have {len(shape)} dimensions, got {result.ndim}")
        for i, (expected, actual) in enumerate(zip(shape, result.shape)):
            if expected is not None and expected != actual:
                raise ValidationError(f"{name} dimension {i} must be {expected}, got {actual}")

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
            raise ValidationError(f"{name} must be a row vector (1, n), got shape {result.shape}")
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
            f"{name} must be positive definite, " f"minimum eigenvalue is {min_eigenvalue:.2e}"
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
        raise ValidationError(f"Arrays must have the same shape. Got: {', '.join(shape_strs)}")


def validated_array_input(
    param_name: str,
    *,
    dtype: type | np.dtype | None = None,
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
