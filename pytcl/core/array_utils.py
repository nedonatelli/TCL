"""
Array utility functions for the Tracker Component Library.

This module provides array manipulation functions that mirror MATLAB behavior,
making it easier to port algorithms while maintaining Pythonic interfaces.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.constants import PI, TWO_PI


def wrap_to_pi(angle: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Wrap angles to the interval [-π, π).

    This is equivalent to MATLAB's wrapToPi function.

    Parameters
    ----------
    angle : array_like
        Angle(s) in radians.

    Returns
    -------
    NDArray
        Angle(s) wrapped to [-π, π).

    Examples
    --------
    >>> wrap_to_pi(3 * np.pi)
    -3.141592653589793

    >>> wrap_to_pi([-4, -3, -2, -1, 0, 1, 2, 3, 4])
    array([ 2.28318531, -3.        , -2.        , -1.        ,  0.        ,
            1.        ,  2.        ,  3.        , -2.28318531])
    """
    angle = np.asarray(angle, dtype=np.float64)
    return np.mod(angle + PI, TWO_PI) - PI


def wrap_to_2pi(angle: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Wrap angles to the interval [0, 2π).

    This is equivalent to MATLAB's wrapTo2Pi function.

    Parameters
    ----------
    angle : array_like
        Angle(s) in radians.

    Returns
    -------
    NDArray
        Angle(s) wrapped to [0, 2π).

    Examples
    --------
    >>> wrap_to_2pi(-np.pi/2)
    4.71238898038469

    >>> wrap_to_2pi(3 * np.pi)
    3.141592653589793
    """
    angle = np.asarray(angle, dtype=np.float64)
    return np.mod(angle, TWO_PI)


def wrap_to_range(
    value: ArrayLike,
    low: float,
    high: float,
) -> NDArray[np.floating[Any]]:
    """
    Wrap values to a specified interval [low, high).

    Parameters
    ----------
    value : array_like
        Value(s) to wrap.
    low : float
        Lower bound of the interval (inclusive).
    high : float
        Upper bound of the interval (exclusive).

    Returns
    -------
    NDArray
        Value(s) wrapped to [low, high).

    Examples
    --------
    >>> wrap_to_range(370, 0, 360)
    10.0

    >>> wrap_to_range(-10, 0, 360)
    350.0
    """
    value = np.asarray(value, dtype=np.float64)
    range_width = high - low
    return np.mod(value - low, range_width) + low


def wrap_to_pm180(angle: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Wrap angles in degrees to the interval [-180, 180).

    Parameters
    ----------
    angle : array_like
        Angle(s) in degrees.

    Returns
    -------
    NDArray
        Angle(s) wrapped to [-180, 180) degrees.

    Examples
    --------
    >>> wrap_to_pm180(270)
    -90.0
    """
    return wrap_to_range(np.asarray(angle, dtype=np.float64), -180.0, 180.0)


def wrap_to_360(angle: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Wrap angles in degrees to the interval [0, 360).

    Parameters
    ----------
    angle : array_like
        Angle(s) in degrees.

    Returns
    -------
    NDArray
        Angle(s) wrapped to [0, 360) degrees.

    Examples
    --------
    >>> wrap_to_360(-90)
    270.0
    """
    return wrap_to_range(np.asarray(angle, dtype=np.float64), 0.0, 360.0)


def column_vector(arr: ArrayLike) -> NDArray[Any]:
    """
    Convert an array-like to a column vector (n, 1).

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    NDArray
        Column vector with shape (n, 1).

    Examples
    --------
    >>> column_vector([1, 2, 3])
    array([[1],
           [2],
           [3]])

    >>> column_vector([[1, 2, 3]])
    array([[1],
           [2],
           [3]])
    """
    arr = np.asarray(arr)
    return arr.flatten().reshape(-1, 1)


def row_vector(arr: ArrayLike) -> NDArray[Any]:
    """
    Convert an array-like to a row vector (1, n).

    Parameters
    ----------
    arr : array_like
        Input array.

    Returns
    -------
    NDArray
        Row vector with shape (1, n).

    Examples
    --------
    >>> row_vector([1, 2, 3])
    array([[1, 2, 3]])

    >>> row_vector([[1], [2], [3]])
    array([[1, 2, 3]])
    """
    arr = np.asarray(arr)
    return arr.flatten().reshape(1, -1)


def vec(arr: ArrayLike, order: Literal["F", "C"] = "F") -> NDArray[Any]:
    """
    Vectorize a matrix (stack columns or rows into a single column).

    This mirrors MATLAB's vec operator which stacks columns.

    Parameters
    ----------
    arr : array_like
        Input matrix.
    order : {'F', 'C'}, optional
        'F' (default): Stack columns (MATLAB-style, column-major).
        'C': Stack rows (row-major).

    Returns
    -------
    NDArray
        Column vector with shape (m*n, 1).

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> vec(A)  # Stack columns: [1, 3, 2, 4]
    array([[1],
           [3],
           [2],
           [4]])

    >>> vec(A, order='C')  # Stack rows: [1, 2, 3, 4]
    array([[1],
           [2],
           [3],
           [4]])
    """
    arr = np.asarray(arr)
    return arr.flatten(order=order).reshape(-1, 1)


def unvec(
    v: ArrayLike,
    shape: tuple[int, int],
    order: Literal["F", "C"] = "F",
) -> NDArray[Any]:
    """
    Reshape a vector back into a matrix.

    Inverse of the vec operation.

    Parameters
    ----------
    v : array_like
        Input vector.
    shape : tuple of int
        Target shape (rows, cols).
    order : {'F', 'C'}, optional
        'F' (default): Unstack to columns (MATLAB-style).
        'C': Unstack to rows.

    Returns
    -------
    NDArray
        Matrix with specified shape.
    """
    v = np.asarray(v).flatten()
    return v.reshape(shape, order=order)


def block_diag(*arrays: ArrayLike) -> NDArray[Any]:
    """
    Create a block diagonal matrix from provided arrays.

    Equivalent to MATLAB's blkdiag function.

    Parameters
    ----------
    *arrays : array_like
        Input arrays to place on the diagonal.

    Returns
    -------
    NDArray
        Block diagonal matrix.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5]])
    >>> block_diag(A, B)
    array([[1, 2, 0],
           [3, 4, 0],
           [0, 0, 5]])
    """
    from scipy.linalg import block_diag as scipy_block_diag

    arrays = [np.atleast_2d(np.asarray(a)) for a in arrays]
    return scipy_block_diag(*arrays)


def skew_symmetric(v: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Create a 3x3 skew-symmetric matrix from a 3D vector.

    The skew-symmetric matrix [v]× satisfies: [v]× @ u = v × u (cross product).

    Parameters
    ----------
    v : array_like
        3-element vector.

    Returns
    -------
    NDArray
        3x3 skew-symmetric matrix.

    Examples
    --------
    >>> v = [1, 2, 3]
    >>> S = skew_symmetric(v)
    >>> S
    array([[ 0., -3.,  2.],
           [ 3.,  0., -1.],
           [-2.,  1.,  0.]])

    >>> u = [4, 5, 6]
    >>> np.allclose(S @ u, np.cross(v, u))
    True
    """
    v = np.asarray(v, dtype=np.float64).flatten()
    if v.size != 3:
        raise ValueError(f"Input must be a 3-element vector, got size {v.size}")

    return np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )


def unskew(S: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Extract the vector from a 3x3 skew-symmetric matrix.

    Inverse of skew_symmetric.

    Parameters
    ----------
    S : array_like
        3x3 skew-symmetric matrix.

    Returns
    -------
    NDArray
        3-element vector.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.shape != (3, 3):
        raise ValueError(f"Input must be 3x3, got shape {S.shape}")

    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def normalize_vector(
    v: ArrayLike,
    axis: int | None = None,
    return_norm: bool = False,
) -> (
    NDArray[np.floating[Any]]
    | tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]
):
    """
    Normalize vector(s) to unit length.

    Parameters
    ----------
    v : array_like
        Vector(s) to normalize.
    axis : int, optional
        Axis along which to compute norms. If None, normalize the flattened array.
    return_norm : bool, optional
        If True, also return the original norm(s). Default is False.

    Returns
    -------
    v_normalized : NDArray
        Unit vector(s).
    norm : NDArray, optional
        Original norm(s), only returned if return_norm=True.

    Examples
    --------
    >>> normalize_vector([3, 4])
    array([0.6, 0.8])

    >>> v_unit, norm = normalize_vector([3, 4], return_norm=True)
    >>> norm
    5.0
    """
    v = np.asarray(v, dtype=np.float64)

    if axis is None and v.ndim == 1:
        norm = np.linalg.norm(v)
    else:
        norm = np.linalg.norm(v, axis=axis, keepdims=True)

    # Handle zero vectors
    with np.errstate(divide="ignore", invalid="ignore"):
        v_normalized = np.where(norm > 0, v / norm, 0.0)

    if axis is not None:
        norm = np.squeeze(norm, axis=axis)

    if return_norm:
        return v_normalized, norm
    return v_normalized


def outer_product(a: ArrayLike, b: ArrayLike) -> NDArray[Any]:
    """
    Compute the outer product of two vectors.

    Parameters
    ----------
    a : array_like
        First vector (m,).
    b : array_like
        Second vector (n,).

    Returns
    -------
    NDArray
        Outer product matrix (m, n).

    Examples
    --------
    >>> outer_product([1, 2], [3, 4, 5])
    array([[ 3,  4,  5],
           [ 6,  8, 10]])
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    return np.outer(a, b)


def repmat(arr: ArrayLike, m: int, n: int) -> NDArray[Any]:
    """
    Replicate and tile an array.

    Equivalent to MATLAB's repmat function.

    Parameters
    ----------
    arr : array_like
        Input array.
    m : int
        Number of times to replicate along rows.
    n : int
        Number of times to replicate along columns.

    Returns
    -------
    NDArray
        Tiled array.

    Examples
    --------
    >>> repmat([1, 2], 2, 3)
    array([[1, 2, 1, 2, 1, 2],
           [1, 2, 1, 2, 1, 2]])
    """
    arr = np.atleast_2d(np.asarray(arr))
    return np.tile(arr, (m, n))


def meshgrid_ij(
    *xi: ArrayLike,
    indexing: Literal["ij", "xy"] = "ij",
) -> tuple[NDArray[Any], ...]:
    """
    Create coordinate matrices from coordinate vectors.

    Wrapper around np.meshgrid with 'ij' indexing as default (MATLAB-style).

    Parameters
    ----------
    *xi : array_like
        1-D arrays representing coordinates.
    indexing : {'ij', 'xy'}, optional
        Cartesian ('xy', default numpy) or matrix ('ij', MATLAB-style) indexing.
        Default is 'ij'.

    Returns
    -------
    tuple of NDArray
        Coordinate matrices.
    """
    return np.meshgrid(*xi, indexing=indexing)


def is_positive_definite(
    A: ArrayLike,
    tol: float = 1e-10,
) -> bool:
    """
    Check if a matrix is positive definite.

    Parameters
    ----------
    A : array_like
        Square matrix to check.
    tol : float, optional
        Tolerance for eigenvalue check. Default is 1e-10.

    Returns
    -------
    bool
        True if matrix is positive definite.

    Examples
    --------
    >>> A = np.array([[4, 2], [2, 5]])
    >>> is_positive_definite(A)
    True
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return False

    # Check symmetry
    if not np.allclose(A, A.T, rtol=tol, atol=tol):
        return False

    try:
        eigenvalues = np.linalg.eigvalsh(A)
        return bool(np.all(eigenvalues > -tol * np.max(np.abs(eigenvalues))))
    except np.linalg.LinAlgError:
        return False


def nearest_positive_definite(A: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    Find the nearest positive definite matrix.

    Uses the method from Higham (1988) "Computing a Nearest Symmetric
    Positive Semidefinite Matrix".

    Parameters
    ----------
    A : array_like
        Input matrix.

    Returns
    -------
    NDArray
        Nearest positive definite matrix.
    """
    A = np.asarray(A, dtype=np.float64)

    # Symmetrize
    B = (A + A.T) / 2

    # Compute SVD
    _, s, Vt = np.linalg.svd(B)

    # Compute positive semi-definite matrix
    H = Vt.T @ np.diag(s) @ Vt

    # Return symmetrized result
    A_pd = (B + H) / 2
    A_pd = (A_pd + A_pd.T) / 2

    # Ensure positive definiteness by adjusting eigenvalues if needed
    eigvals = np.linalg.eigvalsh(A_pd)
    min_eig = np.min(eigvals)
    if min_eig < 0:
        spacing = np.spacing(np.linalg.norm(A_pd))
        A_pd += np.eye(A_pd.shape[0]) * (-min_eig + spacing)

    return A_pd


def safe_cholesky(
    A: ArrayLike,
    max_attempts: int = 10,
) -> NDArray[np.floating[Any]]:
    """
    Compute Cholesky decomposition with fallback for near-singular matrices.

    If standard Cholesky fails, attempts to find nearest positive definite matrix.

    Parameters
    ----------
    A : array_like
        Positive definite matrix.
    max_attempts : int, optional
        Maximum regularization attempts. Default is 10.

    Returns
    -------
    NDArray
        Lower triangular Cholesky factor L such that A = L @ L.T

    Raises
    ------
    np.linalg.LinAlgError
        If Cholesky decomposition fails after all attempts.
    """
    A = np.asarray(A, dtype=np.float64)

    # Ensure symmetry
    A = (A + A.T) / 2

    for attempt in range(max_attempts):
        try:
            return np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            if attempt == max_attempts - 1:
                raise
            # Add small diagonal perturbation
            jitter = 10 ** (attempt - 6) * np.trace(A) / A.shape[0]
            A = A + jitter * np.eye(A.shape[0])

    raise np.linalg.LinAlgError("Cholesky decomposition failed")
