"""
Matrix decomposition utilities.

This module provides matrix decomposition functions that wrap numpy/scipy
with consistent APIs matching the MATLAB TrackerComponentLibrary conventions.
"""

from typing import Literal, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import ArrayLike, NDArray


def chol_semi_def(
    A: ArrayLike,
    upper: bool = False,
    tol: float = 1e-10,
) -> NDArray[np.floating]:
    """
    Compute Cholesky decomposition of a positive semi-definite matrix.

    For positive semi-definite matrices that may be singular or near-singular,
    this function uses eigenvalue decomposition with thresholding to produce
    a valid Cholesky-like factor.

    Parameters
    ----------
    A : array_like
        Symmetric positive semi-definite matrix of shape (n, n).
    upper : bool, optional
        If True, return upper triangular factor R such that A ≈ R.T @ R.
        If False (default), return lower triangular factor L such that A ≈ L @ L.T.
    tol : float, optional
        Eigenvalues below tol * max(eigenvalues) are treated as zero.
        Default is 1e-10.

    Returns
    -------
    L_or_R : ndarray
        Lower (or upper if upper=True) triangular Cholesky factor.
        Shape is (n, n).

    Examples
    --------
    >>> A = np.array([[4, 2], [2, 1]])  # Singular but positive semi-definite
    >>> L = chol_semi_def(A)
    >>> np.allclose(L @ L.T, A)
    True

    See Also
    --------
    numpy.linalg.cholesky : Standard Cholesky for positive definite matrices.
    scipy.linalg.cholesky : Standard Cholesky with more options.
    """
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")

    # Try standard Cholesky first (faster for well-conditioned matrices)
    try:
        L = la.cholesky(A, lower=not upper)
        return L
    except la.LinAlgError:
        pass

    # Fall back to eigenvalue decomposition for semi-definite case
    eigenvalues, eigenvectors = la.eigh(A)

    # Threshold small/negative eigenvalues
    max_eig = np.max(np.abs(eigenvalues))
    threshold = tol * max_eig if max_eig > 0 else tol
    eigenvalues = np.maximum(eigenvalues, threshold)

    # Construct factor: A ≈ V @ diag(λ) @ V.T = (V @ diag(√λ)) @ (V @ diag(√λ)).T
    sqrt_eig = np.sqrt(eigenvalues)
    factor = eigenvectors @ np.diag(sqrt_eig)

    # Convert to triangular form via QR
    if upper:
        Q, R = la.qr(factor.T)
        return R
    else:
        Q, R = la.qr(factor)
        return R.T


def tria(A: ArrayLike) -> NDArray[np.floating]:
    """
    Compute lower triangular square root factor of a symmetric matrix.

    Given a symmetric positive semi-definite matrix A, returns a lower
    triangular matrix S such that A = S @ S.T. This is useful for
    square-root Kalman filtering implementations.

    Parameters
    ----------
    A : array_like
        Symmetric positive semi-definite matrix of shape (n, n).

    Returns
    -------
    S : ndarray
        Lower triangular matrix of shape (n, n) such that A ≈ S @ S.T.

    Notes
    -----
    This function is equivalent to the lower Cholesky factor for positive
    definite matrices. For semi-definite matrices, it uses the eigenvalue-based
    approach from chol_semi_def.

    See Also
    --------
    chol_semi_def : More general function with tolerance control.
    triaSqrt : Square root of concatenated matrices for filter updates.
    """
    return chol_semi_def(A, upper=False)


def tria_sqrt(
    A: ArrayLike,
    B: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Compute triangular square root of [A, B] @ [A, B].T.

    This is commonly used in square-root Kalman filter implementations
    where we need to compute the square root of a sum of outer products.

    Parameters
    ----------
    A : array_like
        Matrix of shape (n, m).
    B : array_like, optional
        Matrix of shape (n, p). If None, computes sqrt of A @ A.T.

    Returns
    -------
    S : ndarray
        Lower triangular matrix of shape (n, n) such that
        S @ S.T = A @ A.T + B @ B.T (or just A @ A.T if B is None).

    Notes
    -----
    Uses QR decomposition for numerical stability:
    [A, B].T = Q @ R implies [A, B] @ [A, B].T = R.T @ R

    Examples
    --------
    >>> A = np.random.randn(3, 4)
    >>> B = np.random.randn(3, 2)
    >>> S = tria_sqrt(A, B)
    >>> expected = A @ A.T + B @ B.T
    >>> np.allclose(S @ S.T, expected)
    True
    """
    A = np.asarray(A, dtype=np.float64)

    if B is not None:
        B = np.asarray(B, dtype=np.float64)
        if A.shape[0] != B.shape[0]:
            raise ValueError(
                f"A and B must have same number of rows: {A.shape[0]} vs {B.shape[0]}"
            )
        combined = np.hstack([A, B])
    else:
        combined = A

    # QR of combined.T gives us the triangular factor
    _, R = la.qr(combined.T, mode="economic")

    # R.T @ R = combined @ combined.T, so R.T is our lower triangular factor
    # But R might have negative diagonal, so we need to ensure proper form
    n = A.shape[0]
    S = R[:n, :n].T

    # Ensure positive diagonal (standard Cholesky convention)
    signs = np.sign(np.diag(S))
    signs[signs == 0] = 1
    S = S * signs

    return S


def pinv_truncated(
    A: ArrayLike,
    tol: Optional[float] = None,
    rank: Optional[int] = None,
) -> NDArray[np.floating]:
    """
    Compute truncated pseudo-inverse using SVD.

    Computes the Moore-Penrose pseudo-inverse with explicit control over
    which singular values are retained.

    Parameters
    ----------
    A : array_like
        Input matrix of shape (m, n).
    tol : float, optional
        Singular values below tol * max(singular values) are set to zero.
        Default is max(m, n) * eps * max(singular values).
    rank : int, optional
        If provided, only the largest `rank` singular values are used.
        Overrides tol if specified.

    Returns
    -------
    A_pinv : ndarray
        Pseudo-inverse of A with shape (n, m).

    See Also
    --------
    numpy.linalg.pinv : Standard pseudo-inverse.
    scipy.linalg.pinv : Scipy version with rcond parameter.
    """
    A = np.asarray(A, dtype=np.float64)

    U, s, Vh = la.svd(A, full_matrices=False)

    if rank is not None:
        # Use only top `rank` singular values
        effective_rank = min(rank, len(s))
        s_inv = np.zeros_like(s)
        s_inv[:effective_rank] = 1.0 / s[:effective_rank]
    else:
        # Use tolerance-based truncation
        if tol is None:
            tol = max(A.shape) * np.finfo(A.dtype).eps * s[0]
        s_inv = np.where(s > tol, 1.0 / s, 0.0)

    return (Vh.T * s_inv) @ U.T


def matrix_sqrt(
    A: ArrayLike,
    method: Literal["schur", "eigenvalue", "denman_beavers"] = "schur",
) -> NDArray[np.floating]:
    """
    Compute the principal matrix square root.

    Finds matrix S such that S @ S = A. This is different from the Cholesky
    factor which satisfies L @ L.T = A.

    Parameters
    ----------
    A : array_like
        Square matrix of shape (n, n).
    method : {'schur', 'eigenvalue', 'denman_beavers'}, optional
        Algorithm to use:
        - 'schur': Uses Schur decomposition (default, most stable).
        - 'eigenvalue': Uses eigenvalue decomposition (faster for normal matrices).
        - 'denman_beavers': Iterative method (good for ill-conditioned cases).

    Returns
    -------
    S : ndarray
        Principal square root matrix of shape (n, n).

    Notes
    -----
    The principal square root has eigenvalues with positive real parts.

    Examples
    --------
    >>> A = np.array([[4, 0], [0, 9]])
    >>> S = matrix_sqrt(A)
    >>> np.allclose(S @ S, A)
    True
    >>> S
    array([[2., 0.],
           [0., 3.]])
    """
    A = np.asarray(A, dtype=np.float64)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {A.shape}")

    if method == "schur":
        return la.sqrtm(A).real

    elif method == "eigenvalue":
        eigenvalues, eigenvectors = la.eig(A)
        sqrt_eig = np.sqrt(eigenvalues.astype(complex))
        S = eigenvectors @ np.diag(sqrt_eig) @ la.inv(eigenvectors)
        return S.real

    elif method == "denman_beavers":
        # Denman-Beavers iteration
        n = A.shape[0]
        Y = A.copy()
        Z = np.eye(n)

        for _ in range(50):  # Max iterations
            Y_new = 0.5 * (Y + la.inv(Z))
            Z_new = 0.5 * (Z + la.inv(Y))

            if np.allclose(Y, Y_new, rtol=1e-12):
                break

            Y, Z = Y_new, Z_new

        return Y

    else:
        raise ValueError(f"Unknown method: {method}")


def rank_revealing_qr(
    A: ArrayLike,
    tol: Optional[float] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp], int]:
    """
    Compute rank-revealing QR decomposition with column pivoting.

    Computes A[:, P] = Q @ R where P is a permutation that reveals the
    numerical rank of A through the diagonal of R.

    Parameters
    ----------
    A : array_like
        Input matrix of shape (m, n).
    tol : float, optional
        Tolerance for determining numerical rank. Diagonal elements of R
        below tol * |R[0,0]| indicate rank deficiency.
        Default is max(m, n) * eps * |R[0,0]|.

    Returns
    -------
    Q : ndarray
        Orthogonal matrix of shape (m, k) where k = min(m, n).
    R : ndarray
        Upper triangular matrix of shape (k, n).
    P : ndarray
        Permutation indices such that A[:, P] = Q @ R.
    rank : int
        Numerical rank determined by tolerance.

    Examples
    --------
    >>> A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # Rank 2
    >>> Q, R, P, rank = rank_revealing_qr(A)
    >>> rank
    2
    """
    A = np.asarray(A, dtype=np.float64)
    m, n = A.shape

    Q, R, P = la.qr(A, pivoting=True, mode="economic")

    # Determine numerical rank
    diag_R = np.abs(np.diag(R))
    if tol is None:
        tol = max(m, n) * np.finfo(A.dtype).eps * diag_R[0] if diag_R[0] > 0 else 0

    rank = np.sum(diag_R > tol)

    return Q, R, P, int(rank)


def null_space(
    A: ArrayLike,
    tol: Optional[float] = None,
) -> NDArray[np.floating]:
    """
    Compute orthonormal basis for the null space of A.

    Parameters
    ----------
    A : array_like
        Input matrix of shape (m, n).
    tol : float, optional
        Singular values below tol are considered zero.
        Default is max(m, n) * eps * max(singular values).

    Returns
    -------
    N : ndarray
        Orthonormal basis for null(A) with shape (n, k) where k is the
        dimension of the null space.

    Examples
    --------
    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> N = null_space(A)
    >>> np.allclose(A @ N, 0, atol=1e-10)
    True
    """
    A = np.asarray(A, dtype=np.float64)

    U, s, Vh = la.svd(A, full_matrices=True)

    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s[0] if len(s) > 0 else 0

    rank = np.sum(s > tol)

    return Vh[rank:].T.conj()


def range_space(
    A: ArrayLike,
    tol: Optional[float] = None,
) -> NDArray[np.floating]:
    """
    Compute orthonormal basis for the range (column space) of A.

    Parameters
    ----------
    A : array_like
        Input matrix of shape (m, n).
    tol : float, optional
        Singular values below tol are considered zero.
        Default is max(m, n) * eps * max(singular values).

    Returns
    -------
    R : ndarray
        Orthonormal basis for range(A) with shape (m, r) where r is the rank.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 6], [5, 10]])  # Rank 1
    >>> R = range_space(A)
    >>> R.shape
    (3, 1)
    """
    A = np.asarray(A, dtype=np.float64)

    U, s, Vh = la.svd(A, full_matrices=False)

    if tol is None:
        tol = max(A.shape) * np.finfo(A.dtype).eps * s[0] if len(s) > 0 else 0

    rank = np.sum(s > tol)

    return U[:, :rank]


__all__ = [
    "chol_semi_def",
    "tria",
    "tria_sqrt",
    "pinv_truncated",
    "matrix_sqrt",
    "rank_revealing_qr",
    "null_space",
    "range_space",
]
