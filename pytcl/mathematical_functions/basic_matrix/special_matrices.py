"""
Special matrix constructions.

This module provides functions for constructing special matrices commonly
used in numerical algorithms and signal processing.
"""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


def vandermonde(
    x: ArrayLike,
    n: Optional[int] = None,
    increasing: bool = False,
) -> NDArray[np.floating]:
    """
    Construct a Vandermonde matrix.

    The Vandermonde matrix has columns that are powers of the input vector.
    By default (decreasing order): V[i,j] = x[i]^(n-1-j)
    With increasing=True: V[i,j] = x[i]^j

    Parameters
    ----------
    x : array_like
        Input vector of length m.
    n : int, optional
        Number of columns. Default is len(x).
    increasing : bool, optional
        If True, powers increase left to right. Default is False.

    Returns
    -------
    V : ndarray
        Vandermonde matrix of shape (m, n).

    Examples
    --------
    >>> vandermonde([1, 2, 3], 3)
    array([[1, 1, 1],
           [4, 2, 1],
           [9, 3, 1]])

    >>> vandermonde([1, 2, 3], 3, increasing=True)
    array([[1, 1, 1],
           [1, 2, 4],
           [1, 3, 9]])
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    m = len(x)

    if n is None:
        n = m

    if increasing:
        return np.vander(x, n, increasing=True)
    else:
        return np.vander(x, n, increasing=False)


def toeplitz(
    c: ArrayLike,
    r: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Construct a Toeplitz matrix.

    A Toeplitz matrix has constant diagonals. It is fully specified by its
    first column and first row.

    Parameters
    ----------
    c : array_like
        First column of the matrix.
    r : array_like, optional
        First row of the matrix. If None, r = conjugate(c) is assumed
        (Hermitian Toeplitz). Note: r[0] is ignored; c[0] is used.

    Returns
    -------
    T : ndarray
        Toeplitz matrix.

    Examples
    --------
    >>> toeplitz([1, 2, 3], [1, 4, 5])
    array([[1, 4, 5],
           [2, 1, 4],
           [3, 2, 1]])

    See Also
    --------
    scipy.linalg.toeplitz : Equivalent scipy function.
    """
    from scipy.linalg import toeplitz as scipy_toeplitz

    c = np.asarray(c, dtype=np.float64)
    if r is not None:
        r = np.asarray(r, dtype=np.float64)

    return scipy_toeplitz(c, r)


def hankel(
    c: ArrayLike,
    r: Optional[ArrayLike] = None,
) -> NDArray[np.floating]:
    """
    Construct a Hankel matrix.

    A Hankel matrix has constant anti-diagonals. It is fully specified by its
    first column and last row.

    Parameters
    ----------
    c : array_like
        First column of the matrix.
    r : array_like, optional
        Last row of the matrix. If None, zeros are used except for c[-1].
        Note: r[0] should equal c[-1].

    Returns
    -------
    H : ndarray
        Hankel matrix.

    Examples
    --------
    >>> hankel([1, 2, 3], [3, 4, 5])
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])

    See Also
    --------
    scipy.linalg.hankel : Equivalent scipy function.
    """
    from scipy.linalg import hankel as scipy_hankel

    c = np.asarray(c, dtype=np.float64)
    if r is not None:
        r = np.asarray(r, dtype=np.float64)

    return scipy_hankel(c, r)


def circulant(c: ArrayLike) -> NDArray[np.floating]:
    """
    Construct a circulant matrix.

    A circulant matrix is a special Toeplitz matrix where each row is a
    cyclic shift of the row above it.

    Parameters
    ----------
    c : array_like
        First column of the matrix.

    Returns
    -------
    C : ndarray
        Circulant matrix of shape (n, n) where n = len(c).

    Examples
    --------
    >>> circulant([1, 2, 3])
    array([[1, 3, 2],
           [2, 1, 3],
           [3, 2, 1]])

    Notes
    -----
    Circulant matrices are diagonalized by the DFT matrix.

    See Also
    --------
    scipy.linalg.circulant : Equivalent scipy function.
    """
    from scipy.linalg import circulant as scipy_circulant

    c = np.asarray(c, dtype=np.float64)
    return scipy_circulant(c)


def block_diag(*arrs: ArrayLike) -> NDArray[np.floating]:
    """
    Create a block diagonal matrix from provided arrays.

    Parameters
    ----------
    *arrs : sequence of array_like
        Input arrays. Each array becomes a block on the diagonal.

    Returns
    -------
    D : ndarray
        Block diagonal matrix.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6, 7]])
    >>> block_diag(A, B)
    array([[1, 2, 0, 0, 0],
           [3, 4, 0, 0, 0],
           [0, 0, 5, 6, 7]])

    See Also
    --------
    scipy.linalg.block_diag : Equivalent scipy function.
    """
    from scipy.linalg import block_diag as scipy_block_diag

    return scipy_block_diag(*arrs).astype(np.float64)


def companion(c: ArrayLike) -> NDArray[np.floating]:
    """
    Create a companion matrix.

    The companion matrix is used for polynomial root finding. For a monic
    polynomial p(x) = x^n + c_{n-1}*x^{n-1} + ... + c_1*x + c_0, the
    eigenvalues of the companion matrix are the roots of p(x).

    Parameters
    ----------
    c : array_like
        Coefficients of the polynomial (excluding leading 1), in order
        [c_{n-1}, c_{n-2}, ..., c_1, c_0] or [c_0, c_1, ..., c_{n-1}]
        depending on convention used.

    Returns
    -------
    C : ndarray
        Companion matrix of shape (n, n) where n = len(c).

    Examples
    --------
    >>> # Polynomial: x^3 - 6x^2 + 11x - 6 = (x-1)(x-2)(x-3)
    >>> c = [6, -11, 6]  # Coefficients (negated, reversed)
    >>> C = companion(c)

    See Also
    --------
    scipy.linalg.companion : Equivalent scipy function.
    """
    from scipy.linalg import companion as scipy_companion

    c = np.asarray(c, dtype=np.float64)
    return scipy_companion(c)


def hilbert(n: int) -> NDArray[np.floating]:
    """
    Create a Hilbert matrix.

    The Hilbert matrix H has entries H[i,j] = 1 / (i + j + 1).
    This matrix is notoriously ill-conditioned.

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    H : ndarray
        Hilbert matrix of shape (n, n).

    Examples
    --------
    >>> hilbert(3)
    array([[1.        , 0.5       , 0.33333333],
           [0.5       , 0.33333333, 0.25      ],
           [0.33333333, 0.25      , 0.2       ]])

    See Also
    --------
    scipy.linalg.hilbert : Equivalent scipy function.
    """
    from scipy.linalg import hilbert as scipy_hilbert

    return scipy_hilbert(n)


def invhilbert(n: int) -> NDArray[np.floating]:
    """
    Compute the inverse of the Hilbert matrix.

    Uses an exact formula to compute the inverse, which is known to have
    integer entries.

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    H_inv : ndarray
        Inverse of the n x n Hilbert matrix.

    See Also
    --------
    scipy.linalg.invhilbert : Equivalent scipy function.
    """
    from scipy.linalg import invhilbert as scipy_invhilbert

    return scipy_invhilbert(n).astype(np.float64)


def hadamard(n: int) -> NDArray[np.floating]:
    """
    Construct a Hadamard matrix.

    A Hadamard matrix H satisfies H @ H.T = n * I, where all entries are
    +1 or -1.

    Parameters
    ----------
    n : int
        Size of the matrix. Must be a power of 2, or 1, 2.

    Returns
    -------
    H : ndarray
        Hadamard matrix of shape (n, n).

    Raises
    ------
    ValueError
        If n is not a power of 2.

    Examples
    --------
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])

    See Also
    --------
    scipy.linalg.hadamard : Equivalent scipy function.
    """
    from scipy.linalg import hadamard as scipy_hadamard

    return scipy_hadamard(n).astype(np.float64)


def dft_matrix(n: int, normalized: bool = False) -> NDArray[np.complexfloating]:
    """
    Construct the DFT (Discrete Fourier Transform) matrix.

    The DFT matrix F has entries F[j,k] = exp(-2*pi*i*j*k/n).

    Parameters
    ----------
    n : int
        Size of the matrix.
    normalized : bool, optional
        If True, return unitary DFT matrix (scaled by 1/sqrt(n)).
        Default is False.

    Returns
    -------
    F : ndarray
        DFT matrix of shape (n, n), complex-valued.

    Examples
    --------
    >>> F = dft_matrix(4)
    >>> x = np.array([1, 2, 3, 4])
    >>> np.allclose(F @ x, np.fft.fft(x))
    True

    See Also
    --------
    scipy.linalg.dft : Equivalent scipy function.
    """
    from scipy.linalg import dft as scipy_dft

    if normalized:
        return scipy_dft(n, scale="sqrtn")
    else:
        return scipy_dft(n, scale=None)


def kron(a: ArrayLike, b: ArrayLike) -> NDArray[np.floating]:
    """
    Compute the Kronecker product of two arrays.

    Parameters
    ----------
    a : array_like
        First input array.
    b : array_like
        Second input array.

    Returns
    -------
    K : ndarray
        Kronecker product of a and b.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[1, 0], [0, 1]])
    >>> kron(a, b)
    array([[1, 0, 2, 0],
           [0, 1, 0, 2],
           [3, 0, 4, 0],
           [0, 3, 0, 4]])

    See Also
    --------
    numpy.kron : Equivalent numpy function.
    """
    return np.kron(a, b).astype(np.float64)


def vec(A: ArrayLike) -> NDArray[np.floating]:
    """
    Vectorize a matrix by stacking its columns.

    This is the standard vec operation from matrix calculus.

    Parameters
    ----------
    A : array_like
        Input matrix of shape (m, n).

    Returns
    -------
    v : ndarray
        Column vector of shape (m*n,) containing columns of A stacked.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> vec(A)
    array([1, 3, 2, 4])

    See Also
    --------
    unvec : Inverse operation.
    """
    A = np.asarray(A, dtype=np.float64)
    return A.flatten(order="F")


def unvec(v: ArrayLike, m: int, n: int) -> NDArray[np.floating]:
    """
    Reshape a vector back to a matrix (inverse of vec).

    Parameters
    ----------
    v : array_like
        Input vector of length m*n.
    m : int
        Number of rows in output matrix.
    n : int
        Number of columns in output matrix.

    Returns
    -------
    A : ndarray
        Matrix of shape (m, n).

    Examples
    --------
    >>> v = np.array([1, 3, 2, 4])
    >>> unvec(v, 2, 2)
    array([[1, 2],
           [3, 4]])

    See Also
    --------
    vec : Forward operation.
    """
    v = np.asarray(v, dtype=np.float64)
    return v.reshape((m, n), order="F")


def commutation_matrix(m: int, n: int) -> NDArray[np.floating]:
    """
    Construct the commutation matrix K_{m,n}.

    The commutation matrix satisfies K @ vec(A) = vec(A.T) for any m x n matrix A.

    Parameters
    ----------
    m : int
        Number of rows of the matrix to be transposed.
    n : int
        Number of columns of the matrix to be transposed.

    Returns
    -------
    K : ndarray
        Commutation matrix of shape (m*n, m*n).

    Examples
    --------
    >>> K = commutation_matrix(2, 3)
    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.allclose(K @ vec(A), vec(A.T))
    True
    """
    mn = m * n
    K = np.zeros((mn, mn), dtype=np.float64)

    for i in range(m):
        for j in range(n):
            # vec(A)[i + m*j] maps to vec(A.T)[j + n*i]
            row_idx = j + n * i
            col_idx = i + m * j
            K[row_idx, col_idx] = 1.0

    return K


def duplication_matrix(n: int) -> NDArray[np.floating]:
    """
    Construct the duplication matrix D_n.

    For a symmetric n x n matrix A, D_n @ vech(A) = vec(A), where vech
    is the half-vectorization operator.

    Parameters
    ----------
    n : int
        Size of the symmetric matrix.

    Returns
    -------
    D : ndarray
        Duplication matrix of shape (n*n, n*(n+1)/2).
    """
    m = n * (n + 1) // 2
    D = np.zeros((n * n, m), dtype=np.float64)

    k = 0
    for j in range(n):
        for i in range(j, n):
            # vech index k corresponds to (i,j) and (j,i) in vec
            vec_idx_ij = i + n * j
            vec_idx_ji = j + n * i

            D[vec_idx_ij, k] = 1.0
            if i != j:
                D[vec_idx_ji, k] = 1.0
            k += 1

    return D


def elimination_matrix(n: int) -> NDArray[np.floating]:
    """
    Construct the elimination matrix L_n.

    For any n x n matrix A, L_n @ vec(A) = vech(A), where vech is the
    half-vectorization operator that extracts the lower triangle.

    Parameters
    ----------
    n : int
        Size of the matrix.

    Returns
    -------
    L : ndarray
        Elimination matrix of shape (n*(n+1)/2, n*n).
    """
    m = n * (n + 1) // 2
    L = np.zeros((m, n * n), dtype=np.float64)

    k = 0
    for j in range(n):
        for i in range(j, n):
            vec_idx = i + n * j
            L[k, vec_idx] = 1.0
            k += 1

    return L


__all__ = [
    "vandermonde",
    "toeplitz",
    "hankel",
    "circulant",
    "block_diag",
    "companion",
    "hilbert",
    "invhilbert",
    "hadamard",
    "dft_matrix",
    "kron",
    "vec",
    "unvec",
    "commutation_matrix",
    "duplication_matrix",
    "elimination_matrix",
]
