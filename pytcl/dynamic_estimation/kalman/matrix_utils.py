"""
Matrix utility functions for Kalman filter implementations.

This module provides numerically stable matrix operations used across
multiple Kalman filter implementations. Separating these utilities prevents
circular imports between filter implementations.

Functions include:
- Cholesky factor update/downdate
- QR-based covariance propagation
- Matrix symmetry enforcement
- Matrix square root computation
- Innovation likelihood computation
"""

from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def cholesky_update(
    S: NDArray[np.floating], v: NDArray[np.floating], sign: float = 1.0
) -> NDArray[np.floating]:
    """
    Rank-1 Cholesky update/downdate.

    Computes the Cholesky factor of P ± v @ v.T given S where P = S @ S.T.

    Parameters
    ----------
    S : ndarray
        Lower triangular Cholesky factor, shape (n, n).
    v : ndarray
        Vector for rank-1 update, shape (n,).
    sign : float
        +1 for update (addition), -1 for downdate (subtraction).

    Returns
    -------
    S_new : ndarray
        Updated lower triangular Cholesky factor.

    Notes
    -----
    Uses the efficient O(n²) algorithm from [1].

    References
    ----------
    .. [1] P. E. Gill, G. H. Golub, W. Murray, and M. A. Saunders,
           "Methods for modifying matrix factorizations,"
           Mathematics of Computation, vol. 28, pp. 505-535, 1974.

    Examples
    --------
    >>> import numpy as np
    >>> S = np.linalg.cholesky(np.eye(2))
    >>> v = np.array([0.5, 0.5])
    >>> S_updated = cholesky_update(S, v, sign=1.0)
    >>> P_updated = S_updated @ S_updated.T
    >>> np.allclose(P_updated, np.eye(2) + np.outer(v, v))
    True
    """
    S = np.asarray(S, dtype=np.float64).copy()
    v = np.asarray(v, dtype=np.float64).flatten().copy()
    n = len(v)

    if sign > 0:
        # Cholesky update
        for k in range(n):
            r = np.sqrt(S[k, k] ** 2 + v[k] ** 2)
            c = r / S[k, k]
            s = v[k] / S[k, k]
            S[k, k] = r
            if k < n - 1:
                S[k + 1 :, k] = (S[k + 1 :, k] + s * v[k + 1 :]) / c
                v[k + 1 :] = c * v[k + 1 :] - s * S[k + 1 :, k]
    else:
        # Cholesky downdate
        for k in range(n):
            r_sq = S[k, k] ** 2 - v[k] ** 2
            if r_sq < 0:
                raise ValueError("Downdate would make matrix non-positive definite")
            r = np.sqrt(r_sq)
            c = r / S[k, k]
            s = v[k] / S[k, k]
            S[k, k] = r
            if k < n - 1:
                S[k + 1 :, k] = (S[k + 1 :, k] - s * v[k + 1 :]) / c
                v[k + 1 :] = c * v[k + 1 :] - s * S[k + 1 :, k]

    return S


def qr_update(
    S_x: NDArray[np.floating],
    S_noise: NDArray[np.floating],
    F: Optional[NDArray[np.floating]] = None,
) -> NDArray[np.floating]:
    """
    QR-based covariance square root update.

    Computes the Cholesky factor of F @ P @ F.T + Q given S_x (where P = S_x @ S_x.T)
    and S_noise (where Q = S_noise @ S_noise.T).

    Parameters
    ----------
    S_x : ndarray
        Lower triangular Cholesky factor of state covariance, shape (n, n).
    S_noise : ndarray
        Lower triangular Cholesky factor of noise covariance, shape (n, n).
    F : ndarray, optional
        State transition matrix, shape (n, n). If None, uses identity.

    Returns
    -------
    S_new : ndarray
        Lower triangular Cholesky factor of the updated covariance.

    Notes
    -----
    Uses QR decomposition for numerical stability. The compound matrix
    [F @ S_x, S_noise].T is QR decomposed, and R.T gives the new Cholesky factor.

    Examples
    --------
    >>> import numpy as np
    >>> S_x = np.linalg.cholesky(np.eye(2) * 0.1)
    >>> S_noise = np.linalg.cholesky(np.eye(2) * 0.01)
    >>> F = np.array([[1, 1], [0, 1]])
    >>> S_new = qr_update(S_x, S_noise, F)
    """
    S_x = np.asarray(S_x, dtype=np.float64)
    S_noise = np.asarray(S_noise, dtype=np.float64)
    n = S_x.shape[0]

    if F is not None:
        F = np.asarray(F, dtype=np.float64)
        FS = F @ S_x
    else:
        FS = S_x

    # Stack the matrices: [F @ S_x; S_noise]
    compound = np.vstack([FS.T, S_noise.T])

    # QR decomposition
    _, R = np.linalg.qr(compound)

    # The upper triangular R gives us the new Cholesky factor
    # Take absolute values on diagonal to ensure positive
    S_new = R[:n, :n].T
    for i in range(n):
        if S_new[i, i] < 0:
            S_new[i:, i] = -S_new[i:, i]

    return S_new


def ensure_symmetric(P: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Enforce symmetry of a covariance matrix.

    Computes (P + P.T) / 2 to ensure the matrix is exactly symmetric,
    which can be lost due to numerical precision issues in matrix operations.

    Parameters
    ----------
    P : ndarray
        Square matrix to symmetrize, shape (n, n).

    Returns
    -------
    P_sym : ndarray
        Symmetric matrix.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[1.0, 0.5 + 1e-15], [0.5, 1.0]])
    >>> P_sym = ensure_symmetric(P)
    >>> np.allclose(P_sym, P_sym.T)
    True
    """
    P = np.asarray(P, dtype=np.float64)
    return (P + P.T) / 2


def compute_matrix_sqrt(
    P: NDArray[np.floating],
    scale: float = 1.0,
    use_eigh_fallback: bool = True,
) -> NDArray[np.floating]:
    """
    Compute the matrix square root using Cholesky or eigendecomposition.

    Attempts Cholesky decomposition first (faster, but requires positive definiteness).
    If that fails and use_eigh_fallback is True, falls back to eigendecomposition
    which is more robust for nearly singular matrices.

    Parameters
    ----------
    P : ndarray
        Symmetric positive semi-definite matrix, shape (n, n).
    scale : float, optional
        Scale factor to multiply P by before taking square root. Default is 1.0.
    use_eigh_fallback : bool, optional
        If True, fall back to eigendecomposition if Cholesky fails. Default is True.

    Returns
    -------
    sqrt_P : ndarray
        Lower triangular matrix such that sqrt_P @ sqrt_P.T ≈ scale * P.

    Raises
    ------
    np.linalg.LinAlgError
        If Cholesky fails and use_eigh_fallback is False.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[4.0, 2.0], [2.0, 3.0]])
    >>> sqrt_P = compute_matrix_sqrt(P)
    >>> np.allclose(sqrt_P @ sqrt_P.T, P)
    True
    """
    P = np.asarray(P, dtype=np.float64)

    try:
        sqrt_P = np.linalg.cholesky(scale * P)
    except np.linalg.LinAlgError:
        if not use_eigh_fallback:
            raise
        # Eigendecomposition fallback for near-singular matrices
        eigvals, eigvecs = np.linalg.eigh(P)
        # Clamp negative eigenvalues to small positive value
        eigvals = np.maximum(eigvals, 1e-10)
        sqrt_P = eigvecs @ np.diag(np.sqrt(scale * eigvals))

    return sqrt_P


def compute_innovation_likelihood(
    innovation: NDArray[np.floating],
    S: NDArray[np.floating],
    S_is_cholesky: bool = False,
) -> float:
    """
    Compute the likelihood of an innovation (measurement residual).

    Computes the multivariate Gaussian probability density for the innovation,
    which is used for track scoring and association in multi-target tracking.

    Parameters
    ----------
    innovation : ndarray
        Innovation (measurement residual) vector, shape (m,).
    S : ndarray
        Innovation covariance matrix, shape (m, m), or its lower triangular
        Cholesky factor if S_is_cholesky is True.
    S_is_cholesky : bool, optional
        If True, S is treated as the lower triangular Cholesky factor.
        Default is False.

    Returns
    -------
    likelihood : float
        Probability density value. Returns 0.0 if covariance is singular.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([0.1, -0.2])
    >>> S = np.array([[0.5, 0.1], [0.1, 0.4]])
    >>> likelihood = compute_innovation_likelihood(y, S)
    >>> likelihood > 0
    True
    """
    innovation = np.asarray(innovation, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)
    m = len(innovation)

    if S_is_cholesky:
        # S is already the lower triangular Cholesky factor
        S_chol = S
        # det(S @ S.T) = det(S)^2
        det_S = np.prod(np.diag(S_chol)) ** 2
        if det_S <= 0:
            return 0.0
        # Solve S @ x = innovation for x, then compute x.T @ x
        import scipy.linalg

        y_normalized = scipy.linalg.solve_triangular(S_chol, innovation, lower=True)
        mahal_sq = np.sum(y_normalized**2)
    else:
        # Compute Cholesky factorization
        try:
            S_chol = np.linalg.cholesky(S)
            det_S = np.prod(np.diag(S_chol)) ** 2
            if det_S <= 0:
                return 0.0
            import scipy.linalg

            y_normalized = scipy.linalg.solve_triangular(S_chol, innovation, lower=True)
            mahal_sq = np.sum(y_normalized**2)
        except np.linalg.LinAlgError:
            # Fallback to direct determinant and solve
            det_S = np.linalg.det(S)
            if det_S <= 0:
                return 0.0
            mahal_sq = innovation @ np.linalg.solve(S, innovation)

    likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    return float(likelihood)


def compute_mahalanobis_distance(
    innovation: NDArray[np.floating],
    S: NDArray[np.floating],
    S_is_cholesky: bool = False,
) -> float:
    """
    Compute the Mahalanobis distance of an innovation.

    The Mahalanobis distance is sqrt(y.T @ S^{-1} @ y), which measures
    how many standard deviations the innovation is from zero.

    Parameters
    ----------
    innovation : ndarray
        Innovation (measurement residual) vector, shape (m,).
    S : ndarray
        Innovation covariance matrix, shape (m, m), or its lower triangular
        Cholesky factor if S_is_cholesky is True.
    S_is_cholesky : bool, optional
        If True, S is treated as the lower triangular Cholesky factor.
        Default is False.

    Returns
    -------
    distance : float
        Mahalanobis distance.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1.0, 0.0])
    >>> S = np.eye(2)
    >>> dist = compute_mahalanobis_distance(y, S)
    >>> np.isclose(dist, 1.0)
    True
    """
    innovation = np.asarray(innovation, dtype=np.float64).flatten()
    S = np.asarray(S, dtype=np.float64)

    if S_is_cholesky:
        import scipy.linalg

        y_normalized = scipy.linalg.solve_triangular(S, innovation, lower=True)
        mahal_sq = np.sum(y_normalized**2)
    else:
        try:
            S_chol = np.linalg.cholesky(S)
            import scipy.linalg

            y_normalized = scipy.linalg.solve_triangular(S_chol, innovation, lower=True)
            mahal_sq = np.sum(y_normalized**2)
        except np.linalg.LinAlgError:
            mahal_sq = innovation @ np.linalg.solve(S, innovation)

    return float(np.sqrt(mahal_sq))


def compute_merwe_weights(
    n: int, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute sigma point weights for the Van der Merwe scaled UKF.

    Parameters
    ----------
    n : int
        State dimension.
    alpha : float, optional
        Spread of sigma points around mean. Default is 1e-3.
    beta : float, optional
        Prior knowledge about distribution. Default is 2.0 (Gaussian).
    kappa : float, optional
        Secondary scaling parameter. Default is 0.0.

    Returns
    -------
    W_m : ndarray
        Mean weights, shape (2n+1,).
    W_c : ndarray
        Covariance weights, shape (2n+1,).

    Examples
    --------
    >>> W_m, W_c = compute_merwe_weights(4, alpha=1e-3, beta=2.0, kappa=0.0)
    >>> np.isclose(W_m.sum(), 1.0)
    True
    """
    lam = alpha**2 * (n + kappa) - n

    W_m = np.zeros(2 * n + 1)
    W_c = np.zeros(2 * n + 1)

    W_m[0] = lam / (n + lam)
    W_c[0] = lam / (n + lam) + (1 - alpha**2 + beta)

    weight = 1 / (2 * (n + lam))
    W_m[1:] = weight
    W_c[1:] = weight

    return W_m, W_c


__all__ = [
    "cholesky_update",
    "qr_update",
    "ensure_symmetric",
    "compute_matrix_sqrt",
    "compute_innovation_likelihood",
    "compute_mahalanobis_distance",
    "compute_merwe_weights",
]
