"""
U-D factorization Kalman filter (Bierman's method).

The U-D filter represents the covariance matrix as P = U @ D @ U.T where
U is unit upper triangular and D is diagonal. This provides excellent
numerical stability with minimal storage requirements.

References
----------
.. [1] G. J. Bierman, "Factorization Methods for Discrete Sequential
       Estimation," Academic Press, 1977.
.. [2] C. L. Thornton and G. J. Bierman, "Gram-Schmidt Algorithms for
       Covariance Propagation," Int. J. Control, 1978.
"""

from typing import NamedTuple

import numpy as np
import scipy.linalg
from numpy.typing import ArrayLike, NDArray


class UDState(NamedTuple):
    """State of a U-D factorization filter.

    The covariance is represented as P = U @ D @ U.T where U is
    unit upper triangular and D is diagonal.

    Attributes
    ----------
    x : ndarray
        State estimate.
    U : ndarray
        Unit upper triangular factor.
    D : ndarray
        Diagonal elements (1D array).
    """

    x: NDArray[np.floating]
    U: NDArray[np.floating]
    D: NDArray[np.floating]


def ud_factorize(P: ArrayLike) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute U-D factorization of a symmetric positive definite matrix.

    Decomposes P = U @ D @ U.T where U is unit upper triangular and D is diagonal.

    Parameters
    ----------
    P : array_like
        Symmetric positive definite matrix, shape (n, n).

    Returns
    -------
    U : ndarray
        Unit upper triangular matrix.
    D : ndarray
        Diagonal elements (1D array).

    Notes
    -----
    The U-D factorization is equivalent to a modified Cholesky decomposition
    and requires only n(n+1)/2 storage elements.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.array([[4.0, 2.0], [2.0, 3.0]])
    >>> U, D = ud_factorize(P)
    >>> np.allclose(U @ np.diag(D) @ U.T, P)
    True
    """
    P = np.asarray(P, dtype=np.float64).copy()  # Make a copy to avoid modifying input
    n = P.shape[0]

    U = np.eye(n)
    D = np.zeros(n)

    for j in range(n - 1, -1, -1):
        D[j] = P[j, j]
        if D[j] > 0:
            alpha = 1.0 / D[j]
            for k in range(j):
                U[k, j] = P[k, j] * alpha
            for i in range(j):
                for k in range(i + 1):
                    P[k, i] = P[k, i] - U[k, j] * D[j] * U[i, j]

    return U, D


def ud_reconstruct(U: ArrayLike, D: ArrayLike) -> NDArray[np.floating]:
    """
    Reconstruct covariance matrix from U-D factors.

    Parameters
    ----------
    U : array_like
        Unit upper triangular matrix.
    D : array_like
        Diagonal elements.

    Returns
    -------
    P : ndarray
        Covariance matrix P = U @ diag(D) @ U.T.

    Examples
    --------
    >>> import numpy as np
    >>> U = np.array([[1.0, 0.5], [0.0, 1.0]])
    >>> D = np.array([2.0, 1.0])
    >>> P = ud_reconstruct(U, D)
    >>> P
    array([[2.5, 0.5],
           [0.5, 1. ]])
    """
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    return U @ np.diag(D) @ U.T


def ud_predict(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    U-D filter prediction step.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    F : array_like
        State transition matrix, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).

    Returns
    -------
    x_pred : ndarray
        Predicted state.
    U_pred : ndarray
        Predicted unit upper triangular factor.
    D_pred : ndarray
        Predicted diagonal elements.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 0.0])
    >>> U = np.eye(2)
    >>> D = np.array([0.1, 0.1])
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.01
    >>> x_pred, U_pred, D_pred = ud_predict(x, U, D, F, Q)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Predicted state
    x_pred = F @ x

    # Predicted covariance: P_pred = F @ P @ F.T + Q
    P = ud_reconstruct(U, D)
    P_pred = F @ P @ F.T + Q

    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2

    # Re-factorize
    U_pred, D_pred = ud_factorize(P_pred)

    return x_pred, U_pred, D_pred


def ud_update_scalar(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    z: float,
    h: ArrayLike,
    r: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    U-D filter scalar measurement update (Bierman's algorithm).

    This is the most efficient form - for vector measurements,
    process each component sequentially.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    z : float
        Scalar measurement.
    h : array_like
        Measurement row vector, shape (n,).
    r : float
        Measurement noise variance.

    Returns
    -------
    x_upd : ndarray
        Updated state.
    U_upd : ndarray
        Updated unit upper triangular factor.
    D_upd : ndarray
        Updated diagonal elements.

    Notes
    -----
    This implements Bierman's sequential scalar update algorithm which
    is numerically stable and efficient for U-D filters.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 0.5])
    >>> U = np.eye(2)
    >>> D = np.array([0.2, 0.1])
    >>> z = 1.1
    >>> h = np.array([1.0, 0.0])
    >>> r = 0.1
    >>> x_upd, U_upd, D_upd = ud_update_scalar(x, U, D, z, h, r)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64).copy()
    D = np.asarray(D, dtype=np.float64).copy()
    h = np.asarray(h, dtype=np.float64).flatten()
    n = len(x)

    # f = U.T @ h
    f = U.T @ h

    # g = D * f (element-wise)
    g = D * f

    # alpha[0] = r + f[0] * g[0]
    alpha = np.zeros(n + 1)
    alpha[0] = r

    for j in range(n):
        alpha[j + 1] = alpha[j] + f[j] * g[j]

    # Innovation
    y = z - h @ x

    # Update D and U
    D_upd = D.copy()
    U_upd = U.copy()

    for j in range(n):
        D_upd[j] = D[j] * alpha[j] / alpha[j + 1]
        if j > 0:
            gamma = g[j]
            for i in range(j):
                U_upd[i, j] = U[i, j] + (gamma / alpha[j]) * (f[i] - U[i, j] * f[j])
                g[i] = g[i] + g[j] * U[i, j]

    # Kalman gain
    K = g / alpha[n]

    # Updated state
    x_upd = x + K * y

    return x_upd, U_upd, D_upd


def ud_update(
    x: ArrayLike,
    U: ArrayLike,
    D: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    NDArray[np.floating],
    float,
]:
    """
    U-D filter vector measurement update.

    Processes measurements sequentially using scalar updates.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    U : array_like
        Unit upper triangular factor, shape (n, n).
    D : array_like
        Diagonal elements, shape (n,).
    z : array_like
        Measurement vector, shape (m,).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
        Should be diagonal for sequential processing.

    Returns
    -------
    x_upd : ndarray
        Updated state.
    U_upd : ndarray
        Updated unit upper triangular factor.
    D_upd : ndarray
        Updated diagonal elements.
    y : ndarray
        Innovation vector.
    likelihood : float
        Measurement likelihood.

    Notes
    -----
    For correlated measurement noise (non-diagonal R), the measurements
    are decorrelated first using a Cholesky decomposition.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 0.5])
    >>> U = np.eye(2)
    >>> D = np.array([0.2, 0.1])
    >>> z = np.array([1.1])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.1]])
    >>> x_upd, U_upd, D_upd, y, likelihood = ud_update(x, U, D, z, H, R)
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    U = np.asarray(U, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    m = len(z)

    # Full innovation before update
    y = z - H @ x

    # Check if R is diagonal
    is_diagonal = np.allclose(R, np.diag(np.diag(R)))

    if is_diagonal:
        # Sequential scalar updates
        x_upd = x.copy()
        U_upd = U.copy()
        D_upd = D.copy()

        for i in range(m):
            x_upd, U_upd, D_upd = ud_update_scalar(
                x_upd, U_upd, D_upd, z[i], H[i, :], R[i, i]
            )
    else:
        # Decorrelate measurements
        S_R = np.linalg.cholesky(R)
        z_dec = scipy.linalg.solve_triangular(S_R, z, lower=True)
        H_dec = scipy.linalg.solve_triangular(S_R, H, lower=True)

        # Sequential scalar updates with unit variance
        x_upd = x.copy()
        U_upd = U.copy()
        D_upd = D.copy()

        for i in range(m):
            x_upd, U_upd, D_upd = ud_update_scalar(
                x_upd, U_upd, D_upd, z_dec[i], H_dec[i, :], 1.0
            )

    # Compute likelihood
    P = ud_reconstruct(U, D)
    S_innov = H @ P @ H.T + R
    det_S = np.linalg.det(S_innov)
    if det_S > 0:
        mahal_sq = y @ np.linalg.solve(S_innov, y)
        likelihood = np.exp(-0.5 * mahal_sq) / np.sqrt((2 * np.pi) ** m * det_S)
    else:
        likelihood = 0.0

    return x_upd, U_upd, D_upd, y, likelihood


__all__ = [
    "UDState",
    "ud_factorize",
    "ud_reconstruct",
    "ud_predict",
    "ud_update_scalar",
    "ud_update",
]
