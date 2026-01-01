"""
Least squares estimation methods.

This module provides ordinary, weighted, and total least squares estimators
commonly used in tracking for state estimation and model fitting.

References
----------
.. [1] S. Van Huffel and J. Vandewalle, "The Total Least Squares Problem:
       Computational Aspects and Analysis," SIAM, 1991.
.. [2] G. H. Golub and C. F. Van Loan, "Matrix Computations," 4th ed.,
       Johns Hopkins University Press, 2013.
"""

from typing import NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class LSResult(NamedTuple):
    """Result of least squares estimation.

    Attributes
    ----------
    x : ndarray
        Estimated parameters.
    residuals : ndarray
        Residual vector (y - A @ x).
    rank : int
        Effective rank of the design matrix.
    singular_values : ndarray
        Singular values of the design matrix.
    """

    x: NDArray[np.floating]
    residuals: NDArray[np.floating]
    rank: int
    singular_values: NDArray[np.floating]


class WLSResult(NamedTuple):
    """Result of weighted least squares estimation.

    Attributes
    ----------
    x : ndarray
        Estimated parameters.
    residuals : ndarray
        Residual vector.
    covariance : ndarray
        Estimated covariance of parameters.
    weighted_residual_sum : float
        Sum of weighted squared residuals.
    """

    x: NDArray[np.floating]
    residuals: NDArray[np.floating]
    covariance: NDArray[np.floating]
    weighted_residual_sum: float


class TLSResult(NamedTuple):
    """Result of total least squares estimation.

    Attributes
    ----------
    x : ndarray
        Estimated parameters.
    residuals_A : ndarray
        Corrections to the design matrix A.
    residuals_b : ndarray
        Corrections to the observation vector b.
    rank : int
        Effective rank used in the solution.
    """

    x: NDArray[np.floating]
    residuals_A: NDArray[np.floating]
    residuals_b: NDArray[np.floating]
    rank: int


def ordinary_least_squares(
    A: ArrayLike,
    b: ArrayLike,
    rcond: Optional[float] = None,
) -> LSResult:
    """
    Ordinary least squares estimation.

    Solves the linear least squares problem:
        min_x ||A @ x - b||_2

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n) where m >= n.
    b : array_like
        Observation vector of shape (m,) or (m, k).
    rcond : float, optional
        Cutoff for small singular values. Values smaller than
        rcond * largest_singular_value are treated as zero.
        Default is machine precision * max(m, n).

    Returns
    -------
    result : LSResult
        Named tuple containing:
        - x: Estimated parameters of shape (n,) or (n, k)
        - residuals: Residual vector
        - rank: Effective rank of A
        - singular_values: Singular values of A

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([1, 2, 2])
    >>> result = ordinary_least_squares(A, b)
    >>> result.x  # Fitted line parameters
    array([0.66666667, 0.5       ])

    Notes
    -----
    Uses SVD-based solution for numerical stability.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Solve using SVD for numerical stability
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=rcond)

    # Compute residuals if not returned by lstsq
    if len(residuals) == 0:
        residuals = b - A @ x

    return LSResult(
        x=x,
        residuals=np.atleast_1d(residuals),
        rank=int(rank),
        singular_values=s,
    )


def weighted_least_squares(
    A: ArrayLike,
    b: ArrayLike,
    W: Optional[ArrayLike] = None,
    weights: Optional[ArrayLike] = None,
) -> WLSResult:
    """
    Weighted least squares estimation.

    Solves the weighted linear least squares problem:
        min_x (b - A @ x)^T W (b - A @ x)

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    W : array_like, optional
        Weight matrix of shape (m, m). If provided, weights is ignored.
        Should be positive definite.
    weights : array_like, optional
        Diagonal weights of shape (m,). Used only if W is None.
        Equivalent to W = diag(weights).

    Returns
    -------
    result : WLSResult
        Named tuple containing:
        - x: Estimated parameters of shape (n,)
        - residuals: Residual vector
        - covariance: Estimated parameter covariance
        - weighted_residual_sum: Weighted sum of squared residuals

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([1, 2, 2])
    >>> weights = np.array([1, 2, 1])  # Higher weight on middle observation
    >>> result = weighted_least_squares(A, b, weights=weights)

    Notes
    -----
    The covariance of the estimated parameters is (A^T W A)^{-1}.

    For measurement noise with covariance R, use W = R^{-1} to get
    the minimum variance unbiased estimator (MVUE).
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape

    # Construct weight matrix
    if W is not None:
        W = np.asarray(W, dtype=np.float64)
    elif weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
        W = np.diag(weights)
    else:
        # Default to identity (OLS)
        W = np.eye(m)

    # Solve normal equations: (A^T W A) x = A^T W b
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ b

    # Solve using Cholesky for positive definite system
    try:
        L = np.linalg.cholesky(AtWA)
        y = np.linalg.solve(L, AtWb)
        x = np.linalg.solve(L.T, y)
        cov = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(n)))
    except np.linalg.LinAlgError:
        # Fall back to pseudoinverse if not positive definite
        cov = np.linalg.pinv(AtWA)
        x = cov @ AtWb

    # Compute residuals and weighted sum
    residuals = b - A @ x
    weighted_residual_sum = float(residuals @ W @ residuals)

    return WLSResult(
        x=x,
        residuals=residuals,
        covariance=cov,
        weighted_residual_sum=weighted_residual_sum,
    )


def total_least_squares(
    A: ArrayLike,
    b: ArrayLike,
    rank: Optional[int] = None,
) -> TLSResult:
    """
    Total least squares (TLS) estimation.

    Solves the errors-in-variables problem where both the design matrix
    and observations may have errors:
        min_{E, r} ||[E | r]||_F  subject to  (A + E) @ x = b + r

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n) where m >= n.
    b : array_like
        Observation vector of shape (m,).
    rank : int, optional
        Truncation rank for regularized TLS. If None, uses full rank.
        Useful when A is rank-deficient.

    Returns
    -------
    result : TLSResult
        Named tuple containing:
        - x: Estimated parameters of shape (n,)
        - residuals_A: Corrections to A of shape (m, n)
        - residuals_b: Corrections to b of shape (m,)
        - rank: Effective rank used

    Examples
    --------
    >>> A = np.array([[1, 1], [2, 1], [3, 1]])
    >>> b = np.array([2.1, 2.9, 4.1])  # Noisy measurements
    >>> result = total_least_squares(A, b)

    Notes
    -----
    TLS is appropriate when both the independent and dependent variables
    are subject to measurement error. It minimizes the orthogonal distance
    from the data points to the fitted model.

    The solution is computed using the SVD of the augmented matrix [A | b].

    References
    ----------
    .. [1] S. Van Huffel and J. Vandewalle, "The Total Least Squares Problem:
           Computational Aspects and Analysis," SIAM, 1991.
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    m, n = A.shape

    if rank is None:
        rank = n

    # Form augmented matrix [A | b]
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    C = np.hstack([A, b])

    # SVD of augmented matrix
    U, s, Vt = np.linalg.svd(C, full_matrices=True)
    V = Vt.T

    # Check if TLS solution exists
    # The solution exists if V[n, n] != 0
    if abs(V[n, n]) < 1e-14:
        raise ValueError(
            "TLS solution does not exist. The smallest singular value "
            "has multiplicity > 1."
        )

    # TLS solution: x = -V[0:n, n] / V[n, n]
    x = -V[:n, n] / V[n, n]

    # Compute corrections using truncated SVD
    # [E | r] = sum_{i=rank}^{n} s_i * u_i * v_i^T
    E_r = np.zeros_like(C)
    for i in range(rank, min(m, n + 1)):
        if i < len(s):
            E_r += s[i] * np.outer(U[:, i], V[:, i])

    residuals_A = E_r[:, :n]
    residuals_b = E_r[:, n].flatten()

    return TLSResult(
        x=x,
        residuals_A=residuals_A,
        residuals_b=residuals_b,
        rank=rank,
    )


def generalized_least_squares(
    A: ArrayLike,
    b: ArrayLike,
    Sigma: ArrayLike,
) -> WLSResult:
    """
    Generalized least squares (GLS) estimation.

    Solves the linear model with correlated errors:
        b = A @ x + e,  where E[e e^T] = Sigma

    This is equivalent to weighted least squares with W = Sigma^{-1}.

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    Sigma : array_like
        Error covariance matrix of shape (m, m).

    Returns
    -------
    result : WLSResult
        Same as weighted_least_squares result.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([1, 2, 2])
    >>> Sigma = np.array([[1, 0.5, 0], [0.5, 1, 0.5], [0, 0.5, 1]])
    >>> result = generalized_least_squares(A, b, Sigma)

    Notes
    -----
    GLS is the BLUE (Best Linear Unbiased Estimator) when the error
    covariance structure is known.
    """
    Sigma = np.asarray(Sigma, dtype=np.float64)
    W = np.linalg.inv(Sigma)
    return weighted_least_squares(A, b, W=W)


def recursive_least_squares(
    x_prev: ArrayLike,
    P_prev: ArrayLike,
    a: ArrayLike,
    y: float,
    forgetting_factor: float = 1.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Recursive least squares (RLS) update.

    Updates parameter estimates when a new observation arrives,
    without reprocessing all previous data.

    Parameters
    ----------
    x_prev : array_like
        Previous parameter estimate of shape (n,).
    P_prev : array_like
        Previous covariance matrix of shape (n, n).
    a : array_like
        New regressor vector of shape (n,).
    y : float
        New observation.
    forgetting_factor : float, optional
        Forgetting factor in (0, 1]. Values < 1 give more weight to
        recent observations. Default is 1.0 (no forgetting).

    Returns
    -------
    x : ndarray
        Updated parameter estimate.
    P : ndarray
        Updated covariance matrix.

    Examples
    --------
    >>> x = np.zeros(2)  # Initial estimate
    >>> P = np.eye(2) * 100  # High initial uncertainty
    >>> # Process observations one at a time
    >>> x, P = recursive_least_squares(x, P, np.array([1, 1]), 2.0)
    >>> x, P = recursive_least_squares(x, P, np.array([1, 2]), 3.0)

    Notes
    -----
    RLS is equivalent to the Kalman filter for the static parameter
    estimation problem. The forgetting factor introduces exponential
    weighting of past data.
    """
    x_prev = np.asarray(x_prev, dtype=np.float64)
    P_prev = np.asarray(P_prev, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    lam = forgetting_factor

    # Innovation
    innovation = y - a @ x_prev

    # Kalman gain
    Pa = P_prev @ a
    denom = lam + a @ Pa
    K = Pa / denom

    # Update estimates
    x = x_prev + K * innovation
    P = (P_prev - np.outer(K, Pa)) / lam

    return x, P


def ridge_regression(
    A: ArrayLike,
    b: ArrayLike,
    alpha: float = 1.0,
) -> NDArray[np.floating]:
    """
    Ridge regression (L2-regularized least squares).

    Solves the regularized problem:
        min_x ||A @ x - b||_2^2 + alpha * ||x||_2^2

    Parameters
    ----------
    A : array_like
        Design matrix of shape (m, n).
    b : array_like
        Observation vector of shape (m,).
    alpha : float, optional
        Regularization parameter. Larger values give more regularization.
        Default is 1.0.

    Returns
    -------
    x : ndarray
        Regularized parameter estimate.

    Examples
    --------
    >>> A = np.array([[1, 1], [1, 2], [1, 3]])
    >>> b = np.array([1, 2, 2])
    >>> x = ridge_regression(A, b, alpha=0.1)

    Notes
    -----
    Ridge regression shrinks parameters toward zero and is useful when:
    - The design matrix is ill-conditioned
    - There are more parameters than observations (n > m)
    - Regularization is desired to prevent overfitting
    """
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    n = A.shape[1]

    # Solve (A^T A + alpha * I) x = A^T b
    AtA = A.T @ A + alpha * np.eye(n)
    Atb = A.T @ b

    return np.linalg.solve(AtA, Atb)


__all__ = [
    "LSResult",
    "WLSResult",
    "TLSResult",
    "ordinary_least_squares",
    "weighted_least_squares",
    "total_least_squares",
    "generalized_least_squares",
    "recursive_least_squares",
    "ridge_regression",
]
