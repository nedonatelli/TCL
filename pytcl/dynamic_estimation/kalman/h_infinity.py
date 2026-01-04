"""
H-infinity filter implementation.

This module provides the H-infinity filter, a robust estimation approach
that provides bounded-error estimation in the presence of model uncertainty.
Unlike the Kalman filter which minimizes mean-squared error assuming
Gaussian noise, the H-infinity filter minimizes the worst-case estimation
error.

The H-infinity filter is particularly useful when:
- Process and measurement noise statistics are uncertain
- The system model contains unmodeled dynamics
- Robustness to worst-case disturbances is required

References
----------
.. [1] Simon, D., "Optimal State Estimation: Kalman, H∞, and Nonlinear
       Approaches," Wiley, 2006.
.. [2] Shen, X. and Deng, L., "Game Theory Approach to Discrete H∞ Filter
       Design," IEEE Trans. Signal Processing, 1997.
.. [3] Shaked, U. and Theodor, Y., "H∞-Optimal Estimation: A Tutorial,"
       Proc. IEEE CDC, 1992.
"""

from typing import Any, Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class HInfinityUpdate(NamedTuple):
    """Result of H-infinity filter update step.

    Attributes
    ----------
    x : ndarray
        Updated state estimate.
    P : ndarray
        Updated state covariance (error bound matrix).
    y : ndarray
        Innovation (measurement residual).
    S : ndarray
        Innovation covariance.
    K : ndarray
        Filter gain.
    gamma : float
        Performance bound parameter used.
    feasible : bool
        Whether the solution satisfies the H-infinity constraint.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]
    y: NDArray[np.floating]
    S: NDArray[np.floating]
    K: NDArray[np.floating]
    gamma: float
    feasible: bool


class HInfinityPrediction(NamedTuple):
    """Result of H-infinity filter prediction step.

    Attributes
    ----------
    x : ndarray
        Predicted state estimate.
    P : ndarray
        Predicted error bound matrix.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


def hinf_predict(
    x: ArrayLike,
    P: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
) -> HInfinityPrediction:
    """
    H-infinity filter prediction (time update) step.

    The prediction step is identical to the standard Kalman filter:
        x_pred = F @ x + B @ u
        P_pred = F @ P @ F' + Q

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    P : array_like
        Current error bound matrix, shape (n, n).
    F : array_like
        State transition matrix, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).
    B : array_like, optional
        Control input matrix, shape (n, m).
    u : array_like, optional
        Control input, shape (m,).

    Returns
    -------
    result : HInfinityPrediction
        Named tuple with predicted state x and error bound matrix P.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0.0, 1.0])
    >>> P = np.eye(2) * 0.1
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.01
    >>> pred = hinf_predict(x, P, F, Q)
    >>> pred.x
    array([1., 1.])

    See Also
    --------
    hinf_update : H-infinity measurement update step.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    # Predicted state
    x_pred = F @ x

    # Add control input if provided
    if B is not None and u is not None:
        B = np.asarray(B, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64).flatten()
        x_pred = x_pred + B @ u

    # Predicted covariance
    P_pred = F @ P @ F.T + Q

    # Ensure symmetry
    P_pred = (P_pred + P_pred.T) / 2

    return HInfinityPrediction(x=x_pred, P=P_pred)


def hinf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    gamma: float,
    L: Optional[ArrayLike] = None,
) -> HInfinityUpdate:
    """
    H-infinity filter measurement update step.

    Computes the updated state estimate that minimizes the worst-case
    estimation error bound. The performance level gamma determines
    the trade-off between robustness and estimation accuracy.

    The H-infinity filter modifies the Kalman update to account for
    worst-case disturbances on the estimation error:

        P_inv_mod = P^{-1} - gamma^{-2} * L' @ L + H' @ R^{-1} @ H
        K = P_new @ H' @ R^{-1}
        x_new = x + K @ (z - H @ x)
        P_new = P_inv_mod^{-1}

    where L is the matrix that weights the estimation error (typically
    the identity matrix or a subset selecting states of interest).

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    P : array_like
        Predicted error bound matrix, shape (n, n).
    z : array_like
        Measurement vector, shape (m,).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    gamma : float
        Performance bound parameter (gamma > 0). Smaller values provide
        more robustness but require the constraint to be feasible.
        As gamma -> infinity, the filter approaches the Kalman filter.
    L : array_like, optional
        Error weighting matrix, shape (p, n). Defines which linear
        combinations of states to bound. Default is identity (all states).

    Returns
    -------
    result : HInfinityUpdate
        Named tuple containing:
        - x: Updated state estimate
        - P: Updated error bound matrix
        - y: Innovation
        - S: Innovation covariance
        - K: Filter gain
        - gamma: Performance bound used
        - feasible: Whether the H-infinity constraint is satisfied

    Notes
    -----
    The H-infinity constraint is feasible if and only if:
        P^{-1} - gamma^{-2} * L' @ L + H' @ R^{-1} @ H > 0

    If the constraint is not feasible (returns feasible=False), the result
    uses a regularized solution and may not satisfy the performance bound.

    The parameter gamma should be chosen based on the desired robustness
    level. Typical values range from 1 to 100. Lower values provide more
    robustness but are more restrictive.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 1.0])
    >>> P = np.eye(2) * 0.1
    >>> z = np.array([1.1])
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.01]])
    >>> gamma = 10.0
    >>> result = hinf_update(x, P, z, H, R, gamma)
    >>> result.feasible
    True

    See Also
    --------
    hinf_predict : H-infinity prediction step.
    hinf_predict_update : Combined predict and update step.

    References
    ----------
    .. [1] Simon, D., "Optimal State Estimation," Chapter 6, Wiley, 2006.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n = len(x)

    # Default L to identity (bound all states equally)
    if L is None:
        L = np.eye(n)
    else:
        L = np.asarray(L, dtype=np.float64)

    # Innovation
    y = z - H @ x

    # Innovation covariance (standard Kalman)
    S = H @ P @ H.T + R

    # H-infinity modification
    # Compute: P^{-1} - gamma^{-2} * L' @ L + H' @ R^{-1} @ H
    try:
        P_inv = np.linalg.inv(P)
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        P_inv = np.linalg.pinv(P)
        R_inv = np.linalg.pinv(R)

    gamma_sq_inv = 1.0 / (gamma * gamma)

    # Modified information matrix
    P_inv_mod = P_inv - gamma_sq_inv * (L.T @ L) + H.T @ R_inv @ H

    # Check feasibility: P_inv_mod must be positive definite
    try:
        eigvals = np.linalg.eigvalsh(P_inv_mod)
        feasible = bool(np.all(eigvals > 0))
    except np.linalg.LinAlgError:
        feasible = False

    if feasible:
        # Standard H-infinity update
        try:
            P_new = np.linalg.inv(P_inv_mod)
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(P_inv_mod)
            feasible = False
    else:
        # Regularize to make feasible
        # Add small regularization to make positive definite
        reg = abs(min(0, float(np.min(eigvals)))) + 1e-6
        P_inv_mod_reg = P_inv_mod + reg * np.eye(n)
        try:
            P_new = np.linalg.inv(P_inv_mod_reg)
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(P_inv_mod_reg)

    # Ensure symmetry
    P_new = (P_new + P_new.T) / 2

    # H-infinity gain
    K = P_new @ H.T @ R_inv

    # Updated state
    x_new = x + K @ y

    return HInfinityUpdate(
        x=x_new,
        P=P_new,
        y=y,
        S=S,
        K=K,
        gamma=gamma,
        feasible=feasible,
    )


def hinf_predict_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    gamma: float,
    B: Optional[ArrayLike] = None,
    u: Optional[ArrayLike] = None,
    L: Optional[ArrayLike] = None,
) -> HInfinityUpdate:
    """
    Combined H-infinity filter prediction and update step.

    Performs prediction followed by measurement update in a single call.

    Parameters
    ----------
    x : array_like
        Current state estimate, shape (n,).
    P : array_like
        Current error bound matrix, shape (n, n).
    z : array_like
        Measurement vector, shape (m,).
    F : array_like
        State transition matrix, shape (n, n).
    Q : array_like
        Process noise covariance, shape (n, n).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    gamma : float
        Performance bound parameter (gamma > 0).
    B : array_like, optional
        Control input matrix, shape (n, m).
    u : array_like, optional
        Control input, shape (m,).
    L : array_like, optional
        Error weighting matrix, shape (p, n).

    Returns
    -------
    result : HInfinityUpdate
        Named tuple with updated state, covariance, and filter quantities.

    See Also
    --------
    hinf_predict : Prediction step only.
    hinf_update : Update step only.
    """
    pred = hinf_predict(x, P, F, Q, B, u)
    return hinf_update(pred.x, pred.P, z, H, R, gamma, L)


def extended_hinf_update(
    x: ArrayLike,
    P: ArrayLike,
    z: ArrayLike,
    h: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    H: ArrayLike,
    R: ArrayLike,
    gamma: float,
    L: Optional[ArrayLike] = None,
) -> HInfinityUpdate:
    """
    Extended H-infinity filter measurement update for nonlinear systems.

    Uses a linearized measurement model around the current estimate,
    similar to the extended Kalman filter approach.

    Parameters
    ----------
    x : array_like
        Predicted state estimate, shape (n,).
    P : array_like
        Predicted error bound matrix, shape (n, n).
    z : array_like
        Measurement vector, shape (m,).
    h : callable
        Nonlinear measurement function h(x) -> z_pred.
    H : array_like
        Measurement Jacobian dh/dx evaluated at x, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    gamma : float
        Performance bound parameter (gamma > 0).
    L : array_like, optional
        Error weighting matrix, shape (p, n).

    Returns
    -------
    result : HInfinityUpdate
        Named tuple with updated state and covariance.

    Notes
    -----
    The innovation is computed using the nonlinear function:
        y = z - h(x)

    while the gain computation uses the linearized Jacobian H.

    See Also
    --------
    hinf_update : Linear H-infinity update.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n = len(x)

    # Default L to identity
    if L is None:
        L = np.eye(n)
    else:
        L = np.asarray(L, dtype=np.float64)

    # Nonlinear innovation
    z_pred = h(x)
    y = z - z_pred

    # Innovation covariance
    S = H @ P @ H.T + R

    # H-infinity modification
    try:
        P_inv = np.linalg.inv(P)
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P)
        R_inv = np.linalg.pinv(R)

    gamma_sq_inv = 1.0 / (gamma * gamma)
    P_inv_mod = P_inv - gamma_sq_inv * (L.T @ L) + H.T @ R_inv @ H

    # Check feasibility
    try:
        eigvals = np.linalg.eigvalsh(P_inv_mod)
        feasible = bool(np.all(eigvals > 0))
    except np.linalg.LinAlgError:
        feasible = False

    if feasible:
        try:
            P_new = np.linalg.inv(P_inv_mod)
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(P_inv_mod)
            feasible = False
    else:
        reg = abs(min(0, float(np.min(eigvals)))) + 1e-6
        P_inv_mod_reg = P_inv_mod + reg * np.eye(n)
        try:
            P_new = np.linalg.inv(P_inv_mod_reg)
        except np.linalg.LinAlgError:
            P_new = np.linalg.pinv(P_inv_mod_reg)

    P_new = (P_new + P_new.T) / 2

    K = P_new @ H.T @ R_inv
    x_new = x + K @ y

    return HInfinityUpdate(
        x=x_new,
        P=P_new,
        y=y,
        S=S,
        K=K,
        gamma=gamma,
        feasible=feasible,
    )


def find_min_gamma(
    P: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    L: Optional[ArrayLike] = None,
    tol: float = 1e-6,
) -> float:
    """
    Find the minimum feasible gamma for H-infinity filtering.

    Computes the minimum value of gamma for which the H-infinity
    constraint is satisfied (P_inv_mod is positive definite).

    Parameters
    ----------
    P : array_like
        Predicted error bound matrix, shape (n, n).
    H : array_like
        Measurement matrix, shape (m, n).
    R : array_like
        Measurement noise covariance, shape (m, m).
    L : array_like, optional
        Error weighting matrix, shape (p, n). Default is identity.
    tol : float, optional
        Tolerance for feasibility check. Default 1e-6.

    Returns
    -------
    gamma_min : float
        Minimum feasible gamma value.

    Notes
    -----
    The minimum gamma is found by solving for when the minimum
    eigenvalue of P_inv_mod equals zero:

        min_eig(P^{-1} - gamma^{-2} * L' @ L + H' @ R^{-1} @ H) = 0

    This is solved via bisection search.

    Examples
    --------
    >>> import numpy as np
    >>> P = np.eye(2) * 0.1
    >>> H = np.array([[1.0, 0.0]])
    >>> R = np.array([[0.01]])
    >>> gamma_min = find_min_gamma(P, H, R)
    >>> gamma_min < 1.0  # Typical result
    True
    """
    P = np.asarray(P, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n = P.shape[0]

    if L is None:
        L = np.eye(n)
    else:
        L = np.asarray(L, dtype=np.float64)

    try:
        P_inv = np.linalg.inv(P)
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P)
        R_inv = np.linalg.pinv(R)

    # Base term (without gamma contribution)
    base = P_inv + H.T @ R_inv @ H

    # L term
    LtL = L.T @ L

    def min_eigenvalue(gamma: float) -> float:
        """Minimum eigenvalue of P_inv_mod."""
        gamma_sq_inv = 1.0 / (gamma * gamma)
        P_inv_mod = base - gamma_sq_inv * LtL
        return np.min(np.linalg.eigvalsh(P_inv_mod))

    # Binary search for minimum gamma
    # Start with a wide range
    gamma_low = 0.01
    gamma_high = 1000.0

    # Ensure we bracket the solution
    while min_eigenvalue(gamma_high) < 0:
        gamma_high *= 2
        if gamma_high > 1e10:
            return float("inf")

    while min_eigenvalue(gamma_low) > tol:
        gamma_low /= 2
        if gamma_low < 1e-10:
            return 0.0

    # Bisection search
    while gamma_high - gamma_low > tol:
        gamma_mid = (gamma_low + gamma_high) / 2
        if min_eigenvalue(gamma_mid) > 0:
            gamma_high = gamma_mid
        else:
            gamma_low = gamma_mid

    return gamma_high


__all__ = [
    "HInfinityUpdate",
    "HInfinityPrediction",
    "hinf_predict",
    "hinf_update",
    "hinf_predict_update",
    "extended_hinf_update",
    "find_min_gamma",
]
