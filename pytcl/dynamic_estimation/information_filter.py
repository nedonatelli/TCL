"""
Information filter implementations.

The information filter is an alternative formulation of the Kalman filter
that uses the information matrix Y = P^{-1} and information vector y = Y @ x
instead of the state covariance P and state x.

Advantages:
- Update step is additive (easy to incorporate multiple measurements)
- Natural initialization for unknown initial state (Y = 0)
- Multi-sensor fusion is simpler
- Better for some numerical conditioning

This module provides:
- Information filter (standard form)
- Square-Root Information Filter (SRIF) for improved numerical stability
"""

from typing import List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.linear import (
    information_filter_predict,
    information_filter_update,
)


class InformationState(NamedTuple):
    """State in information form.

    Attributes
    ----------
    y : ndarray
        Information vector (Y @ x).
    Y : ndarray
        Information matrix (P^{-1}).
    """

    y: NDArray[np.floating]
    Y: NDArray[np.floating]


class InformationFilterResult(NamedTuple):
    """Result of running an information filter.

    Attributes
    ----------
    y_filt : list of ndarray
        Filtered information vectors at each time step.
    Y_filt : list of ndarray
        Filtered information matrices at each time step.
    x_filt : list of ndarray
        Filtered state estimates (converted from information form).
    P_filt : list of ndarray
        Filtered covariances (converted from information form).
    """

    y_filt: List[NDArray[np.floating]]
    Y_filt: List[NDArray[np.floating]]
    x_filt: List[NDArray[np.floating]]
    P_filt: List[NDArray[np.floating]]


class SRIFState(NamedTuple):
    """State for Square-Root Information Filter.

    Attributes
    ----------
    r : ndarray
        Information vector (R^{-T} @ x where Y = R @ R.T).
    R : ndarray
        Upper triangular square root of information matrix (Y = R @ R.T).
    """

    r: NDArray[np.floating]
    R: NDArray[np.floating]


class SRIFResult(NamedTuple):
    """Result of Square-Root Information Filter.

    Attributes
    ----------
    r_filt : list of ndarray
        Filtered information vectors.
    R_filt : list of ndarray
        Filtered square-root information matrices.
    x_filt : list of ndarray
        Filtered state estimates.
    P_filt : list of ndarray
        Filtered covariances.
    """

    r_filt: List[NDArray[np.floating]]
    R_filt: List[NDArray[np.floating]]
    x_filt: List[NDArray[np.floating]]
    P_filt: List[NDArray[np.floating]]


def information_to_state(
    y: ArrayLike,
    Y: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert information form to state form.

    Parameters
    ----------
    y : array_like
        Information vector.
    Y : array_like
        Information matrix.

    Returns
    -------
    x : ndarray
        State estimate.
    P : ndarray
        State covariance.
    """
    y = np.asarray(y, dtype=np.float64).flatten()
    Y = np.asarray(Y, dtype=np.float64)

    P = np.linalg.inv(Y)
    x = P @ y

    return x, P


def state_to_information(
    x: ArrayLike,
    P: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Convert state form to information form.

    Parameters
    ----------
    x : array_like
        State estimate.
    P : array_like
        State covariance.

    Returns
    -------
    y : ndarray
        Information vector.
    Y : ndarray
        Information matrix.
    """
    x = np.asarray(x, dtype=np.float64).flatten()
    P = np.asarray(P, dtype=np.float64)

    Y = np.linalg.inv(P)
    y = Y @ x

    return y, Y


def information_filter(
    y0: ArrayLike,
    Y0: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    F_list: Optional[List[ArrayLike]] = None,
    Q_list: Optional[List[ArrayLike]] = None,
    H_list: Optional[List[ArrayLike]] = None,
    R_list: Optional[List[ArrayLike]] = None,
) -> InformationFilterResult:
    """
    Run information filter on a sequence of measurements.

    Parameters
    ----------
    y0 : array_like
        Initial information vector.
    Y0 : array_like
        Initial information matrix.
    measurements : list of array_like
        List of measurements. Use None for missing measurements.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.
    F_list : list of array_like, optional
        Time-varying transition matrices.
    Q_list : list of array_like, optional
        Time-varying process noise.
    H_list : list of array_like, optional
        Time-varying measurement matrices.
    R_list : list of array_like, optional
        Time-varying measurement noise.

    Returns
    -------
    result : InformationFilterResult
        Information filter results including both information form
        and state form estimates.

    Examples
    --------
    >>> import numpy as np
    >>> # Start with unknown state (zero information)
    >>> n = 2
    >>> y0 = np.zeros(n)
    >>> Y0 = np.zeros((n, n))  # Unknown initial state
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.1
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[1.0]])
    >>> measurements = [np.array([1.0]), np.array([2.1]), np.array([3.3])]
    >>> result = information_filter(y0, Y0, measurements, F, Q, H, R)
    >>> len(result.x_filt)
    3

    Notes
    -----
    The information filter is particularly useful when:
    - The initial state is completely unknown (set Y0 = 0)
    - Fusing measurements from multiple sensors
    - Measurements arrive from distributed sources

    References
    ----------
    .. [1] Mutambara, A.G.O., "Decentralized Estimation and Control for
           Multisensor Systems", CRC Press, 1998.
    """
    y0 = np.asarray(y0, dtype=np.float64).flatten()
    Y0 = np.asarray(Y0, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n_steps = len(measurements)

    # Storage
    y_filt: List[NDArray[np.floating]] = []
    Y_filt: List[NDArray[np.floating]] = []
    x_filt: List[NDArray[np.floating]] = []
    P_filt: List[NDArray[np.floating]] = []

    y = y0.copy()
    Y = Y0.copy()

    for k in range(n_steps):
        # Get time-varying matrices
        F_k = np.asarray(F_list[k], dtype=np.float64) if F_list else F
        Q_k = np.asarray(Q_list[k], dtype=np.float64) if Q_list else Q
        H_k = np.asarray(H_list[k], dtype=np.float64) if H_list else H
        R_k = np.asarray(R_list[k], dtype=np.float64) if R_list else R

        # Predict (requires positive definite Y)
        if np.linalg.matrix_rank(Y) == Y.shape[0]:
            y, Y = information_filter_predict(y, Y, F_k, Q_k)
        else:
            # For singular Y (unknown state), just add process noise
            # This is a special case for initialization
            pass

        # Update if measurement available
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            y, Y = information_filter_update(y, Y, z, H_k, R_k)

        y_filt.append(y.copy())
        Y_filt.append(Y.copy())

        # Convert to state form for output
        if np.linalg.matrix_rank(Y) == Y.shape[0]:
            x, P = information_to_state(y, Y)
        else:
            # State is still uncertain - use pseudo-inverse
            P = np.linalg.pinv(Y)
            x = P @ y
        x_filt.append(x)
        P_filt.append(P)

    return InformationFilterResult(
        y_filt=y_filt,
        Y_filt=Y_filt,
        x_filt=x_filt,
        P_filt=P_filt,
    )


def srif_predict(
    r: ArrayLike,
    R: ArrayLike,
    F: ArrayLike,
    Q: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Square-Root Information Filter prediction step.

    Uses QR decomposition to maintain numerical stability.

    Parameters
    ----------
    r : array_like
        Information vector (related to R @ x).
    R : array_like
        Upper triangular square root of information matrix.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.

    Returns
    -------
    r_pred : ndarray
        Predicted information vector.
    R_pred : ndarray
        Predicted square root information matrix.

    Notes
    -----
    The SRIF prediction uses:
    - Convert to state space
    - Apply prediction
    - Use Cholesky/QR to get square root form back
    """
    r = np.asarray(r, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)

    n = len(r)

    # Convert to state form
    if np.linalg.matrix_rank(R) == n:
        Y = R.T @ R
        P = np.linalg.inv(Y)
        x = np.linalg.solve(R.T @ R, R.T @ r)
    else:
        # Handle singular case
        Y = R.T @ R
        P = np.linalg.pinv(Y)
        x = P @ (R.T @ r)

    # Predict in state space
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # Convert back to square root information form
    # Y_pred = inv(P_pred), R_pred s.t. R_pred.T @ R_pred = Y_pred
    try:
        L_pred = np.linalg.cholesky(P_pred)
        R_pred = np.linalg.inv(L_pred).T  # Upper triangular
    except np.linalg.LinAlgError:
        # Fallback using SVD
        U, s, Vt = np.linalg.svd(P_pred)
        S_sqrt_inv = np.diag(1.0 / np.sqrt(s))
        R_pred = S_sqrt_inv @ Vt

    r_pred = R_pred @ x_pred

    return r_pred, R_pred


def srif_update(
    r: ArrayLike,
    R: ArrayLike,
    z: ArrayLike,
    H: ArrayLike,
    R_meas: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Square-Root Information Filter update step.

    Uses Householder or Givens rotations for numerical stability.

    Parameters
    ----------
    r : array_like
        Predicted information vector.
    R : array_like
        Predicted square root information matrix.
    z : array_like
        Measurement.
    H : array_like
        Measurement matrix.
    R_meas : array_like
        Measurement noise covariance.

    Returns
    -------
    r_upd : ndarray
        Updated information vector.
    R_upd : ndarray
        Updated square root information matrix.

    Notes
    -----
    The update uses QR decomposition to combine the prior information
    with the measurement information while maintaining numerical stability.
    """
    r = np.asarray(r, dtype=np.float64).flatten()
    R = np.asarray(R, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).flatten()
    H = np.asarray(H, dtype=np.float64)
    R_meas = np.asarray(R_meas, dtype=np.float64)

    n = len(r)
    m = len(z)

    # Get square root of measurement information
    try:
        L_meas = np.linalg.cholesky(R_meas)
        R_meas_inv = np.linalg.inv(L_meas).T
    except np.linalg.LinAlgError:
        R_meas_inv = np.linalg.inv(np.linalg.cholesky(R_meas + np.eye(m) * 1e-10)).T

    # Form the information from measurement
    H_bar = R_meas_inv @ H
    z_bar = R_meas_inv @ z

    # Stack for QR decomposition
    # [R   r  ]     [R_upd   r_upd]
    # [H_bar z_bar] -> [0      e    ]
    # where e is the residual

    # Build augmented matrix
    A = np.vstack([R, H_bar])
    b = np.concatenate([r, z_bar])

    # QR decomposition
    Q, R_aug = np.linalg.qr(A, mode="reduced")

    # Extract updated values
    R_upd = R_aug[:n, :n]
    r_upd = Q[:, :n].T @ b

    return r_upd, R_upd


def srif_filter(
    r0: ArrayLike,
    R0: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R_meas: ArrayLike,
) -> SRIFResult:
    """
    Run Square-Root Information Filter on a sequence of measurements.

    The SRIF maintains the square root of the information matrix,
    providing better numerical stability than the standard information
    filter, especially for large or ill-conditioned problems.

    Parameters
    ----------
    r0 : array_like
        Initial information vector (R0 @ x0).
    R0 : array_like
        Initial square root information matrix (upper triangular,
        such that R0.T @ R0 = Y0 = P0^{-1}).
    measurements : list of array_like
        List of measurements. Use None for missing measurements.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R_meas : array_like
        Measurement noise covariance.

    Returns
    -------
    result : SRIFResult
        SRIF results including square root information form
        and state form estimates.

    Examples
    --------
    >>> import numpy as np
    >>> n = 2
    >>> # Initialize with some prior knowledge
    >>> P0 = np.eye(n) * 10.0
    >>> R0 = np.linalg.inv(np.linalg.cholesky(P0)).T
    >>> x0 = np.array([0.0, 0.0])
    >>> r0 = R0 @ x0
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.1
    >>> H = np.array([[1, 0]])
    >>> R_meas = np.array([[1.0]])
    >>> measurements = [np.array([1.0]), np.array([2.0])]
    >>> result = srif_filter(r0, R0, measurements, F, Q, H, R_meas)
    >>> len(result.x_filt)
    2

    Notes
    -----
    The SRIF is algebraically equivalent to the standard Kalman filter
    but uses orthogonal transformations (QR decomposition) instead of
    matrix inversions. This provides:

    - Better numerical stability
    - Guaranteed positive semi-definiteness
    - More accurate results for ill-conditioned problems

    References
    ----------
    .. [1] Bierman, G.J., "Factorization Methods for Discrete Sequential
           Estimation", Academic Press, 1977.
    .. [2] Kaminski, P.G., Bryson, A.E., and Schmidt, S.F., "Discrete
           Square Root Filtering: A Survey of Current Techniques",
           IEEE Trans. Automatic Control, 1971.
    """
    r0 = np.asarray(r0, dtype=np.float64).flatten()
    R0 = np.asarray(R0, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R_meas = np.asarray(R_meas, dtype=np.float64)

    n = len(r0)
    n_steps = len(measurements)

    # Storage
    r_filt: List[NDArray[np.floating]] = []
    R_filt: List[NDArray[np.floating]] = []
    x_filt: List[NDArray[np.floating]] = []
    P_filt: List[NDArray[np.floating]] = []

    r = r0.copy()
    R = R0.copy()

    for k in range(n_steps):
        # Predict
        r, R = srif_predict(r, R, F, Q)

        # Update if measurement available
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            r, R = srif_update(r, R, z, H, R_meas)

        r_filt.append(r.copy())
        R_filt.append(R.copy())

        # Convert to state form
        Y = R.T @ R
        if np.linalg.matrix_rank(Y) == n:
            P = np.linalg.inv(Y)
            x = np.linalg.solve(Y, R.T @ r)
        else:
            P = np.linalg.pinv(Y)
            x = P @ (R.T @ r)

        x_filt.append(x)
        P_filt.append(P)

    return SRIFResult(
        r_filt=r_filt,
        R_filt=R_filt,
        x_filt=x_filt,
        P_filt=P_filt,
    )


def fuse_information(
    info_states: List[InformationState],
) -> InformationState:
    """
    Fuse multiple information states.

    This is useful for multi-sensor fusion where each sensor produces
    its own information contribution.

    Parameters
    ----------
    info_states : list of InformationState
        List of information states to fuse.

    Returns
    -------
    fused : InformationState
        Fused information state.

    Examples
    --------
    >>> import numpy as np
    >>> # Two sensors with different measurements
    >>> state1 = InformationState(
    ...     y=np.array([1.0, 0.5]),
    ...     Y=np.array([[0.5, 0], [0, 0.1]])
    ... )
    >>> state2 = InformationState(
    ...     y=np.array([0.8, 0.6]),
    ...     Y=np.array([[0.3, 0], [0, 0.2]])
    ... )
    >>> fused = fuse_information([state1, state2])
    >>> fused.Y[0, 0]  # Should be 0.5 + 0.3 = 0.8
    0.8

    Notes
    -----
    Information fusion is additive:
        Y_fused = Y_1 + Y_2 + ... + Y_n
        y_fused = y_1 + y_2 + ... + y_n

    This is a major advantage of the information form for multi-sensor
    systems - each sensor can independently compute its contribution,
    and fusion is simply addition.
    """
    if not info_states:
        raise ValueError("At least one information state required")

    n = len(info_states[0].y)

    Y_fused = np.zeros((n, n))
    y_fused = np.zeros(n)

    for state in info_states:
        Y_fused += state.Y
        y_fused += state.y

    return InformationState(y=y_fused, Y=Y_fused)


__all__ = [
    "InformationState",
    "InformationFilterResult",
    "SRIFState",
    "SRIFResult",
    "information_to_state",
    "state_to_information",
    "information_filter",
    "srif_predict",
    "srif_update",
    "srif_filter",
    "fuse_information",
]
