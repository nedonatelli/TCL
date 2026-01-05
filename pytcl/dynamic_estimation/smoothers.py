"""
Smoothing algorithms for state estimation.

This module provides optimal fixed-interval, fixed-lag, and fixed-point
smoothers for linear and nonlinear systems. Smoothers use both past and
future measurements to produce optimal state estimates.

The main algorithms are:
- RTS (Rauch-Tung-Striebel) smoother for linear systems
- Fixed-lag smoother for real-time applications
- Fixed-interval smoother for batch processing
- Two-filter smoother for parallel processing
"""

from typing import List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_smooth, kf_update


class SmoothedState(NamedTuple):
    """Smoothed state estimate.

    Attributes
    ----------
    x : ndarray
        Smoothed state estimate.
    P : ndarray
        Smoothed state covariance.
    """

    x: NDArray[np.floating]
    P: NDArray[np.floating]


class RTSResult(NamedTuple):
    """Result of RTS (Rauch-Tung-Striebel) smoother.

    Attributes
    ----------
    x_smooth : list of ndarray
        Smoothed state estimates at each time step.
    P_smooth : list of ndarray
        Smoothed covariances at each time step.
    x_filt : list of ndarray
        Filtered state estimates (forward pass).
    P_filt : list of ndarray
        Filtered covariances (forward pass).
    """

    x_smooth: List[NDArray[np.floating]]
    P_smooth: List[NDArray[np.floating]]
    x_filt: List[NDArray[np.floating]]
    P_filt: List[NDArray[np.floating]]


class FixedLagResult(NamedTuple):
    """Result of fixed-lag smoother.

    Attributes
    ----------
    x_smooth : list of ndarray
        Smoothed state estimates with lag L.
    P_smooth : list of ndarray
        Smoothed covariances with lag L.
    lag : int
        Smoothing lag used.
    """

    x_smooth: List[NDArray[np.floating]]
    P_smooth: List[NDArray[np.floating]]
    lag: int


def rts_smoother(
    x0: ArrayLike,
    P0: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    F_list: Optional[List[ArrayLike]] = None,
    Q_list: Optional[List[ArrayLike]] = None,
    H_list: Optional[List[ArrayLike]] = None,
    R_list: Optional[List[ArrayLike]] = None,
) -> RTSResult:
    """
    Rauch-Tung-Striebel (RTS) fixed-interval smoother.

    Runs a forward Kalman filter pass followed by a backward smoothing
    pass to produce optimal smoothed estimates using all measurements.

    Parameters
    ----------
    x0 : array_like
        Initial state estimate, shape (n,).
    P0 : array_like
        Initial state covariance, shape (n, n).
    measurements : list of array_like
        List of measurements at each time step. Use None for missing.
    F : array_like
        State transition matrix, shape (n, n). Used if F_list not provided.
    Q : array_like
        Process noise covariance. Used if Q_list not provided.
    H : array_like
        Measurement matrix. Used if H_list not provided.
    R : array_like
        Measurement noise covariance. Used if R_list not provided.
    F_list : list of array_like, optional
        Time-varying state transition matrices.
    Q_list : list of array_like, optional
        Time-varying process noise covariances.
    H_list : list of array_like, optional
        Time-varying measurement matrices.
    R_list : list of array_like, optional
        Time-varying measurement noise covariances.

    Returns
    -------
    result : RTSResult
        Named tuple containing smoothed states/covariances and
        filtered states/covariances.

    Examples
    --------
    >>> import numpy as np
    >>> # Simple 1D position tracking with velocity
    >>> x0 = np.array([0.0, 0.0])  # [position, velocity]
    >>> P0 = np.eye(2) * 10.0
    >>> F = np.array([[1, 1], [0, 1]])  # CV model
    >>> Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.1
    >>> H = np.array([[1, 0]])  # Position measurement
    >>> R = np.array([[1.0]])
    >>> measurements = [np.array([1.1]), np.array([2.3]), np.array([3.2])]
    >>> result = rts_smoother(x0, P0, measurements, F, Q, H, R)
    >>> len(result.x_smooth)
    3

    Notes
    -----
    The RTS smoother provides the optimal linear estimate using all
    available data. It consists of two passes:

    Forward pass (Kalman filter):
        x_k|k, P_k|k from measurements z_1, ..., z_k

    Backward pass:
        G_k = P_k|k @ F_k' @ P_{k+1|k}^{-1}
        x_k|N = x_k|k + G_k @ (x_{k+1|N} - x_{k+1|k})
        P_k|N = P_k|k + G_k @ (P_{k+1|N} - P_{k+1|k}) @ G_k'

    where N is the total number of measurements.

    References
    ----------
    .. [1] Rauch, H.E., Tung, F., and Striebel, C.T., "Maximum likelihood
           estimates of linear dynamic systems", AIAA Journal, 1965.
    """
    x0 = np.asarray(x0, dtype=np.float64).flatten()
    P0 = np.asarray(P0, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n_steps = len(measurements)

    # Storage for forward pass
    x_filt: List[NDArray[np.floating]] = []
    P_filt: List[NDArray[np.floating]] = []
    x_pred: List[NDArray[np.floating]] = []
    P_pred: List[NDArray[np.floating]] = []
    F_used: List[NDArray[np.floating]] = []

    # Forward pass (Kalman filter)
    x = x0.copy()
    P = P0.copy()

    for k in range(n_steps):
        # Get time-varying matrices if provided
        F_k = np.asarray(F_list[k], dtype=np.float64) if F_list else F
        Q_k = np.asarray(Q_list[k], dtype=np.float64) if Q_list else Q
        H_k = np.asarray(H_list[k], dtype=np.float64) if H_list else H
        R_k = np.asarray(R_list[k], dtype=np.float64) if R_list else R

        # Predict
        pred = kf_predict(x, P, F_k, Q_k)
        x_pred.append(pred.x)
        P_pred.append(pred.P)
        F_used.append(F_k)

        # Update (if measurement available)
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            upd = kf_update(pred.x, pred.P, z, H_k, R_k)
            x = upd.x
            P = upd.P
        else:
            x = pred.x
            P = pred.P

        x_filt.append(x.copy())
        P_filt.append(P.copy())

    # Backward pass (RTS smoother)
    x_smooth: List[NDArray[np.floating]] = [None] * n_steps  # type: ignore
    P_smooth: List[NDArray[np.floating]] = [None] * n_steps  # type: ignore

    # Initialize with last filtered estimate
    x_smooth[n_steps - 1] = x_filt[n_steps - 1]
    P_smooth[n_steps - 1] = P_filt[n_steps - 1]

    # Backward recursion
    for k in range(n_steps - 2, -1, -1):
        x_s, P_s = kf_smooth(
            x_filt[k],
            P_filt[k],
            x_pred[k + 1],
            P_pred[k + 1],
            x_smooth[k + 1],
            P_smooth[k + 1],
            F_used[k],
        )
        x_smooth[k] = x_s
        P_smooth[k] = P_s

    return RTSResult(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        x_filt=x_filt,
        P_filt=P_filt,
    )


def fixed_lag_smoother(
    x0: ArrayLike,
    P0: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
    lag: int = 5,
) -> FixedLagResult:
    """
    Fixed-lag smoother.

    Produces smoothed estimates with a fixed delay of L time steps.
    At time k, outputs the smoothed estimate for time k-L using
    measurements up to time k.

    Parameters
    ----------
    x0 : array_like
        Initial state estimate.
    P0 : array_like
        Initial state covariance.
    measurements : list of array_like
        List of measurements.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.
    lag : int, optional
        Smoothing lag in time steps (default: 5).

    Returns
    -------
    result : FixedLagResult
        Smoothed estimates with specified lag.

    Examples
    --------
    >>> import numpy as np
    >>> x0 = np.array([0.0, 0.0])
    >>> P0 = np.eye(2) * 10.0
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.1
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[1.0]])
    >>> measurements = [np.array([i + 0.1*np.random.randn()]) for i in range(10)]
    >>> result = fixed_lag_smoother(x0, P0, measurements, F, Q, H, R, lag=3)
    >>> len(result.x_smooth)
    10

    Notes
    -----
    The fixed-lag smoother is suitable for real-time applications where
    a delay of L steps is acceptable. It requires storing only the last
    L filter results instead of the entire sequence.

    At each time k:
    - Run forward filter to get x_{k|k}, P_{k|k}
    - Apply backward smoothing for L steps
    - Output x_{k-L|k}, P_{k-L|k}
    """
    x0 = np.asarray(x0, dtype=np.float64).flatten()
    P0 = np.asarray(P0, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n_steps = len(measurements)
    lag = min(lag, n_steps)

    # Ring buffer for filter results (size = lag + 1)
    buffer_size = lag + 1
    x_buffer: List[Optional[NDArray[np.floating]]] = [None] * buffer_size
    P_buffer: List[Optional[NDArray[np.floating]]] = [None] * buffer_size
    x_pred_buffer: List[Optional[NDArray[np.floating]]] = [None] * buffer_size
    P_pred_buffer: List[Optional[NDArray[np.floating]]] = [None] * buffer_size

    x_smooth: List[NDArray[np.floating]] = []
    P_smooth: List[NDArray[np.floating]] = []

    x = x0.copy()
    P = P0.copy()

    for k in range(n_steps):
        # Store index in ring buffer
        buf_idx = k % buffer_size

        # Predict
        pred = kf_predict(x, P, F, Q)
        x_pred_buffer[buf_idx] = pred.x.copy()
        P_pred_buffer[buf_idx] = pred.P.copy()

        # Update
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            upd = kf_update(pred.x, pred.P, z, H, R)
            x = upd.x
            P = upd.P
        else:
            x = pred.x
            P = pred.P

        x_buffer[buf_idx] = x.copy()
        P_buffer[buf_idx] = P.copy()

        # Compute smoothed estimate for time k-lag (if available)
        if k >= lag:
            # Run backward smoothing from k to k-lag
            x_s = x.copy()
            P_s = P.copy()

            for j in range(lag):
                back_k = k - j
                back_idx = back_k % buffer_size
                prev_idx = (back_k - 1) % buffer_size

                if x_buffer[prev_idx] is None:
                    break

                x_s, P_s = kf_smooth(
                    x_buffer[prev_idx],
                    P_buffer[prev_idx],
                    x_pred_buffer[back_idx],
                    P_pred_buffer[back_idx],
                    x_s,
                    P_s,
                    F,
                )

            x_smooth.append(x_s)
            P_smooth.append(P_s)
        else:
            # Not enough data for full lag, use filtered estimate
            x_smooth.append(x.copy())
            P_smooth.append(P.copy())

    return FixedLagResult(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        lag=lag,
    )


def fixed_interval_smoother(
    x0: ArrayLike,
    P0: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> RTSResult:
    """
    Fixed-interval smoother (alias for RTS smoother).

    Produces optimal smoothed estimates over a fixed time interval
    using all measurements in that interval.

    Parameters
    ----------
    x0 : array_like
        Initial state estimate.
    P0 : array_like
        Initial state covariance.
    measurements : list of array_like
        List of measurements in the interval.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    result : RTSResult
        Smoothed estimates over the interval.

    Examples
    --------
    >>> import numpy as np
    >>> # 1D constant velocity model
    >>> x0 = np.array([0.0, 1.0])  # [position, velocity]
    >>> P0 = np.eye(2) * 5.0
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.array([[0.25, 0.5], [0.5, 1.0]]) * 0.01
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[0.5]])
    >>> measurements = [np.array([0.9]), np.array([2.1]), np.array([3.0]),
    ...                 np.array([4.2]), np.array([4.9])]
    >>> result = fixed_interval_smoother(x0, P0, measurements, F, Q, H, R)
    >>> len(result.x_smooth)
    5
    >>> # Smoothed estimates have lower uncertainty
    >>> np.trace(result.P_smooth[2]) < np.trace(result.P_filt[2])
    True

    See Also
    --------
    rts_smoother : Full RTS smoother with time-varying parameters.

    Notes
    -----
    This is equivalent to the RTS smoother but with a simpler interface
    for the common case of time-invariant system matrices.
    """
    return rts_smoother(x0, P0, measurements, F, Q, H, R)


def two_filter_smoother(
    x0_fwd: ArrayLike,
    P0_fwd: ArrayLike,
    x0_bwd: ArrayLike,
    P0_bwd: ArrayLike,
    measurements: List[ArrayLike],
    F: ArrayLike,
    Q: ArrayLike,
    H: ArrayLike,
    R: ArrayLike,
) -> RTSResult:
    """
    Two-filter smoother (Fraser-Potter form).

    Combines forward and backward filter passes to produce smoothed
    estimates. Useful for parallel implementation.

    Parameters
    ----------
    x0_fwd : array_like
        Initial state for forward filter.
    P0_fwd : array_like
        Initial covariance for forward filter.
    x0_bwd : array_like
        Initial state for backward filter (typically diffuse).
    P0_bwd : array_like
        Initial covariance for backward filter (typically large).
    measurements : list of array_like
        List of measurements.
    F : array_like
        State transition matrix.
    Q : array_like
        Process noise covariance.
    H : array_like
        Measurement matrix.
    R : array_like
        Measurement noise covariance.

    Returns
    -------
    result : RTSResult
        Smoothed estimates.

    Examples
    --------
    >>> import numpy as np
    >>> # 1D position-velocity system
    >>> x0_fwd = np.array([0.0, 1.0])
    >>> P0_fwd = np.eye(2) * 5.0
    >>> # Backward filter starts with diffuse (high uncertainty) prior
    >>> x0_bwd = np.array([5.0, 1.0])  # approximate final state
    >>> P0_bwd = np.eye(2) * 100.0     # diffuse prior
    >>> F = np.array([[1, 1], [0, 1]])
    >>> Q = np.eye(2) * 0.1
    >>> H = np.array([[1, 0]])
    >>> R = np.array([[1.0]])
    >>> measurements = [np.array([0.8]), np.array([1.9]), np.array([3.1]),
    ...                 np.array([4.0]), np.array([5.2])]
    >>> result = two_filter_smoother(x0_fwd, P0_fwd, x0_bwd, P0_bwd,
    ...                               measurements, F, Q, H, R)
    >>> len(result.x_smooth)
    5

    Notes
    -----
    The two-filter smoother runs two independent Kalman filters:
    - Forward filter: uses measurements z_1, ..., z_k
    - Backward filter: uses measurements z_N, ..., z_{k+1}

    The smoothed estimate combines both filters using information
    fusion:
        Y_k|N = Y_k|k^{fwd} + Y_k|k^{bwd}
        y_k|N = y_k|k^{fwd} + y_k|k^{bwd}

    This form is useful for parallel computation since both filters
    can run simultaneously.

    References
    ----------
    .. [1] Fraser, D.C. and Potter, J.E., "The optimum linear smoother
           as a combination of two optimum linear filters", IEEE Trans.
           Automatic Control, 1969.
    """
    x0_fwd = np.asarray(x0_fwd, dtype=np.float64).flatten()
    P0_fwd = np.asarray(P0_fwd, dtype=np.float64)
    x0_bwd = np.asarray(x0_bwd, dtype=np.float64).flatten()
    P0_bwd = np.asarray(P0_bwd, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)

    n_steps = len(measurements)

    # Forward filter pass
    x_fwd: List[NDArray[np.floating]] = []
    P_fwd: List[NDArray[np.floating]] = []

    x = x0_fwd.copy()
    P = P0_fwd.copy()

    for k in range(n_steps):
        pred = kf_predict(x, P, F, Q)
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            upd = kf_update(pred.x, pred.P, z, H, R)
            x = upd.x
            P = upd.P
        else:
            x = pred.x
            P = pred.P
        x_fwd.append(x.copy())
        P_fwd.append(P.copy())

    # Backward filter pass (using inverse transition)
    x_bwd: List[NDArray[np.floating]] = [None] * n_steps  # type: ignore
    P_bwd: List[NDArray[np.floating]] = [None] * n_steps  # type: ignore

    x = x0_bwd.copy()
    P = P0_bwd.copy()

    # Inverse transition for backward filter
    try:
        F_inv = np.linalg.inv(F)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse
        F_inv = np.linalg.pinv(F)

    for k in range(n_steps - 1, -1, -1):
        # Backward prediction (using inverse dynamics)
        x_pred = F_inv @ x
        P_pred = F_inv @ P @ F_inv.T + F_inv @ Q @ F_inv.T

        # Backward update
        z = measurements[k]
        if z is not None:
            z = np.asarray(z, dtype=np.float64).flatten()
            upd = kf_update(x_pred, P_pred, z, H, R)
            x = upd.x
            P = upd.P
        else:
            x = x_pred
            P = P_pred

        x_bwd[k] = x.copy()
        P_bwd[k] = P.copy()

    # Fuse forward and backward estimates using information form
    x_smooth: List[NDArray[np.floating]] = []
    P_smooth: List[NDArray[np.floating]] = []

    for k in range(n_steps):
        # Convert to information form
        try:
            Y_fwd = np.linalg.inv(P_fwd[k])
            y_fwd = Y_fwd @ x_fwd[k]
        except np.linalg.LinAlgError:
            Y_fwd = np.linalg.pinv(P_fwd[k])
            y_fwd = Y_fwd @ x_fwd[k]

        try:
            Y_bwd = np.linalg.inv(P_bwd[k])
            y_bwd = Y_bwd @ x_bwd[k]
        except np.linalg.LinAlgError:
            Y_bwd = np.linalg.pinv(P_bwd[k])
            y_bwd = Y_bwd @ x_bwd[k]

        # Fuse
        Y_smooth = Y_fwd + Y_bwd
        y_smooth = y_fwd + y_bwd

        # Convert back to state form
        try:
            P_s = np.linalg.inv(Y_smooth)
        except np.linalg.LinAlgError:
            P_s = np.linalg.pinv(Y_smooth)

        x_s = P_s @ y_smooth

        # Ensure symmetry
        P_s = (P_s + P_s.T) / 2

        x_smooth.append(x_s)
        P_smooth.append(P_s)

    return RTSResult(
        x_smooth=x_smooth,
        P_smooth=P_smooth,
        x_filt=x_fwd,
        P_filt=P_fwd,
    )


def rts_smoother_single_step(
    x_filt: ArrayLike,
    P_filt: ArrayLike,
    x_pred_next: ArrayLike,
    P_pred_next: ArrayLike,
    x_smooth_next: ArrayLike,
    P_smooth_next: ArrayLike,
    F: ArrayLike,
) -> SmoothedState:
    """
    Single backward step of RTS smoother.

    This is a convenience wrapper around kf_smooth that returns a
    SmoothedState named tuple.

    Parameters
    ----------
    x_filt : array_like
        Filtered state at current time.
    P_filt : array_like
        Filtered covariance at current time.
    x_pred_next : array_like
        Predicted state at next time.
    P_pred_next : array_like
        Predicted covariance at next time.
    x_smooth_next : array_like
        Smoothed state at next time.
    P_smooth_next : array_like
        Smoothed covariance at next time.
    F : array_like
        State transition matrix.

    Returns
    -------
    result : SmoothedState
        Smoothed state and covariance at current time.

    Examples
    --------
    >>> import numpy as np
    >>> # After running forward filter and getting smoothed estimate at k+1
    >>> x_filt = np.array([1.0, 0.5])      # filtered state at k
    >>> P_filt = np.eye(2) * 0.5           # filtered covariance at k
    >>> x_pred_next = np.array([1.5, 0.5]) # predicted state at k+1
    >>> P_pred_next = np.eye(2) * 0.6      # predicted covariance at k+1
    >>> x_smooth_next = np.array([1.4, 0.6])  # smoothed state at k+1
    >>> P_smooth_next = np.eye(2) * 0.3       # smoothed covariance at k+1
    >>> F = np.array([[1, 1], [0, 1]])
    >>> result = rts_smoother_single_step(x_filt, P_filt, x_pred_next,
    ...                                    P_pred_next, x_smooth_next,
    ...                                    P_smooth_next, F)
    >>> result.x.shape
    (2,)
    >>> result.P.shape
    (2, 2)
    """
    x_s, P_s = kf_smooth(
        x_filt, P_filt, x_pred_next, P_pred_next, x_smooth_next, P_smooth_next, F
    )
    return SmoothedState(x=x_s, P=P_s)


__all__ = [
    "SmoothedState",
    "RTSResult",
    "FixedLagResult",
    "rts_smoother",
    "fixed_lag_smoother",
    "fixed_interval_smoother",
    "two_filter_smoother",
    "rts_smoother_single_step",
]
