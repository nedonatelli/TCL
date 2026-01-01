"""
Single-target tracker implementation.

This module provides a simple single-target tracker using Kalman filtering.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray


class TrackState(NamedTuple):
    """
    State of a single target track.

    Attributes
    ----------
    state : ndarray
        State estimate vector.
    covariance : ndarray
        State covariance matrix.
    time : float
        Time of state estimate.
    """

    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    time: float


class SingleTargetTracker:
    """
    Single-target tracker using Kalman filtering.

    This tracker maintains a single track and provides predict/update
    functionality with optional gating.

    Parameters
    ----------
    state_dim : int
        Dimension of state vector.
    meas_dim : int
        Dimension of measurement vector.
    F : callable or ndarray
        State transition matrix or function F(dt) -> ndarray.
    H : ndarray
        Measurement matrix.
    Q : callable or ndarray
        Process noise covariance or function Q(dt) -> ndarray.
    R : ndarray
        Measurement noise covariance.
    gate_threshold : float, optional
        Chi-squared gate threshold (default: None, no gating).

    Examples
    --------
    >>> import numpy as np
    >>> # Constant velocity model in 2D
    >>> F = lambda dt: np.array([[1, dt, 0, 0],
    ...                          [0, 1, 0, 0],
    ...                          [0, 0, 1, dt],
    ...                          [0, 0, 0, 1]])
    >>> H = np.array([[1, 0, 0, 0],
    ...               [0, 0, 1, 0]])
    >>> Q = lambda dt: 0.1 * np.eye(4)
    >>> R = np.eye(2) * 0.5
    >>> tracker = SingleTargetTracker(4, 2, F, H, Q, R)
    >>> tracker.initialize(np.array([0, 1, 0, 1]), np.eye(4))
    >>> tracker.predict(1.0)
    >>> tracker.update(np.array([1.1, 1.2]))
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        F: Callable[[float], NDArray[np.float64]] | NDArray[np.float64],
        H: NDArray[np.float64],
        Q: Callable[[float], NDArray[np.float64]] | NDArray[np.float64],
        R: NDArray[np.float64],
        gate_threshold: Optional[float] = None,
    ) -> None:
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Store dynamics
        self._F = F if callable(F) else lambda dt: F
        self.H = np.asarray(H, dtype=np.float64)
        self._Q = Q if callable(Q) else lambda dt: Q
        self.R = np.asarray(R, dtype=np.float64)
        self.gate_threshold = gate_threshold

        # Track state
        self._state: Optional[NDArray[np.float64]] = None
        self._covariance: Optional[NDArray[np.float64]] = None
        self._time: float = 0.0
        self._initialized: bool = False

    def initialize(
        self,
        state: ArrayLike,
        covariance: ArrayLike,
        time: float = 0.0,
    ) -> None:
        """
        Initialize the tracker with initial state.

        Parameters
        ----------
        state : array_like
            Initial state estimate.
        covariance : array_like
            Initial state covariance.
        time : float, optional
            Initial time (default: 0).
        """
        self._state = np.asarray(state, dtype=np.float64)
        self._covariance = np.asarray(covariance, dtype=np.float64)
        self._time = time
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        """Check if tracker is initialized."""
        return self._initialized

    @property
    def state(self) -> Optional[TrackState]:
        """Get current track state."""
        if not self._initialized:
            return None
        return TrackState(
            state=self._state.copy(),
            covariance=self._covariance.copy(),
            time=self._time,
        )

    def predict(self, dt: float) -> TrackState:
        """
        Predict state to new time.

        Parameters
        ----------
        dt : float
            Time step.

        Returns
        -------
        TrackState
            Predicted state.

        Raises
        ------
        RuntimeError
            If tracker is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Tracker not initialized")

        F = self._F(dt)
        Q = self._Q(dt)

        # Kalman prediction
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + Q
        self._time += dt

        return self.state

    def update(
        self,
        measurement: ArrayLike,
    ) -> tuple[TrackState, float]:
        """
        Update state with measurement.

        Parameters
        ----------
        measurement : array_like
            Measurement vector.

        Returns
        -------
        state : TrackState
            Updated state.
        likelihood : float
            Measurement likelihood (Mahalanobis distance).

        Raises
        ------
        RuntimeError
            If tracker is not initialized.
        """
        if not self._initialized:
            raise RuntimeError("Tracker not initialized")

        z = np.asarray(measurement, dtype=np.float64)

        # Innovation
        z_pred = self.H @ self._state
        innovation = z - z_pred
        S = self.H @ self._covariance @ self.H.T + self.R

        # Mahalanobis distance
        S_inv = np.linalg.inv(S)
        d2 = float(innovation @ S_inv @ innovation)

        # Gating check
        if self.gate_threshold is not None and d2 > self.gate_threshold:
            # Measurement rejected
            return self.state, d2

        # Kalman gain
        K = self._covariance @ self.H.T @ S_inv

        # Update
        self._state = self._state + K @ innovation
        self._covariance = (np.eye(self.state_dim) - K @ self.H) @ self._covariance

        return self.state, d2

    def predict_measurement(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predict measurement and innovation covariance.

        Returns
        -------
        z_pred : ndarray
            Predicted measurement.
        S : ndarray
            Innovation covariance.
        """
        if not self._initialized:
            raise RuntimeError("Tracker not initialized")

        z_pred = self.H @ self._state
        S = self.H @ self._covariance @ self.H.T + self.R
        return z_pred, S


__all__ = ["SingleTargetTracker", "TrackState"]
