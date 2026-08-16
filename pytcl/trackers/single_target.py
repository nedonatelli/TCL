"""
Single-target tracker implementation.

This module provides a simple single-target tracker using Kalman filtering.
"""

from typing import Callable, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.exceptions import ConfigurationError
from pytcl.trackers.configs import SingleTargetConfig


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
    config : SingleTargetConfig, optional
        Typed configuration. Mutually exclusive with the individual
        keyword arguments above. ``config.F``/``config.Q`` must be set
        (matrix dynamics) -- a config snapshotting callable dynamics
        cannot rebuild the tracker.

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
    >>> predicted = tracker.predict(1.0)
    >>> predicted.state
    array([1., 1., 1., 1.])
    >>> updated = tracker.update(np.array([1.1, 1.2]))
    """

    def __init__(
        self,
        state_dim: Optional[int] = None,
        meas_dim: Optional[int] = None,
        F: Optional[
            Callable[[float], NDArray[np.float64]] | NDArray[np.float64]
        ] = None,
        H: Optional[NDArray[np.float64]] = None,
        Q: Optional[
            Callable[[float], NDArray[np.float64]] | NDArray[np.float64]
        ] = None,
        R: Optional[NDArray[np.float64]] = None,
        gate_threshold: Optional[float] = None,
        *,
        config: Optional[SingleTargetConfig] = None,
    ) -> None:
        if config is not None:
            if any(
                v is not None for v in (state_dim, meas_dim, F, H, Q, R, gate_threshold)
            ):
                raise ConfigurationError(
                    "pass either config= or individual arguments, not both"
                )
            if config.F is None or config.Q is None:
                raise ConfigurationError(
                    "config lacks matrix dynamics; construct with callables "
                    "and use load_session's model arguments instead"
                )
            state_dim = config.state_dim
            meas_dim = config.meas_dim
            H = np.asarray(config.H, dtype=np.float64)
            R = np.asarray(config.R, dtype=np.float64)
            F = np.asarray(config.F, dtype=np.float64)
            Q = np.asarray(config.Q, dtype=np.float64)
            gate_threshold = config.gate_threshold

        if (
            state_dim is None
            or meas_dim is None
            or F is None
            or H is None
            or Q is None
            or R is None
        ):
            raise ConfigurationError("state_dim, meas_dim, F, H, Q and R are required")

        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Store dynamics, retaining the pre-normalization matrix form (None
        # for callables) so session snapshots can recover it -- see
        # pytcl.io.session.
        self._F_matrix = None if callable(F) else np.asarray(F, dtype=np.float64)
        self._Q_matrix = None if callable(Q) else np.asarray(Q, dtype=np.float64)
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

    def _current_state(self) -> TrackState:
        """Return the current state.

        Callers within this class use this instead of the ``state``
        property after already checking ``self._initialized`` (typically
        via the ``RuntimeError`` guard at the top of ``predict``/``update``),
        so the ``None`` branch of the property is provably unreachable here.
        """
        result = self.state
        assert result is not None, "Tracker not initialized"
        return result

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

        F = self._F(dt)  # ty: ignore[call-top-callable]
        Q = self._Q(dt)  # ty: ignore[call-top-callable]

        # Kalman prediction
        self._state = F @ self._state
        self._covariance = F @ self._covariance @ F.T + Q
        self._time += dt

        return self._current_state()

    def update(
        self,
        measurement: ArrayLike,
        measurement_covariance: Optional[ArrayLike] = None,
    ) -> tuple[TrackState, float]:
        """
        Update state with measurement.

        Parameters
        ----------
        measurement : array_like
            Measurement vector.
        measurement_covariance : array_like, optional
            Covariance for this detection, shape ``(meas_dim, meas_dim)``. When
            omitted the tracker's fixed ``R`` is used.

            Supply this when the measurement error varies between detections --
            a converted polar detection, for instance, has a Cartesian
            covariance that is anisotropic and grows with range, so no single
            ``R`` describes it. Both the gate and the Kalman gain then use the
            covariance that actually applies.

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
        ValueError
            If ``measurement_covariance`` is not ``(meas_dim, meas_dim)``.
        """
        if not self._initialized:
            raise RuntimeError("Tracker not initialized")

        z = np.asarray(measurement, dtype=np.float64)

        if measurement_covariance is None:
            R = self.R
        else:
            R = np.asarray(measurement_covariance, dtype=np.float64)
            expected = (self.meas_dim, self.meas_dim)
            if R.shape != expected:
                raise ValueError(
                    f"measurement_covariance has shape {R.shape}, expected {expected}"
                )

        # Innovation
        z_pred = self.H @ self._state
        innovation = z - z_pred
        S = self.H @ self._covariance @ self.H.T + R

        # Mahalanobis distance
        S_inv = np.linalg.inv(S)
        d2 = float(innovation @ S_inv @ innovation)

        # Gating check
        if self.gate_threshold is not None and d2 > self.gate_threshold:
            # Measurement rejected
            return self._current_state(), d2

        # Kalman gain
        K = self._covariance @ self.H.T @ S_inv

        # Update
        self._state = self._state + K @ innovation
        self._covariance = (np.eye(self.state_dim) - K @ self.H) @ self._covariance

        return self._current_state(), d2

    def predict_measurement(
        self,
        measurement_covariance: Optional[ArrayLike] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Predict measurement and innovation covariance.

        Parameters
        ----------
        measurement_covariance : array_like, optional
            Covariance of the detection being considered. Omit to use the
            tracker's fixed ``R``.

        Returns
        -------
        z_pred : ndarray
            Predicted measurement.
        S : ndarray
            Innovation covariance.
        """
        if not self._initialized:
            raise RuntimeError("Tracker not initialized")

        R = (
            self.R
            if measurement_covariance is None
            else np.asarray(measurement_covariance, dtype=np.float64)
        )
        z_pred = self.H @ self._state
        S = self.H @ self._covariance @ self.H.T + R
        return z_pred, S


__all__ = ["SingleTargetTracker", "TrackState"]
