"""
Multi-target tracker implementation.

This module provides a multi-target tracker using GNN data association
and Kalman filtering with track management (initiation, maintenance, deletion).
"""

from enum import Enum
from typing import Callable, List, NamedTuple, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.assignment_algorithms import chi2_gate_threshold, gnn_association


class TrackStatus(Enum):
    """Track status enumeration."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    DELETED = "deleted"


class Track(NamedTuple):
    """
    Multi-target track.

    Attributes
    ----------
    id : int
        Unique track identifier.
    state : ndarray
        State estimate vector.
    covariance : ndarray
        State covariance matrix.
    status : TrackStatus
        Track status.
    hits : int
        Number of measurement updates.
    misses : int
        Number of consecutive missed detections.
    time : float
        Time of last update.
    """

    id: int
    state: NDArray[np.float64]
    covariance: NDArray[np.float64]
    status: TrackStatus
    hits: int
    misses: int
    time: float


class MultiTargetTracker:
    """
    Multi-target tracker with GNN data association.

    This tracker maintains multiple tracks and handles:
    - Track initiation from unassociated measurements
    - Track update via GNN data association
    - Track confirmation (M-of-N logic)
    - Track deletion (miss count)

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
    gate_probability : float, optional
        Gate probability for association (default: 0.99).
    confirm_hits : int, optional
        Hits needed to confirm track (default: 3).
    confirm_window : int, optional
        Window for M-of-N confirmation (default: 5).
    max_misses : int, optional
        Consecutive misses before deletion (default: 5).
    init_covariance : ndarray, optional
        Initial covariance for new tracks. If None, uses 100*R projected to state.

    Examples
    --------
    >>> import numpy as np
    >>> # Constant velocity model
    >>> F = lambda dt: np.array([[1, dt, 0, 0],
    ...                          [0, 1, 0, 0],
    ...                          [0, 0, 1, dt],
    ...                          [0, 0, 0, 1]])
    >>> H = np.array([[1, 0, 0, 0],
    ...               [0, 0, 1, 0]])
    >>> Q = lambda dt: 0.1 * np.eye(4)
    >>> R = np.eye(2) * 0.5
    >>> tracker = MultiTargetTracker(4, 2, F, H, Q, R)
    >>> # Process measurements
    >>> measurements = [np.array([1, 2]), np.array([5, 6])]
    >>> tracks = tracker.process(measurements, dt=1.0)
    """

    def __init__(
        self,
        state_dim: int,
        meas_dim: int,
        F: Callable[[float], NDArray[np.float64]] | NDArray[np.float64],
        H: NDArray[np.float64],
        Q: Callable[[float], NDArray[np.float64]] | NDArray[np.float64],
        R: NDArray[np.float64],
        gate_probability: float = 0.99,
        confirm_hits: int = 3,
        confirm_window: int = 5,
        max_misses: int = 5,
        init_covariance: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        self._F = F if callable(F) else lambda dt: F
        self.H = np.asarray(H, dtype=np.float64)
        self._Q = Q if callable(Q) else lambda dt: Q
        self.R = np.asarray(R, dtype=np.float64)

        self.gate_threshold = chi2_gate_threshold(gate_probability, meas_dim)
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_misses = max_misses

        if init_covariance is not None:
            self.init_covariance = np.asarray(init_covariance, dtype=np.float64)
        else:
            # Default: large uncertainty
            self.init_covariance = np.eye(state_dim) * 100.0

        # Track storage
        self._tracks: List[_InternalTrack] = []
        self._next_id: int = 0
        self._time: float = 0.0

    @property
    def tracks(self) -> List[Track]:
        """Get list of active tracks."""
        return [t.to_track() for t in self._tracks if t.status != TrackStatus.DELETED]

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Get list of confirmed tracks only."""
        return [t.to_track() for t in self._tracks if t.status == TrackStatus.CONFIRMED]

    def process(
        self,
        measurements: List[ArrayLike],
        dt: float,
    ) -> List[Track]:
        """
        Process measurements at new time step.

        Parameters
        ----------
        measurements : list of array_like
            List of measurement vectors.
        dt : float
            Time step since last update.

        Returns
        -------
        list of Track
            Active tracks after update.
        """
        self._time += dt

        # Predict all tracks
        self._predict_all(dt)

        # Convert measurements to array
        if len(measurements) == 0:
            Z = np.zeros((0, self.meas_dim))
        else:
            Z = np.array([np.asarray(m) for m in measurements])

        # Data association
        if len(self._tracks) > 0 and len(measurements) > 0:
            associations = self._associate(Z)
        else:
            associations = {}

        # Update associated tracks
        associated_meas = set()
        for track_idx, meas_idx in associations.items():
            self._update_track(track_idx, Z[meas_idx])
            associated_meas.add(meas_idx)

        # Handle missed tracks
        for i, track in enumerate(self._tracks):
            if i not in associations and track.status != TrackStatus.DELETED:
                track.misses += 1
                if track.misses >= self.max_misses:
                    track.status = TrackStatus.DELETED

        # Initiate new tracks from unassociated measurements
        for j in range(len(measurements)):
            if j not in associated_meas:
                self._initiate_track(Z[j])

        # Remove deleted tracks
        self._tracks = [t for t in self._tracks if t.status != TrackStatus.DELETED]

        return self.tracks

    def _predict_all(self, dt: float) -> None:
        """Predict all tracks."""
        F = self._F(dt)
        Q = self._Q(dt)

        for track in self._tracks:
            if track.status != TrackStatus.DELETED:
                track.state = F @ track.state
                track.covariance = F @ track.covariance @ F.T + Q
                track.time = self._time

    def _associate(self, Z: NDArray[np.float64]) -> dict[int, int]:
        """
        Associate measurements to tracks using GNN.

        Returns dict mapping track_idx -> meas_idx.
        """
        n_tracks = len(self._tracks)
        n_meas = Z.shape[0]

        # Build cost matrix
        cost_matrix = np.full((n_tracks, n_meas), np.inf)

        for i, track in enumerate(self._tracks):
            if track.status == TrackStatus.DELETED:
                continue

            z_pred = self.H @ track.state
            S = self.H @ track.covariance @ self.H.T + self.R
            S_inv = np.linalg.inv(S)

            for j in range(n_meas):
                innovation = Z[j] - z_pred
                d2 = float(innovation @ S_inv @ innovation)
                cost_matrix[i, j] = d2

        # Run GNN
        result = gnn_association(
            cost_matrix,
            gate_threshold=self.gate_threshold,
            cost_of_non_assignment=self.gate_threshold,
        )

        # Build association dict
        associations = {}
        for i in range(n_tracks):
            meas_idx = result.track_to_measurement[i]
            if meas_idx >= 0:
                associations[i] = meas_idx

        return associations

    def _update_track(self, track_idx: int, measurement: NDArray[np.float64]) -> None:
        """Update a single track with measurement."""
        track = self._tracks[track_idx]

        # Innovation
        z_pred = self.H @ track.state
        innovation = measurement - z_pred
        S = self.H @ track.covariance @ self.H.T + self.R

        # Kalman gain
        K = track.covariance @ self.H.T @ np.linalg.inv(S)

        # Update
        track.state = track.state + K @ innovation
        track.covariance = (np.eye(self.state_dim) - K @ self.H) @ track.covariance

        # Update counts
        track.hits += 1
        track.misses = 0

        # Check confirmation
        if track.status == TrackStatus.TENTATIVE:
            if track.hits >= self.confirm_hits:
                track.status = TrackStatus.CONFIRMED

    def _initiate_track(self, measurement: NDArray[np.float64]) -> None:
        """Initiate new track from measurement."""
        # Initialize state from measurement
        # Use pseudoinverse of H to map measurement to state
        H_pinv = np.linalg.pinv(self.H)
        state = H_pinv @ measurement

        # Create track
        track = _InternalTrack(
            id=self._next_id,
            state=state,
            covariance=self.init_covariance.copy(),
            status=TrackStatus.TENTATIVE,
            hits=1,
            misses=0,
            time=self._time,
        )
        self._tracks.append(track)
        self._next_id += 1


class _InternalTrack:
    """Internal mutable track representation."""

    def __init__(
        self,
        id: int,
        state: NDArray[np.float64],
        covariance: NDArray[np.float64],
        status: TrackStatus,
        hits: int,
        misses: int,
        time: float,
    ) -> None:
        self.id = id
        self.state = state
        self.covariance = covariance
        self.status = status
        self.hits = hits
        self.misses = misses
        self.time = time

    def to_track(self) -> Track:
        """Convert to immutable Track."""
        return Track(
            id=self.id,
            state=self.state.copy(),
            covariance=self.covariance.copy(),
            status=self.status,
            hits=self.hits,
            misses=self.misses,
            time=self.time,
        )


__all__ = ["MultiTargetTracker", "Track", "TrackStatus"]
