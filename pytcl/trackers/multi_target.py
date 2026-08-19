"""
Multi-target tracker implementation.

This module provides a multi-target tracker using GNN data association
and Kalman filtering with track management (initiation, maintenance, deletion).
"""

from collections import deque
from enum import Enum
from typing import Callable, List, NamedTuple, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.assignment_algorithms import chi2_gate_threshold, gnn_association
from pytcl.core.exceptions import ConfigurationError
from pytcl.diagnostics import NIS_WINDOW, diagnostics_enabled, log_filter_health, logger
from pytcl.trackers.configs import MultiTargetConfig


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
        Hits required within ``confirm_window`` to confirm a track
        (the M of M-of-N; default: 3). The initiating detection counts.
    confirm_window : int, optional
        Number of most recent association outcomes examined when deciding
        confirmation (the N of M-of-N; default: 5).
    max_misses : int, optional
        Consecutive misses before deletion (default: 5).
    init_covariance : ndarray, optional
        Initial covariance for new tracks, shape ``(state_dim, state_dim)``.
        If None, uses ``100 * I``.
    config : MultiTargetConfig, optional
        Typed configuration. Mutually exclusive with the individual
        keyword arguments above. ``config.F``/``config.Q`` must be set
        (matrix dynamics) -- a config snapshotting callable dynamics
        cannot rebuild the tracker.

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
        gate_probability: Optional[float] = None,
        confirm_hits: Optional[int] = None,
        confirm_window: Optional[int] = None,
        max_misses: Optional[int] = None,
        init_covariance: Optional[NDArray[np.float64]] = None,
        *,
        config: Optional[MultiTargetConfig] = None,
    ) -> None:
        if config is not None:
            if any(
                v is not None
                for v in (
                    state_dim,
                    meas_dim,
                    F,
                    H,
                    Q,
                    R,
                    gate_probability,
                    confirm_hits,
                    confirm_window,
                    max_misses,
                    init_covariance,
                )
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
            gate_probability = config.gate_probability
            confirm_hits = config.confirm_hits
            confirm_window = config.confirm_window
            max_misses = config.max_misses
            init_covariance = (
                None
                if config.init_covariance is None
                else np.asarray(config.init_covariance, dtype=np.float64)
            )
        else:
            # Mirrors MultiTargetConfig's field defaults.
            if gate_probability is None:
                gate_probability = 0.99
            if confirm_hits is None:
                confirm_hits = 3
            if confirm_window is None:
                confirm_window = 5
            if max_misses is None:
                max_misses = 5

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

        self.gate_threshold = chi2_gate_threshold(gate_probability, meas_dim)
        self.gate_probability = gate_probability
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
        measurement_covariances: Optional[Sequence[ArrayLike]] = None,
    ) -> List[Track]:
        """
        Process measurements at new time step.

        Parameters
        ----------
        measurements : list of array_like
            List of measurement vectors.
        dt : float
            Time step since last update.
        measurement_covariances : sequence of array_like, optional
            Per-detection measurement covariance, one ``(meas_dim, meas_dim)``
            matrix per entry in ``measurements``. When omitted the tracker's
            fixed ``R`` is used for every detection.

            Supply this when the measurement error is not the same for every
            detection. The usual case is a converted polar detection: its
            Cartesian covariance is ``J R_polar J^T``, which is anisotropic and
            grows with range, so no single ``R`` describes it. Forcing one makes
            the gate either too tight at long range -- true detections fall
            outside it and the tracker spawns duplicate tracks -- or too loose
            at short range, which admits clutter and inflates the covariance.

        Returns
        -------
        list of Track
            Active tracks after update.

        Raises
        ------
        ValueError
            If ``measurement_covariances`` is given and its length does not
            match ``measurements``, or a matrix is not ``(meas_dim, meas_dim)``.
        """
        self._time += dt

        # Predict all tracks
        self._predict_all(dt)

        # Convert measurements to array
        if len(measurements) == 0:
            Z = np.zeros((0, self.meas_dim))
        else:
            Z = np.array([np.asarray(m) for m in measurements])

        R_list = self._validate_covariances(measurement_covariances, len(measurements))

        # Data association
        if len(self._tracks) > 0 and len(measurements) > 0:
            associations = self._associate(Z, R_list)
        else:
            associations = {}

        # Update associated tracks
        associated_meas = set()
        for track_idx, meas_idx in associations.items():
            self._update_track(
                track_idx, Z[meas_idx], None if R_list is None else R_list[meas_idx]
            )
            associated_meas.add(meas_idx)

        # Handle missed tracks
        for i, track in enumerate(self._tracks):
            if i not in associations and track.status != TrackStatus.DELETED:
                track.misses += 1
                self._record_outcome(track, hit=False)
                if track.misses >= self.max_misses:
                    track.status = TrackStatus.DELETED

        # Initiate new tracks from unassociated measurements
        for j in range(len(measurements)):
            if j not in associated_meas:
                self._initiate_track(Z[j])

        # Remove deleted tracks
        self._tracks = [t for t in self._tracks if t.status != TrackStatus.DELETED]

        return self.tracks

    def _validate_covariances(
        self,
        measurement_covariances: Optional[Sequence[ArrayLike]],
        n_measurements: int,
    ) -> Optional[List[NDArray[np.float64]]]:
        """Check and normalize per-detection covariances, or return None."""
        if measurement_covariances is None:
            return None

        covariances = [np.asarray(R, dtype=np.float64) for R in measurement_covariances]
        if len(covariances) != n_measurements:
            raise ValueError(
                f"measurement_covariances has {len(covariances)} entries for "
                f"{n_measurements} measurements"
            )
        expected = (self.meas_dim, self.meas_dim)
        for i, R in enumerate(covariances):
            if R.shape != expected:
                raise ValueError(
                    f"measurement_covariances[{i}] has shape {R.shape}, "
                    f"expected {expected}"
                )
        return covariances

    def _predict_all(self, dt: float) -> None:
        """Predict all tracks."""
        F = self._F(dt)  # ty: ignore[call-top-callable]
        Q = self._Q(dt)  # ty: ignore[call-top-callable]

        for track in self._tracks:
            if track.status != TrackStatus.DELETED:
                track.state = F @ track.state
                track.covariance = F @ track.covariance @ F.T + Q
                track.time = self._time

    def _associate(
        self,
        Z: NDArray[np.float64],
        R_list: Optional[List[NDArray[np.float64]]] = None,
    ) -> dict[int, int]:
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
            HPHt = self.H @ track.covariance @ self.H.T

            if R_list is None:
                # One innovation covariance for the whole row.
                S_inv = np.linalg.inv(HPHt + self.R)
                for j in range(n_meas):
                    innovation = Z[j] - z_pred
                    cost_matrix[i, j] = float(innovation @ S_inv @ innovation)
            else:
                # The gate has to be evaluated in each detection's own metric.
                for j in range(n_meas):
                    innovation = Z[j] - z_pred
                    S = HPHt + R_list[j]
                    cost_matrix[i, j] = float(
                        innovation @ np.linalg.solve(S, innovation)
                    )

        if diagnostics_enabled():
            for i, track in enumerate(self._tracks):
                if track.status == TrackStatus.DELETED:
                    continue
                rejected = [
                    (j, float(cost_matrix[i, j]))
                    for j in range(n_meas)
                    if cost_matrix[i, j] > self.gate_threshold
                ]
                if rejected:
                    logger.bind(site="gating").debug(
                        "track {}: gated out {} of {} measurements: {}",
                        track.id,
                        len(rejected),
                        n_meas,
                        "; ".join(
                            f"m{j} d={d:.2f}>thr={self.gate_threshold:.2f}"
                            for j, d in rejected
                        ),
                    )

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

        if diagnostics_enabled():
            pairs = [
                (self._tracks[i].id, int(meas_idx))
                for i, meas_idx in associations.items()
            ]
            logger.bind(site="association", algo="gnn").debug(
                "gnn: assigned {} pair(s) {}, total_cost={:.4f}",
                len(pairs),
                pairs,
                float(result.total_cost),
            )

        return associations

    def _update_track(
        self,
        track_idx: int,
        measurement: NDArray[np.float64],
        R: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update a single track with measurement."""
        track = self._tracks[track_idx]

        # Innovation
        z_pred = self.H @ track.state
        innovation = measurement - z_pred
        S = self.H @ track.covariance @ self.H.T + (self.R if R is None else R)
        S_inv = np.linalg.inv(S)

        # Kalman gain
        K = track.covariance @ self.H.T @ S_inv

        # Update
        track.state = track.state + K @ innovation
        track.covariance = (np.eye(self.state_dim) - K @ self.H) @ track.covariance

        # Update counts
        track.hits += 1
        track.misses = 0
        self._record_outcome(track, hit=True)

        # Confirmation is M-of-N over the recent window, not a cumulative
        # lifetime hit count.
        if track.status == TrackStatus.TENTATIVE:
            if sum(track.recent) >= self.confirm_hits:
                track.status = TrackStatus.CONFIRMED

        if diagnostics_enabled():
            # Reuses S_inv already computed above for the Kalman gain -- no
            # extra inversion, and none of this touches state/covariance.
            # _nis_history is not cleared on disable, so a later re-enable
            # resumes the same deque -- the health window can blend NIS
            # values from before the disable/enable toggle.
            nis = float(innovation @ S_inv @ innovation)
            history = getattr(track, "_nis_history", None)
            if history is None:
                history = deque(maxlen=NIS_WINDOW)
                track._nis_history = history
            history.append(nis)
            cov_condition = float(np.linalg.cond(track.covariance))
            log_filter_health(track.id, nis, list(history), cov_condition)

    def _record_outcome(self, track: "_InternalTrack", hit: bool) -> None:
        """Append one association outcome to a track's M-of-N window.

        ``confirm_window`` bounds the window; ``confirm_hits`` of the
        outcomes within it must be hits before a tentative track confirms.
        """
        track.recent.append(hit)
        while len(track.recent) > self.confirm_window:
            track.recent.popleft()

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
        # The initiating detection counts toward confirmation: it is already
        # counted in ``hits``, and leaving it out of the M-of-N window would
        # silently require confirm_hits + 1 detections.
        self._record_outcome(track, hit=True)
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
        #: Recent association outcomes, most recent last, bounded to
        #: ``confirm_window`` by the tracker. ``hits`` remains a cumulative
        #: lifetime count and is reported as such on `Track`; confirmation
        #: reads this window instead.
        self.recent: deque = deque()

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
