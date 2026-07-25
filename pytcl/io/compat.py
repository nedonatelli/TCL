"""Backward compatibility adapters for v1.x filter integration.

Provides adapter classes that seamlessly connect pytcl's existing filter
outputs (KF, EKF, UKF, IMM, particle filters, MHT, JPDA) to the
TrackDatabaseManager and TrackHDF5Storage backends.

Examples
--------
Wrap a Kalman filter loop with SQL persistence:

>>> from pytcl.io.compat import KalmanTrackAdapter
>>> adapter = KalmanTrackAdapter(db, "trk_001", F, H, Q, R)  # doctest: +SKIP
>>> adapter.initialize(x0, P0, timestamp=0.0)  # doctest: +SKIP
>>> for k, z in enumerate(measurements):  # doctest: +SKIP
...     adapter.predict_update(z, timestamp=float(k + 1))

Use with MultiTargetTracker:

>>> from pytcl.io.compat import TrackerDatabaseAdapter
>>> adapter = TrackerDatabaseAdapter(db, tracker)  # doctest: +SKIP
>>> for k, measurements in enumerate(scans):  # doctest: +SKIP
...     adapter.process_scan(measurements, dt=1.0, timestamp=float(k))
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.io.track_database import TrackDatabaseManager, TrackDatabaseStatus


class KalmanTrackAdapter:
    """Adapter connecting Kalman filter predict/update to SQL storage.

    Wraps a single track's filter state and persists every predict/update
    into a TrackDatabaseManager.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Unique track identifier.
    F : NDArray
        State transition matrix.
    H : NDArray
        Measurement matrix.
    Q : NDArray
        Process noise covariance.
    R : NDArray
        Measurement noise covariance.

    Examples
    --------
    >>> adapter = KalmanTrackAdapter(db, "trk_001", F, H, Q, R)  # doctest: +SKIP
    >>> adapter.initialize(x0, P0, timestamp=0.0)  # doctest: +SKIP
    >>> adapter.predict(dt=1.0, timestamp=1.0)  # doctest: +SKIP
    >>> adapter.update(z, timestamp=1.0)  # doctest: +SKIP
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        track_id: str,
        F: ArrayLike,
        H: ArrayLike,
        Q: ArrayLike,
        R: ArrayLike,
    ) -> None:
        self._db = db
        self._track_id = track_id
        self._F = np.asarray(F, dtype=np.float64)
        self._H = np.asarray(H, dtype=np.float64)
        self._Q = np.asarray(Q, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._x: Optional[NDArray[np.float64]] = None
        self._P: Optional[NDArray[np.float64]] = None

    @property
    def state(self) -> Optional[NDArray[np.float64]]:
        """Current state estimate."""
        return self._x

    @property
    def covariance(self) -> Optional[NDArray[np.float64]]:
        """Current covariance estimate."""
        return self._P

    @property
    def track_id(self) -> str:
        """Track identifier."""
        return self._track_id

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the track with an initial state estimate.

        Parameters
        ----------
        x0 : ArrayLike
            Initial state vector.
        P0 : ArrayLike
            Initial covariance matrix.
        timestamp : float
            Birth time.
        metadata : dict, optional
            Additional metadata.
        """
        self._x = np.asarray(x0, dtype=np.float64)
        self._P = np.asarray(P0, dtype=np.float64)
        self._db.initiate_track(
            self._track_id, self._x, self._P, timestamp, metadata=metadata
        )

    def predict(self, timestamp: float) -> None:
        """Run Kalman prediction and store result.

        Parameters
        ----------
        timestamp : float
            Time of prediction.
        """
        from pytcl.dynamic_estimation.kalman.linear import kf_predict

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized. Call initialize() first.")

        pred = kf_predict(self._x, self._P, self._F, self._Q)
        self._x, self._P = pred.x, pred.P
        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            update_type="prediction",
        )

    def update(
        self,
        measurement: ArrayLike,
        timestamp: float,
        detection_id: Optional[str] = None,
        sensor_id: str = "default",
    ) -> None:
        """Run Kalman update and store result.

        Parameters
        ----------
        measurement : ArrayLike
            Measurement vector.
        timestamp : float
            Time of measurement.
        detection_id : str, optional
            Detection identifier. Auto-generated if not provided.
        sensor_id : str
            Sensor identifier.
        """
        from pytcl.dynamic_estimation.kalman.linear import kf_update

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized. Call initialize() first.")

        z = np.asarray(measurement, dtype=np.float64)
        upd = kf_update(self._x, self._P, z, self._H, self._R)
        residual = upd.y if hasattr(upd, "y") else None
        self._x, self._P = upd.x, upd.P

        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            residual=residual,
            update_type="update",
        )

        # Store and associate detection
        if detection_id is None:
            detection_id = f"{self._track_id}_det_{timestamp:.6f}"
        self._db.store_detection(
            detection_id,
            z,
            sensor_id,
            timestamp,
            covariance=self._R,
        )
        self._db.associate_detection(detection_id, self._track_id)

    def predict_update(
        self,
        measurement: ArrayLike,
        timestamp: float,
        detection_id: Optional[str] = None,
        sensor_id: str = "default",
    ) -> None:
        """Combined predict + update step.

        Parameters
        ----------
        measurement : ArrayLike
            Measurement vector.
        timestamp : float
            Time of measurement.
        detection_id : str, optional
            Detection identifier.
        sensor_id : str
            Sensor identifier.
        """
        self.predict(timestamp)
        self.update(measurement, timestamp, detection_id, sensor_id)


class EKFTrackAdapter:
    """Adapter connecting Extended Kalman filter to SQL storage.

    Similar to KalmanTrackAdapter but uses nonlinear dynamics/measurement
    functions with Jacobians.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Unique track identifier.
    f : callable
        State transition function f(x) -> x_pred.
    F_func : callable
        Jacobian of f, F_func(x) -> F matrix.
    h : callable
        Measurement function h(x) -> z_pred.
    H_func : callable
        Jacobian of h, H_func(x) -> H matrix.
    Q : NDArray
        Process noise covariance.
    R : NDArray
        Measurement noise covariance.
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        track_id: str,
        f: Callable[..., Any],
        F_func: Callable[..., Any],
        h: Callable[..., Any],
        H_func: Callable[..., Any],
        Q: ArrayLike,
        R: ArrayLike,
    ) -> None:
        self._db = db
        self._track_id = track_id
        self._f = f
        self._F_func = F_func
        self._h = h
        self._H_func = H_func
        self._Q = np.asarray(Q, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._x: Optional[NDArray[np.float64]] = None
        self._P: Optional[NDArray[np.float64]] = None

    @property
    def state(self) -> Optional[NDArray[np.float64]]:
        """Current state estimate."""
        return self._x

    @property
    def covariance(self) -> Optional[NDArray[np.float64]]:
        """Current covariance estimate."""
        return self._P

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize the track."""
        self._x = np.asarray(x0, dtype=np.float64)
        self._P = np.asarray(P0, dtype=np.float64)
        self._db.initiate_track(self._track_id, self._x, self._P, timestamp)

    def predict(self, timestamp: float) -> None:
        """EKF prediction step."""
        from pytcl.dynamic_estimation.kalman.extended import ekf_predict

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized.")

        F = self._F_func(self._x)
        pred = ekf_predict(self._x, self._P, self._f, F, self._Q)
        self._x, self._P = pred.x, pred.P
        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            update_type="prediction",
        )

    def update(self, measurement: ArrayLike, timestamp: float) -> None:
        """EKF update step."""
        from pytcl.dynamic_estimation.kalman.extended import ekf_update

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized.")

        z = np.asarray(measurement, dtype=np.float64)
        H = self._H_func(self._x)
        upd = ekf_update(self._x, self._P, z, self._h, H, self._R)
        self._x, self._P = upd.x, upd.P
        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            update_type="update",
        )


class UKFTrackAdapter:
    """Adapter connecting Unscented Kalman filter to SQL storage.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Unique track identifier.
    f : callable
        State transition function.
    h : callable
        Measurement function.
    Q : NDArray
        Process noise covariance.
    R : NDArray
        Measurement noise covariance.
    alpha : float
        UKF spread parameter.
    beta : float
        UKF distribution parameter.
    kappa : float
        UKF scaling parameter.
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        track_id: str,
        f: Callable[..., Any],
        h: Callable[..., Any],
        Q: ArrayLike,
        R: ArrayLike,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self._db = db
        self._track_id = track_id
        self._f = f
        self._h = h
        self._Q = np.asarray(Q, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._alpha = alpha
        self._beta = beta
        self._kappa = kappa
        self._x: Optional[NDArray[np.float64]] = None
        self._P: Optional[NDArray[np.float64]] = None

    @property
    def state(self) -> Optional[NDArray[np.float64]]:
        """Current state estimate."""
        return self._x

    @property
    def covariance(self) -> Optional[NDArray[np.float64]]:
        """Current covariance estimate."""
        return self._P

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize the track."""
        self._x = np.asarray(x0, dtype=np.float64)
        self._P = np.asarray(P0, dtype=np.float64)
        self._db.initiate_track(self._track_id, self._x, self._P, timestamp)

    def predict(self, timestamp: float) -> None:
        """UKF prediction step."""
        from pytcl.dynamic_estimation.kalman.unscented import ukf_predict

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized.")

        pred = ukf_predict(
            self._x,
            self._P,
            self._f,
            self._Q,
            self._alpha,
            self._beta,
            self._kappa,
        )
        self._x, self._P = pred.x, pred.P
        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            update_type="prediction",
        )

    def update(self, measurement: ArrayLike, timestamp: float) -> None:
        """UKF update step."""
        from pytcl.dynamic_estimation.kalman.unscented import ukf_update

        if self._x is None or self._P is None:
            raise RuntimeError("Track not initialized.")

        z = np.asarray(measurement, dtype=np.float64)
        upd = ukf_update(
            self._x,
            self._P,
            z,
            self._h,
            self._R,
            self._alpha,
            self._beta,
            self._kappa,
        )
        self._x, self._P = upd.x, upd.P
        self._db.update_track_state(
            self._track_id,
            self._x,
            self._P,
            timestamp,
            update_type="update",
        )


class TrackerDatabaseAdapter:
    """Adapter connecting MultiTargetTracker or MHTTracker to SQL storage.

    Automatically persists track state after each scan, handles initiation
    and deletion, and keeps the database in sync with the tracker.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    tracker : object
        A pytcl tracker with a ``process`` or ``process_scan`` method
        that returns a list of Track-like objects.
    confirm_hits : int
        Number of hits to confirm a track. Default is 3.
    max_misses : int
        Consecutive misses before marking dead. Default is 5.

    Examples
    --------
    >>> adapter = TrackerDatabaseAdapter(db, tracker)  # doctest: +SKIP
    >>> for k, meas in enumerate(scans):  # doctest: +SKIP
    ...     tracks = adapter.process_scan(meas, dt=1.0, timestamp=float(k))
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        tracker: Any,
        confirm_hits: int = 3,
        max_misses: int = 5,
    ) -> None:
        self._db = db
        self._tracker = tracker
        self._confirm_hits = confirm_hits
        self._max_misses = max_misses
        self._known_tracks: Dict[int, str] = {}

    def process_scan(
        self,
        measurements: Sequence[ArrayLike],
        dt: float,
        timestamp: float,
        sensor_id: str = "default",
    ) -> List[Any]:
        """Process a scan of measurements and persist results.

        Parameters
        ----------
        measurements : sequence of ArrayLike
            Measurement vectors for this scan.
        dt : float
            Time step since last scan.
        timestamp : float
            Current time.
        sensor_id : str
            Sensor identifier for stored detections.

        Returns
        -------
        list
            Track objects from the underlying tracker.
        """
        # Store detections
        for j, z in enumerate(measurements):
            det_id = f"det_{timestamp:.6f}_{j:04d}"
            self._db.store_detection(
                det_id,
                np.asarray(z),
                sensor_id,
                timestamp,
            )

        # Run tracker
        result = self._tracker.process(measurements, dt)
        tracks = result if isinstance(result, list) else list(result)

        # Sync database with tracker state
        active_ids = set()
        for track in tracks:
            tid = track.id
            db_tid = f"trk_{tid}"
            active_ids.add(tid)

            if tid not in self._known_tracks:
                # New track
                self._db.initiate_track(
                    db_tid,
                    track.state,
                    track.covariance,
                    timestamp,
                )
                self._known_tracks[tid] = db_tid
            else:
                # Existing track: update state
                self._db.update_track_state(
                    db_tid,
                    track.state,
                    track.covariance,
                    timestamp,
                )

            # Status management
            if track.hits >= self._confirm_hits:
                self._db.confirm_track(db_tid)
            if track.misses >= self._max_misses:
                self._db.mark_track_dead(db_tid)

        return tracks

    def get_track_list(
        self,
        status: Optional[TrackDatabaseStatus] = None,
    ) -> Any:
        """Get current tracks as a pytcl TrackList.

        Parameters
        ----------
        status : TrackDatabaseStatus, optional
            Filter by status.

        Returns
        -------
        TrackList
            Container of pytcl Track objects.
        """
        return self._db.tracks_to_tracklist(status)


class IMMTrackAdapter:
    """Adapter connecting IMM estimator to SQL storage.

    Persists the combined IMM state after each predict/update and stores
    mode probabilities in track metadata.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Unique track identifier.
    imm : object
        An IMMEstimator instance.
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        track_id: str,
        imm: Any,
    ) -> None:
        self._db = db
        self._track_id = track_id
        self._imm = imm
        self._initialized = False

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        timestamp: float = 0.0,
        mode_probs: Optional[ArrayLike] = None,
    ) -> None:
        """Initialize the IMM and store initial state."""
        x = np.asarray(x0, dtype=np.float64)
        P = np.asarray(P0, dtype=np.float64)
        self._imm.initialize(x, P, mode_probs)
        self._db.initiate_track(self._track_id, x, P, timestamp)
        self._initialized = True

    def predict(self, timestamp: float) -> None:
        """IMM prediction step."""
        if not self._initialized:
            raise RuntimeError("Track not initialized.")

        self._imm.predict()
        x, P = self._imm.x, self._imm.P
        self._db.update_track_state(
            self._track_id,
            x,
            P,
            timestamp,
            update_type="prediction",
        )

    def update(self, measurement: ArrayLike, timestamp: float) -> None:
        """IMM update step."""
        if not self._initialized:
            raise RuntimeError("Track not initialized.")

        z = np.asarray(measurement, dtype=np.float64)
        self._imm.update(z)
        x, P = self._imm.x, self._imm.P
        self._db.update_track_state(
            self._track_id,
            x,
            P,
            timestamp,
            update_type="update",
        )

    @property
    def mode_probs(self) -> NDArray[np.float64]:
        """Current mode probabilities."""
        return np.asarray(self._imm.mode_probs)


class ParticleFilterTrackAdapter:
    """Adapter connecting particle filter to SQL/HDF5 storage.

    Stores the weighted mean and covariance from the particle cloud
    as the track state after each step.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Unique track identifier.
    f : callable
        State transition function.
    h : callable
        Measurement function.
    Q : NDArray
        Process noise covariance.
    R : NDArray
        Measurement noise covariance.
    n_particles : int
        Number of particles.
    """

    def __init__(
        self,
        db: TrackDatabaseManager,
        track_id: str,
        f: Callable[..., Any],
        h: Callable[..., Any],
        Q: ArrayLike,
        R: ArrayLike,
        n_particles: int = 200,
    ) -> None:
        self._db = db
        self._track_id = track_id
        self._f = f
        self._h = h
        self._Q = np.asarray(Q, dtype=np.float64)
        self._R = np.asarray(R, dtype=np.float64)
        self._n_particles = n_particles
        self._particles: Optional[NDArray[np.float64]] = None
        self._weights: Optional[NDArray[np.float64]] = None

    def initialize(
        self,
        x0: ArrayLike,
        P0: ArrayLike,
        timestamp: float = 0.0,
    ) -> None:
        """Initialize particles around an initial state."""
        from pytcl.dynamic_estimation.particle_filters import initialize_particles

        x = np.asarray(x0, dtype=np.float64)
        P = np.asarray(P0, dtype=np.float64)
        ps = initialize_particles(x, P, self._n_particles)
        self._particles = ps.particles
        self._weights = ps.weights
        self._db.initiate_track(self._track_id, x, P, timestamp)

    def predict_update(self, measurement: ArrayLike, timestamp: float) -> None:
        """Particle filter step: predict, update, store."""
        from pytcl.dynamic_estimation.particle_filters import (
            bootstrap_pf_predict,
            bootstrap_pf_update,
            gaussian_likelihood,
            particle_covariance,
            particle_mean,
            resample_systematic,
        )

        if self._particles is None or self._weights is None:
            raise RuntimeError("Track not initialized.")

        z = np.asarray(measurement, dtype=np.float64)
        Q_mat = self._Q
        R_mat = self._R
        h_func = self._h

        # Q_sample: draws process noise for N particles
        def q_sample(
            n: int,
            rng: Optional[np.random.Generator] = None,
        ) -> NDArray[np.float64]:
            if rng is None:
                rng = np.random.default_rng()
            return rng.multivariate_normal(np.zeros(Q_mat.shape[0]), Q_mat, n)

        # likelihood_func: evaluates p(z | x) for a single particle
        def lik_func(z_obs: NDArray[np.float64], x: NDArray[np.float64]) -> float:
            z_pred = h_func(x)
            return float(gaussian_likelihood(z_obs, z_pred, R_mat))

        # Predict
        self._particles = bootstrap_pf_predict(
            self._particles,
            self._f,
            q_sample,
        )

        # Update weights
        self._weights, _ = bootstrap_pf_update(
            self._particles,
            self._weights,
            z,
            lik_func,
        )

        # Resample
        self._particles = resample_systematic(self._particles, self._weights)
        self._weights = np.ones(self._n_particles) / self._n_particles

        # Extract state and covariance for storage
        x_est = particle_mean(self._particles, self._weights)
        P_est = particle_covariance(self._particles, self._weights, x_est)

        self._db.update_track_state(
            self._track_id,
            x_est,
            P_est,
            timestamp,
            update_type="update",
        )

    @property
    def particles(self) -> Optional[NDArray[np.float64]]:
        """Current particle cloud."""
        return self._particles

    @property
    def weights(self) -> Optional[NDArray[np.float64]]:
        """Current particle weights."""
        return self._weights


def store_filter_result(
    db: TrackDatabaseManager,
    track_id: str,
    result: Any,
    timestamp: float,
    update_type: str = "update",
) -> None:
    """Store any filter result (KalmanUpdate, IMMUpdate, etc.) in the database.

    Extracts `x` and `P` attributes from the result and stores them.
    Works with KalmanPrediction, KalmanUpdate, IMMUpdate, SRKalmanUpdate, etc.

    Parameters
    ----------
    db : TrackDatabaseManager
        Open database connection.
    track_id : str
        Track identifier.
    result : object
        Filter result with `.x` and `.P` attributes. For square-root
        results with `.S`, computes P = S @ S.T.
    timestamp : float
        Time of the result.
    update_type : str
        'prediction', 'update', or 'smoothed'.
    """
    x = np.asarray(result.x, dtype=np.float64)

    if hasattr(result, "P"):
        P = np.asarray(result.P, dtype=np.float64)
    elif hasattr(result, "S"):
        S = np.asarray(result.S, dtype=np.float64)
        P = S @ S.T
    else:
        raise ValueError(
            "Filter result must have .P (covariance) or .S (Cholesky factor)"
        )

    residual = None
    if hasattr(result, "y") and result.y is not None:
        residual = np.asarray(result.y, dtype=np.float64)

    db.update_track_state(
        track_id,
        x,
        P,
        timestamp,
        residual=residual,
        update_type=update_type,
    )
