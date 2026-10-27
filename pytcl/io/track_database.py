"""SQL-backed track lifecycle database manager.

Provides detection management, track initiation, state maintenance,
and lifecycle operations using SQLite for real-time tracking scenarios.
"""

import hashlib
import json
import math
import sqlite3
import struct
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.exceptions import FormatError
from pytcl.io.serialize import _reject_unknown_schema_version
from pytcl.io.sql_storage import _connect, _truncate_sqlite

# Struct formats for the array shape header, indexed by ndim. numpy 2 allows
# up to 64 dimensions; a table that stops short makes such arrays store fine
# and then fail on read.
_SHAPE_FORMATS = tuple(f"<{n}Q" for n in range(65))


class TrackDatabaseStatus(Enum):
    """Track lifecycle status for database persistence.

    Extends the runtime TrackStatus with COASTING and DEAD states
    for full lifecycle management.
    """

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DEAD = "dead"


_TRACK_STATUS_TO_DB = {
    "tentative": TrackDatabaseStatus.TENTATIVE,
    "confirmed": TrackDatabaseStatus.CONFIRMED,
    "deleted": TrackDatabaseStatus.DEAD,
}


def _track_row_id(track_id: str) -> int:
    """Return a process-independent integer id in ``[0, 2**31)`` for a track id.

    ``hash()`` of a str is salted per process, so the same track id mapped to
    a different integer in every interpreter.
    """
    digest = hashlib.blake2b(track_id.encode(), digest_size=4).digest()
    return int.from_bytes(digest, "big") % 2**31


_CURRENT_SCHEMA_VERSION = 2
_CURRENT_SCHEMA_VERSION_STR = "2.0"


class TrackDatabaseManager:
    """SQL-backed track lifecycle database manager.

    Provides detection management, track initiation, state maintenance,
    and lifecycle operations using SQLite.

    Parameters
    ----------
    path : str
        Path to SQLite database file.

    Examples
    --------
    >>> import os
    >>> import tempfile
    >>> from pytcl.io import TrackDatabaseManager
    >>> tmpdir = tempfile.TemporaryDirectory()
    >>> path = os.path.join(tmpdir.name, "tracking.db")
    >>> with TrackDatabaseManager(path) as db:
    ...     db.open(mode="w")
    ...     db.store_detection("det_001", np.array([1.0, 2.0]), "radar", 0.0)
    ...     db.initiate_track("trk_001", np.array([1, 2, 0, 0]),
    ...                       np.eye(4), 0.0)
    ...     db.update_track_state("trk_001", np.array([1.1, 2.1, 0, 0]),
    ...                           np.eye(4) * 0.9, 1.0)
    >>> tmpdir.cleanup()
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._mode: Optional[str] = None
        self._schema_version = _CURRENT_SCHEMA_VERSION

    def open(self, mode: str = "a") -> None:
        """Open database connection.

        Parameters
        ----------
        mode : str
            'r' (read-only), 'w' (create), 'a' (create if absent, else
            append). Default is 'a'. 'w' creates the file, discarding any
            existing file and its SQLite journal files (matching
            ``HDF5Storage``).

        Raises
        ------
        ValueError
            If ``mode`` is not one of 'r', 'w', 'a'.
        FileNotFoundError
            If ``mode='r'`` and the database does not exist.
        FormatError
            If the database records a schema version this pytcl does not
            read. The connection is closed and the file left unmodified.

        Notes
        -----
        Read mode is enforced: a write through a handle opened with ``'r'``
        raises :class:`PermissionError` (it used to persist).

        ``'w'`` unlinks the path before connecting. On Windows that raises
        :class:`PermissionError` while another process holds the file open.
        A symlinked path is replaced by a regular file; the symlink's target
        is left intact. Calling ``open('w')`` again without :meth:`close`
        leaks the old connection, and on Windows the unlink then raises
        :class:`PermissionError` because that connection still holds the
        file.

        Read mode does not create the database. It used to: ``sqlite3.connect``
        was called unconditionally, so opening a mistyped path for reading
        produced an empty file and the first query then failed with
        ``no such table: detections`` -- reporting a missing table rather than
        the missing database the caller actually had. This is the same defect
        gh-21 fixed for :class:`~pytcl.io.sql_storage.SQLStorage`, which was
        not applied here at the time.
        """
        if mode not in ("r", "w", "a"):
            raise ValueError(f"mode must be 'r', 'w' or 'a', got {mode!r}")

        if mode == "r" and not Path(self._path).exists():
            raise FileNotFoundError(
                f"No such database: {self._path}. Opening for reading does "
                f"not create one; use mode='w' or mode='a' to create it."
            )

        if mode == "w":
            _truncate_sqlite(Path(self._path))

        self._mode = mode
        self._connection, self._cursor = _connect(str(self._path), mode)

        try:
            if mode in ("w", "a"):
                self._initialize_tables()
            else:
                self._schema_version = self._read_schema_version()
        except FormatError:
            self._connection.close()
            self._connection = None
            self._cursor = None
            raise

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._cursor = None

    def __enter__(self) -> "TrackDatabaseManager":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    # =========================================================================
    # Detection Management
    # =========================================================================

    def store_detection(
        self,
        detection_id: str,
        measurement: ArrayLike,
        sensor_id: str,
        timestamp: float,
        covariance: Optional[ArrayLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a raw detection/measurement.

        Parameters
        ----------
        detection_id : str
            Unique detection identifier.
        measurement : ArrayLike
            Measurement vector. An (n, 1) column is stored, and read back, as
            (n,); on a schema 2.0 file any other shape is kept.
        sensor_id : str
            Sensor source identifier.
        timestamp : float
            Time of detection.
        covariance : ArrayLike, optional
            Measurement covariance matrix.
        metadata : dict, optional
            Additional metadata.
        """
        self._check_writable()
        meas = np.asarray(measurement, dtype=np.float64)
        if meas.ndim > 1 and meas.size == meas.shape[0]:
            # An (n, 1) column read back as (n,) before schema 2.0; keep that.
            meas = meas.reshape(meas.shape[0])
        legacy = self._schema_version < 2
        if legacy:
            self._check_legacy_shapes(meas, covariance)
            meas_bytes, meas_dim, meas_dtype = meas.tobytes(), len(meas), "float64"
        else:
            meas_bytes, meas_dim, meas_dtype = self._serialize_array(meas)

        cov_bytes = None
        if covariance is not None:
            cov = np.asarray(covariance, dtype=np.float64)
            cov_bytes = cov.tobytes() if legacy else self._serialize_array(cov)[0]

        meta_json = self._sanitize_metadata(metadata)

        self._cursor.execute(
            """INSERT INTO detections
               (detection_id, timestamp, sensor_id, measurement,
                measurement_dim, measurement_dtype, covariance, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                detection_id,
                timestamp,
                sensor_id,
                meas_bytes,
                meas_dim,
                meas_dtype,
                cov_bytes,
                meta_json,
            ),
        )
        self._connection.commit()

    def retrieve_detections(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        sensor_id: Optional[str] = None,
        association_status: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Query detections by time range, sensor, or association status.

        Parameters
        ----------
        start_time : float, optional
            Minimum timestamp (inclusive).
        end_time : float, optional
            Maximum timestamp (inclusive).
        sensor_id : str, optional
            Filter by sensor.
        association_status : str, optional
            Filter by status ('unassociated', 'associated', 'clutter').
        limit : int, optional
            Maximum number of results. A negative value means no limit.

        Returns
        -------
        list of dict
            Detection records with keys: detection_id, timestamp,
            sensor_id, measurement, covariance, association_status,
            associated_track_id, association_confidence, metadata --
            everything ``_row_to_detection`` builds, not the subset
            previously listed here.
        """
        self._check_open()
        conditions: List[str] = []
        params: List[Any] = []

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)
        if sensor_id is not None:
            conditions.append("sensor_id = ?")
            params.append(sensor_id)
        if association_status is not None:
            conditions.append("association_status = ?")
            params.append(association_status)

        query = "SELECT * FROM detections"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY timestamp"
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
                raise ValueError(f"limit must be an int, got {limit!r}")
            query += " LIMIT ?"
            params.append(max(int(limit), -1))

        self._cursor.execute(query, params)
        return [self._row_to_detection(row) for row in self._cursor.fetchall()]

    def retrieve_detection(self, detection_id: str) -> Dict[str, Any]:
        """Retrieve a single detection by ID.

        Parameters
        ----------
        detection_id : str
            Detection identifier.

        Returns
        -------
        dict
            Detection record.

        Raises
        ------
        KeyError
            If detection not found.
        """
        self._check_open()
        self._cursor.execute(
            "SELECT * FROM detections WHERE detection_id = ?", (detection_id,)
        )
        row = self._cursor.fetchone()
        if row is None:
            raise KeyError(f"Detection '{detection_id}' not found")
        return self._row_to_detection(row)

    def retrieve_all_detections(self) -> List[Dict[str, Any]]:
        """Retrieve all detections.

        Returns
        -------
        list of dict
            All detection records.
        """
        return self.retrieve_detections()

    def associate_detection(
        self,
        detection_id: str,
        track_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Link a detection to a track.

        Parameters
        ----------
        detection_id : str
            Detection identifier.
        track_id : str
            Track identifier.
        confidence : float
            Association confidence score (0-1). Default is 1.0.

        Raises
        ------
        KeyError
            If the detection or the track is not in this database. An unknown
            track used to be written into ``track_associations`` anyway,
            leaving a reference to a track that does not exist.
        """
        self._check_writable()

        self._cursor.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,))
        if self._cursor.fetchone() is None:
            raise KeyError(f"no such track: {track_id!r}")

        # Get detection timestamp
        self._cursor.execute(
            "SELECT timestamp FROM detections WHERE detection_id = ?",
            (detection_id,),
        )
        row = self._cursor.fetchone()
        if row is None:
            raise KeyError(f"Detection '{detection_id}' not found")
        timestamp = row[0]

        # Update detection status
        self._cursor.execute(
            """UPDATE detections SET association_status = 'associated',
               associated_track_id = ?, association_confidence = ?
               WHERE detection_id = ?""",
            (track_id, confidence, detection_id),
        )

        # Record in associations table
        self._cursor.execute(
            """INSERT OR REPLACE INTO track_associations
               (track_id, detection_id, timestamp, confidence)
               VALUES (?, ?, ?, ?)""",
            (track_id, detection_id, timestamp, confidence),
        )
        self._connection.commit()

    # =========================================================================
    # Track Initiation
    # =========================================================================

    def initiate_track(
        self,
        track_id: str,
        initial_state: ArrayLike,
        initial_covariance: ArrayLike,
        timestamp: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a new tentative track with initial state estimate.

        Parameters
        ----------
        track_id : str
            Unique track identifier.
        initial_state : ArrayLike
            Initial state vector.
        initial_covariance : ArrayLike
            Initial covariance matrix.
        timestamp : float
            Track birth time.
        metadata : dict, optional
            Additional track metadata.
        """
        self._check_writable()
        state = np.asarray(initial_state, dtype=np.float64)
        cov = np.asarray(initial_covariance, dtype=np.float64)
        state = self._validate_state_shapes(state, cov)
        meta_json = self._sanitize_metadata(metadata)

        self._cursor.execute(
            """INSERT INTO tracks
               (track_id, status, birth_time, last_update_time,
                state_dim, hits, misses, total_misses,
                confidence_score, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                TrackDatabaseStatus.TENTATIVE.value,
                timestamp,
                timestamp,
                len(state),
                0,
                0,
                0,
                0.0,
                meta_json,
            ),
        )

        # Store initial state
        self._store_state_row(track_id, state, cov, timestamp, update_type="update")
        self._connection.commit()

    def get_initiation_queue(
        self,
        max_age: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Get unassociated detections awaiting track initiation.

        Parameters
        ----------
        max_age : float, optional
            Maximum age in seconds. If provided, only returns detections
            newer than (max_timestamp - max_age).

        Returns
        -------
        list of dict
            Unassociated detection records.
        """
        self._check_open()

        if max_age is not None:
            self._cursor.execute("SELECT MAX(timestamp) FROM detections")
            max_ts = self._cursor.fetchone()[0]
            if max_ts is None:
                return []
            cutoff = max_ts - max_age
            return self.retrieve_detections(
                start_time=cutoff, association_status="unassociated"
            )

        return self.retrieve_detections(association_status="unassociated")

    def confirm_track(self, track_id: str) -> None:
        """Promote a tentative track to confirmed status.

        Parameters
        ----------
        track_id : str
            Track identifier.
        """
        self._set_track_status(track_id, TrackDatabaseStatus.CONFIRMED)

    # =========================================================================
    # Track State Maintenance
    # =========================================================================

    def update_track_state(
        self,
        track_id: str,
        state: ArrayLike,
        covariance: ArrayLike,
        timestamp: float,
        residual: Optional[ArrayLike] = None,
        update_type: str = "update",
    ) -> None:
        """Record a filter state update.

        Parameters
        ----------
        track_id : str
            Track identifier.
        state : ArrayLike
            Updated state vector.
        covariance : ArrayLike
            Updated covariance matrix.
        timestamp : float
            Time of update.
        residual : ArrayLike, optional
            Innovation/residual vector.
        update_type : str
            Type of update: 'prediction', 'update', or 'smoothed'.
            Default is 'update'.

        Raises
        ------
        KeyError
            If ``track_id`` is not a track in this database.

        Notes
        -----
        An unknown ``track_id`` used to insert the state row anyway and then
        update zero rows in the ``tracks`` table, leaving history that belongs
        to no track (gh-21). Nothing surfaced: no error, and the state was
        retrievable by id, so a typo produced a track that existed in every
        respect except the one that counts.
        """
        self._check_writable()
        s = np.asarray(state, dtype=np.float64)
        c = np.asarray(covariance, dtype=np.float64)

        self._cursor.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,))
        if self._cursor.fetchone() is None:
            raise KeyError(
                f"No track {track_id!r} in this database. Create it with "
                f"initiate_track() before recording state updates."
            )

        self._store_state_row(track_id, s, c, timestamp, residual, update_type)

        # Update track metadata
        if update_type == "update":
            self._cursor.execute(
                """UPDATE tracks SET last_update_time = ?,
                   hits = hits + 1, misses = 0
                   WHERE track_id = ?""",
                (timestamp, track_id),
            )
        elif update_type == "prediction":
            self._cursor.execute(
                """UPDATE tracks SET last_update_time = ?,
                   misses = misses + 1, total_misses = total_misses + 1
                   WHERE track_id = ?""",
                (timestamp, track_id),
            )
        else:
            self._cursor.execute(
                """UPDATE tracks SET last_update_time = ?
                   WHERE track_id = ?""",
                (timestamp, track_id),
            )
        self._connection.commit()

    def store_track_history(
        self,
        track_id: str,
        states: ArrayLike,
        covariances: ArrayLike,
        timestamps: ArrayLike,
        residuals: Optional[ArrayLike] = None,
    ) -> None:
        """Batch-store an entire state timeline for a track.

        Parameters
        ----------
        track_id : str
            Track identifier.
        states : ArrayLike
            State history, shape (N, state_dim).
        covariances : ArrayLike
            Covariance history, shape (N, state_dim, state_dim).
        timestamps : ArrayLike
            Timestamps, shape (N,).
        residuals : ArrayLike, optional
            Residual history, shape (N, meas_dim).

        Raises
        ------
        ValueError
            If the inputs differ in length, or any row is rejected. Nothing
            is stored in either case.
        """
        self._check_writable()
        s_arr = np.asarray(states, dtype=np.float64)
        c_arr = np.asarray(covariances, dtype=np.float64)
        t_arr = np.asarray(timestamps, dtype=np.float64)
        r_arr = None if residuals is None else np.asarray(residuals, dtype=np.float64)

        lengths = {
            "states": len(s_arr),
            "covariances": len(c_arr),
            "timestamps": len(t_arr),
        }
        if r_arr is not None:
            lengths["residuals"] = len(r_arr)
        if len(set(lengths.values())) != 1:
            detail = ", ".join(f"{k}={v}" for k, v in lengths.items())
            raise ValueError(
                f"states, covariances, timestamps and residuals must have the "
                f"same length, got {detail}"
            )

        self._cursor.execute("SAVEPOINT store_track_history")
        try:
            for i in range(len(t_arr)):
                self._store_state_row(
                    track_id,
                    s_arr[i],
                    c_arr[i],
                    float(t_arr[i]),
                    residual=None if r_arr is None else r_arr[i],
                    update_type="update",
                )
        except BaseException:
            self._cursor.execute("ROLLBACK TO store_track_history")
            self._cursor.execute("RELEASE store_track_history")
            raise
        self._cursor.execute("RELEASE store_track_history")
        self._connection.commit()

    def get_track_state(self, track_id: str) -> Dict[str, Any]:
        """Get the most recent state estimate for a track.

        Parameters
        ----------
        track_id : str
            Track identifier.

        Returns
        -------
        dict
            Keys: track_id, state, covariance, timestamp, status, hits, misses.
        """
        self._check_open()

        # Get latest state
        self._cursor.execute(
            """SELECT state, covariance, state_dim, timestamp, residual
               FROM track_states WHERE track_id = ?
               ORDER BY timestamp DESC LIMIT 1""",
            (track_id,),
        )
        state_row = self._cursor.fetchone()
        if state_row is None:
            raise KeyError(f"No state found for track '{track_id}'")

        # Get track metadata
        track_info = self.get_track(track_id)

        state = self._deserialize_vector(state_row[0], state_row[2])
        cov = self._deserialize_matrix(state_row[1], state_row[2])

        return {
            "track_id": track_id,
            "state": state,
            "covariance": cov,
            "timestamp": state_row[3],
            "status": TrackDatabaseStatus(track_info["status"]),
            "hits": track_info["hits"],
            "misses": track_info["misses"],
        }

    def get_track_history(
        self,
        track_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Get state history for a track within a time range.

        Parameters
        ----------
        track_id : str
            Track identifier.
        start_time : float, optional
            Minimum timestamp (inclusive).
        end_time : float, optional
            Maximum timestamp (inclusive).

        Returns
        -------
        dict
            Keys: states (N, state_dim), covariances (N, state_dim, state_dim),
            timestamps (N,), residuals (N, meas_dim) or None.

            ``residuals`` is row-aligned with ``timestamps``. Rows that carry
            no residual -- predictions and initiations -- hold ``NaN``, so
            ``np.isnan(residuals).any(axis=1)`` identifies them. It is ``None``
            only when no row in the range has a residual at all.

        Raises
        ------
        KeyError
            If the track has no history in the given range.

        Notes
        -----
        Residuals used to be keyed off the first row. A window beginning with a
        prediction reported ``residuals=None`` even when later rows had them,
        and a window mixing the two returned an array *shorter* than
        ``timestamps`` with no indication of which rows it belonged to --
        breaking the documented ``(N, meas_dim)`` shape and silently
        misaligning every residual with the wrong timestamp (gh-21).

        That is the shape a predict-then-update filter produces on every step,
        which is what ``KalmanTrackAdapter`` does, so it was the normal case
        rather than an edge one.
        """
        self._check_open()
        conditions = ["track_id = ?"]
        params: List[Any] = [track_id]

        if start_time is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time)
        if end_time is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time)

        query = (
            "SELECT state, covariance, state_dim, timestamp, residual "
            "FROM track_states WHERE "
            + " AND ".join(conditions)
            + " ORDER BY timestamp"
        )
        self._cursor.execute(query, params)
        rows = self._cursor.fetchall()

        if not rows:
            raise KeyError(f"No history found for track '{track_id}'")

        state_dim = rows[0][2]
        states = np.array([self._deserialize_vector(r[0], state_dim) for r in rows])
        covariances = np.array(
            [self._deserialize_matrix(r[1], state_dim) for r in rows]
        )
        timestamps = np.array([r[3] for r in rows])

        # Scan every row, not just the first: a window can begin with a
        # prediction and still contain residuals further in.
        stored = [
            np.frombuffer(r[4], dtype=np.float64) if r[4] is not None else None
            for r in rows
        ]
        present = [res for res in stored if res is not None]

        residuals = None
        if present:
            # One row per state, NaN where there is no residual, so the result
            # stays aligned with `timestamps` and keeps the documented shape.
            meas_dim = len(present[0])
            residuals = np.full((len(rows), meas_dim), np.nan)
            for index, res in enumerate(stored):
                if res is not None:
                    residuals[index] = res

        return {
            "states": states,
            "covariances": covariances,
            "timestamps": timestamps,
            "residuals": residuals,
        }

    def get_track(self, track_id: str) -> Dict[str, Any]:
        """Get full track metadata.

        Parameters
        ----------
        track_id : str
            Track identifier.

        Returns
        -------
        dict
            Track metadata with keys: track_id, status, birth_time,
            last_update_time, state_dim, hits, misses, total_misses,
            confidence_score, metadata.
        """
        self._check_open()
        self._cursor.execute("SELECT * FROM tracks WHERE track_id = ?", (track_id,))
        row = self._cursor.fetchone()
        if row is None:
            raise KeyError(f"Track '{track_id}' not found")
        return self._row_to_track(row)

    def retrieve_all_tracks(
        self,
        status: Optional[TrackDatabaseStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve all tracks, optionally filtered by status.

        Parameters
        ----------
        status : TrackDatabaseStatus, optional
            Filter by status.

        Returns
        -------
        list of dict
            Track metadata records.
        """
        self._check_open()
        if status is not None:
            self._cursor.execute(
                "SELECT * FROM tracks WHERE status = ?", (status.value,)
            )
        else:
            self._cursor.execute("SELECT * FROM tracks")
        return [self._row_to_track(row) for row in self._cursor.fetchall()]

    # =========================================================================
    # Track Lifecycle Management
    # =========================================================================

    def mark_track_tentative(self, track_id: str) -> None:
        """Set track status to TENTATIVE."""
        self._set_track_status(track_id, TrackDatabaseStatus.TENTATIVE)

    def mark_track_confirmed(self, track_id: str) -> None:
        """Set track status to CONFIRMED."""
        self._set_track_status(track_id, TrackDatabaseStatus.CONFIRMED)

    def mark_track_coasting(self, track_id: str) -> None:
        """Set track status to COASTING."""
        self._set_track_status(track_id, TrackDatabaseStatus.COASTING)

    def mark_track_dead(self, track_id: str) -> None:
        """Set track status to DEAD."""
        self._set_track_status(track_id, TrackDatabaseStatus.DEAD)

    def prune_old_detections(self, age_threshold: float) -> int:
        """Remove unassociated detections older than threshold.

        Parameters
        ----------
        age_threshold : float
            Maximum age in seconds relative to newest detection.

        Returns
        -------
        int
            Number of detections pruned.
        """
        self._check_writable()
        self._cursor.execute("SELECT MAX(timestamp) FROM detections")
        max_ts = self._cursor.fetchone()[0]
        if max_ts is None:
            return 0

        cutoff = max_ts - age_threshold
        self._cursor.execute(
            """DELETE FROM detections
               WHERE association_status = 'unassociated'
               AND timestamp < ?""",
            (cutoff,),
        )
        count = self._cursor.rowcount
        self._connection.commit()
        return count

    def prune_dead_tracks(self, age_threshold: float) -> int:
        """Remove tracks that have been DEAD for longer than threshold.

        Removes the track record, its state history, and associations.

        Parameters
        ----------
        age_threshold : float
            Maximum age in seconds relative to newest track update.

        Returns
        -------
        int
            Number of tracks pruned.
        """
        self._check_writable()
        self._cursor.execute("SELECT MAX(last_update_time) FROM tracks")
        max_ts = self._cursor.fetchone()[0]
        if max_ts is None:
            return 0

        cutoff = max_ts - age_threshold

        # Find dead tracks to prune
        self._cursor.execute(
            """SELECT track_id FROM tracks
               WHERE status = ? AND last_update_time < ?""",
            (TrackDatabaseStatus.DEAD.value, cutoff),
        )
        track_ids = [row[0] for row in self._cursor.fetchall()]

        if not track_ids:
            return 0

        placeholders = ",".join("?" for _ in track_ids)

        # Delete state history
        self._cursor.execute(
            f"DELETE FROM track_states WHERE track_id IN ({placeholders})",
            track_ids,
        )
        # Delete associations
        self._cursor.execute(
            f"DELETE FROM track_associations WHERE track_id IN ({placeholders})",
            track_ids,
        )
        # Delete tracks
        self._cursor.execute(
            f"DELETE FROM tracks WHERE track_id IN ({placeholders})",
            track_ids,
        )
        self._connection.commit()
        return len(track_ids)

    def merge_tracks(
        self,
        track_id_keep: str,
        track_id_merge: str,
    ) -> None:
        """Merge one track into another.

        Combines state histories, re-associates detections from the merged
        track to the kept track, and marks the merged track DEAD.

        Parameters
        ----------
        track_id_keep : str
            Track to keep.
        track_id_merge : str
            Track to merge in and mark dead.

        Raises
        ------
        KeyError
            If either track is absent from the database.
        ValueError
            If both ids name the same track. A self-merge doubled ``hits``
            and then marked the surviving track DEAD.

        Notes
        -----
        Track-level fields are combined as well as the history: ``birth_time``
        becomes the earlier of the two, ``last_update_time`` the later, and the
        merged track's metadata keys are folded in without overwriting keys the
        kept track already has.

        None of that used to happen (gh-21). History, associations and
        detections were re-assigned and the hit counters summed, but the kept
        track's ``last_update_time`` was left behind -- so a merge that brought
        in newer states made the track look stale, and any staleness-based
        pruning would then delete the track that had just been reinforced.
        """
        self._check_writable()

        if track_id_keep == track_id_merge:
            raise ValueError(f"cannot merge track {track_id_keep!r} into itself")

        for track_id in (track_id_keep, track_id_merge):
            self._cursor.execute("SELECT 1 FROM tracks WHERE track_id = ?", (track_id,))
            if self._cursor.fetchone() is None:
                raise KeyError(f"No track {track_id!r} in this database")

        # Re-assign state history
        self._cursor.execute(
            """UPDATE track_states SET track_id = ?
               WHERE track_id = ?""",
            (track_id_keep, track_id_merge),
        )

        # Re-assign associations
        self._cursor.execute(
            """UPDATE track_associations SET track_id = ?
               WHERE track_id = ?""",
            (track_id_keep, track_id_merge),
        )

        # Re-assign detections
        self._cursor.execute(
            """UPDATE detections SET associated_track_id = ?
               WHERE associated_track_id = ?""",
            (track_id_keep, track_id_merge),
        )

        # Fold the merged track's counters, lifetime and metadata into the
        # kept one. The lifetime is the union of the two, so birth_time takes
        # the earlier and last_update_time the later.
        self._cursor.execute(
            "SELECT hits, total_misses, birth_time, last_update_time, metadata "
            "FROM tracks WHERE track_id = ?",
            (track_id_merge,),
        )
        merged_hits, merged_misses, merged_birth, merged_last, merged_meta = (
            self._cursor.fetchone()
        )

        self._cursor.execute(
            "SELECT metadata FROM tracks WHERE track_id = ?", (track_id_keep,)
        )
        kept_meta = self._cursor.fetchone()[0]

        # The kept track's own keys win: merging is additive, not a takeover.
        combined = {**json.loads(merged_meta or "{}"), **json.loads(kept_meta or "{}")}

        self._cursor.execute(
            """UPDATE tracks SET
               hits = hits + ?,
               total_misses = total_misses + ?,
               birth_time = MIN(birth_time, ?),
               last_update_time = MAX(last_update_time, ?),
               metadata = ?
               WHERE track_id = ?""",
            (
                merged_hits,
                merged_misses,
                merged_birth,
                merged_last,
                json.dumps(combined),
                track_id_keep,
            ),
        )

        # Mark merged track dead
        self._set_track_status(track_id_merge, TrackDatabaseStatus.DEAD)
        self._connection.commit()

    # =========================================================================
    # Conversion Helpers
    # =========================================================================

    def track_to_pytcl(self, track_id: str) -> Any:
        """Convert stored track to pytcl Track NamedTuple.

        Parameters
        ----------
        track_id : str
            Track identifier.

        Returns
        -------
        Track
            pytcl Track NamedTuple.
        """
        from pytcl.trackers.multi_target import Track, TrackStatus

        track_info = self.get_track(track_id)
        state_info = self.get_track_state(track_id)

        # Map database status to TrackStatus
        status_map = {
            "tentative": TrackStatus.TENTATIVE,
            "confirmed": TrackStatus.CONFIRMED,
            "coasting": TrackStatus.CONFIRMED,
            "dead": TrackStatus.DELETED,
        }
        status = status_map.get(track_info["status"], TrackStatus.DELETED)

        return Track(
            id=_track_row_id(track_id),
            state=state_info["state"],
            covariance=state_info["covariance"],
            status=status,
            hits=track_info["hits"],
            misses=track_info["misses"],
            time=state_info["timestamp"],
        )

    def tracks_to_tracklist(
        self,
        status: Optional[TrackDatabaseStatus] = None,
    ) -> Any:
        """Convert stored tracks to pytcl TrackList.

        Parameters
        ----------
        status : TrackDatabaseStatus, optional
            Filter by status.

        Returns
        -------
        TrackList
            pytcl TrackList container.
        """
        from pytcl.containers.track_list import TrackList

        tracks_info = self.retrieve_all_tracks(status)
        track_objects = []
        for t in tracks_info:
            try:
                track_objects.append(self.track_to_pytcl(t["track_id"]))
            except KeyError:
                continue

        return TrackList(track_objects)

    def store_from_track(self, track: Any, timestamp: Optional[float] = None) -> None:
        """Store a pytcl Track NamedTuple into the database.

        Parameters
        ----------
        track : Track
            pytcl Track NamedTuple, or any object with ``id``, ``state`` and
            ``covariance`` (``MHTTrack``, ``SimpleTrack``, ...).
        timestamp : float, optional
            Override timestamp. Uses track.time if not provided.

        Notes
        -----
        ``status`` and the hit and miss counts (``hits``/``misses``, or
        ``n_hits``/``n_misses`` on ``MHTTrack``) are stored with the state.
        They used to be dropped, so a confirmed track with 5 hits and 1 miss
        read back tentative with 0 and 0. A field the object does not carry,
        or a status that is not tentative, confirmed or deleted (including
        ``None``), keeps the value the state update gives it, as before.
        """
        self._check_writable()
        t = timestamp if timestamp is not None else track.time
        track_id = f"trk_{track.id}"

        # Check if track exists
        self._cursor.execute(
            "SELECT track_id FROM tracks WHERE track_id = ?", (track_id,)
        )
        if self._cursor.fetchone() is None:
            self.initiate_track(track_id, track.state, track.covariance, t)
        else:
            self.update_track_state(track_id, track.state, track.covariance, t)

        status = getattr(track, "status", None)
        db_status = _TRACK_STATUS_TO_DB.get(getattr(status, "value", status))
        fields = {
            "status": None if db_status is None else db_status.value,
            "hits": getattr(track, "hits", getattr(track, "n_hits", None)),
            "misses": getattr(track, "misses", getattr(track, "n_misses", None)),
        }
        for column, value in fields.items():
            if value is not None:
                if column != "status":
                    value = int(value)
                self._cursor.execute(
                    f"UPDATE tracks SET {column} = ? WHERE track_id = ?",
                    (value, track_id),
                )
        self._connection.commit()

    def store_from_tracklist(self, track_list: Any) -> None:
        """Store all tracks from a TrackList.

        Parameters
        ----------
        track_list : TrackList
            pytcl TrackList container.
        """
        for track in track_list:
            self.store_from_track(track)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _check_open(self) -> None:
        """Verify the database is open."""
        if self._cursor is None:
            raise RuntimeError("Database not open. Call open() first.")

    def _check_writable(self) -> None:
        """Verify the database is open and not in read mode."""
        self._check_open()
        if self._mode == "r":
            raise PermissionError(
                "Database was opened read-only (mode='r'); reopen with "
                "mode='a' or mode='w' to write."
            )

    @staticmethod
    def _sanitize_metadata(metadata: Optional[Dict[str, Any]]) -> str:
        """Serialize metadata to JSON, converting numpy types to native Python."""
        meta = metadata or {}
        cleaned: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (np.integer,)):
                cleaned[k] = int(v)
            elif isinstance(v, (np.floating,)):
                cleaned[k] = float(v)
            elif isinstance(v, np.ndarray):
                cleaned[k] = v.tolist()
            else:
                cleaned[k] = v
        return json.dumps(cleaned)

    def _set_track_status(self, track_id: str, status: TrackDatabaseStatus) -> None:
        """Update a track's status."""
        self._check_writable()
        self._cursor.execute(
            "UPDATE tracks SET status = ? WHERE track_id = ?",
            (status.value, track_id),
        )
        self._connection.commit()

    def _store_state_row(
        self,
        track_id: str,
        state: NDArray[np.float64],
        covariance: NDArray[np.float64],
        timestamp: float,
        residual: Optional[ArrayLike] = None,
        update_type: str = "update",
    ) -> None:
        """Insert a single state row into track_states."""
        state = self._validate_state_shapes(state, covariance)
        res_bytes = None
        if residual is not None:
            res_bytes = np.asarray(residual, dtype=np.float64).tobytes()

        self._cursor.execute(
            """INSERT INTO track_states
               (track_id, timestamp, state, covariance, state_dim,
                residual, update_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                track_id,
                timestamp,
                state.tobytes(),
                covariance.tobytes(),
                len(state),
                res_bytes,
                update_type,
            ),
        )

    @staticmethod
    def _serialize_array(
        arr: NDArray[np.float64],
    ) -> Tuple[bytes, int, str]:
        """Serialize ndarray to (bytes, element_count, dtype_str).

        The byte string is a shape header (``uint8`` ndim, then ``uint64``
        extents, little-endian) followed by the C-order data, so the shape
        survives the round trip. The element count is ``arr.size``; the
        schema-1 layout used ``arr.shape[0]`` and truncated 2-D arrays.
        """
        arr = np.ascontiguousarray(arr)
        header = struct.pack(f"<B{arr.ndim}Q", arr.ndim, *arr.shape)
        return header + arr.tobytes(), int(arr.size), str(arr.dtype)

    @staticmethod
    def _deserialize_array(data: bytes, dtype: str) -> NDArray[Any]:
        """Inverse of :meth:`_serialize_array` (schema 2)."""
        ndim = data[0]
        if ndim >= len(_SHAPE_FORMATS):
            raise ValueError(f"corrupt array header: ndim={ndim}")
        shape = struct.unpack_from(_SHAPE_FORMATS[ndim], data, 1)
        count = math.prod(shape)
        arr = np.frombuffer(data, dtype=dtype, count=count, offset=1 + 8 * ndim)
        return (arr if ndim == 1 else arr.reshape(shape)).copy()

    @staticmethod
    def _validate_state_shapes(
        state: NDArray[np.float64], covariance: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return ``state`` as (n,); raise unless it is (n,) or (n, 1) and
        ``covariance`` is (n, n).

        Track-state rows store ``len(state)`` as the dimension and read the
        covariance back as ``dim * dim`` elements, so any other shape was
        stored under the wrong dimension and garbled on read. An (n, 1)
        column holds the same n values in the same order and always read
        back as (n,).
        """
        if state.ndim == 0 or state.size != state.shape[0]:
            raise ValueError(
                f"state must be a vector of shape (n,) or (n, 1), got shape "
                f"{state.shape}"
            )
        n = state.shape[0]
        if covariance.shape != (n, n):
            raise ValueError(
                f"covariance must have shape ({n}, {n}) for a state of length "
                f"{n}, got shape {covariance.shape}"
            )
        return state.reshape(n)

    @staticmethod
    def _check_legacy_shapes(meas: NDArray[np.float64], cov: Any) -> None:
        """Raise for data the schema 1.0 detection layout cannot represent."""
        n = meas.shape[0] if meas.ndim == 1 else None
        bad_cov = cov is not None and np.shape(cov) != (n, n)
        if n is None or bad_cov:
            raise ValueError(
                "This database is schema 1.0, whose detection layout stores a "
                "1-D measurement and an (n, n) covariance only; got measurement "
                f"shape {meas.shape} and covariance shape "
                f"{None if cov is None else np.shape(cov)}. A new database "
                "(created at schema 2.0) stores any shape."
            )

    def _read_schema_version(self) -> int:
        """Return the file's major schema version; absence means 1."""
        try:
            self._cursor.execute(
                "SELECT value FROM _pytcl_track_db_metadata "
                "WHERE key = 'schema_version'"
            )
        except sqlite3.OperationalError:
            return 1
        row = self._cursor.fetchone()
        if not row:
            return 1
        _reject_unknown_schema_version(
            f"track database {str(self._path)!r}",
            str(row[0]),
            ("1.0", _CURRENT_SCHEMA_VERSION_STR),
        )
        return int(str(row[0]).split(".")[0])

    @staticmethod
    def _deserialize_vector(data: bytes, dim: int) -> NDArray[np.float64]:
        """Deserialize bytes to 1D ndarray."""
        return np.frombuffer(data, dtype=np.float64)[:dim].copy()

    @staticmethod
    def _deserialize_matrix(data: bytes, dim: int) -> NDArray[np.float64]:
        """Deserialize bytes to 2D ndarray (dim x dim)."""
        return (
            np.frombuffer(data, dtype=np.float64)[: dim * dim].reshape(dim, dim).copy()
        )

    def _row_to_detection(self, row: tuple[Any, ...]) -> Dict[str, Any]:
        """Convert a detections table row to a dict."""
        cov = None
        if self._schema_version < 2:
            meas = np.frombuffer(row[3], dtype=np.dtype(row[5]))[: row[4]].copy()
            if row[6] is not None:
                cov_dim = row[4]
                cov = (
                    np.frombuffer(row[6], dtype=np.float64)[: cov_dim * cov_dim]
                    .reshape(cov_dim, cov_dim)
                    .copy()
                )
        else:
            meas = self._deserialize_array(row[3], row[5])
            if row[6] is not None:
                cov = self._deserialize_array(row[6], "float64")
        return {
            "detection_id": row[0],
            "timestamp": row[1],
            "sensor_id": row[2],
            "measurement": meas,
            "covariance": cov,
            "association_status": row[7],
            "associated_track_id": row[8],
            "association_confidence": row[9],
            "metadata": json.loads(row[10]) if row[10] else {},
        }

    @staticmethod
    def _row_to_track(row: tuple[Any, ...]) -> Dict[str, Any]:
        """Convert a tracks table row to a dict."""
        return {
            "track_id": row[0],
            "status": row[1],
            "birth_time": row[2],
            "last_update_time": row[3],
            "state_dim": row[4],
            "hits": row[5],
            "misses": row[6],
            "total_misses": row[7],
            "confidence_score": row[8],
            "metadata": json.loads(row[9]) if row[9] else {},
        }

    def _initialize_tables(self) -> None:
        """Create all tables and indices if they do not exist."""
        if self._cursor is None:
            return

        # Reject an unknown version before any CREATE, so the file is untouched.
        self._read_schema_version()

        self._cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='detections'"
        )
        preexisting = self._cursor.fetchone() is not None

        self._cursor.execute("""CREATE TABLE IF NOT EXISTS detections (
            detection_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            sensor_id TEXT NOT NULL,
            measurement BLOB NOT NULL,
            measurement_dim INTEGER NOT NULL,
            measurement_dtype TEXT NOT NULL,
            covariance BLOB,
            association_status TEXT DEFAULT 'unassociated',
            associated_track_id TEXT,
            association_confidence REAL,
            metadata TEXT DEFAULT '{}'
        )""")

        self._cursor.execute("""CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            birth_time REAL NOT NULL,
            last_update_time REAL NOT NULL,
            state_dim INTEGER NOT NULL,
            hits INTEGER DEFAULT 0,
            misses INTEGER DEFAULT 0,
            total_misses INTEGER DEFAULT 0,
            confidence_score REAL DEFAULT 0.0,
            metadata TEXT DEFAULT '{}'
        )""")

        self._cursor.execute("""CREATE TABLE IF NOT EXISTS track_states (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            state BLOB NOT NULL,
            covariance BLOB NOT NULL,
            state_dim INTEGER NOT NULL,
            residual BLOB,
            update_type TEXT DEFAULT 'update'
        )""")

        self._cursor.execute("""CREATE TABLE IF NOT EXISTS track_associations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id TEXT NOT NULL,
            detection_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            confidence REAL DEFAULT 1.0,
            UNIQUE(track_id, detection_id)
        )""")

        self._cursor.execute("""CREATE TABLE IF NOT EXISTS _pytcl_track_db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )""")

        # Create indices
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_det_time ON detections(timestamp)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_det_sensor ON detections(sensor_id)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_det_status "
            "ON detections(association_status)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_ts_track_time "
            "ON track_states(track_id, timestamp)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trk_status ON tracks(status)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_trk_update ON tracks(last_update_time)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_assoc_track ON track_associations(track_id)"
        )
        self._cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_assoc_det "
            "ON track_associations(detection_id)"
        )

        # A file with no version row is new (every schema-1 file wrote one);
        # an existing file keeps its version, and with it its row format, so
        # rows of two formats never share one version label.
        self._cursor.execute(
            "INSERT OR IGNORE INTO _pytcl_track_db_metadata (key, value) "
            "VALUES ('schema_version', ?)",
            ("1.0" if preexisting else _CURRENT_SCHEMA_VERSION_STR,),
        )
        self._connection.commit()
        self._schema_version = self._read_schema_version()
