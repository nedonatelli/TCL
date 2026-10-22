"""HDF5-backed storage for large-scale tracking datasets.

Optimized for archival and post-analysis of tracking scenarios with
efficient time-series access and compression. h5py is a core pytcl
dependency (see pyproject.toml); no optional install step is needed.
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.io.serialize import _warn_unknown_schema_version

_SCHEMA_VERSION = "1.0"


def _cast_checked(x: ArrayLike, dtype: np.dtype, name: str) -> NDArray[Any]:
    """Cast `x` to `dtype`, refusing silent loss of a complex part.

    Raises ``TypeError`` on complex input to a non-complex dtype and warns
    on integer input that a floating dtype may not hold exactly.
    """
    arr = np.asarray(x)
    if np.iscomplexobj(arr) and dtype.kind != "c":
        raise TypeError(
            f"{name} is complex but the storage dtype is {dtype}; casting "
            "would discard the imaginary part"
        )
    if arr.dtype.kind in "iu" and dtype.kind in "fc" and arr.size:
        limit = 2 ** (np.finfo(dtype).nmant + 1)
        if int(arr.max()) > limit or int(arr.min()) < -limit:
            warnings.warn(
                f"{name} has integer values beyond 2**{np.finfo(dtype).nmant + 1} "
                f"in magnitude; the {dtype} cast may not store them exactly",
                UserWarning,
                stacklevel=3,
            )
    return np.asarray(arr, dtype=dtype)


class TrackHDF5Storage:
    """HDF5-backed storage for large-scale tracking datasets.

    Optimized for archival and post-analysis of tracking scenarios with
    efficient time-series access and compression.

    Parameters
    ----------
    path : str
        Path to HDF5 file.
    chunk_size : int
        Chunk size for time-series datasets, in rows along the time axis.
        Datasets shorter than this are stored as a single chunk spanning
        their full history (time-aligned, not split across tracks), which
        is the common case for archival writes. Default is 1000.
    compression : str
        Compression algorithm. Default is 'gzip'.
    compression_level : int
        Compression level (1-9). Default is 4.
    dtype : str
        Default dtype for stored arrays. Default is 'float64'.
    shuffle : bool
        Enable HDF5's byte-shuffle filter before compression. Reorders each
        chunk's bytes so that same-significance bytes of adjacent float64
        values are contiguous, which measurably improves gzip's ratio on
        slowly-varying track data (measured +7% on the benchmark scenario
        in ``tests/unit/test_hdf5_compression.py`` -- see that file's
        module docstring for the reproduction command and full figures).
        Default is True.

    Examples
    --------
    >>> from pytcl.io import TrackHDF5Storage
    >>> with TrackHDF5Storage("scenario.h5") as store:  # doctest: +SKIP
    ...     store.open(mode="w")
    ...     store.store_track("trk_001", states, covariances, timestamps)
    ...     traj = store.get_track_trajectory("trk_001", start_time=0.0)
    """

    def __init__(
        self,
        path: str,
        chunk_size: int = 1000,
        compression: str = "gzip",
        compression_level: int = 4,
        dtype: str = "float64",
        shuffle: bool = True,
    ) -> None:
        self._path = path
        self._chunk_size = chunk_size
        self._compression = compression
        self._compression_level = compression_level
        self._shuffle = shuffle
        self._dtype = np.dtype(dtype)
        self._file: Optional[Any] = None
        self._mode: Optional[str] = None

    def open(self, mode: str = "r") -> None:
        """Open HDF5 file.

        Parameters
        ----------
        mode : str
            'r' (read), 'w' (write/create), 'a' (append). Default is 'r'.
        """
        self._mode = mode
        self._file = h5py.File(self._path, mode)

        if mode in ("w", "a"):
            if "_metadata" not in self._file:
                meta = self._file.create_group("_metadata")
                meta.attrs["schema_version"] = _SCHEMA_VERSION

        if "_metadata" in self._file:
            found = self._file["_metadata"].attrs.get("schema_version")
            if isinstance(found, bytes):
                found = found.decode()
            _warn_unknown_schema_version(
                f"HDF5 track file {str(self._path)!r}", found, (_SCHEMA_VERSION,)
            )

    def close(self) -> None:
        """Close HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "TrackHDF5Storage":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self.close()

    def flush(self) -> None:
        """Ensure all data is written to disk."""
        if self._file is not None:
            self._file.flush()

    # =========================================================================
    # Single Track Storage
    # =========================================================================

    def store_track(
        self,
        track_id: str,
        states: ArrayLike,
        covariances: ArrayLike,
        timestamps: ArrayLike,
        metadata: Optional[Dict[str, Any]] = None,
        residuals: Optional[ArrayLike] = None,
        scenario_id: Optional[str] = None,
    ) -> None:
        """Store a complete track trajectory.

        Parameters
        ----------
        track_id : str
            Unique track identifier.
        states : ArrayLike
            State history, shape (N, state_dim).
        covariances : ArrayLike
            Covariance history, shape (N, state_dim, state_dim).
        timestamps : ArrayLike
            Timestamps, shape (N,).
        metadata : dict, optional
            Track metadata (status, birth_time, etc.).
        residuals : ArrayLike, optional
            Innovation residuals, shape (N, meas_dim).
        scenario_id : str, optional
            Store under /scenarios/{scenario_id}/tracks/.

        Raises
        ------
        TypeError
            If any input is complex and the storage dtype is not; the cast
            would discard the imaginary part.

        Warns
        -----
        UserWarning
            If an integer input has a value beyond ``2**53`` in magnitude,
            which a float64 cast cannot represent exactly.
        """
        self._check_open()
        s = _cast_checked(states, self._dtype, "states")
        c = _cast_checked(covariances, self._dtype, "covariances")
        t = _cast_checked(timestamps, np.dtype(np.float64), "timestamps")

        group_path = self._resolve_track_group(track_id, scenario_id)
        grp = self._ensure_group(group_path)

        self._create_chunked_dataset(grp, "state_history", s)
        self._create_chunked_dataset(grp, "covariance_history", c)
        self._create_chunked_dataset(grp, "timestamps", t)

        if residuals is not None:
            r = _cast_checked(residuals, self._dtype, "residuals")
            self._create_chunked_dataset(grp, "residuals", r)

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    grp.attrs[k] = v
                else:
                    grp.attrs[k] = json.dumps(v)

        self._file.flush()

    def append_track_state(
        self,
        track_id: str,
        state: ArrayLike,
        covariance: ArrayLike,
        timestamp: float,
        residual: Optional[ArrayLike] = None,
        scenario_id: Optional[str] = None,
    ) -> None:
        """Append a single state to an existing track's history.

        Parameters
        ----------
        track_id : str
            Track identifier.
        state : ArrayLike
            State vector.
        covariance : ArrayLike
            Covariance matrix.
        timestamp : float
            Timestamp.
        residual : ArrayLike, optional
            Innovation residual. If the track has no residuals dataset yet,
            one is created with NaN rows for the earlier states; if the track
            has residuals and this is omitted, a NaN row is appended. Either
            way ``residuals`` stays row-aligned with ``timestamps``.
        scenario_id : str, optional
            Scenario identifier.

        Raises
        ------
        TypeError
            If any input is complex and the storage dtype is not, or if a
            NaN residual pad is needed (see `residual`) and the storage
            dtype is not floating or complex.
        """
        self._check_open()
        group_path = self._resolve_track_group(track_id, scenario_id)
        grp = self._file[group_path]

        rows = [
            ("state_history", _cast_checked(state, self._dtype, "state")),
            (
                "covariance_history",
                _cast_checked(covariance, self._dtype, "covariance"),
            ),
            (
                "timestamps",
                _cast_checked(timestamp, np.dtype(np.float64), "timestamp"),
            ),
        ]
        needs_nan_pad = (residual is None) == ("residuals" in grp)
        if needs_nan_pad and self._dtype.kind not in "fc":
            # Checked before any resize so the append stays atomic; numpy
            # would store the NaN pad as 0 or INT_MIN with only a warning.
            raise TypeError(
                f"residuals must be NaN-padded to stay row-aligned, but the "
                f"storage dtype {self._dtype} cannot hold NaN"
            )
        if residual is not None:
            r = _cast_checked(residual, self._dtype, "residual")
            if "residuals" not in grp:
                # Backfill the earlier rows so residuals stays row-aligned
                # with timestamps (it used to drop the residual silently).
                n_prev = grp["timestamps"].shape[0]
                backfill = np.full((n_prev,) + r.shape, np.nan, dtype=self._dtype)
                self._create_chunked_dataset(grp, "residuals", backfill)
            rows.append(("residuals", r))
        elif "residuals" in grp:
            nan_row = np.full(grp["residuals"].shape[1:], np.nan, dtype=self._dtype)
            rows.append(("residuals", nan_row))

        for name, data in rows:
            ds = grp[name]
            new_shape = list(ds.shape)
            new_shape[0] += 1
            ds.resize(new_shape)
            ds[-1] = data

    def retrieve_track(
        self,
        track_id: str,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve a complete track.

        Parameters
        ----------
        track_id : str
            Track identifier.
        scenario_id : str, optional
            Scenario identifier.

        Returns
        -------
        dict
            Keys: states, covariances, timestamps, residuals (or None),
            metadata.
        """
        self._check_open()
        group_path = self._resolve_track_group(track_id, scenario_id)

        if group_path not in self._file:
            raise KeyError(f"Track '{track_id}' not found")

        grp = self._file[group_path]

        result: Dict[str, Any] = {
            "states": np.array(grp["state_history"]),
            "covariances": np.array(grp["covariance_history"]),
            "timestamps": np.array(grp["timestamps"]),
            "residuals": None,
            "metadata": dict(grp.attrs),
        }

        if "residuals" in grp:
            result["residuals"] = np.array(grp["residuals"])

        return result

    # =========================================================================
    # Single Detection Storage
    # =========================================================================

    def store_detection(
        self,
        detection_id: str,
        measurement: ArrayLike,
        timestamp: float,
        sensor_id: str,
        covariance: Optional[ArrayLike] = None,
        metadata: Optional[Dict[str, Any]] = None,
        scenario_id: Optional[str] = None,
    ) -> None:
        """Store a single detection.

        Parameters
        ----------
        detection_id : str
            Detection identifier.
        measurement : ArrayLike
            Measurement vector.
        timestamp : float
            Time of detection.
        sensor_id : str
            Sensor source.
        covariance : ArrayLike, optional
            Measurement covariance.
        metadata : dict, optional
            Additional metadata.
        scenario_id : str, optional
            Scenario identifier.
        """
        self._check_open()
        group_path = self._resolve_detection_group(detection_id, scenario_id)
        grp = self._ensure_group(group_path)

        grp.create_dataset(
            "measurement", data=np.asarray(measurement, dtype=self._dtype)
        )

        if covariance is not None:
            grp.create_dataset(
                "covariance", data=np.asarray(covariance, dtype=self._dtype)
            )

        grp.attrs["timestamp"] = timestamp
        grp.attrs["sensor_id"] = sensor_id

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    grp.attrs[k] = v
                else:
                    grp.attrs[k] = json.dumps(v)

    def retrieve_detection(
        self,
        detection_id: str,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve a single detection.

        Parameters
        ----------
        detection_id : str
            Detection identifier.
        scenario_id : str, optional
            Scenario identifier.

        Returns
        -------
        dict
            Keys: detection_id, measurement, timestamp, sensor_id,
            covariance (or None), metadata.
        """
        self._check_open()
        group_path = self._resolve_detection_group(detection_id, scenario_id)

        if group_path not in self._file:
            raise KeyError(f"Detection '{detection_id}' not found")

        grp = self._file[group_path]

        result: Dict[str, Any] = {
            "detection_id": detection_id,
            "measurement": np.array(grp["measurement"]),
            "timestamp": float(grp.attrs["timestamp"]),
            "sensor_id": str(grp.attrs["sensor_id"]),
            "covariance": None,
            "metadata": {
                k: v
                for k, v in grp.attrs.items()
                if k not in ("timestamp", "sensor_id")
            },
        }

        if "covariance" in grp:
            result["covariance"] = np.array(grp["covariance"])

        return result

    # =========================================================================
    # Time-Series Queries
    # =========================================================================

    def get_track_trajectory(
        self,
        track_id: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, NDArray[np.float64]]:
        """Extract a track segment within a time range.

        Selects by boolean mask over the full timestamp array -- a linear
        scan, not the binary search previously claimed here.
        (``get_state_at_time`` is the method that uses ``searchsorted``.)

        Parameters
        ----------
        track_id : str
            Track identifier.
        start_time : float, optional
            Minimum timestamp (inclusive).
        end_time : float, optional
            Maximum timestamp (inclusive).
        scenario_id : str, optional
            Scenario identifier.

        Returns
        -------
        dict
            Keys: states, covariances, timestamps.
        """
        self._check_open()
        group_path = self._resolve_track_group(track_id, scenario_id)

        if group_path not in self._file:
            raise KeyError(f"Track '{track_id}' not found")

        grp = self._file[group_path]
        timestamps = np.array(grp["timestamps"])

        # Find time range indices
        if start_time is not None or end_time is not None:
            mask = np.ones(len(timestamps), dtype=bool)
            if start_time is not None:
                mask &= timestamps >= start_time
            if end_time is not None:
                mask &= timestamps <= end_time

            idx = np.where(mask)[0]
            if len(idx) == 0:
                return {
                    "states": np.empty((0,)),
                    "covariances": np.empty((0,)),
                    "timestamps": np.empty((0,)),
                }
            # Index with the mask, not idx[0]:idx[-1]+1: after an out-of-order
            # append the in-range rows are not contiguous.
            return {
                "states": np.array(grp["state_history"])[idx],
                "covariances": np.array(grp["covariance_history"])[idx],
                "timestamps": timestamps[idx],
            }

        return {
            "states": np.array(grp["state_history"]),
            "covariances": np.array(grp["covariance_history"]),
            "timestamps": timestamps,
        }

    def get_state_at_time(
        self,
        track_id: str,
        time: float,
        interpolate: bool = False,
        scenario_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get state at a specific time.

        Parameters
        ----------
        track_id : str
            Track identifier.
        time : float
            Query time.
        interpolate : bool
            If True, linearly interpolate between nearest states.
            If False, return nearest state. Default is False.
        scenario_id : str, optional
            Scenario identifier.

        Returns
        -------
        dict
            Keys: state, covariance, timestamp.
        """
        self._check_open()
        group_path = self._resolve_track_group(track_id, scenario_id)

        if group_path not in self._file:
            raise KeyError(f"Track '{track_id}' not found")

        grp = self._file[group_path]
        timestamps = np.array(grp["timestamps"])

        if not interpolate:
            # Nearest neighbor
            idx = int(np.argmin(np.abs(timestamps - time)))
            return {
                "state": np.array(grp["state_history"][idx]),
                "covariance": np.array(grp["covariance_history"][idx]),
                "timestamp": float(timestamps[idx]),
            }

        # Linear interpolation
        if time <= timestamps[0]:
            return {
                "state": np.array(grp["state_history"][0]),
                "covariance": np.array(grp["covariance_history"][0]),
                "timestamp": float(timestamps[0]),
            }
        if time >= timestamps[-1]:
            return {
                "state": np.array(grp["state_history"][-1]),
                "covariance": np.array(grp["covariance_history"][-1]),
                "timestamp": float(timestamps[-1]),
            }

        # Find bracketing indices
        idx = int(np.searchsorted(timestamps, time))
        t0, t1 = timestamps[idx - 1], timestamps[idx]
        alpha = (time - t0) / (t1 - t0)

        s0 = np.array(grp["state_history"][idx - 1])
        s1 = np.array(grp["state_history"][idx])
        c0 = np.array(grp["covariance_history"][idx - 1])
        c1 = np.array(grp["covariance_history"][idx])

        return {
            "state": (1 - alpha) * s0 + alpha * s1,
            "covariance": (1 - alpha) * c0 + alpha * c1,
            "timestamp": time,
        }

    def get_tracks_in_region(
        self,
        bbox: List[float],
        time_range: Optional[List[float]] = None,
        state_indices: Tuple[int, int] = (0, 2),
        scenario_id: Optional[str] = None,
    ) -> List[str]:
        """Find track IDs with states inside a bounding box.

        Parameters
        ----------
        bbox : list of float
            [x_min, y_min, x_max, y_max] bounding box.
        time_range : list of float, optional
            [t_min, t_max] time range filter.
        state_indices : tuple of int
            Indices of x and y components in state vector.
            Default is (0, 2).
        scenario_id : str, optional
            Scenario identifier.

        Returns
        -------
        list of str
            Track IDs with states in the region.
        """
        self._check_open()
        x_min, y_min, x_max, y_max = bbox
        ix, iy = state_indices

        track_ids = self.list_tracks(scenario_id)
        result = []

        for tid in track_ids:
            group_path = self._resolve_track_group(tid, scenario_id)
            grp = self._file[group_path]
            timestamps = np.array(grp["timestamps"])
            states = np.array(grp["state_history"])

            # Time filter
            if time_range is not None:
                mask = (timestamps >= time_range[0]) & (timestamps <= time_range[1])
                states = states[mask]

            if len(states) == 0:
                continue

            # Spatial filter
            x_vals = states[:, ix]
            y_vals = states[:, iy]
            if np.any(
                (x_vals >= x_min)
                & (x_vals <= x_max)
                & (y_vals >= y_min)
                & (y_vals <= y_max)
            ):
                result.append(tid)

        return result

    # =========================================================================
    # Scenario Operations
    # =========================================================================

    def store_tracking_scenario(
        self,
        scenario_id: str,
        tracks: Dict[str, Dict[str, Any]],
        detections: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a complete tracking scenario.

        Parameters
        ----------
        scenario_id : str
            Unique scenario identifier.
        tracks : dict
            {track_id: {"states": ndarray, "covariances": ndarray,
                         "timestamps": ndarray, ...}}.
        detections : dict, optional
            {detection_id: {"measurement": ndarray, "timestamp": float,
                            "sensor_id": str, ...}}.
        metadata : dict, optional
            Scenario-level metadata.
        """
        self._check_open()
        scenario_grp = self._ensure_group(f"scenarios/{scenario_id}")

        if metadata:
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    scenario_grp.attrs[k] = v
                else:
                    scenario_grp.attrs[k] = json.dumps(v)

        scenario_grp.attrs["n_tracks"] = len(tracks)

        for tid, tdata in tracks.items():
            self.store_track(
                tid,
                tdata["states"],
                tdata["covariances"],
                tdata["timestamps"],
                metadata=tdata.get("metadata"),
                residuals=tdata.get("residuals"),
                scenario_id=scenario_id,
            )

        if detections:
            scenario_grp.attrs["n_detections"] = len(detections)
            for did, ddata in detections.items():
                self.store_detection(
                    did,
                    ddata["measurement"],
                    ddata["timestamp"],
                    ddata.get("sensor_id", "unknown"),
                    covariance=ddata.get("covariance"),
                    metadata=ddata.get("metadata"),
                    scenario_id=scenario_id,
                )

        self._file.flush()

    def retrieve_tracking_scenario(
        self,
        scenario_id: str,
    ) -> Dict[str, Any]:
        """Retrieve a complete scenario.

        Parameters
        ----------
        scenario_id : str
            Scenario identifier.

        Returns
        -------
        dict
            Keys: tracks, detections, metadata.
        """
        self._check_open()
        scenario_path = f"scenarios/{scenario_id}"

        if scenario_path not in self._file:
            raise KeyError(f"Scenario '{scenario_id}' not found")

        scenario_grp = self._file[scenario_path]

        tracks = {}
        for tid in self.list_tracks(scenario_id):
            tracks[tid] = self.retrieve_track(tid, scenario_id)

        detections = {}
        for did in self.list_detections(scenario_id):
            detections[did] = self.retrieve_detection(did, scenario_id)

        return {
            "tracks": tracks,
            "detections": detections,
            "metadata": dict(scenario_grp.attrs),
        }

    def list_scenarios(self) -> List[str]:
        """List all stored scenario IDs.

        Returns
        -------
        list of str
            Scenario identifiers.
        """
        self._check_open()
        if "scenarios" not in self._file:
            return []
        return list(self._file["scenarios"].keys())

    def list_tracks(
        self,
        scenario_id: Optional[str] = None,
    ) -> List[str]:
        """List all track IDs.

        Parameters
        ----------
        scenario_id : str, optional
            If provided, list tracks within this scenario.

        Returns
        -------
        list of str
            Track identifiers.
        """
        self._check_open()
        if scenario_id is not None:
            path = f"scenarios/{scenario_id}/tracks"
        else:
            path = "tracks"

        if path not in self._file:
            return []
        return list(self._file[path].keys())

    def list_detections(
        self,
        scenario_id: Optional[str] = None,
    ) -> List[str]:
        """List all detection IDs.

        Parameters
        ----------
        scenario_id : str, optional
            If provided, list detections within this scenario.

        Returns
        -------
        list of str
            Detection identifiers.
        """
        self._check_open()
        if scenario_id is not None:
            path = f"scenarios/{scenario_id}/detections"
        else:
            path = "detections"

        if path not in self._file:
            return []
        return list(self._file[path].keys())

    def compare_scenarios(
        self,
        scenario_id1: str,
        scenario_id2: str,
    ) -> Dict[str, Any]:
        """Compare two scenarios.

        Parameters
        ----------
        scenario_id1 : str
            First scenario.
        scenario_id2 : str
            Second scenario.

        Returns
        -------
        dict
            Keys: common_tracks, unique_to_1, unique_to_2,
            state_differences (dict of track_id -> RMSE).

        Warns
        -----
        RuntimeWarning
            For each common track whose two histories share no timestamp;
            its RMSE is NaN.

        Notes
        -----
        The RMSE for a common track is taken over the states at timestamps
        present in both histories, matched by exact equality (no
        tolerance), so timestamps produced by different float arithmetic
        (``0.1 * k`` against a cumulative sum of ``0.1``) may not coincide
        and the RMSE may cover only a subset of the nominally shared
        scans. Within one history, a repeated timestamp contributes its
        first row. Until v2.11.1 rows were paired by index, so two
        identical tracks offset by one scan reported a nonzero RMSE.
        """
        self._check_open()
        tracks1 = set(self.list_tracks(scenario_id1))
        tracks2 = set(self.list_tracks(scenario_id2))

        common = tracks1 & tracks2
        differences: Dict[str, float] = {}

        for tid in common:
            t1 = self.retrieve_track(tid, scenario_id1)
            t2 = self.retrieve_track(tid, scenario_id2)

            if len(t1["states"]) == 0 or len(t2["states"]) == 0:
                continue
            _, i1, i2 = np.intersect1d(
                t1["timestamps"], t2["timestamps"], return_indices=True
            )
            if len(i1) == 0:
                warnings.warn(
                    f"track '{tid}' has no timestamps in common between "
                    f"scenarios '{scenario_id1}' and '{scenario_id2}'; "
                    "its RMSE is NaN",
                    RuntimeWarning,
                    stacklevel=2,
                )
                differences[tid] = float("nan")
                continue
            diff = t1["states"][i1] - t2["states"][i2]
            differences[tid] = float(np.sqrt(np.mean(diff**2)))

        return {
            "common_tracks": sorted(common),
            "unique_to_1": sorted(tracks1 - tracks2),
            "unique_to_2": sorted(tracks2 - tracks1),
            "state_differences": differences,
        }

    # =========================================================================
    # Export/Import
    # =========================================================================

    def export_to_sql(
        self,
        db_manager: Any,
        scenario_id: Optional[str] = None,
    ) -> None:
        """Export HDF5 track data to a TrackDatabaseManager.

        Parameters
        ----------
        db_manager : TrackDatabaseManager
            Target SQL database manager (must be open for writing).
        scenario_id : str, optional
            Scenario to export. If None, exports standalone tracks.
        """
        self._check_open()

        for tid in self.list_tracks(scenario_id):
            track_data = self.retrieve_track(tid, scenario_id)
            states = track_data["states"]
            covs = track_data["covariances"]
            timestamps = track_data["timestamps"]

            if len(states) > 0:
                db_manager.initiate_track(
                    tid,
                    states[0],
                    covs[0],
                    float(timestamps[0]),
                    metadata=track_data.get("metadata"),
                )

                if len(states) > 1:
                    residuals = track_data.get("residuals")
                    db_manager.store_track_history(
                        tid,
                        states[1:],
                        covs[1:],
                        timestamps[1:],
                        residuals=residuals[1:] if residuals is not None else None,
                    )

        for did in self.list_detections(scenario_id):
            det = self.retrieve_detection(did, scenario_id)
            db_manager.store_detection(
                did,
                det["measurement"],
                det["sensor_id"],
                det["timestamp"],
                covariance=det.get("covariance"),
            )

    def import_from_sql(
        self,
        db_manager: Any,
        scenario_id: str,
    ) -> None:
        """Import tracks and detections from SQL into HDF5 as a scenario.

        Parameters
        ----------
        db_manager : TrackDatabaseManager
            Source SQL database manager (must be open for reading).
        scenario_id : str
            Scenario ID for the imported data.
        """
        self._check_open()

        tracks_data: Dict[str, Dict[str, Any]] = {}
        for track_info in db_manager.retrieve_all_tracks():
            tid = track_info["track_id"]
            try:
                history = db_manager.get_track_history(tid)
                tracks_data[tid] = {
                    "states": history["states"],
                    "covariances": history["covariances"],
                    "timestamps": history["timestamps"],
                    "residuals": history.get("residuals"),
                    "metadata": {
                        "status": track_info["status"],
                        "birth_time": track_info["birth_time"],
                        "hits": track_info["hits"],
                        "misses": track_info["misses"],
                    },
                }
            except KeyError:
                continue

        detections_data: Dict[str, Dict[str, Any]] = {}
        for det in db_manager.retrieve_all_detections():
            detections_data[det["detection_id"]] = {
                "measurement": det["measurement"],
                "timestamp": det["timestamp"],
                "sensor_id": det["sensor_id"],
                "covariance": det.get("covariance"),
            }

        self.store_tracking_scenario(scenario_id, tracks_data, detections_data or None)

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _check_open(self) -> None:
        """Verify the file is open."""
        if self._file is None:
            raise RuntimeError("File not open. Call open() first.")

    def _ensure_group(self, path: str) -> Any:
        """Create group and all parent groups if needed."""
        if path in self._file:
            return self._file[path]
        return self._file.create_group(path)

    def _resolve_track_group(
        self, track_id: str, scenario_id: Optional[str] = None
    ) -> str:
        """Resolve HDF5 group path for a track."""
        if scenario_id is not None:
            return f"scenarios/{scenario_id}/tracks/{track_id}"
        return f"tracks/{track_id}"

    def _resolve_detection_group(
        self, detection_id: str, scenario_id: Optional[str] = None
    ) -> str:
        """Resolve HDF5 group path for a detection."""
        if scenario_id is not None:
            return f"scenarios/{scenario_id}/detections/{detection_id}"
        return f"detections/{detection_id}"

    def _create_chunked_dataset(
        self,
        group: Any,
        name: str,
        data: NDArray[np.float64],
    ) -> Any:
        """Create a chunked, compressed, resizable dataset.

        The chunk's first axis is time (each track's own history), capped
        at ``chunk_size`` -- so a track shorter than ``chunk_size`` (the
        common case) lands in a single chunk spanning its whole series,
        letting gzip see the full smooth trajectory rather than an
        arbitrary slice of it.
        """
        # Calculate chunk shape
        shape = data.shape
        chunks = list(shape)
        chunks[0] = min(shape[0], self._chunk_size)
        chunk_tuple = tuple(chunks)

        # Maxshape allows resizing along first axis
        maxshape = list(shape)
        maxshape[0] = None
        maxshape_tuple = tuple(maxshape)

        return group.create_dataset(
            name,
            data=data,
            chunks=chunk_tuple,
            maxshape=maxshape_tuple,
            compression=self._compression,
            compression_opts=self._compression_level,
            shuffle=self._shuffle,
        )
