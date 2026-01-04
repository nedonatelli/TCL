"""
Measurement set container.

This module provides a time-indexed collection of measurements
with spatial query support.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.containers.kd_tree import KDTree


class Measurement(NamedTuple):
    """
    Single measurement with metadata.

    Attributes
    ----------
    value : ndarray
        Measurement vector.
    time : float
        Time of measurement.
    covariance : ndarray, optional
        Measurement covariance matrix.
    sensor_id : int
        ID of the sensor that produced this measurement.
    id : int
        Unique measurement identifier.
    """

    value: NDArray[np.float64]
    time: float
    covariance: Optional[NDArray[np.float64]] = None
    sensor_id: int = 0
    id: int = -1


class MeasurementQuery(NamedTuple):
    """
    Result of a measurement set query.

    Attributes
    ----------
    measurements : List[Measurement]
        List of measurements matching the query.
    indices : List[int]
        Original indices of the matching measurements.
    """

    measurements: List[Measurement]
    indices: List[int]


class MeasurementSet:
    """
    Collection of measurements with time and spatial indexing.

    Provides:
    - Time-windowed queries
    - Spatial region queries
    - Sensor filtering
    - Batch value extraction

    Parameters
    ----------
    measurements : Iterable[Measurement], optional
        Initial measurements to add.

    Examples
    --------
    >>> import numpy as np
    >>> # Create measurements
    >>> m1 = Measurement(value=np.array([1.0, 2.0]), time=0.0, id=0)
    >>> m2 = Measurement(value=np.array([3.0, 4.0]), time=0.0, id=1)
    >>> m3 = Measurement(value=np.array([5.0, 6.0]), time=1.0, id=2)
    >>> # Create measurement set
    >>> mset = MeasurementSet([m1, m2, m3])
    >>> len(mset)
    3
    >>> # Filter by time
    >>> at_t0 = mset.at_time(0.0)
    >>> len(at_t0)
    2
    >>> # Get values
    >>> values = mset.values()
    >>> values.shape
    (3, 2)
    """

    def __init__(self, measurements: Optional[Iterable[Measurement]] = None) -> None:
        """Initialize measurement set."""
        if measurements is None:
            self._measurements: List[Measurement] = []
        else:
            self._measurements = list(measurements)

        # Spatial index (built lazily)
        self._spatial_index: Optional[KDTree] = None
        self._index_valid: bool = False

    @classmethod
    def from_arrays(
        cls,
        values: ArrayLike,
        times: ArrayLike,
        covariances: Optional[ArrayLike] = None,
        sensor_ids: Optional[ArrayLike] = None,
    ) -> MeasurementSet:
        """
        Create MeasurementSet from arrays.

        Parameters
        ----------
        values : array_like
            Array of shape (n_meas, meas_dim) containing measurement values.
        times : array_like
            Array of length n_meas containing measurement times.
        covariances : array_like, optional
            Array of shape (n_meas, meas_dim, meas_dim) containing covariances.
        sensor_ids : array_like, optional
            Array of length n_meas containing sensor IDs.

        Returns
        -------
        MeasurementSet
            New MeasurementSet containing the measurements.
        """
        values = np.asarray(values, dtype=np.float64)
        times = np.asarray(times, dtype=np.float64)

        n_meas = len(values)

        if covariances is not None:
            covariances = np.asarray(covariances, dtype=np.float64)
        if sensor_ids is not None:
            sensor_ids = np.asarray(sensor_ids, dtype=np.int64)
        else:
            sensor_ids = np.zeros(n_meas, dtype=np.int64)

        measurements = []
        for i in range(n_meas):
            cov = covariances[i] if covariances is not None else None
            m = Measurement(
                value=values[i].copy(),
                time=float(times[i]),
                covariance=cov.copy() if cov is not None else None,
                sensor_id=int(sensor_ids[i]),
                id=i,
            )
            measurements.append(m)

        return cls(measurements)

    def __len__(self) -> int:
        """Return number of measurements."""
        return len(self._measurements)

    def __iter__(self) -> Iterator[Measurement]:
        """Iterate over measurements."""
        return iter(self._measurements)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Measurement, MeasurementSet]:
        """
        Get measurement by index or slice.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Measurement or MeasurementSet
            Single measurement if int, MeasurementSet if slice.
        """
        if isinstance(idx, int):
            return self._measurements[idx]
        else:
            return MeasurementSet(self._measurements[idx])

    def __repr__(self) -> str:
        """String representation."""
        return f"MeasurementSet(n_meas={len(self)})"

    def at_time(self, time: float, tolerance: float = 1e-9) -> MeasurementSet:
        """
        Get measurements at a specific time.

        Parameters
        ----------
        time : float
            Time to query.
        tolerance : float, optional
            Tolerance for time matching (default: 1e-9).

        Returns
        -------
        MeasurementSet
            Measurements at the specified time.
        """
        measurements = [
            m for m in self._measurements if abs(m.time - time) <= tolerance
        ]
        return MeasurementSet(measurements)

    def in_time_window(self, start: float, end: float) -> MeasurementSet:
        """
        Get measurements in a time window.

        Parameters
        ----------
        start : float
            Start time (inclusive).
        end : float
            End time (inclusive).

        Returns
        -------
        MeasurementSet
            Measurements within the time window.
        """
        measurements = [m for m in self._measurements if start <= m.time <= end]
        return MeasurementSet(measurements)

    def in_region(self, center: ArrayLike, radius: float) -> MeasurementSet:
        """
        Get measurements within a spatial region.

        Parameters
        ----------
        center : array_like
            Center point of the region.
        radius : float
            Radius of the region.

        Returns
        -------
        MeasurementSet
            Measurements within the region.
        """
        center = np.asarray(center, dtype=np.float64)

        measurements = []
        for m in self._measurements:
            # Use only dimensions that match the center
            meas_val = m.value[: len(center)]
            dist = np.linalg.norm(meas_val - center)
            if dist <= radius:
                measurements.append(m)

        return MeasurementSet(measurements)

    def by_sensor(self, sensor_id: int) -> MeasurementSet:
        """
        Get measurements from a specific sensor.

        Parameters
        ----------
        sensor_id : int
            Sensor ID to filter by.

        Returns
        -------
        MeasurementSet
            Measurements from the specified sensor.
        """
        measurements = [m for m in self._measurements if m.sensor_id == sensor_id]
        return MeasurementSet(measurements)

    def nearest_to(self, point: ArrayLike, k: int = 1) -> MeasurementQuery:
        """
        Find k nearest measurements to a point.

        Parameters
        ----------
        point : array_like
            Query point.
        k : int, optional
            Number of nearest neighbors (default: 1).

        Returns
        -------
        MeasurementQuery
            Query result with measurements and indices.

        Notes
        -----
        This method builds a spatial index on first call if not
        already built.
        """
        if len(self._measurements) == 0:
            return MeasurementQuery(measurements=[], indices=[])

        self.build_spatial_index()

        point = np.asarray(point, dtype=np.float64).reshape(1, -1)
        result = self._spatial_index.query(point, k=min(k, len(self)))

        indices = result.indices[0].tolist()
        measurements = [self._measurements[i] for i in indices]

        return MeasurementQuery(measurements=measurements, indices=indices)

    @property
    def times(self) -> NDArray[np.float64]:
        """Get unique measurement times."""
        if len(self._measurements) == 0:
            return np.array([])
        return np.unique([m.time for m in self._measurements])

    @property
    def sensors(self) -> List[int]:
        """Get unique sensor IDs."""
        return list(set(m.sensor_id for m in self._measurements))

    @property
    def time_range(self) -> Tuple[float, float]:
        """
        Get time range of measurements.

        Returns
        -------
        tuple of float
            (min_time, max_time) or (0.0, 0.0) if empty.
        """
        if len(self._measurements) == 0:
            return (0.0, 0.0)
        times = [m.time for m in self._measurements]
        return (min(times), max(times))

    def values(self) -> NDArray[np.float64]:
        """
        Extract all measurement values as array.

        Returns
        -------
        ndarray
            Array of shape (n_meas, meas_dim).
        """
        if len(self._measurements) == 0:
            return np.zeros((0, 0))
        return np.array([m.value for m in self._measurements])

    def values_at_time(
        self, time: float, tolerance: float = 1e-9
    ) -> NDArray[np.float64]:
        """
        Extract measurement values at a specific time.

        Parameters
        ----------
        time : float
            Time to query.
        tolerance : float, optional
            Tolerance for time matching (default: 1e-9).

        Returns
        -------
        ndarray
            Array of shape (n_meas_at_time, meas_dim).
        """
        return self.at_time(time, tolerance).values()

    def add(self, measurement: Measurement) -> MeasurementSet:
        """
        Add a measurement and return a new MeasurementSet.

        Parameters
        ----------
        measurement : Measurement
            Measurement to add.

        Returns
        -------
        MeasurementSet
            New MeasurementSet with the measurement added.
        """
        return MeasurementSet(self._measurements + [measurement])

    def add_batch(self, measurements: Iterable[Measurement]) -> MeasurementSet:
        """
        Add multiple measurements and return a new MeasurementSet.

        Parameters
        ----------
        measurements : Iterable[Measurement]
            Measurements to add.

        Returns
        -------
        MeasurementSet
            New MeasurementSet with the measurements added.
        """
        return MeasurementSet(self._measurements + list(measurements))

    def merge(self, other: MeasurementSet) -> MeasurementSet:
        """
        Merge with another MeasurementSet.

        Parameters
        ----------
        other : MeasurementSet
            MeasurementSet to merge with.

        Returns
        -------
        MeasurementSet
            New MeasurementSet containing measurements from both.
        """
        return MeasurementSet(list(self._measurements) + list(other._measurements))

    def copy(self) -> MeasurementSet:
        """
        Create a copy of this MeasurementSet.

        Returns
        -------
        MeasurementSet
            A new MeasurementSet with the same measurements.
        """
        return MeasurementSet(self._measurements)

    def build_spatial_index(self) -> None:
        """
        Build spatial index for efficient nearest neighbor queries.

        The index is built from measurement values. This is called
        automatically by nearest_to() if needed.
        """
        if self._index_valid and self._spatial_index is not None:
            return

        if len(self._measurements) == 0:
            self._spatial_index = None
            self._index_valid = True
            return

        # Build KDTree from measurement values
        values = self.values()
        self._spatial_index = KDTree(values)
        self._index_valid = True


__all__ = [
    "Measurement",
    "MeasurementSet",
    "MeasurementQuery",
]
