"""
Track list container.

This module provides a collection class for managing multiple tracks
with filtering, querying, and batch operations.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.trackers.multi_target import Track, TrackStatus

if TYPE_CHECKING:
    from pytcl.trackers.multi_target import MultiTargetTracker


class TrackQuery(NamedTuple):
    """
    Result of a track list query.

    Attributes
    ----------
    tracks : List[Track]
        List of tracks matching the query.
    indices : List[int]
        Original indices of the matching tracks.
    """

    tracks: List[Track]
    indices: List[int]


class TrackListStats(NamedTuple):
    """
    Statistics about a track list.

    Attributes
    ----------
    n_tracks : int
        Total number of tracks.
    n_confirmed : int
        Number of confirmed tracks.
    n_tentative : int
        Number of tentative tracks.
    n_deleted : int
        Number of deleted tracks.
    mean_hits : float
        Average number of hits per track.
    mean_misses : float
        Average number of misses per track.
    """

    n_tracks: int
    n_confirmed: int
    n_tentative: int
    n_deleted: int
    mean_hits: float
    mean_misses: float


class TrackList:
    """
    Collection of tracks with query and filter capabilities.

    Provides:
    - Filtering by status, time, region
    - Iteration and indexing
    - Batch state extraction
    - Statistics computation

    Parameters
    ----------
    tracks : Iterable[Track], optional
        Initial tracks to add to the list.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers.multi_target import Track, TrackStatus
    >>> # Create some tracks
    >>> t1 = Track(id=0, state=np.array([1, 0, 2, 0]),
    ...            covariance=np.eye(4), status=TrackStatus.CONFIRMED,
    ...            hits=5, misses=0, time=1.0)
    >>> t2 = Track(id=1, state=np.array([5, 0, 6, 0]),
    ...            covariance=np.eye(4), status=TrackStatus.TENTATIVE,
    ...            hits=2, misses=1, time=1.0)
    >>> # Create track list
    >>> tracks = TrackList([t1, t2])
    >>> len(tracks)
    2
    >>> # Filter by status
    >>> confirmed = tracks.confirmed
    >>> len(confirmed)
    1
    >>> # Get positions
    >>> positions = tracks.positions(indices=(0, 2))
    >>> positions.shape
    (2, 2)
    """

    def __init__(self, tracks: Optional[Iterable[Track]] = None) -> None:
        """Initialize track list."""
        if tracks is None:
            self._tracks: List[Track] = []
        else:
            self._tracks = list(tracks)

        # Build ID lookup for fast access
        self._id_to_idx: dict[int, int] = {t.id: i for i, t in enumerate(self._tracks)}

    @classmethod
    def from_tracker(cls, tracker: MultiTargetTracker) -> TrackList:
        """
        Create a TrackList from a MultiTargetTracker.

        Parameters
        ----------
        tracker : MultiTargetTracker
            The tracker to extract tracks from.

        Returns
        -------
        TrackList
            A new TrackList containing the tracker's current tracks.
        """
        return cls(tracker.tracks)

    def __len__(self) -> int:
        """Return number of tracks."""
        return len(self._tracks)

    def __iter__(self) -> Iterator[Track]:
        """Iterate over tracks."""
        return iter(self._tracks)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Track, TrackList]:
        """
        Get track by index or slice.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        Track or TrackList
            Single track if int, TrackList if slice.
        """
        if isinstance(idx, int):
            return self._tracks[idx]
        else:
            return TrackList(self._tracks[idx])

    def __contains__(self, track_id: int) -> bool:
        """Check if track ID exists in list."""
        return track_id in self._id_to_idx

    def __repr__(self) -> str:
        """String representation."""
        return f"TrackList(n_tracks={len(self)})"

    def get_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID.

        Parameters
        ----------
        track_id : int
            Track ID to find.

        Returns
        -------
        Track or None
            The track if found, None otherwise.
        """
        idx = self._id_to_idx.get(track_id)
        if idx is not None:
            return self._tracks[idx]
        return None

    def get_by_ids(self, track_ids: Iterable[int]) -> TrackList:
        """
        Get multiple tracks by their IDs.

        Parameters
        ----------
        track_ids : Iterable[int]
            Track IDs to find.

        Returns
        -------
        TrackList
            New TrackList containing the found tracks.
        """
        tracks = []
        for tid in track_ids:
            track = self.get_by_id(tid)
            if track is not None:
                tracks.append(track)
        return TrackList(tracks)

    def filter_by_status(self, status: TrackStatus) -> TrackList:
        """
        Filter tracks by status.

        Parameters
        ----------
        status : TrackStatus
            Status to filter by.

        Returns
        -------
        TrackList
            New TrackList containing only tracks with the given status.
        """
        return TrackList([t for t in self._tracks if t.status == status])

    def filter_by_time(
        self,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
    ) -> TrackList:
        """
        Filter tracks by time range.

        Parameters
        ----------
        min_time : float, optional
            Minimum time (inclusive).
        max_time : float, optional
            Maximum time (inclusive).

        Returns
        -------
        TrackList
            New TrackList containing tracks within the time range.
        """
        tracks = []
        for t in self._tracks:
            if min_time is not None and t.time < min_time:
                continue
            if max_time is not None and t.time > max_time:
                continue
            tracks.append(t)
        return TrackList(tracks)

    def filter_by_region(
        self,
        center: ArrayLike,
        radius: float,
        state_indices: Tuple[int, int] = (0, 2),
    ) -> TrackList:
        """
        Filter tracks by spatial region.

        Parameters
        ----------
        center : array_like
            Center point [x, y].
        radius : float
            Radius of the region.
        state_indices : tuple of int, optional
            Indices of x and y in state vector (default: (0, 2)).

        Returns
        -------
        TrackList
            New TrackList containing tracks within the region.
        """
        center = np.asarray(center, dtype=np.float64)
        ix, iy = state_indices

        tracks = []
        for t in self._tracks:
            pos = np.array([t.state[ix], t.state[iy]])
            dist = np.linalg.norm(pos - center)
            if dist <= radius:
                tracks.append(t)
        return TrackList(tracks)

    def filter_by_predicate(self, predicate: Callable[[Track], bool]) -> TrackList:
        """
        Filter tracks using a custom predicate.

        Parameters
        ----------
        predicate : callable
            Function that takes a Track and returns True to include it.

        Returns
        -------
        TrackList
            New TrackList containing tracks that pass the predicate.
        """
        return TrackList([t for t in self._tracks if predicate(t)])

    @property
    def confirmed(self) -> TrackList:
        """Get confirmed tracks only."""
        return self.filter_by_status(TrackStatus.CONFIRMED)

    @property
    def tentative(self) -> TrackList:
        """Get tentative tracks only."""
        return self.filter_by_status(TrackStatus.TENTATIVE)

    @property
    def track_ids(self) -> List[int]:
        """Get list of all track IDs."""
        return [t.id for t in self._tracks]

    def states(self) -> NDArray[np.float64]:
        """
        Extract all track states as array.

        Returns
        -------
        ndarray
            Array of shape (n_tracks, state_dim) containing all states.
        """
        if len(self._tracks) == 0:
            return np.zeros((0, 0))
        return np.array([t.state for t in self._tracks])

    def covariances(self) -> NDArray[np.float64]:
        """
        Extract all track covariances as array.

        Returns
        -------
        ndarray
            Array of shape (n_tracks, state_dim, state_dim).
        """
        if len(self._tracks) == 0:
            return np.zeros((0, 0, 0))
        return np.array([t.covariance for t in self._tracks])

    def positions(self, indices: Tuple[int, int] = (0, 2)) -> NDArray[np.float64]:
        """
        Extract track positions.

        Parameters
        ----------
        indices : tuple of int, optional
            Indices of x and y in state vector (default: (0, 2)).

        Returns
        -------
        ndarray
            Array of shape (n_tracks, 2) containing positions.
        """
        if len(self._tracks) == 0:
            return np.zeros((0, 2))
        ix, iy = indices
        return np.array([[t.state[ix], t.state[iy]] for t in self._tracks])

    def stats(self) -> TrackListStats:
        """
        Compute statistics about the track list.

        Returns
        -------
        TrackListStats
            Statistics about the tracks.
        """
        n_tracks = len(self._tracks)
        if n_tracks == 0:
            return TrackListStats(
                n_tracks=0,
                n_confirmed=0,
                n_tentative=0,
                n_deleted=0,
                mean_hits=0.0,
                mean_misses=0.0,
            )

        n_confirmed = sum(1 for t in self._tracks if t.status == TrackStatus.CONFIRMED)
        n_tentative = sum(1 for t in self._tracks if t.status == TrackStatus.TENTATIVE)
        n_deleted = sum(1 for t in self._tracks if t.status == TrackStatus.DELETED)
        mean_hits = sum(t.hits for t in self._tracks) / n_tracks
        mean_misses = sum(t.misses for t in self._tracks) / n_tracks

        return TrackListStats(
            n_tracks=n_tracks,
            n_confirmed=n_confirmed,
            n_tentative=n_tentative,
            n_deleted=n_deleted,
            mean_hits=mean_hits,
            mean_misses=mean_misses,
        )

    def add(self, track: Track) -> TrackList:
        """
        Add a track and return a new TrackList.

        Parameters
        ----------
        track : Track
            Track to add.

        Returns
        -------
        TrackList
            New TrackList with the track added.
        """
        return TrackList(self._tracks + [track])

    def remove(self, track_id: int) -> TrackList:
        """
        Remove a track by ID and return a new TrackList.

        Parameters
        ----------
        track_id : int
            ID of track to remove.

        Returns
        -------
        TrackList
            New TrackList without the specified track.
        """
        return TrackList([t for t in self._tracks if t.id != track_id])

    def merge(self, other: TrackList) -> TrackList:
        """
        Merge with another TrackList.

        Parameters
        ----------
        other : TrackList
            TrackList to merge with.

        Returns
        -------
        TrackList
            New TrackList containing tracks from both lists.

        Notes
        -----
        Duplicate track IDs are not handled specially - both tracks
        will be included. Use get_by_id on the result to access
        specific tracks.
        """
        return TrackList(list(self._tracks) + list(other._tracks))

    def copy(self) -> TrackList:
        """
        Create a copy of this TrackList.

        Returns
        -------
        TrackList
            A new TrackList with the same tracks.
        """
        return TrackList(self._tracks)


__all__ = [
    "TrackList",
    "TrackQuery",
    "TrackListStats",
]
