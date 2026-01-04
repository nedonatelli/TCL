"""
Cluster set container.

This module provides a container for managing groups of tracks
that move together (formations, convoys, etc.).
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
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

from pytcl.clustering.dbscan import dbscan
from pytcl.clustering.kmeans import kmeans
from pytcl.containers.track_list import TrackList
from pytcl.trackers.multi_target import Track


class TrackCluster(NamedTuple):
    """
    A cluster of related tracks.

    Attributes
    ----------
    id : int
        Unique cluster identifier.
    track_ids : Tuple[int, ...]
        Immutable tuple of track IDs in this cluster.
    centroid : ndarray
        Cluster center position.
    covariance : ndarray
        Cluster spread covariance matrix.
    time : float
        Time at which cluster was computed.
    """

    id: int
    track_ids: Tuple[int, ...]
    centroid: NDArray[np.float64]
    covariance: NDArray[np.float64]
    time: float


class ClusterStats(NamedTuple):
    """
    Statistics for a cluster.

    Attributes
    ----------
    n_tracks : int
        Number of tracks in the cluster.
    mean_separation : float
        Average distance between tracks and centroid.
    max_separation : float
        Maximum distance from any track to centroid.
    velocity_coherence : float
        Measure of how aligned velocities are (0-1).
        1.0 means perfectly aligned, 0.0 means random directions.
    """

    n_tracks: int
    mean_separation: float
    max_separation: float
    velocity_coherence: float


def compute_cluster_centroid(
    tracks: Iterable[Track],
    state_indices: Tuple[int, int] = (0, 2),
) -> NDArray[np.float64]:
    """
    Compute the centroid of a group of tracks.

    Parameters
    ----------
    tracks : Iterable[Track]
        Tracks to compute centroid for.
    state_indices : tuple of int, optional
        Indices of x and y in state vector (default: (0, 2)).

    Returns
    -------
    centroid : ndarray
        Centroid position [x, y].
    """
    track_list = list(tracks)
    if len(track_list) == 0:
        return np.array([0.0, 0.0])

    ix, iy = state_indices
    positions = np.array([[t.state[ix], t.state[iy]] for t in track_list])
    return np.mean(positions, axis=0)


def compute_cluster_covariance(
    tracks: Iterable[Track],
    state_indices: Tuple[int, int] = (0, 2),
) -> NDArray[np.float64]:
    """
    Compute the covariance of track positions in a cluster.

    Parameters
    ----------
    tracks : Iterable[Track]
        Tracks to compute covariance for.
    state_indices : tuple of int, optional
        Indices of x and y in state vector (default: (0, 2)).

    Returns
    -------
    covariance : ndarray
        Position covariance matrix (2x2).
    """
    track_list = list(tracks)
    if len(track_list) < 2:
        return np.eye(2)

    ix, iy = state_indices
    positions = np.array([[t.state[ix], t.state[iy]] for t in track_list])
    return np.cov(positions.T)


def cluster_tracks_dbscan(
    tracks: TrackList,
    eps: float,
    min_samples: int = 2,
    state_indices: Tuple[int, int] = (0, 2),
) -> "ClusterSet":
    """
    Cluster tracks using DBSCAN algorithm.

    Parameters
    ----------
    tracks : TrackList
        Tracks to cluster.
    eps : float
        Maximum distance between two tracks to be considered neighbors.
    min_samples : int, optional
        Minimum number of tracks to form a cluster (default: 2).
    state_indices : tuple of int, optional
        Indices of x and y in state vector (default: (0, 2)).

    Returns
    -------
    ClusterSet
        Set of track clusters.
    """
    if len(tracks) == 0:
        return ClusterSet()

    # Extract positions
    positions = tracks.positions(indices=state_indices)

    # Run DBSCAN
    result = dbscan(positions, eps=eps, min_samples=min_samples)

    # Build clusters from labels
    clusters = []
    track_list = list(tracks)
    time = track_list[0].time if track_list else 0.0

    for cluster_id in range(result.n_clusters):
        mask = result.labels == cluster_id
        cluster_tracks = [track_list[i] for i in range(len(track_list)) if mask[i]]
        track_ids = tuple(t.id for t in cluster_tracks)

        centroid = compute_cluster_centroid(cluster_tracks, state_indices)
        covariance = compute_cluster_covariance(cluster_tracks, state_indices)

        clusters.append(
            TrackCluster(
                id=cluster_id,
                track_ids=track_ids,
                centroid=centroid,
                covariance=covariance,
                time=time,
            )
        )

    return ClusterSet(clusters)


def cluster_tracks_kmeans(
    tracks: TrackList,
    n_clusters: int,
    state_indices: Tuple[int, int] = (0, 2),
    rng: Optional[np.random.Generator] = None,
) -> "ClusterSet":
    """
    Cluster tracks using K-means algorithm.

    Parameters
    ----------
    tracks : TrackList
        Tracks to cluster.
    n_clusters : int
        Number of clusters to form.
    state_indices : tuple of int, optional
        Indices of x and y in state vector (default: (0, 2)).
    rng : numpy.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    ClusterSet
        Set of track clusters.
    """
    if len(tracks) == 0:
        return ClusterSet()

    if n_clusters > len(tracks):
        n_clusters = len(tracks)

    # Extract positions
    positions = tracks.positions(indices=state_indices)

    # Run K-means
    result = kmeans(positions, n_clusters=n_clusters, rng=rng)

    # Build clusters from labels
    clusters = []
    track_list = list(tracks)
    time = track_list[0].time if track_list else 0.0

    for cluster_id in range(n_clusters):
        mask = result.labels == cluster_id
        cluster_tracks = [track_list[i] for i in range(len(track_list)) if mask[i]]

        if len(cluster_tracks) == 0:
            continue

        track_ids = tuple(t.id for t in cluster_tracks)
        centroid = compute_cluster_centroid(cluster_tracks, state_indices)
        covariance = compute_cluster_covariance(cluster_tracks, state_indices)

        clusters.append(
            TrackCluster(
                id=cluster_id,
                track_ids=track_ids,
                centroid=centroid,
                covariance=covariance,
                time=time,
            )
        )

    return ClusterSet(clusters)


class ClusterSet:
    """
    Collection of track clusters.

    Provides:
    - Cluster creation from tracks
    - Cluster queries
    - Cluster merging/splitting

    Parameters
    ----------
    clusters : Iterable[TrackCluster], optional
        Initial clusters to add.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.trackers.multi_target import Track, TrackStatus
    >>> from pytcl.containers.track_list import TrackList
    >>> # Create some tracks in two groups
    >>> t1 = Track(id=0, state=np.array([0, 0, 0, 0]),
    ...            covariance=np.eye(4), status=TrackStatus.CONFIRMED,
    ...            hits=5, misses=0, time=1.0)
    >>> t2 = Track(id=1, state=np.array([1, 0, 1, 0]),
    ...            covariance=np.eye(4), status=TrackStatus.CONFIRMED,
    ...            hits=5, misses=0, time=1.0)
    >>> t3 = Track(id=2, state=np.array([10, 0, 10, 0]),
    ...            covariance=np.eye(4), status=TrackStatus.CONFIRMED,
    ...            hits=5, misses=0, time=1.0)
    >>> tracks = TrackList([t1, t2, t3])
    >>> # Cluster using DBSCAN
    >>> clusters = cluster_tracks_dbscan(tracks, eps=5.0, min_samples=2)
    """

    def __init__(self, clusters: Optional[Iterable[TrackCluster]] = None) -> None:
        """Initialize cluster set."""
        if clusters is None:
            self._clusters: List[TrackCluster] = []
        else:
            self._clusters = list(clusters)

        # Build lookups
        self._id_to_idx: Dict[int, int] = {
            c.id: i for i, c in enumerate(self._clusters)
        }
        self._track_to_cluster: Dict[int, int] = {}
        for cluster in self._clusters:
            for track_id in cluster.track_ids:
                self._track_to_cluster[track_id] = cluster.id

    @classmethod
    def from_tracks(
        cls,
        tracks: TrackList,
        method: str = "dbscan",
        **kwargs: Any,
    ) -> ClusterSet:
        """
        Create a ClusterSet by clustering tracks.

        Parameters
        ----------
        tracks : TrackList
            Tracks to cluster.
        method : str
            Clustering method: 'dbscan' or 'kmeans'.
        **kwargs
            Additional arguments passed to the clustering function.
            For DBSCAN: eps, min_samples, state_indices
            For K-means: n_clusters, state_indices, rng

        Returns
        -------
        ClusterSet
            New ClusterSet containing the computed clusters.

        Examples
        --------
        >>> clusters = ClusterSet.from_tracks(tracks, method='dbscan', eps=2.0)
        >>> clusters = ClusterSet.from_tracks(tracks, method='kmeans', n_clusters=3)
        """
        if method == "dbscan":
            return cluster_tracks_dbscan(tracks, **kwargs)
        elif method == "kmeans":
            return cluster_tracks_kmeans(tracks, **kwargs)
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def __len__(self) -> int:
        """Return number of clusters."""
        return len(self._clusters)

    def __iter__(self) -> Iterator[TrackCluster]:
        """Iterate over clusters."""
        return iter(self._clusters)

    def __getitem__(self, idx: Union[int, slice]) -> Union[TrackCluster, "ClusterSet"]:
        """
        Get cluster by index or slice.

        Parameters
        ----------
        idx : int or slice
            Index or slice to retrieve.

        Returns
        -------
        TrackCluster or ClusterSet
            Single cluster if int, ClusterSet if slice.
        """
        if isinstance(idx, int):
            return self._clusters[idx]
        else:
            return ClusterSet(self._clusters[idx])

    def __contains__(self, cluster_id: int) -> bool:
        """Check if cluster ID exists in set."""
        return cluster_id in self._id_to_idx

    def __repr__(self) -> str:
        """String representation."""
        return f"ClusterSet(n_clusters={len(self)})"

    def get_cluster(self, cluster_id: int) -> Optional[TrackCluster]:
        """
        Get cluster by ID.

        Parameters
        ----------
        cluster_id : int
            Cluster ID to find.

        Returns
        -------
        TrackCluster or None
            The cluster if found, None otherwise.
        """
        idx = self._id_to_idx.get(cluster_id)
        if idx is not None:
            return self._clusters[idx]
        return None

    def get_cluster_for_track(self, track_id: int) -> Optional[TrackCluster]:
        """
        Get the cluster containing a specific track.

        Parameters
        ----------
        track_id : int
            Track ID to find.

        Returns
        -------
        TrackCluster or None
            The cluster containing the track, or None if not found.
        """
        cluster_id = self._track_to_cluster.get(track_id)
        if cluster_id is not None:
            return self.get_cluster(cluster_id)
        return None

    def clusters_in_region(
        self, center: ArrayLike, radius: float
    ) -> List[TrackCluster]:
        """
        Get clusters with centroids within a spatial region.

        Parameters
        ----------
        center : array_like
            Center point [x, y].
        radius : float
            Radius of the region.

        Returns
        -------
        list of TrackCluster
            Clusters within the region.
        """
        center = np.asarray(center, dtype=np.float64)
        result = []

        for cluster in self._clusters:
            dist = np.linalg.norm(cluster.centroid - center)
            if dist <= radius:
                result.append(cluster)

        return result

    @property
    def cluster_ids(self) -> List[int]:
        """Get list of all cluster IDs."""
        return [c.id for c in self._clusters]

    @property
    def n_tracks_total(self) -> int:
        """Get total number of tracks across all clusters."""
        return sum(len(c.track_ids) for c in self._clusters)

    def cluster_stats(
        self,
        cluster_id: int,
        tracks: Optional[TrackList] = None,
        state_indices: Tuple[int, int] = (0, 2),
        velocity_indices: Tuple[int, int] = (1, 3),
    ) -> Optional[ClusterStats]:
        """
        Compute statistics for a cluster.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster.
        tracks : TrackList, optional
            TrackList containing the tracks. Required for velocity coherence.
        state_indices : tuple of int, optional
            Indices of x and y in state vector (default: (0, 2)).
        velocity_indices : tuple of int, optional
            Indices of vx and vy in state vector (default: (1, 3)).

        Returns
        -------
        ClusterStats or None
            Statistics for the cluster, or None if cluster not found.
        """
        cluster = self.get_cluster(cluster_id)
        if cluster is None:
            return None

        n_tracks = len(cluster.track_ids)
        if n_tracks == 0:
            return ClusterStats(
                n_tracks=0,
                mean_separation=0.0,
                max_separation=0.0,
                velocity_coherence=0.0,
            )

        # Compute separations if we have track data
        mean_separation = 0.0
        max_separation = 0.0
        velocity_coherence = 0.0

        if tracks is not None:
            ix, iy = state_indices
            positions = []
            velocities = []

            for tid in cluster.track_ids:
                track = tracks.get_by_id(tid)
                if track is not None:
                    positions.append([track.state[ix], track.state[iy]])
                    ivx, ivy = velocity_indices
                    if len(track.state) > max(ivx, ivy):
                        velocities.append([track.state[ivx], track.state[ivy]])

            if len(positions) > 0:
                positions = np.array(positions)
                centroid = cluster.centroid

                # Compute separations
                separations = np.sqrt(np.sum((positions - centroid) ** 2, axis=1))
                mean_separation = float(np.mean(separations))
                max_separation = float(np.max(separations))

            # Compute velocity coherence
            if len(velocities) > 1:
                velocities = np.array(velocities)
                # Normalize velocities
                norms = np.linalg.norm(velocities, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                unit_velocities = velocities / norms

                # Mean velocity direction
                mean_vel = np.mean(unit_velocities, axis=0)
                mean_vel_norm = np.linalg.norm(mean_vel)

                # Coherence is magnitude of mean unit velocity
                velocity_coherence = float(mean_vel_norm)

        return ClusterStats(
            n_tracks=n_tracks,
            mean_separation=mean_separation,
            max_separation=max_separation,
            velocity_coherence=velocity_coherence,
        )

    def all_stats(
        self,
        tracks: Optional[TrackList] = None,
        state_indices: Tuple[int, int] = (0, 2),
        velocity_indices: Tuple[int, int] = (1, 3),
    ) -> Dict[int, ClusterStats]:
        """
        Compute statistics for all clusters.

        Parameters
        ----------
        tracks : TrackList, optional
            TrackList containing the tracks.
        state_indices : tuple of int, optional
            Indices of x and y in state vector (default: (0, 2)).
        velocity_indices : tuple of int, optional
            Indices of vx and vy in state vector (default: (1, 3)).

        Returns
        -------
        dict
            Mapping from cluster ID to ClusterStats.
        """
        result = {}
        for cluster in self._clusters:
            stats = self.cluster_stats(
                cluster.id, tracks, state_indices, velocity_indices
            )
            if stats is not None:
                result[cluster.id] = stats
        return result

    def add_cluster(self, cluster: TrackCluster) -> ClusterSet:
        """
        Add a cluster and return a new ClusterSet.

        Parameters
        ----------
        cluster : TrackCluster
            Cluster to add.

        Returns
        -------
        ClusterSet
            New ClusterSet with the cluster added.
        """
        return ClusterSet(self._clusters + [cluster])

    def remove_cluster(self, cluster_id: int) -> ClusterSet:
        """
        Remove a cluster by ID and return a new ClusterSet.

        Parameters
        ----------
        cluster_id : int
            ID of cluster to remove.

        Returns
        -------
        ClusterSet
            New ClusterSet without the specified cluster.
        """
        return ClusterSet([c for c in self._clusters if c.id != cluster_id])

    def merge_clusters(
        self,
        id1: int,
        id2: int,
        new_id: Optional[int] = None,
    ) -> ClusterSet:
        """
        Merge two clusters into one.

        Parameters
        ----------
        id1 : int
            ID of first cluster.
        id2 : int
            ID of second cluster.
        new_id : int, optional
            ID for the merged cluster. Defaults to id1.

        Returns
        -------
        ClusterSet
            New ClusterSet with merged cluster.

        Raises
        ------
        ValueError
            If either cluster ID is not found.
        """
        c1 = self.get_cluster(id1)
        c2 = self.get_cluster(id2)

        if c1 is None:
            raise ValueError(f"Cluster {id1} not found")
        if c2 is None:
            raise ValueError(f"Cluster {id2} not found")

        if new_id is None:
            new_id = id1

        # Merge track IDs
        merged_track_ids = tuple(set(c1.track_ids) | set(c2.track_ids))

        # Compute new centroid (weighted average)
        n1, n2 = len(c1.track_ids), len(c2.track_ids)
        new_centroid = (n1 * c1.centroid + n2 * c2.centroid) / (n1 + n2)

        # Combine covariances (simple average for now)
        new_covariance = (c1.covariance + c2.covariance) / 2

        merged = TrackCluster(
            id=new_id,
            track_ids=merged_track_ids,
            centroid=new_centroid,
            covariance=new_covariance,
            time=max(c1.time, c2.time),
        )

        # Build new cluster list
        new_clusters = [c for c in self._clusters if c.id not in (id1, id2)]
        new_clusters.append(merged)

        return ClusterSet(new_clusters)

    def split_cluster(
        self,
        cluster_id: int,
        track_ids_1: Iterable[int],
        track_ids_2: Iterable[int],
        new_id_1: Optional[int] = None,
        new_id_2: Optional[int] = None,
        tracks: Optional[TrackList] = None,
        state_indices: Tuple[int, int] = (0, 2),
    ) -> ClusterSet:
        """
        Split a cluster into two.

        Parameters
        ----------
        cluster_id : int
            ID of cluster to split.
        track_ids_1 : Iterable[int]
            Track IDs for first new cluster.
        track_ids_2 : Iterable[int]
            Track IDs for second new cluster.
        new_id_1 : int, optional
            ID for first new cluster. Defaults to cluster_id.
        new_id_2 : int, optional
            ID for second new cluster. Defaults to max(cluster_ids) + 1.
        tracks : TrackList, optional
            TrackList for computing centroids. If None, uses existing centroid.
        state_indices : tuple of int, optional
            Indices of x and y in state vector (default: (0, 2)).

        Returns
        -------
        ClusterSet
            New ClusterSet with split clusters.

        Raises
        ------
        ValueError
            If cluster ID is not found.
        """
        original = self.get_cluster(cluster_id)
        if original is None:
            raise ValueError(f"Cluster {cluster_id} not found")

        ids_1 = tuple(track_ids_1)
        ids_2 = tuple(track_ids_2)

        if new_id_1 is None:
            new_id_1 = cluster_id
        if new_id_2 is None:
            new_id_2 = max(self.cluster_ids) + 1 if self.cluster_ids else 0

        # Compute centroids
        if tracks is not None:
            tracks_1 = [tracks.get_by_id(tid) for tid in ids_1]
            tracks_1 = [t for t in tracks_1 if t is not None]
            tracks_2 = [tracks.get_by_id(tid) for tid in ids_2]
            tracks_2 = [t for t in tracks_2 if t is not None]

            centroid_1 = compute_cluster_centroid(tracks_1, state_indices)
            centroid_2 = compute_cluster_centroid(tracks_2, state_indices)
            cov_1 = compute_cluster_covariance(tracks_1, state_indices)
            cov_2 = compute_cluster_covariance(tracks_2, state_indices)
        else:
            # Use original centroid for both
            centroid_1 = original.centroid.copy()
            centroid_2 = original.centroid.copy()
            cov_1 = original.covariance.copy()
            cov_2 = original.covariance.copy()

        cluster_1 = TrackCluster(
            id=new_id_1,
            track_ids=ids_1,
            centroid=centroid_1,
            covariance=cov_1,
            time=original.time,
        )
        cluster_2 = TrackCluster(
            id=new_id_2,
            track_ids=ids_2,
            centroid=centroid_2,
            covariance=cov_2,
            time=original.time,
        )

        # Build new cluster list
        new_clusters = [c for c in self._clusters if c.id != cluster_id]
        new_clusters.extend([cluster_1, cluster_2])

        return ClusterSet(new_clusters)

    def copy(self) -> ClusterSet:
        """
        Create a copy of this ClusterSet.

        Returns
        -------
        ClusterSet
            A new ClusterSet with the same clusters.
        """
        return ClusterSet(self._clusters)


__all__ = [
    "TrackCluster",
    "ClusterSet",
    "ClusterStats",
    "cluster_tracks_dbscan",
    "cluster_tracks_kmeans",
    "compute_cluster_centroid",
]
