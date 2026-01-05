"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

DBSCAN is a density-based clustering algorithm that groups points
that are closely packed together, marking points in low-density
regions as outliers.

References
----------
.. [1] M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, "A Density-Based
       Algorithm for Discovering Clusters in Large Spatial Databases
       with Noise," KDD 1996.
"""

from typing import Any, List, NamedTuple, Set

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


class DBSCANResult(NamedTuple):
    """Result of DBSCAN clustering.

    Attributes
    ----------
    labels : ndarray
        Cluster labels for each point, shape (n_samples,).
        -1 indicates noise points.
    n_clusters : int
        Number of clusters found (excluding noise).
    core_sample_indices : ndarray
        Indices of core samples.
    n_noise : int
        Number of noise points.
    """

    labels: NDArray[np.intp]
    n_clusters: int
    core_sample_indices: NDArray[np.intp]
    n_noise: int


@njit(cache=True)
def _compute_distance_matrix(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Compute pairwise Euclidean distance matrix (JIT-compiled)."""
    n = X.shape[0]
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(X.shape[1]):
                diff = X[i, k] - X[j, k]
                d += diff * diff
            d = np.sqrt(d)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def compute_neighbors(
    X: NDArray[np.floating],
    eps: float,
) -> List[NDArray[np.intp]]:
    """
    Compute neighbors within eps distance for all points.

    Parameters
    ----------
    X : ndarray
        Data points, shape (n_samples, n_features).
    eps : float
        Maximum distance between neighbors.

    Returns
    -------
    neighbors : list of ndarray
        neighbors[i] contains indices of points within eps of point i.

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [0.5, 0.0], [3.0, 0.0]])
    >>> neighbors = compute_neighbors(X, eps=1.0)
    >>> 0 in neighbors[1] and 1 in neighbors[0]  # Points 0 and 1 are neighbors
    True
    >>> 2 in neighbors[0]  # Point 2 is far from point 0
    False
    """
    n_samples = X.shape[0]

    # Use JIT-compiled distance matrix computation
    dist_matrix = _compute_distance_matrix(X)

    neighbors = []
    for i in range(n_samples):
        neighbor_indices = np.where(dist_matrix[i] <= eps)[0]
        neighbors.append(neighbor_indices)

    return neighbors


def dbscan(
    X: ArrayLike,
    eps: float = 0.5,
    min_samples: int = 5,
) -> DBSCANResult:
    """
    DBSCAN clustering algorithm.

    Finds core samples of high density and expands clusters from them.
    Points that are in low-density regions are marked as noise.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    eps : float
        Maximum distance between two samples to be considered neighbors.
        Default 0.5.
    min_samples : int
        Minimum number of samples in a neighborhood to form a core point.
        Default 5.

    Returns
    -------
    result : DBSCANResult
        Clustering result with labels and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> # Two dense clusters with noise
    >>> rng = np.random.default_rng(42)
    >>> cluster1 = rng.normal(0, 0.3, (30, 2))
    >>> cluster2 = rng.normal(3, 0.3, (30, 2))
    >>> noise = rng.uniform(-2, 5, (5, 2))
    >>> X = np.vstack([cluster1, cluster2, noise])
    >>> result = dbscan(X, eps=0.5, min_samples=5)
    >>> result.n_clusters
    2

    Notes
    -----
    A point is a core point if it has at least min_samples points
    within distance eps (including itself). A point is reachable from
    a core point if it is within eps distance. Noise points are not
    reachable from any core point.
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples == 0:
        return DBSCANResult(
            labels=np.array([], dtype=np.intp),
            n_clusters=0,
            core_sample_indices=np.array([], dtype=np.intp),
            n_noise=0,
        )

    # Compute neighborhoods
    neighbors = compute_neighbors(X, eps)

    # Identify core points
    core_samples = np.array(
        [i for i in range(n_samples) if len(neighbors[i]) >= min_samples], dtype=np.intp
    )

    core_set = set(core_samples)

    # Initialize labels (-1 = unvisited/noise)
    labels = np.full(n_samples, -1, dtype=np.intp)

    # Cluster expansion
    cluster_id = 0

    for i in core_samples:
        if labels[i] != -1:
            # Already assigned
            continue

        # Start new cluster with BFS
        labels[i] = cluster_id
        queue: List[int] = [i]
        visited: Set[int] = {i}

        while queue:
            current = queue.pop(0)

            for neighbor in neighbors[current]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)

                if labels[neighbor] == -1:
                    # Assign to cluster
                    labels[neighbor] = cluster_id

                    # If neighbor is core point, expand from it
                    if neighbor in core_set:
                        queue.append(neighbor)

        cluster_id += 1

    n_noise = np.sum(labels == -1)

    return DBSCANResult(
        labels=labels,
        n_clusters=cluster_id,
        core_sample_indices=core_samples,
        n_noise=int(n_noise),
    )


def dbscan_predict(
    X_new: ArrayLike,
    X_train: ArrayLike,
    labels_train: ArrayLike,
    eps: float,
) -> NDArray[np.intp]:
    """
    Predict cluster labels for new points based on trained DBSCAN.

    Assigns new points to the cluster of the nearest core point
    within eps distance, or -1 if no core point is within range.

    Parameters
    ----------
    X_new : array_like
        New data points, shape (n_new, n_features).
    X_train : array_like
        Training data points, shape (n_train, n_features).
    labels_train : array_like
        Cluster labels from training, shape (n_train,).
    eps : float
        Maximum distance threshold.

    Returns
    -------
    labels : ndarray
        Predicted cluster labels, shape (n_new,).
        -1 indicates no cluster assignment.

    Examples
    --------
    >>> # After running dbscan on X_train
    >>> X_new = np.array([[0.1, 0.1], [10.0, 10.0]])
    >>> labels_new = dbscan_predict(X_new, X_train, result.labels, eps=0.5)
    """
    X_new = np.asarray(X_new, dtype=np.float64)
    X_train = np.asarray(X_train, dtype=np.float64)
    labels_train = np.asarray(labels_train, dtype=np.intp)

    n_new = X_new.shape[0]
    labels = np.full(n_new, -1, dtype=np.intp)

    # Find non-noise points in training data
    valid_mask = labels_train >= 0

    if not np.any(valid_mask):
        return labels

    X_valid = X_train[valid_mask]
    labels_valid = labels_train[valid_mask]

    for i in range(n_new):
        # Compute distances to all valid training points
        distances = np.sqrt(np.sum((X_valid - X_new[i]) ** 2, axis=1))

        # Find nearest point within eps
        within_eps = distances <= eps
        if np.any(within_eps):
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] <= eps:
                labels[i] = labels_valid[nearest_idx]

    return labels


__all__ = [
    "DBSCANResult",
    "compute_neighbors",
    "dbscan",
    "dbscan_predict",
]
