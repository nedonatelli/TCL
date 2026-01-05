"""
Hierarchical (Agglomerative) Clustering.

Hierarchical clustering builds a tree of clusters by iteratively
merging the closest pairs. This is useful for track fusion and
understanding cluster relationships.

References
----------
.. [1] S. C. Johnson, "Hierarchical clustering schemes,"
       Psychometrika, 1967.
"""

from enum import Enum
from typing import Any, List, Literal, NamedTuple, Optional

import numpy as np
from numba import njit
from numpy.typing import ArrayLike, NDArray


class LinkageType(Enum):
    """Linkage methods for hierarchical clustering."""

    SINGLE = "single"  # Minimum distance
    COMPLETE = "complete"  # Maximum distance
    AVERAGE = "average"  # Average distance
    WARD = "ward"  # Ward's minimum variance


class DendrogramNode(NamedTuple):
    """A node in the dendrogram (merge tree).

    Attributes
    ----------
    left : int
        Index of left child (negative for original samples).
    right : int
        Index of right child (negative for original samples).
    distance : float
        Distance/dissimilarity at which merge occurred.
    count : int
        Number of original samples in this cluster.
    """

    left: int
    right: int
    distance: float
    count: int


class HierarchicalResult(NamedTuple):
    """Result of hierarchical clustering.

    Attributes
    ----------
    labels : ndarray
        Cluster labels for each point, shape (n_samples,).
    n_clusters : int
        Number of clusters.
    linkage_matrix : ndarray
        Linkage matrix of shape (n_samples-1, 4).
        Each row [i, j, dist, count] represents a merge.
    dendrogram : list of DendrogramNode
        List of merge operations.
    """

    labels: NDArray[np.intp]
    n_clusters: int
    linkage_matrix: NDArray[np.floating]
    dendrogram: List[DendrogramNode]


@njit(cache=True)
def _compute_distance_matrix_jit(X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """JIT-compiled pairwise Euclidean distance computation."""
    n = X.shape[0]
    n_features = X.shape[1]
    distances = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1, n):
            d = 0.0
            for k in range(n_features):
                diff = X[i, k] - X[j, k]
                d += diff * diff
            d = np.sqrt(d)
            distances[i, j] = d
            distances[j, i] = d

    return distances


def compute_distance_matrix(
    X: NDArray[np.floating],
) -> NDArray[np.floating]:
    """
    Compute pairwise Euclidean distance matrix.

    Parameters
    ----------
    X : ndarray
        Data points, shape (n_samples, n_features).

    Returns
    -------
    distances : ndarray
        Distance matrix, shape (n_samples, n_samples).

    Examples
    --------
    >>> X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> D = compute_distance_matrix(X)
    >>> D.shape
    (3, 3)
    >>> D[0, 1]  # Distance between points 0 and 1
    1.0
    """
    X = np.asarray(X, dtype=np.float64)
    return _compute_distance_matrix_jit(X)


def _single_linkage(
    dist_i: NDArray[Any],
    dist_j: NDArray[Any],
    size_i: int,
    size_j: int,
) -> NDArray[Any]:
    """Single linkage: minimum of distances."""
    return np.minimum(dist_i, dist_j)


def _complete_linkage(
    dist_i: NDArray[Any],
    dist_j: NDArray[Any],
    size_i: int,
    size_j: int,
) -> NDArray[Any]:
    """Complete linkage: maximum of distances."""
    return np.maximum(dist_i, dist_j)


def _average_linkage(
    dist_i: NDArray[Any],
    dist_j: NDArray[Any],
    size_i: int,
    size_j: int,
) -> NDArray[Any]:
    """Average linkage: weighted average of distances."""
    return (size_i * dist_i + size_j * dist_j) / (size_i + size_j)


def _ward_linkage(
    dist_i: NDArray[Any],
    dist_j: NDArray[Any],
    size_i: int,
    size_j: int,
    size_k: NDArray[Any],
    dist_ij: float,
) -> NDArray[Any]:
    """Ward's linkage: minimum variance merge."""
    total = size_i + size_j + size_k
    return np.sqrt(
        (
            (size_i + size_k) * dist_i**2
            + (size_j + size_k) * dist_j**2
            - size_k * dist_ij**2
        )
        / total
    )


def agglomerative_clustering(
    X: ArrayLike,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
    linkage: Literal["single", "complete", "average", "ward"] = "ward",
) -> HierarchicalResult:
    """
    Agglomerative hierarchical clustering.

    Recursively merges pairs of clusters that minimize the linkage
    criterion until the desired number of clusters is reached.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    n_clusters : int, optional
        Number of clusters to find. If None, uses distance_threshold.
    distance_threshold : float, optional
        Distance threshold for cluster merging. Merging stops when
        the minimum linkage distance exceeds this. If None, uses n_clusters.
    linkage : {'single', 'complete', 'average', 'ward'}
        Linkage criterion. Default 'ward'.

    Returns
    -------
    result : HierarchicalResult
        Clustering result with labels and dendrogram.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal(0, 0.5, (20, 2)),
    ...                rng.normal(3, 0.5, (20, 2))])
    >>> result = agglomerative_clustering(X, n_clusters=2)
    >>> result.n_clusters
    2

    Notes
    -----
    Linkage methods:
    - 'single': Distance between closest points (can create chains)
    - 'complete': Distance between farthest points (tends to create compact clusters)
    - 'average': Average distance between all pairs
    - 'ward': Minimizes within-cluster variance (requires Euclidean distance)
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples = X.shape[0]

    if n_samples == 0:
        return HierarchicalResult(
            labels=np.array([], dtype=np.intp),
            n_clusters=0,
            linkage_matrix=np.zeros((0, 4)),
            dendrogram=[],
        )

    if n_samples == 1:
        return HierarchicalResult(
            labels=np.array([0], dtype=np.intp),
            n_clusters=1,
            linkage_matrix=np.zeros((0, 4)),
            dendrogram=[],
        )

    if n_clusters is None and distance_threshold is None:
        n_clusters = 2

    # Compute initial distance matrix
    distances = compute_distance_matrix(X)

    # Initialize clusters
    # cluster_id -> list of original sample indices
    clusters = {i: [i] for i in range(n_samples)}
    cluster_sizes = {i: 1 for i in range(n_samples)}

    # Active cluster IDs
    active = set(range(n_samples))

    # Linkage matrix and dendrogram
    linkage_matrix = []
    dendrogram = []

    # Current cluster distance matrix (upper triangular, inf on diagonal)
    cluster_dist = distances.copy()
    np.fill_diagonal(cluster_dist, np.inf)

    next_cluster_id = n_samples

    # Merge until stopping criterion
    while len(active) > 1:
        # Check stopping criteria
        if n_clusters is not None and len(active) <= n_clusters:
            break

        # Find minimum distance pair among active clusters
        min_dist = np.inf
        merge_i, merge_j = -1, -1

        active_list = list(active)
        for idx_a, i in enumerate(active_list):
            for j in active_list[idx_a + 1 :]:
                if cluster_dist[i, j] < min_dist:
                    min_dist = cluster_dist[i, j]
                    merge_i, merge_j = i, j

        if merge_i == -1 or min_dist == np.inf:
            break

        if distance_threshold is not None and min_dist > distance_threshold:
            break

        # Ensure merge_i < merge_j for consistency
        if merge_i > merge_j:
            merge_i, merge_j = merge_j, merge_i

        # Record merge
        size_new = cluster_sizes[merge_i] + cluster_sizes[merge_j]
        linkage_matrix.append([merge_i, merge_j, min_dist, size_new])
        dendrogram.append(
            DendrogramNode(
                left=merge_i,
                right=merge_j,
                distance=min_dist,
                count=size_new,
            )
        )

        # Create new cluster
        new_cluster_id = next_cluster_id
        next_cluster_id += 1

        clusters[new_cluster_id] = clusters[merge_i] + clusters[merge_j]
        cluster_sizes[new_cluster_id] = size_new

        # Update distance matrix for new cluster
        # Expand matrix if needed
        if new_cluster_id >= cluster_dist.shape[0]:
            new_size = new_cluster_id + 1
            new_dist = np.full((new_size, new_size), np.inf)
            old_size = cluster_dist.shape[0]
            new_dist[:old_size, :old_size] = cluster_dist
            cluster_dist = new_dist

        # Compute distances from new cluster to all other active clusters
        for k in active:
            if k == merge_i or k == merge_j:
                continue

            if linkage == "single":
                new_d = _single_linkage(
                    cluster_dist[merge_i, k],
                    cluster_dist[merge_j, k],
                    cluster_sizes[merge_i],
                    cluster_sizes[merge_j],
                )
            elif linkage == "complete":
                new_d = _complete_linkage(
                    cluster_dist[merge_i, k],
                    cluster_dist[merge_j, k],
                    cluster_sizes[merge_i],
                    cluster_sizes[merge_j],
                )
            elif linkage == "average":
                new_d = _average_linkage(
                    cluster_dist[merge_i, k],
                    cluster_dist[merge_j, k],
                    cluster_sizes[merge_i],
                    cluster_sizes[merge_j],
                )
            elif linkage == "ward":
                new_d = _ward_linkage(
                    cluster_dist[merge_i, k],
                    cluster_dist[merge_j, k],
                    cluster_sizes[merge_i],
                    cluster_sizes[merge_j],
                    np.array([cluster_sizes[k]]),
                    cluster_dist[merge_i, merge_j],
                )[0]
            else:
                raise ValueError(f"Unknown linkage: {linkage}")

            cluster_dist[new_cluster_id, k] = new_d
            cluster_dist[k, new_cluster_id] = new_d

        # Update active set
        active.remove(merge_i)
        active.remove(merge_j)
        active.add(new_cluster_id)

        # Clean up old clusters
        del clusters[merge_i]
        del clusters[merge_j]
        del cluster_sizes[merge_i]
        del cluster_sizes[merge_j]

    # Assign labels based on final clusters
    labels = np.zeros(n_samples, dtype=np.intp)
    for cluster_label, cluster_id in enumerate(active):
        for sample_idx in clusters[cluster_id]:
            labels[sample_idx] = cluster_label

    # Convert linkage matrix to array
    if linkage_matrix:
        linkage_array = np.array(linkage_matrix)
    else:
        linkage_array = np.zeros((0, 4))

    return HierarchicalResult(
        labels=labels,
        n_clusters=len(active),
        linkage_matrix=linkage_array,
        dendrogram=dendrogram,
    )


def cut_dendrogram(
    linkage_matrix: ArrayLike,
    n_samples: int,
    n_clusters: Optional[int] = None,
    distance_threshold: Optional[float] = None,
) -> NDArray[np.intp]:
    """
    Cut a dendrogram to obtain cluster labels.

    Parameters
    ----------
    linkage_matrix : array_like
        Linkage matrix from agglomerative_clustering, shape (n-1, 4).
    n_samples : int
        Number of original samples.
    n_clusters : int, optional
        Number of clusters to form.
    distance_threshold : float, optional
        Maximum linkage distance.

    Returns
    -------
    labels : ndarray
        Cluster labels, shape (n_samples,).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal(0, 0.5, (10, 2)), rng.normal(3, 0.5, (10, 2))])
    >>> result = agglomerative_clustering(X)
    >>> labels = cut_dendrogram(result.linkage_matrix, n_samples=20, n_clusters=2)
    >>> len(np.unique(labels))
    2
    """
    linkage_matrix = np.asarray(linkage_matrix)

    if len(linkage_matrix) == 0:
        return np.zeros(n_samples, dtype=np.intp)

    if n_clusters is None and distance_threshold is None:
        n_clusters = 2

    # Build cluster membership by replaying merges
    # Initially each sample is its own cluster
    cluster_members = {i: {i} for i in range(n_samples)}
    next_id = n_samples

    n_current_clusters = n_samples

    for row in linkage_matrix:
        left, right, dist, _ = int(row[0]), int(row[1]), row[2], int(row[3])

        # Check stopping criteria
        if n_clusters is not None and n_current_clusters <= n_clusters:
            break
        if distance_threshold is not None and dist > distance_threshold:
            break

        # Merge clusters
        new_members = cluster_members[left] | cluster_members[right]
        cluster_members[next_id] = new_members
        del cluster_members[left]
        del cluster_members[right]

        next_id += 1
        n_current_clusters -= 1

    # Assign labels
    labels = np.zeros(n_samples, dtype=np.intp)
    for label, (cluster_id, members) in enumerate(cluster_members.items()):
        for sample_idx in members:
            labels[sample_idx] = label

    return labels


def fcluster(
    linkage_matrix: ArrayLike,
    n_samples: int,
    t: float,
    criterion: Literal["distance", "maxclust"] = "distance",
) -> NDArray[np.intp]:
    """
    Form flat clusters from hierarchical clustering.

    Compatible interface with scipy.cluster.hierarchy.fcluster.

    Parameters
    ----------
    linkage_matrix : array_like
        Linkage matrix from agglomerative_clustering.
    n_samples : int
        Number of original samples.
    t : float
        Threshold for forming clusters.
    criterion : {'distance', 'maxclust'}
        'distance': t is maximum cophenetic distance
        'maxclust': t is maximum number of clusters

    Returns
    -------
    labels : ndarray
        Cluster labels (1-indexed for scipy compatibility).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal(0, 0.5, (10, 2)), rng.normal(3, 0.5, (10, 2))])
    >>> result = agglomerative_clustering(X)
    >>> labels = fcluster(result.linkage_matrix, n_samples=20, t=2, criterion='maxclust')
    >>> labels.min()  # 1-indexed
    1
    """
    if criterion == "distance":
        labels = cut_dendrogram(linkage_matrix, n_samples, distance_threshold=t)
    elif criterion == "maxclust":
        labels = cut_dendrogram(linkage_matrix, n_samples, n_clusters=int(t))
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # Convert to 1-indexed for scipy compatibility
    return labels + 1


__all__ = [
    "LinkageType",
    "DendrogramNode",
    "HierarchicalResult",
    "compute_distance_matrix",
    "agglomerative_clustering",
    "cut_dendrogram",
    "fcluster",
]
