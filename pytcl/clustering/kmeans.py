"""
K-means clustering for tracking applications.

This module provides K-means clustering with K-means++ initialization,
commonly used in track clustering and data association scenarios.

References
----------
.. [1] D. Arthur and S. Vassilvitskii, "k-means++: The Advantages of
       Careful Seeding," SODA 2007.
"""

from typing import Any, Literal, NamedTuple, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.distance import cdist


class KMeansResult(NamedTuple):
    """Result of K-means clustering.

    Attributes
    ----------
    labels : ndarray
        Cluster assignment for each point, shape (n_samples,).
    centers : ndarray
        Cluster centroids, shape (n_clusters, n_features).
    inertia : float
        Sum of squared distances to nearest centroid.
    n_iter : int
        Number of iterations run.
    converged : bool
        Whether the algorithm converged before max_iter.
    """

    labels: NDArray[np.intp]
    centers: NDArray[np.floating]
    inertia: float
    n_iter: int
    converged: bool


def kmeans_plusplus_init(
    X: ArrayLike,
    n_clusters: int,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.floating]:
    """
    K-means++ initialization for cluster centers.

    Selects initial centers by sampling proportional to squared distance
    from nearest existing center, providing better initialization than
    random selection.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    centers : ndarray
        Initial cluster centers, shape (n_clusters, n_features).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])
    >>> centers = kmeans_plusplus_init(X, n_clusters=2, rng=np.random.default_rng(42))
    >>> centers.shape
    (2, 2)
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if rng is None:
        rng = np.random.default_rng()

    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) > n_samples ({n_samples})")

    centers = np.empty((n_clusters, n_features), dtype=np.float64)

    # First center: random choice
    idx = rng.integers(0, n_samples)
    centers[0] = X[idx].copy()

    # Subsequent centers: sample proportional to D^2
    for k in range(1, n_clusters):
        # Compute squared distances to nearest center (vectorized via cdist)
        distances_sq = cdist(X, centers[:k], metric="sqeuclidean").min(axis=1)

        # Sample proportional to D^2
        probs = distances_sq / distances_sq.sum()
        idx = rng.choice(n_samples, p=probs)
        centers[k] = X[idx].copy()

    return centers


def assign_clusters(
    X: ArrayLike,
    centers: ArrayLike,
) -> tuple[NDArray[np.intp], float]:
    """
    Assign each point to its nearest cluster center.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    centers : array_like
        Cluster centers, shape (n_clusters, n_features).

    Returns
    -------
    labels : ndarray
        Cluster assignment for each point, shape (n_samples,).
    inertia : float
        Sum of squared distances to assigned centers.

    Examples
    --------
    >>> X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
    >>> centers = np.array([[0.5, 0], [10.5, 10]])
    >>> labels, inertia = assign_clusters(X, centers)
    >>> labels
    array([0, 0, 1, 1])
    """
    X = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)

    n_samples = X.shape[0]

    # Compute squared distances to all centers (vectorized via cdist)
    distances_sq = cdist(X, centers, metric="sqeuclidean")

    # Assign to nearest center
    labels = np.argmin(distances_sq, axis=1).astype(np.intp)

    # Compute inertia
    inertia = np.sum(distances_sq[np.arange(n_samples), labels])

    return labels, inertia


def update_centers(
    X: ArrayLike,
    labels: ArrayLike,
    n_clusters: int,
) -> NDArray[np.floating]:
    """
    Update cluster centers as mean of assigned points.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    labels : array_like
        Cluster assignments, shape (n_samples,).
    n_clusters : int
        Number of clusters.

    Returns
    -------
    centers : ndarray
        Updated cluster centers, shape (n_clusters, n_features).
        Empty clusters retain their previous position (zeros).
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.intp)

    n_features = X.shape[1]
    centers = np.zeros((n_clusters, n_features), dtype=np.float64)

    for k in range(n_clusters):
        mask = labels == k
        if np.any(mask):
            centers[k] = X[mask].mean(axis=0)

    return centers


def kmeans(
    X: ArrayLike,
    n_clusters: int,
    init: Union[Literal["kmeans++", "random"], ArrayLike] = "kmeans++",
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> KMeansResult:
    """
    K-means clustering algorithm.

    Partitions data into k clusters by minimizing within-cluster
    sum of squared distances.

    Parameters
    ----------
    X : array_like
        Data points, shape (n_samples, n_features).
    n_clusters : int
        Number of clusters.
    init : {'kmeans++', 'random'} or array_like
        Initialization method or explicit initial centers.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance. Algorithm stops when center
        movement is below this threshold.
    n_init : int
        Number of random initializations. Best result is returned.
    rng : numpy.random.Generator, optional
        Random number generator.

    Returns
    -------
    result : KMeansResult
        Clustering result with labels, centers, and diagnostics.

    Examples
    --------
    >>> import numpy as np
    >>> # Two well-separated clusters
    >>> rng = np.random.default_rng(42)
    >>> X1 = rng.normal(0, 0.5, (50, 2))
    >>> X2 = rng.normal(5, 0.5, (50, 2))
    >>> X = np.vstack([X1, X2])
    >>> result = kmeans(X, n_clusters=2, rng=rng)
    >>> result.converged
    True
    >>> len(np.unique(result.labels))
    2
    """
    X = np.asarray(X, dtype=np.float64)
    n_samples, n_features = X.shape

    if rng is None:
        rng = np.random.default_rng()

    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if n_clusters > n_samples:
        raise ValueError(f"n_clusters ({n_clusters}) > n_samples ({n_samples})")

    # Check if initial centers are provided
    if isinstance(init, np.ndarray) or (
        isinstance(init, (list, tuple)) and len(init) > 0
    ):
        init_centers = np.asarray(init, dtype=np.float64)
        if init_centers.shape != (n_clusters, n_features):
            raise ValueError(
                f"init shape {init_centers.shape} doesn't match "
                f"expected ({n_clusters}, {n_features})"
            )
        # Run single time with provided centers
        return _kmeans_single(X, init_centers, max_iter, tol)

    # Multiple random initializations
    best_result: Optional[KMeansResult] = None
    best_inertia = np.inf

    for _ in range(n_init):
        # Initialize centers
        if init == "kmeans++":
            centers = kmeans_plusplus_init(X, n_clusters, rng)
        elif init == "random":
            indices = rng.choice(n_samples, size=n_clusters, replace=False)
            centers = X[indices].copy()
        else:
            raise ValueError(f"Unknown init method: {init}")

        # Run K-means
        result = _kmeans_single(X, centers, max_iter, tol)

        if result.inertia < best_inertia:
            best_inertia = result.inertia
            best_result = result

    assert best_result is not None
    return best_result


def _kmeans_single(
    X: NDArray[np.floating],
    centers: NDArray[np.floating],
    max_iter: int,
    tol: float,
) -> KMeansResult:
    """Run K-means with given initial centers."""
    n_clusters = centers.shape[0]
    centers = centers.copy()

    labels, inertia = assign_clusters(X, centers)

    for iteration in range(max_iter):
        # Update centers
        new_centers = update_centers(X, labels, n_clusters)

        # Handle empty clusters: keep old center
        for k in range(n_clusters):
            if np.all(new_centers[k] == 0) and not np.any(labels == k):
                new_centers[k] = centers[k]

        # Check convergence
        center_shift = np.sqrt(np.sum((new_centers - centers) ** 2))
        centers = new_centers

        # Reassign
        labels, inertia = assign_clusters(X, centers)

        if center_shift < tol:
            return KMeansResult(
                labels=labels,
                centers=centers,
                inertia=inertia,
                n_iter=iteration + 1,
                converged=True,
            )

    return KMeansResult(
        labels=labels,
        centers=centers,
        inertia=inertia,
        n_iter=max_iter,
        converged=False,
    )


def kmeans_elbow(
    X: ArrayLike,
    k_range: Optional[range] = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Compute K-means for a range of k values for elbow method.

    Parameters
    ----------
    X : array_like
        Data points.
    k_range : range, optional
        Range of k values to try. Default is range(1, 11).
    **kwargs
        Additional arguments passed to kmeans().

    Returns
    -------
    results : dict
        Dictionary with keys 'k_values' and 'inertias'.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = np.vstack([rng.normal(0, 1, (50, 2)), rng.normal(5, 1, (50, 2))])
    >>> results = kmeans_elbow(X, k_range=range(1, 6), rng=rng)
    >>> len(results['inertias'])
    5
    """
    X = np.asarray(X, dtype=np.float64)

    if k_range is None:
        k_range = range(1, min(11, X.shape[0] + 1))

    k_values = []
    inertias = []

    for k in k_range:
        if k > X.shape[0]:
            break
        result = kmeans(X, n_clusters=k, **kwargs)
        k_values.append(k)
        inertias.append(result.inertia)

    return {"k_values": k_values, "inertias": inertias}


__all__ = [
    "KMeansResult",
    "kmeans_plusplus_init",
    "assign_clusters",
    "update_centers",
    "kmeans",
    "kmeans_elbow",
]
