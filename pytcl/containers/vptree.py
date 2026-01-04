"""
Vantage Point Tree (VP-tree) implementation.

VP-trees are metric trees that partition data based on distance to
selected vantage points. They are effective for nearest neighbor
search in metric spaces, particularly with high-dimensional data.

References
----------
.. [1] P. N. Yianilos, "Data structures and algorithms for nearest
       neighbor search in general metric spaces," SODA 1993.
"""

import logging
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.containers.base import MetricSpatialIndex, validate_query_input

# Module logger
_logger = logging.getLogger("pytcl.containers.vptree")


class VPTreeResult(NamedTuple):
    """Result of VP-tree query.

    Attributes
    ----------
    indices : ndarray
        Indices of nearest neighbors.
    distances : ndarray
        Distances to nearest neighbors.
    """

    indices: NDArray[np.intp]
    distances: NDArray[np.floating]


class VPNode:
    """Node in a VP-tree.

    Attributes
    ----------
    index : int
        Index of the vantage point in the original data.
    radius : float
        Median distance to vantage point (splitting threshold).
    left : VPNode or None
        Left subtree (points closer than radius).
    right : VPNode or None
        Right subtree (points farther than radius).
    """

    __slots__ = ["index", "radius", "left", "right"]

    def __init__(self, index: int, radius: float = 0.0):
        self.index = index
        self.radius = radius
        self.left: Optional["VPNode"] = None
        self.right: Optional["VPNode"] = None


class VPTree(MetricSpatialIndex):
    """
    Vantage Point Tree for metric space nearest neighbor search.

    A VP-tree recursively partitions space by selecting a vantage point
    and dividing remaining points into those closer than a threshold
    (median distance) and those farther.

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).
    metric : callable, optional
        Distance function metric(x, y) -> float.
        Default is Euclidean distance.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.random.rand(100, 3)
    >>> tree = VPTree(points)
    >>> result = tree.query(points[:5], k=3)
    >>> result.indices.shape
    (5, 3)

    Notes
    -----
    VP-trees can use any metric distance function, making them useful
    for non-Euclidean spaces (e.g., edit distance for strings, geodesic
    distance on manifolds).

    Query complexity is O(log n) on average but can degrade to O(n)
    for pathological distance distributions.

    See Also
    --------
    MetricSpatialIndex : Abstract base class for metric-based spatial indices.
    CoverTree : Alternative metric space index with theoretical guarantees.
    """

    def __init__(
        self,
        data: ArrayLike,
        metric: Optional[
            Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float]
        ] = None,
    ):
        super().__init__(data, metric)

        # Build tree
        indices = np.arange(self.n_samples)
        self.root = self._build_tree(indices)
        metric_name = metric.__name__ if metric else "euclidean"
        _logger.debug("VPTree built with metric=%s", metric_name)

    def _build_tree(self, indices: NDArray[np.intp]) -> Optional[VPNode]:
        """Recursively build the VP-tree."""
        if len(indices) == 0:
            return None

        if len(indices) == 1:
            return VPNode(indices[0], 0.0)

        # Select vantage point (use first point for simplicity)
        # Better strategies: random selection, spread-based selection
        vp_idx = indices[0]
        vp = self.data[vp_idx]

        # Compute distances to vantage point
        remaining = indices[1:]
        distances = np.array([self.metric(vp, self.data[i]) for i in remaining])

        # Split at median distance
        median_dist = float(np.median(distances))

        # Partition into left (closer) and right (farther)
        left_mask = distances <= median_dist
        right_mask = ~left_mask

        left_indices = remaining[left_mask]
        right_indices = remaining[right_mask]

        node = VPNode(vp_idx, median_dist)
        node.left = self._build_tree(left_indices)
        node.right = self._build_tree(right_indices)

        return node

    def query(
        self,
        X: ArrayLike,
        k: int = 1,
    ) -> VPTreeResult:
        """
        Query the tree for k nearest neighbors.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        k : int, optional
            Number of nearest neighbors. Default 1.

        Returns
        -------
        result : VPTreeResult
            Indices and distances of k nearest neighbors.
        """
        X = validate_query_input(X, self.n_features)
        n_queries = X.shape[0]

        all_indices = np.zeros((n_queries, k), dtype=np.intp)
        all_distances = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            neighbors = self._query_single(X[i], k)
            n_found = len(neighbors)
            if n_found > 0:
                indices, distances = zip(*neighbors)
                all_indices[i, :n_found] = indices
                all_distances[i, :n_found] = distances

        return VPTreeResult(indices=all_indices, distances=all_distances)

    def _query_single(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a single query."""
        # List of (index, distance) tuples, maintained sorted
        neighbors: List[Tuple[int, float]] = []
        tau = np.inf  # Current kth nearest distance

        def search(node: Optional[VPNode]) -> None:
            nonlocal tau

            if node is None:
                return

            # Distance to vantage point
            dist = self.metric(query, self.data[node.index])

            # Check if vantage point is a neighbor
            if dist < tau:
                if len(neighbors) < k:
                    neighbors.append((node.index, dist))
                    neighbors.sort(key=lambda x: x[1])
                    if len(neighbors) == k:
                        tau = neighbors[-1][1]
                else:
                    neighbors[-1] = (node.index, dist)
                    neighbors.sort(key=lambda x: x[1])
                    tau = neighbors[-1][1]

            # Decide which subtrees to search
            if dist < node.radius:
                # Query is closer to vantage point than radius
                # Search left first (closer points)
                if dist - tau <= node.radius:
                    search(node.left)
                if dist + tau >= node.radius:
                    search(node.right)
            else:
                # Query is farther from vantage point than radius
                # Search right first (farther points)
                if dist + tau >= node.radius:
                    search(node.right)
                if dist - tau <= node.radius:
                    search(node.left)

        search(self.root)
        return neighbors

    def query_radius(
        self,
        X: ArrayLike,
        r: float,
    ) -> List[List[int]]:
        """
        Find all points within radius r of query points.

        Parameters
        ----------
        X : array_like
            Query points.
        r : float
            Query radius.

        Returns
        -------
        indices : list of lists
            For each query, list of indices within radius.
        """
        X = validate_query_input(X, self.n_features)
        n_queries = X.shape[0]
        results: List[List[int]] = []

        for i in range(n_queries):
            indices = self._query_radius_single(X[i], r)
            results.append(indices)

        return results

    def _query_radius_single(
        self,
        query: NDArray[np.floating],
        r: float,
    ) -> List[int]:
        """Find all points within radius r of query."""
        indices: List[int] = []

        def search(node: Optional[VPNode]) -> None:
            if node is None:
                return

            dist = self.metric(query, self.data[node.index])

            # Check vantage point
            if dist <= r:
                indices.append(node.index)

            # Check subtrees
            if dist - r <= node.radius:
                search(node.left)
            if dist + r >= node.radius:
                search(node.right)

        search(self.root)
        return indices


__all__ = [
    "VPTreeResult",
    "VPNode",
    "VPTree",
]
