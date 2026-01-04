"""
K-D Tree implementation.

A k-dimensional tree (k-d tree) is a space-partitioning data structure
for organizing points in k-dimensional space. Useful for efficient
nearest neighbor searches in tracking applications.

References
----------
.. [1] J. L. Bentley, "Multidimensional binary search trees used for
       associative searching," Communications of the ACM, 1975.
.. [2] J. H. Friedman, J. L. Bentley, R. A. Finkel, "An Algorithm for
       Finding Best Matches in Logarithmic Expected Time," ACM TOMS, 1977.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.containers.base import NearestNeighborResult  # Backward compatibility alias
from pytcl.containers.base import (
    BaseSpatialIndex,
    NeighborResult,
    validate_query_input,
)

# Module logger
_logger = logging.getLogger("pytcl.containers.kd_tree")


class KDNode:
    """A node in the k-d tree.

    Attributes
    ----------
    point : ndarray
        The point stored at this node.
    index : int
        Original index of this point in the data array.
    split_dim : int
        The dimension used for splitting at this node.
    left : KDNode or None
        Left child (points with smaller split dimension value).
    right : KDNode or None
        Right child (points with larger split dimension value).
    """

    __slots__ = ["point", "index", "split_dim", "left", "right"]

    def __init__(
        self,
        point: NDArray[np.floating],
        index: int,
        split_dim: int,
    ):
        self.point = point
        self.index = index
        self.split_dim = split_dim
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree(BaseSpatialIndex):
    """
    K-D Tree for efficient spatial queries.

    A k-d tree recursively partitions space by splitting along different
    dimensions at each level. This enables O(log n) average-case nearest
    neighbor queries.

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).
    leaf_size : int, optional
        Maximum number of points in a leaf node. Default 10.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> tree = KDTree(points)
    >>> result = tree.query(np.array([[0.1, 0.1]]), k=2)
    >>> result.indices
    array([[0, 1]])

    Notes
    -----
    The tree is balanced by choosing the median point along the split
    dimension at each level, giving O(n log n) construction time.

    Query complexity is O(log n) on average for nearest neighbor search,
    though worst case is O(n) for highly unbalanced queries.

    See Also
    --------
    BaseSpatialIndex : Abstract base class defining the spatial index interface.
    BallTree : Alternative spatial index using hyperspheres.
    """

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 10,
    ):
        super().__init__(data)
        self.leaf_size = leaf_size

        # Build the tree
        indices = np.arange(self.n_samples)
        self.root = self._build_tree(indices, depth=0)
        _logger.debug("KDTree built with leaf_size=%d", leaf_size)

    def _build_tree(
        self,
        indices: NDArray[np.intp],
        depth: int,
    ) -> Optional[KDNode]:
        """Recursively build the k-d tree."""
        if len(indices) == 0:
            return None

        # Choose split dimension (cycle through dimensions)
        split_dim = depth % self.n_features

        # Sort indices by split dimension and find median
        sorted_indices = indices[np.argsort(self.data[indices, split_dim])]
        median_idx = len(sorted_indices) // 2

        # Create node
        node_index = sorted_indices[median_idx]
        node = KDNode(
            point=self.data[node_index],
            index=node_index,
            split_dim=split_dim,
        )

        # Recursively build subtrees
        node.left = self._build_tree(sorted_indices[:median_idx], depth + 1)
        node.right = self._build_tree(sorted_indices[median_idx + 1 :], depth + 1)

        return node

    def query(
        self,
        X: ArrayLike,
        k: int = 1,
    ) -> NeighborResult:
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
        result : NeighborResult
            Indices and distances of k nearest neighbors for each query.

        Examples
        --------
        >>> tree = KDTree(np.array([[0, 0], [1, 1], [2, 2]]))
        >>> result = tree.query([[0.5, 0.5]], k=2)
        >>> result.indices
        array([[0, 1]])
        """
        X = validate_query_input(X, self.n_features)
        n_queries = X.shape[0]
        _logger.debug("KDTree.query: %d queries, k=%d", n_queries, k)

        all_indices = np.zeros((n_queries, k), dtype=np.intp)
        all_distances = np.full((n_queries, k), np.inf)

        for i in range(n_queries):
            neighbors = self._query_single(X[i], k)
            n_found = len(neighbors)
            if n_found > 0:
                indices, distances = zip(*neighbors)
                all_indices[i, :n_found] = indices
                all_distances[i, :n_found] = distances

        return NeighborResult(indices=all_indices, distances=all_distances)

    def _query_single(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a single query point."""
        # List of (index, distance) tuples, maintained as a bounded heap
        neighbors: List[Tuple[int, float]] = []

        def _search(node: Optional[KDNode], depth: int) -> None:
            if node is None:
                return

            # Distance to current node
            dist = np.sqrt(np.sum((query - node.point) ** 2))

            # Add to neighbors if room or better than worst
            if len(neighbors) < k:
                neighbors.append((node.index, dist))
                neighbors.sort(key=lambda x: x[1])
            elif dist < neighbors[-1][1]:
                neighbors[-1] = (node.index, dist)
                neighbors.sort(key=lambda x: x[1])

            # Decide which subtree to search first
            split_dim = node.split_dim
            diff = query[split_dim] - node.point[split_dim]

            if diff <= 0:
                first, second = node.left, node.right
            else:
                first, second = node.right, node.left

            # Search closer subtree first
            _search(first, depth + 1)

            # Check if we need to search the other subtree
            # Only if the splitting plane is closer than our worst neighbor
            if len(neighbors) < k or abs(diff) < neighbors[-1][1]:
                _search(second, depth + 1)

        _search(self.root, 0)
        return neighbors

    def query_radius(
        self,
        X: ArrayLike,
        r: float,
    ) -> List[List[int]]:
        """
        Query the tree for all points within radius r.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        r : float
            Query radius.

        Returns
        -------
        indices : list of lists
            For each query, a list of indices of points within radius r.

        Examples
        --------
        >>> tree = KDTree(np.array([[0, 0], [1, 0], [0, 1], [5, 5]]))
        >>> tree.query_radius([[0, 0]], r=1.5)
        [[0, 1, 2]]
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
        """Find all points within radius r of query point."""
        indices: List[int] = []

        def _search(node: Optional[KDNode]) -> None:
            if node is None:
                return

            # Distance to current node
            dist = np.sqrt(np.sum((query - node.point) ** 2))

            if dist <= r:
                indices.append(node.index)

            # Check if subtrees might contain points in radius
            split_dim = node.split_dim
            diff = query[split_dim] - node.point[split_dim]

            # Search appropriate subtrees
            if diff - r <= 0:
                _search(node.left)
            if diff + r >= 0:
                _search(node.right)

        _search(self.root)
        return indices

    # query_ball_point inherited from BaseSpatialIndex


class BallTree(BaseSpatialIndex):
    """
    Ball Tree for efficient spatial queries.

    A ball tree partitions space using hyperspheres (balls), which
    can be more efficient than k-d trees for high-dimensional data
    or non-Euclidean metrics.

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).
    leaf_size : int, optional
        Maximum number of points in a leaf node. Default 10.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    >>> tree = BallTree(points)
    >>> result = tree.query(np.array([[0.1, 0.1]]), k=2)

    Notes
    -----
    Ball trees have O(n log n) construction and O(log n) average-case
    query time. They can outperform k-d trees in high dimensions.

    See Also
    --------
    BaseSpatialIndex : Abstract base class defining the spatial index interface.
    KDTree : Alternative spatial index using axis-aligned splits.
    """

    def __init__(
        self,
        data: ArrayLike,
        leaf_size: int = 10,
    ):
        super().__init__(data)
        self.leaf_size = leaf_size

        # Build tree using indices
        self._indices = np.arange(self.n_samples)
        self._centroids: List[NDArray[np.floating]] = []
        self._radii: List[float] = []
        self._left: List[int] = []
        self._right: List[int] = []
        self._is_leaf: List[bool] = []
        self._leaf_indices: List[Optional[NDArray[np.intp]]] = []

        self._build_tree(self._indices)
        _logger.debug("BallTree built with leaf_size=%d", leaf_size)

    def _build_tree(
        self,
        indices: NDArray[np.intp],
    ) -> int:
        """Build the ball tree recursively."""
        node_id = len(self._centroids)

        points = self.data[indices]
        centroid = np.mean(points, axis=0)
        radius = float(np.max(np.sqrt(np.sum((points - centroid) ** 2, axis=1))))

        self._centroids.append(centroid)
        self._radii.append(radius)

        if len(indices) <= self.leaf_size:
            # Leaf node
            self._is_leaf.append(True)
            self._leaf_indices.append(indices.copy())
            self._left.append(-1)
            self._right.append(-1)
        else:
            # Internal node - split along dimension with largest spread
            spread = np.max(points, axis=0) - np.min(points, axis=0)
            split_dim = np.argmax(spread)

            # Split at median
            split_values = self.data[indices, split_dim]
            median = np.median(split_values)

            left_mask = split_values <= median
            right_mask = ~left_mask

            # Ensure non-empty splits
            if not np.any(left_mask) or not np.any(right_mask):
                mid = len(indices) // 2
                sorted_idx = indices[np.argsort(split_values)]
                left_indices = sorted_idx[:mid]
                right_indices = sorted_idx[mid:]
            else:
                left_indices = indices[left_mask]
                right_indices = indices[right_mask]

            self._is_leaf.append(False)
            self._leaf_indices.append(None)
            self._left.append(-1)  # Placeholder
            self._right.append(-1)  # Placeholder

            # Build subtrees
            left_id = self._build_tree(left_indices)
            right_id = self._build_tree(right_indices)

            self._left[node_id] = left_id
            self._right[node_id] = right_id

        return node_id

    def query(
        self,
        X: ArrayLike,
        k: int = 1,
    ) -> NeighborResult:
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
        result : NeighborResult
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

        return NeighborResult(indices=all_indices, distances=all_distances)

    # query_ball_point inherited from BaseSpatialIndex

    def query_radius(
        self,
        X: ArrayLike,
        r: float,
    ) -> List[List[int]]:
        """
        Query the tree for all points within radius r.

        Parameters
        ----------
        X : array_like
            Query points of shape (n_queries, n_features) or (n_features,).
        r : float
            Query radius.

        Returns
        -------
        indices : list of lists
            For each query, a list of indices of points within radius r.
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
        """Find all points within radius r of query point."""
        indices: List[int] = []

        def _search(node_id: int) -> None:
            if node_id < 0:
                return

            centroid = self._centroids[node_id]
            radius = self._radii[node_id]

            # Distance to ball surface
            dist_to_center = np.sqrt(np.sum((query - centroid) ** 2))

            # Prune if ball is farther than radius
            if dist_to_center - radius > r:
                return

            if self._is_leaf[node_id]:
                # Check all points in leaf
                leaf_indices = self._leaf_indices[node_id]
                if leaf_indices is not None:
                    for idx in leaf_indices:
                        dist = np.sqrt(np.sum((query - self.data[idx]) ** 2))
                        if dist <= r:
                            indices.append(idx)
            else:
                # Visit both children
                _search(self._left[node_id])
                _search(self._right[node_id])

        _search(0)
        return indices

    def _query_single(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a single point."""
        neighbors: List[Tuple[int, float]] = []

        def _search(node_id: int) -> None:
            if node_id < 0:
                return

            centroid = self._centroids[node_id]
            radius = self._radii[node_id]

            # Distance to ball surface
            dist_to_center = np.sqrt(np.sum((query - centroid) ** 2))

            # Prune if ball is farther than current worst neighbor
            if len(neighbors) >= k and dist_to_center - radius >= neighbors[-1][1]:
                return

            if self._is_leaf[node_id]:
                # Check all points in leaf
                leaf_indices = self._leaf_indices[node_id]
                if leaf_indices is not None:
                    for idx in leaf_indices:
                        dist = np.sqrt(np.sum((query - self.data[idx]) ** 2))
                        if len(neighbors) < k:
                            neighbors.append((idx, dist))
                            neighbors.sort(key=lambda x: x[1])
                        elif dist < neighbors[-1][1]:
                            neighbors[-1] = (idx, dist)
                            neighbors.sort(key=lambda x: x[1])
            else:
                # Visit closer child first
                left_id = self._left[node_id]
                right_id = self._right[node_id]

                left_dist = (
                    np.sqrt(np.sum((query - self._centroids[left_id]) ** 2))
                    if left_id >= 0
                    else np.inf
                )
                right_dist = (
                    np.sqrt(np.sum((query - self._centroids[right_id]) ** 2))
                    if right_id >= 0
                    else np.inf
                )

                if left_dist <= right_dist:
                    _search(left_id)
                    _search(right_id)
                else:
                    _search(right_id)
                    _search(left_id)

        _search(0)
        return neighbors


__all__ = [
    "KDNode",
    "NeighborResult",
    "NearestNeighborResult",  # Backward compatibility alias
    "KDTree",
    "BallTree",
]
