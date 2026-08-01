"""
Cover Tree implementation for nearest neighbor search.

Cover trees are data structures for nearest neighbor search in metric
spaces with a theoretical guarantee of O(c^12 log n) query time, where
c is the expansion constant of the data.

References
----------
- A. Beygelzimer, S. Kakade, J. Langford, "Cover trees for nearest
  neighbor," ICML 2006.
"""

import logging
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.containers.base import (
    CoverTreeResult,  # Backward compatibility alias
    MetricSpatialIndex,
    NeighborResult,
    validate_neighbor_count,
    validate_query_input,
)

# Module logger
_logger = logging.getLogger("pytcl.containers.covertree")


class CoverTreeNode:
    """Node in a Cover tree.

    Attributes
    ----------
    index : int
        Index of the point in the original data.
    level : int
        Level in the tree (determines covering radius 2^level).
    children : dict
        Children organized by level.
    max_desc : float
        Covering radius: maximum distance from this node's point to any
        point in its subtree. Used for exact query pruning.
    """

    __slots__ = ["index", "level", "children", "max_desc"]

    def __init__(self, index: int, level: int):
        self.index = index
        self.level = level
        # Children at each level
        self.children: dict[int, List["CoverTreeNode"]] = {}
        self.max_desc = 0.0

    def add_child(self, level: int, child: "CoverTreeNode") -> None:
        """Add a child at the specified level."""
        if level not in self.children:
            self.children[level] = []
        self.children[level].append(child)


class CoverTree(MetricSpatialIndex):
    """
    Cover Tree for metric space nearest neighbor search.

    A cover tree maintains a hierarchy of nested coverings of the data,
    where points at level i are a subset of points at level i-1 and
    cover all points within distance 2^i.

    Parameters
    ----------
    data : array_like
        Data points of shape (n_samples, n_features).
    metric : callable, optional
        Distance function metric(x, y) -> float.
        Default is Euclidean distance.
    base : float, optional
        Base for the exponential scale. Default 2.0.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.random.rand(100, 3)
    >>> tree = CoverTree(points)
    >>> result = tree.query(points[:5], k=3)

    Notes
    -----
    Cover trees provide theoretical guarantees based on the expansion
    constant of the data. For well-distributed data, queries are
    efficient even in high dimensions.

    The implementation uses a simplified version of the original
    algorithm for clarity.

    See Also
    --------
    MetricSpatialIndex : Abstract base class for metric-based spatial indices.
    VPTree : Alternative metric space index using vantage points.
    """

    def __init__(
        self,
        data: ArrayLike,
        metric: Optional[
            Callable[[np.ndarray[Any, Any], np.ndarray[Any, Any]], float]
        ] = None,
        base: float = 2.0,
    ):
        super().__init__(data, metric)
        self.base = base

        # Compute distance cache for small datasets
        self._distance_cache: dict[Tuple[int, int], float] = {}

        # Build tree
        self.root: Optional[CoverTreeNode] = None
        self.max_level = 0
        self.min_level = 0

        if self.n_samples > 0:
            self._build_tree()
            _logger.debug(
                "CoverTree built with base=%.1f, levels=%d to %d",
                base,
                self.min_level,
                self.max_level,
            )

    def _distance(self, i: int, j: int) -> float:
        """Get distance between points i and j (with caching)."""
        if i == j:
            return 0.0
        key = (min(i, j), max(i, j))
        if key not in self._distance_cache:
            self._distance_cache[key] = self.metric(self.data[i], self.data[j])
        return self._distance_cache[key]

    def _distance_to_point(self, idx: int, query: NDArray[np.floating]) -> float:
        """Distance from data point to query point."""
        return self.metric(self.data[idx], query)

    def _cover_distance(self, level: int) -> float:
        """Get the cover distance for a level (base^level)."""
        return self.base**level

    def _build_tree(self) -> None:
        """Build the cover tree using batch insertion."""
        # Find max distance to set initial level
        max_dist = 0.0
        for i in range(min(self.n_samples, 100)):  # Sample for large datasets
            for j in range(i + 1, min(self.n_samples, 100)):
                d = self._distance(i, j)
                max_dist = max(max_dist, d)

        # Set initial level
        if max_dist > 0:
            self.max_level = int(np.ceil(np.log(max_dist) / np.log(self.base))) + 1
        else:
            self.max_level = 0
        self.min_level = self.max_level

        # Create root with first point
        self.root = CoverTreeNode(0, self.max_level)

        # Insert remaining points
        for i in range(1, self.n_samples):
            self._insert(i)

        # Compute exact covering radii for query pruning. The simplified
        # insertion does not maintain the strict cover invariant
        # (d(parent, child) <= base^level), so queries prune using the
        # actual maximum descendant distance instead of level bounds.
        self._compute_covering_radii(self.root)

    def _compute_covering_radii(self, node: CoverTreeNode) -> float:
        """Compute max distance from node's point to any point in its subtree."""
        max_desc = 0.0
        for children in node.children.values():
            for child in children:
                child_radius = self._compute_covering_radii(child)
                d = self._distance(node.index, child.index)
                max_desc = max(max_desc, d + child_radius)
        node.max_desc = max_desc
        return max_desc

    def _insert(self, point_idx: int) -> None:
        """Insert a point into the cover tree."""
        if self.root is None:
            self.root = CoverTreeNode(point_idx, self.max_level)
            return

        # Find the level at which to insert
        # Start from max_level and descend
        level = self.max_level

        # Find nodes at each level that cover this point
        cover_sets: dict[int, List[CoverTreeNode]] = {level: [self.root]}

        while level > self.min_level - 1:
            cover_dist = self._cover_distance(level)
            next_level = level - 1
            next_cover: List[CoverTreeNode] = []

            for node in cover_sets.get(level, []):
                # Check if this node covers the new point
                d = self._distance(node.index, point_idx)

                if d <= cover_dist:
                    # Node covers point, add to candidates for next level
                    next_cover.append(node)
                    # Also add children as candidates
                    for child in node.children.get(next_level, []):
                        if self._distance(child.index, point_idx) <= cover_dist:
                            next_cover.append(child)

            if not next_cover:
                # No nodes at next level cover this point
                # Insert here
                break

            cover_sets[next_level] = next_cover
            level = next_level

        # Insert point as child of closest covering node
        min_dist = np.inf
        parent = self.root

        for node in cover_sets.get(level, [self.root]):
            d = self._distance(node.index, point_idx)
            if d < min_dist:
                min_dist = d
                parent = node

        # Create new node
        new_level = level - 1
        new_node = CoverTreeNode(point_idx, new_level)
        parent.add_child(new_level, new_node)

        # Update min level
        self.min_level = min(self.min_level, new_level)

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
        validate_neighbor_count(k, self.n_samples)
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

    def _query_single(
        self,
        query: NDArray[np.floating],
        k: int,
    ) -> List[Tuple[int, float]]:
        """Find k nearest neighbors for a single query."""
        if self.root is None:
            return []

        neighbors: List[Tuple[int, float]] = []

        def search(node: CoverTreeNode, dist: float) -> None:
            # Each data point appears in exactly one node, so visiting
            # each node once yields no duplicate indices.
            if len(neighbors) < k:
                neighbors.append((node.index, dist))
                neighbors.sort(key=lambda x: x[1])
            elif dist < neighbors[-1][1]:
                neighbors[-1] = (node.index, dist)
                neighbors.sort(key=lambda x: x[1])

            # Gather children with their distances, visit closest first
            child_dists: List[Tuple[float, CoverTreeNode]] = []
            for children in node.children.values():
                for child in children:
                    child_dists.append(
                        (self._distance_to_point(child.index, query), child)
                    )
            child_dists.sort(key=lambda x: x[0])

            for child_dist, child in child_dists:
                # Any point in child's subtree is at distance
                # >= child_dist - child.max_desc from the query.
                if len(neighbors) < k or child_dist - child.max_desc < neighbors[-1][1]:
                    search(child, child_dist)

        search(self.root, self._distance_to_point(self.root.index, query))
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
        if self.root is None:
            return []

        indices: List[int] = []

        def search(node: CoverTreeNode, dist: float) -> None:
            # Check if this point is within radius
            if dist <= r:
                indices.append(node.index)

            # Any point in a child's subtree is at distance
            # >= d(query, child) - child.max_desc, so descend only into
            # children whose subtree could intersect the query ball.
            for children in node.children.values():
                for child in children:
                    child_dist = self._distance_to_point(child.index, query)
                    if child_dist - child.max_desc <= r:
                        search(child, child_dist)

        search(self.root, self._distance_to_point(self.root.index, query))
        return indices


__all__ = [
    "NeighborResult",
    "CoverTreeResult",  # Backward compatibility alias
    "CoverTreeNode",
    "CoverTree",
]
