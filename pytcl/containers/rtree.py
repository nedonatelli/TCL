"""
R-tree implementation for spatial indexing of bounding boxes.

R-trees are tree data structures used for spatial access methods,
i.e., for indexing multi-dimensional information such as geographical
coordinates, rectangles, or polygons.

References
----------
.. [1] A. Guttman, "R-trees: A Dynamic Index Structure for Spatial
       Searching," ACM SIGMOD, 1984.
.. [2] N. Beckmann et al., "The R*-tree: An Efficient and Robust Access
       Method for Points and Rectangles," ACM SIGMOD, 1990.
"""

from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


class BoundingBox(NamedTuple):
    """Axis-aligned bounding box.

    Attributes
    ----------
    min_coords : ndarray
        Minimum coordinates in each dimension.
    max_coords : ndarray
        Maximum coordinates in each dimension.
    """

    min_coords: NDArray[np.floating]
    max_coords: NDArray[np.floating]

    @property
    def center(self) -> NDArray[np.floating]:
        """Center point of the bounding box."""
        return (self.min_coords + self.max_coords) / 2

    @property
    def dimensions(self) -> NDArray[np.floating]:
        """Size in each dimension."""
        return self.max_coords - self.min_coords

    @property
    def volume(self) -> float:
        """Volume (area in 2D) of the bounding box."""
        dims = self.dimensions
        if np.all(dims == 0):
            return 0.0
        return float(np.prod(dims[dims > 0]))

    def contains_point(self, point: ArrayLike) -> bool:
        """Check if box contains a point."""
        p = np.asarray(point)
        return bool(np.all(p >= self.min_coords) and np.all(p <= self.max_coords))

    def intersects(self, other: "BoundingBox") -> bool:
        """Check if this box intersects another."""
        return bool(
            np.all(self.max_coords >= other.min_coords)
            and np.all(self.min_coords <= other.max_coords)
        )

    def contains_box(self, other: "BoundingBox") -> bool:
        """Check if this box fully contains another."""
        return bool(
            np.all(self.min_coords <= other.min_coords)
            and np.all(self.max_coords >= other.max_coords)
        )


def merge_boxes(boxes: List[BoundingBox]) -> BoundingBox:
    """Compute minimum bounding box containing all boxes."""
    if not boxes:
        raise ValueError("Cannot merge empty list of boxes")

    min_coords = boxes[0].min_coords.copy()
    max_coords = boxes[0].max_coords.copy()

    for box in boxes[1:]:
        min_coords = np.minimum(min_coords, box.min_coords)
        max_coords = np.maximum(max_coords, box.max_coords)

    return BoundingBox(min_coords, max_coords)


def box_from_point(point: ArrayLike) -> BoundingBox:
    """Create a zero-volume bounding box from a point."""
    p = np.asarray(point, dtype=np.float64)
    return BoundingBox(p.copy(), p.copy())


def box_from_points(points: ArrayLike) -> BoundingBox:
    """Create minimum bounding box containing all points."""
    pts = np.asarray(points, dtype=np.float64)
    return BoundingBox(
        min_coords=np.min(pts, axis=0),
        max_coords=np.max(pts, axis=0),
    )


class RTreeNode:
    """Node in an R-tree.

    Attributes
    ----------
    bbox : BoundingBox
        Bounding box of this node.
    is_leaf : bool
        Whether this is a leaf node.
    children : list
        Child nodes (for internal nodes).
    entries : list
        (bbox, data_index) pairs (for leaf nodes).
    """

    __slots__ = ["bbox", "is_leaf", "children", "entries", "parent"]

    def __init__(self, is_leaf: bool = True):
        self.is_leaf = is_leaf
        self.children: List["RTreeNode"] = []
        self.entries: List[Tuple[BoundingBox, int]] = []
        self.bbox: Optional[BoundingBox] = None
        self.parent: Optional["RTreeNode"] = None

    def update_bbox(self) -> None:
        """Update bounding box to contain all children/entries."""
        if self.is_leaf:
            if self.entries:
                boxes = [entry[0] for entry in self.entries]
                self.bbox = merge_boxes(boxes)
        else:
            if self.children:
                boxes = [child.bbox for child in self.children if child.bbox]
                if boxes:
                    self.bbox = merge_boxes(boxes)


class RTreeResult(NamedTuple):
    """Result of R-tree query.

    Attributes
    ----------
    indices : list
        Indices of matching entries.
    boxes : list
        Bounding boxes of matching entries.
    """

    indices: List[int]
    boxes: List[BoundingBox]


class RTree:
    """
    R-tree for spatial indexing of bounding boxes.

    An R-tree groups nearby objects and represents them with their
    minimum bounding rectangle. This allows efficient spatial queries.

    Parameters
    ----------
    max_entries : int, optional
        Maximum entries per node. Default 10.
    min_entries : int, optional
        Minimum entries per node (except root). Default max_entries // 2.

    Examples
    --------
    >>> tree = RTree()
    >>> tree.insert(BoundingBox(np.array([0, 0]), np.array([1, 1])), 0)
    >>> tree.insert(BoundingBox(np.array([2, 2]), np.array([3, 3])), 1)
    >>> query_box = BoundingBox(np.array([0.5, 0.5]), np.array([2.5, 2.5]))
    >>> result = tree.query_intersect(query_box)

    Notes
    -----
    This implementation uses a simplified insertion algorithm.
    For production use, consider using R*-tree or packed R-tree variants.
    """

    def __init__(
        self,
        max_entries: int = 10,
        min_entries: Optional[int] = None,
    ):
        self.max_entries = max_entries
        self.min_entries = min_entries or max_entries // 2
        self.root = RTreeNode(is_leaf=True)
        self.n_entries = 0
        self._data: List[BoundingBox] = []

    def __len__(self) -> int:
        return self.n_entries

    def insert(self, bbox: BoundingBox, data_index: Optional[int] = None) -> int:
        """
        Insert a bounding box into the tree.

        Parameters
        ----------
        bbox : BoundingBox
            Bounding box to insert.
        data_index : int, optional
            Index to associate with this box. If None, uses next available.

        Returns
        -------
        index : int
            Index of the inserted entry.
        """
        if data_index is None:
            data_index = self.n_entries

        self._data.append(bbox)

        # Find leaf to insert into
        leaf = self._choose_leaf(self.root, bbox)

        # Insert entry
        leaf.entries.append((bbox, data_index))
        leaf.update_bbox()

        # Handle overflow
        if len(leaf.entries) > self.max_entries:
            self._split_node(leaf)

        self.n_entries += 1

        # Propagate bbox updates to root
        self._update_path(leaf)

        return data_index

    def insert_point(self, point: ArrayLike, data_index: Optional[int] = None) -> int:
        """Insert a point as a zero-volume bounding box."""
        return self.insert(box_from_point(point), data_index)

    def insert_points(self, points: ArrayLike) -> List[int]:
        """Insert multiple points."""
        points = np.asarray(points, dtype=np.float64)
        indices = []
        for point in points:
            indices.append(self.insert_point(point))
        return indices

    def _choose_leaf(self, node: RTreeNode, bbox: BoundingBox) -> RTreeNode:
        """Choose best leaf node for insertion."""
        if node.is_leaf:
            return node

        # Choose child that needs least enlargement
        best_child = None
        best_enlargement = np.inf

        for child in node.children:
            if child.bbox is None:
                return self._choose_leaf(child, bbox)

            # Compute enlargement needed
            merged = merge_boxes([child.bbox, bbox])
            enlargement = merged.volume - child.bbox.volume

            if enlargement < best_enlargement:
                best_enlargement = enlargement
                best_child = child

        if best_child is None:
            best_child = node.children[0]

        return self._choose_leaf(best_child, bbox)

    def _split_node(self, node: RTreeNode) -> None:
        """Split an overflowing node."""
        if node.is_leaf:
            entries = node.entries
        else:
            entries = [(child.bbox, child) for child in node.children]

        # Simple split: sort by center x-coordinate and split in half
        if node.is_leaf:
            sorted_entries = sorted(entries, key=lambda e: e[0].center[0] if e[0] else 0)
        else:
            sorted_entries = sorted(
                entries,
                key=lambda e: e[0].center[0] if e[0] else 0,
            )

        mid = len(sorted_entries) // 2

        # Create new sibling node
        sibling = RTreeNode(is_leaf=node.is_leaf)

        if node.is_leaf:
            node.entries = sorted_entries[:mid]
            sibling.entries = sorted_entries[mid:]
        else:
            node.children = [e[1] for e in sorted_entries[:mid]]
            sibling.children = [e[1] for e in sorted_entries[mid:]]
            for child in sibling.children:
                child.parent = sibling

        node.update_bbox()
        sibling.update_bbox()

        # Handle root split
        if node.parent is None:
            new_root = RTreeNode(is_leaf=False)
            new_root.children = [node, sibling]
            node.parent = new_root
            sibling.parent = new_root
            new_root.update_bbox()
            self.root = new_root
        else:
            sibling.parent = node.parent
            node.parent.children.append(sibling)
            node.parent.update_bbox()

            if len(node.parent.children) > self.max_entries:
                self._split_node(node.parent)

    def _update_path(self, node: RTreeNode) -> None:
        """Update bounding boxes along path to root."""
        current = node.parent
        while current is not None:
            current.update_bbox()
            current = current.parent

    def query_intersect(self, query_bbox: BoundingBox) -> RTreeResult:
        """
        Find all entries intersecting a query box.

        Parameters
        ----------
        query_bbox : BoundingBox
            Query bounding box.

        Returns
        -------
        result : RTreeResult
            Indices and boxes of intersecting entries.
        """
        indices: List[int] = []
        boxes: List[BoundingBox] = []

        def search(node: RTreeNode) -> None:
            if node.bbox is None or not node.bbox.intersects(query_bbox):
                return

            if node.is_leaf:
                for bbox, idx in node.entries:
                    if bbox.intersects(query_bbox):
                        indices.append(idx)
                        boxes.append(bbox)
            else:
                for child in node.children:
                    search(child)

        search(self.root)
        return RTreeResult(indices=indices, boxes=boxes)

    def query_contains(self, query_bbox: BoundingBox) -> RTreeResult:
        """
        Find all entries contained within a query box.

        Parameters
        ----------
        query_bbox : BoundingBox
            Query bounding box.

        Returns
        -------
        result : RTreeResult
            Indices and boxes of contained entries.
        """
        indices: List[int] = []
        boxes: List[BoundingBox] = []

        def search(node: RTreeNode) -> None:
            if node.bbox is None or not node.bbox.intersects(query_bbox):
                return

            if node.is_leaf:
                for bbox, idx in node.entries:
                    if query_bbox.contains_box(bbox):
                        indices.append(idx)
                        boxes.append(bbox)
            else:
                for child in node.children:
                    search(child)

        search(self.root)
        return RTreeResult(indices=indices, boxes=boxes)

    def query_point(self, point: ArrayLike) -> RTreeResult:
        """
        Find all entries containing a point.

        Parameters
        ----------
        point : array_like
            Query point.

        Returns
        -------
        result : RTreeResult
            Indices and boxes of entries containing the point.
        """
        p = np.asarray(point, dtype=np.float64)
        indices: List[int] = []
        boxes: List[BoundingBox] = []

        def search(node: RTreeNode) -> None:
            if node.bbox is None or not node.bbox.contains_point(p):
                return

            if node.is_leaf:
                for bbox, idx in node.entries:
                    if bbox.contains_point(p):
                        indices.append(idx)
                        boxes.append(bbox)
            else:
                for child in node.children:
                    search(child)

        search(self.root)
        return RTreeResult(indices=indices, boxes=boxes)

    def nearest(
        self,
        query_point: ArrayLike,
        k: int = 1,
    ) -> Tuple[List[int], List[float]]:
        """
        Find k nearest entries to a query point.

        Parameters
        ----------
        query_point : array_like
            Query point.
        k : int, optional
            Number of nearest neighbors. Default 1.

        Returns
        -------
        indices : list
            Indices of nearest entries.
        distances : list
            Distances to nearest entries.
        """
        query = np.asarray(query_point, dtype=np.float64)
        neighbors: List[Tuple[float, int]] = []

        def min_dist_to_box(point: NDArray, bbox: BoundingBox) -> float:
            """Minimum distance from point to bounding box."""
            clamped = np.clip(point, bbox.min_coords, bbox.max_coords)
            return float(np.sqrt(np.sum((point - clamped) ** 2)))

        def search(node: RTreeNode) -> None:
            if node.bbox is None:
                return

            # Prune if node is farther than worst neighbor
            if len(neighbors) >= k:
                if min_dist_to_box(query, node.bbox) >= neighbors[-1][0]:
                    return

            if node.is_leaf:
                for bbox, idx in node.entries:
                    dist = min_dist_to_box(query, bbox)
                    if len(neighbors) < k:
                        neighbors.append((dist, idx))
                        neighbors.sort(key=lambda x: x[0])
                    elif dist < neighbors[-1][0]:
                        neighbors[-1] = (dist, idx)
                        neighbors.sort(key=lambda x: x[0])
            else:
                # Sort children by distance and search closest first
                child_dists = [
                    (min_dist_to_box(query, c.bbox) if c.bbox else np.inf, c) for c in node.children
                ]
                child_dists.sort(key=lambda x: x[0])
                for _, child in child_dists:
                    search(child)

        search(self.root)

        indices = [n[1] for n in neighbors]
        distances = [n[0] for n in neighbors]
        return indices, distances


__all__ = [
    "BoundingBox",
    "merge_boxes",
    "box_from_point",
    "box_from_points",
    "RTreeNode",
    "RTreeResult",
    "RTree",
]
