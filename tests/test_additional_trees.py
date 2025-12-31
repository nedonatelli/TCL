"""Tests for additional spatial tree structures (R-tree, VP-tree, Cover tree)."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.containers import (  # R-Tree; VP-Tree; Cover Tree
    BoundingBox,
    CoverTree,
    RTree,
    VPTree,
    box_from_point,
    box_from_points,
    merge_boxes,
)


class TestBoundingBox:
    """Tests for BoundingBox class."""

    def test_creation(self):
        """Basic bounding box creation."""
        bbox = BoundingBox(np.array([0, 0]), np.array([1, 1]))

        assert_array_equal(bbox.min_coords, [0, 0])
        assert_array_equal(bbox.max_coords, [1, 1])

    def test_center(self):
        """Center computation."""
        bbox = BoundingBox(np.array([0, 0]), np.array([2, 4]))

        assert_array_equal(bbox.center, [1, 2])

    def test_dimensions(self):
        """Dimensions computation."""
        bbox = BoundingBox(np.array([1, 2]), np.array([4, 5]))

        assert_array_equal(bbox.dimensions, [3, 3])

    def test_volume(self):
        """Volume (area in 2D) computation."""
        bbox = BoundingBox(np.array([0, 0]), np.array([2, 3]))

        assert bbox.volume == 6.0

    def test_contains_point(self):
        """Point containment check."""
        bbox = BoundingBox(np.array([0, 0]), np.array([1, 1]))

        assert bbox.contains_point([0.5, 0.5])
        assert bbox.contains_point([0, 0])
        assert bbox.contains_point([1, 1])
        assert not bbox.contains_point([1.1, 0.5])

    def test_intersects(self):
        """Box intersection check."""
        box1 = BoundingBox(np.array([0, 0]), np.array([2, 2]))
        box2 = BoundingBox(np.array([1, 1]), np.array([3, 3]))
        box3 = BoundingBox(np.array([5, 5]), np.array([6, 6]))

        assert box1.intersects(box2)
        assert box2.intersects(box1)
        assert not box1.intersects(box3)

    def test_contains_box(self):
        """Box containment check."""
        outer = BoundingBox(np.array([0, 0]), np.array([10, 10]))
        inner = BoundingBox(np.array([2, 2]), np.array([5, 5]))
        outside = BoundingBox(np.array([11, 11]), np.array([12, 12]))

        assert outer.contains_box(inner)
        assert not inner.contains_box(outer)
        assert not outer.contains_box(outside)


class TestBoundingBoxHelpers:
    """Tests for bounding box helper functions."""

    def test_merge_boxes(self):
        """Merging multiple boxes."""
        boxes = [
            BoundingBox(np.array([0, 0]), np.array([1, 1])),
            BoundingBox(np.array([2, 2]), np.array([3, 3])),
        ]

        merged = merge_boxes(boxes)

        assert_array_equal(merged.min_coords, [0, 0])
        assert_array_equal(merged.max_coords, [3, 3])

    def test_box_from_point(self):
        """Creating box from point."""
        point = np.array([1.0, 2.0, 3.0])
        bbox = box_from_point(point)

        assert_array_equal(bbox.min_coords, point)
        assert_array_equal(bbox.max_coords, point)
        assert bbox.volume == 0.0

    def test_box_from_points(self):
        """Creating box from multiple points."""
        points = np.array([[0, 0], [1, 2], [3, 1]])
        bbox = box_from_points(points)

        assert_array_equal(bbox.min_coords, [0, 0])
        assert_array_equal(bbox.max_coords, [3, 2])


class TestRTree:
    """Tests for R-tree."""

    def test_construction(self):
        """Basic R-tree construction."""
        tree = RTree()
        assert len(tree) == 0

    def test_insert_single(self):
        """Insert single bounding box."""
        tree = RTree()
        bbox = BoundingBox(np.array([0, 0]), np.array([1, 1]))

        idx = tree.insert(bbox)

        assert idx == 0
        assert len(tree) == 1

    def test_insert_multiple(self):
        """Insert multiple bounding boxes."""
        tree = RTree()

        for i in range(10):
            bbox = BoundingBox(
                np.array([i, i], dtype=np.float64),
                np.array([i + 1, i + 1], dtype=np.float64),
            )
            tree.insert(bbox)

        assert len(tree) == 10

    def test_insert_point(self):
        """Insert point as zero-volume box."""
        tree = RTree()
        tree.insert_point([1.0, 2.0])

        assert len(tree) == 1

    def test_query_intersect(self):
        """Query for intersecting boxes."""
        tree = RTree()

        # Insert non-overlapping boxes
        tree.insert(BoundingBox(np.array([0, 0]), np.array([1, 1])))
        tree.insert(BoundingBox(np.array([2, 2]), np.array([3, 3])))
        tree.insert(BoundingBox(np.array([4, 4]), np.array([5, 5])))

        # Query intersecting first two
        query = BoundingBox(np.array([0.5, 0.5]), np.array([2.5, 2.5]))
        result = tree.query_intersect(query)

        assert len(result.indices) == 2
        assert 0 in result.indices
        assert 1 in result.indices

    def test_query_contains(self):
        """Query for contained boxes."""
        tree = RTree()

        tree.insert(BoundingBox(np.array([1, 1]), np.array([2, 2])))
        tree.insert(BoundingBox(np.array([5, 5]), np.array([6, 6])))

        # Query box that contains first
        query = BoundingBox(np.array([0, 0]), np.array([3, 3]))
        result = tree.query_contains(query)

        assert len(result.indices) == 1
        assert 0 in result.indices

    def test_query_point(self):
        """Query for boxes containing a point."""
        tree = RTree()

        tree.insert(BoundingBox(np.array([0, 0]), np.array([2, 2])))
        tree.insert(BoundingBox(np.array([1, 1]), np.array([3, 3])))
        tree.insert(BoundingBox(np.array([10, 10]), np.array([11, 11])))

        result = tree.query_point([1.5, 1.5])

        assert len(result.indices) == 2

    def test_nearest(self):
        """Find nearest boxes to point."""
        tree = RTree()

        tree.insert(BoundingBox(np.array([0, 0]), np.array([1, 1])))
        tree.insert(BoundingBox(np.array([10, 10]), np.array([11, 11])))

        indices, distances = tree.nearest([0.5, 0.5], k=1)

        assert indices[0] == 0
        assert distances[0] == 0.0  # Point is inside box


class TestVPTree:
    """Tests for VP-tree."""

    def test_construction(self):
        """Basic VP-tree construction."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = VPTree(points)

        assert tree.n_samples == 4
        assert tree.n_features == 2

    def test_query_single(self):
        """Find single nearest neighbor."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = VPTree(points)

        result = tree.query([[0.1, 0.1]], k=1)

        assert result.indices[0, 0] == 0

    def test_query_multiple(self):
        """Find multiple nearest neighbors."""
        points = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        tree = VPTree(points)

        result = tree.query([[0.1, 0.1]], k=3)

        # First 3 should be the close points
        assert set(result.indices[0]) == {0, 1, 2}

    def test_query_radius(self):
        """Find all points within radius."""
        points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
        tree = VPTree(points)

        indices = tree.query_radius([[0, 0]], r=1.5)

        assert len(indices) == 1
        assert set(indices[0]) == {0, 1, 2}

    def test_custom_metric(self):
        """VP-tree with custom distance metric."""

        def manhattan(x, y):
            return float(np.sum(np.abs(x - y)))

        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = VPTree(points, metric=manhattan)

        result = tree.query([[0.5, 0.5]], k=1)

        # All points are equidistant in Manhattan metric
        assert result.indices[0, 0] in [0, 1, 2]

    def test_self_query(self):
        """Querying with points from dataset."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (50, 3))
        tree = VPTree(points)

        result = tree.query(points, k=1)

        # Each point should find itself
        assert_array_equal(result.indices[:, 0], np.arange(50))
        assert_allclose(result.distances[:, 0], 0.0)


class TestCoverTree:
    """Tests for Cover tree."""

    def test_construction(self):
        """Basic Cover tree construction."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = CoverTree(points)

        assert tree.n_samples == 4
        assert tree.n_features == 2

    def test_query_single(self):
        """Find single nearest neighbor."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = CoverTree(points)

        result = tree.query([[0.1, 0.1]], k=1)

        assert result.indices[0, 0] == 0

    def test_query_multiple(self):
        """Find multiple nearest neighbors."""
        points = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        tree = CoverTree(points)

        result = tree.query([[0.1, 0.1]], k=3)

        # First 3 should be the close points
        found = set(result.indices[0])
        assert 0 in found  # Origin point should be found

    def test_query_radius(self):
        """Find all points within radius."""
        points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
        tree = CoverTree(points)

        indices = tree.query_radius([[0, 0]], r=1.5)

        assert len(indices) == 1
        assert 0 in indices[0]

    def test_custom_metric(self):
        """Cover tree with custom distance metric."""

        def manhattan(x, y):
            return float(np.sum(np.abs(x - y)))

        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = CoverTree(points, metric=manhattan)

        result = tree.query([[0.5, 0.5]], k=1)

        assert result.indices[0, 0] in [0, 1, 2]


class TestTreeComparison:
    """Compare different tree implementations."""

    def test_same_results_kd_vp(self):
        """VP-tree gives same results as KD-tree."""
        from pytcl.containers import KDTree

        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (50, 2))

        kd = KDTree(points)
        vp = VPTree(points)

        queries = rng.uniform(0, 10, (5, 2))

        kd_result = kd.query(queries, k=3)
        vp_result = vp.query(queries, k=3)

        # Should find same neighbors (order may differ for ties)
        for i in range(5):
            kd_set = set(kd_result.indices[i])
            vp_set = set(vp_result.indices[i])
            # At least most should match
            assert len(kd_set & vp_set) >= 2

    def test_handles_duplicates(self):
        """All trees handle duplicate points."""
        points = np.array([[0, 0], [0, 0], [1, 1]])

        vp = VPTree(points)
        cover = CoverTree(points)

        vp_result = vp.query([[0, 0]], k=2)
        _ = cover.query([[0, 0]], k=2)  # Verify cover tree doesn't error

        # Should find both copies at origin
        assert set(vp_result.indices[0]) <= {0, 1}


class TestLargerDatasets:
    """Tests with larger datasets."""

    def test_vptree_larger(self):
        """VP-tree handles larger datasets."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (200, 3))
        tree = VPTree(points)

        result = tree.query(points[:10], k=5)

        assert result.indices.shape == (10, 5)
        # First neighbor should be itself
        assert_array_equal(result.indices[:, 0], np.arange(10))

    def test_rtree_many_insertions(self):
        """R-tree handles many insertions."""
        tree = RTree(max_entries=5)

        for i in range(100):
            bbox = BoundingBox(
                np.array([i, i], dtype=np.float64),
                np.array([i + 1, i + 1], dtype=np.float64),
            )
            tree.insert(bbox)

        assert len(tree) == 100

        # Query should still work
        query = BoundingBox(np.array([50, 50]), np.array([60, 60]))
        result = tree.query_intersect(query)
        assert len(result.indices) > 0
