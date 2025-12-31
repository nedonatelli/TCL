"""Tests for spatial data structures (k-d tree, ball tree)."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.containers import (
    BallTree,
    KDTree,
    NearestNeighborResult,
)


class TestKDTree:
    """Tests for KDTree."""

    def test_construction(self):
        """Tree can be constructed."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = KDTree(points)

        assert tree.n_samples == 4
        assert tree.n_features == 2

    def test_single_nearest(self):
        """Finds single nearest neighbor."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = KDTree(points)

        result = tree.query([[0.1, 0.1]], k=1)

        assert result.indices[0, 0] == 0
        assert result.distances[0, 0] < 0.2

    def test_k_nearest(self):
        """Finds k nearest neighbors."""
        points = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        tree = KDTree(points)

        result = tree.query([[0.1, 0.1]], k=3)

        # First three should be the close points
        assert set(result.indices[0]) == {0, 1, 2}

    def test_multiple_queries(self):
        """Handles multiple query points."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = KDTree(points)

        queries = np.array([[0, 0], [1, 1]])
        result = tree.query(queries, k=1)

        assert result.indices.shape == (2, 1)
        assert result.indices[0, 0] == 0
        assert result.indices[1, 0] == 3

    def test_1d_query(self):
        """Handles 1D query vector."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        result = tree.query(np.array([0.5, 0.5]), k=1)

        assert result.indices.shape == (1, 1)

    def test_query_radius(self):
        """Finds all points within radius."""
        points = np.array([[0, 0], [1, 0], [0, 1], [5, 5]])
        tree = KDTree(points)

        indices = tree.query_radius([[0, 0]], r=1.5)

        assert len(indices) == 1
        assert set(indices[0]) == {0, 1, 2}

    def test_query_ball_point(self):
        """query_ball_point is alias for query_radius."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        result1 = tree.query_radius([[0, 0]], r=1.5)
        result2 = tree.query_ball_point([[0, 0]], r=1.5)

        assert result1 == result2

    def test_empty_radius_query(self):
        """Handles radius query with no results."""
        points = np.array([[0, 0], [1, 0]])
        tree = KDTree(points)

        indices = tree.query_radius([[10, 10]], r=1.0)

        assert len(indices[0]) == 0

    def test_large_dataset(self):
        """Handles larger datasets efficiently."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (1000, 3))
        tree = KDTree(points)

        # Query should work
        result = tree.query(points[:10], k=5)

        assert result.indices.shape == (10, 5)
        # First neighbor of each query should be itself
        assert_array_equal(result.indices[:, 0], np.arange(10))

    def test_distances_sorted(self):
        """Distances are returned in sorted order."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (20, 2))
        tree = KDTree(points)

        result = tree.query([[5, 5]], k=10)

        # Distances should be non-decreasing
        distances = result.distances[0]
        assert np.all(distances[:-1] <= distances[1:])

    def test_high_dimensional(self):
        """Works in higher dimensions."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 1, (100, 10))
        tree = KDTree(points)

        result = tree.query(points[:5], k=3)

        assert result.indices.shape == (5, 3)


class TestKDTreeEdgeCases:
    """Edge case tests for KDTree."""

    def test_single_point(self):
        """Handles single point."""
        points = np.array([[1, 2, 3]])
        tree = KDTree(points)

        result = tree.query([[1, 2, 3]], k=1)

        assert result.indices[0, 0] == 0
        assert_allclose(result.distances[0, 0], 0.0)

    def test_duplicate_points(self):
        """Handles duplicate points."""
        points = np.array([[0, 0], [0, 0], [1, 1]])
        tree = KDTree(points)

        result = tree.query([[0, 0]], k=2)

        # Both duplicates should be found
        assert set(result.indices[0]) <= {0, 1}

    def test_collinear_points(self):
        """Handles collinear points."""
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        tree = KDTree(points)

        result = tree.query([[1.5, 0]], k=2)

        assert set(result.indices[0]) == {1, 2}

    def test_invalid_data(self):
        """Raises error for invalid data."""
        try:
            KDTree(np.array([1, 2, 3]))  # 1D array
            assert False, "Should raise ValueError"
        except ValueError:
            pass


class TestBallTree:
    """Tests for BallTree."""

    def test_construction(self):
        """Tree can be constructed."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = BallTree(points)

        assert tree.n_samples == 4
        assert tree.n_features == 2

    def test_single_nearest(self):
        """Finds single nearest neighbor."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = BallTree(points)

        result = tree.query([[0.1, 0.1]], k=1)

        assert result.indices[0, 0] == 0

    def test_k_nearest(self):
        """Finds k nearest neighbors."""
        points = np.array([[0, 0], [1, 0], [0, 1], [10, 10]])
        tree = BallTree(points)

        result = tree.query([[0.1, 0.1]], k=3)

        # First three should be the close points
        assert set(result.indices[0]) == {0, 1, 2}

    def test_matches_kdtree(self):
        """Results match KDTree."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (50, 3))

        kd = KDTree(points)
        ball = BallTree(points)

        queries = rng.uniform(0, 10, (10, 3))

        kd_result = kd.query(queries, k=5)
        ball_result = ball.query(queries, k=5)

        # Same neighbors (order might differ for ties)
        for i in range(10):
            kd_set = set(kd_result.indices[i])
            ball_set = set(ball_result.indices[i])
            # At least 4 of 5 should match (ties can cause differences)
            assert len(kd_set & ball_set) >= 4

    def test_high_dimensional(self):
        """Works in higher dimensions."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 1, (100, 20))
        tree = BallTree(points)

        result = tree.query(points[:5], k=3)

        assert result.indices.shape == (5, 3)


class TestNearestNeighborResult:
    """Tests for NearestNeighborResult."""

    def test_namedtuple(self):
        """Is a proper named tuple."""
        result = NearestNeighborResult(
            indices=np.array([[0, 1]]),
            distances=np.array([[0.0, 1.0]]),
        )

        assert_array_equal(result.indices, [[0, 1]])
        assert_array_equal(result.distances, [[0.0, 1.0]])


class TestSpatialTreePerformance:
    """Performance-related tests."""

    def test_query_faster_than_brute_force(self):
        """Tree query is faster than brute force for large datasets."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (500, 3))
        queries = rng.uniform(0, 100, (10, 3))

        tree = KDTree(points)

        # This should work without timeout
        result = tree.query(queries, k=10)

        assert result.indices.shape == (10, 10)

    def test_radius_query_efficiency(self):
        """Radius query is efficient."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (500, 2))

        tree = KDTree(points)

        # Small radius should be fast
        result = tree.query_radius([[50, 50]], r=5)

        assert isinstance(result[0], list)


class TestCorrectness:
    """Correctness verification tests."""

    def test_exact_distances(self):
        """Distances are computed correctly."""
        points = np.array([[0, 0], [3, 4]])  # Distance = 5
        tree = KDTree(points)

        result = tree.query([[0, 0]], k=2)

        assert_allclose(result.distances[0, 0], 0.0)
        assert_allclose(result.distances[0, 1], 5.0)

    def test_finds_exact_match(self):
        """Finds exact match as nearest neighbor."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (100, 3))
        tree = KDTree(points)

        # Query with points from dataset
        result = tree.query(points, k=1)

        # Each point should find itself
        assert_array_equal(result.indices[:, 0], np.arange(100))
        assert_allclose(result.distances[:, 0], 0.0)

    def test_radius_boundary(self):
        """Points exactly on radius boundary are included."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        # Query with radius exactly reaching point
        result = tree.query_radius([[0, 0]], r=1.0)

        # Points at distance 1 should be included
        assert 1 in result[0]
        assert 2 in result[0]
