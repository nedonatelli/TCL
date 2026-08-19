"""Tests for spatial container data structures.

Comprehensive parametrized tests for:
- KDTree, BallTree, VPTree, CoverTree
- BaseSpatialIndex and MetricSpatialIndex contracts
- Input validation, numerical edge cases, query patterns
- Custom metrics, radius queries, correctness verification
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.containers import (
    BallTree,
    BaseSpatialIndex,
    CoverTree,
    KDTree,
    MetricSpatialIndex,
    NearestNeighborResult,
    VPTree,
)

# =============================================================================
# Fixtures for parametrized tests
# =============================================================================


@pytest.fixture(params=[KDTree, BallTree])
def euclidean_tree_class(request):
    """Fixture providing Euclidean-only tree classes."""
    return request.param


@pytest.fixture(params=[VPTree, CoverTree])
def metric_tree_class(request):
    """Fixture providing metric tree classes."""
    return request.param


@pytest.fixture(params=[KDTree, BallTree, VPTree, CoverTree])
def tree_class(request):
    """Fixture providing all tree classes."""
    return request.param


# =============================================================================
# Parametrized: Dimensionality Tests
# =============================================================================


class TestDimensionalityParametrized:
    """Test spatial structures across different dimensionalities."""

    @pytest.mark.parametrize("n_dims", [1, 2, 3, 5, 10, 20])
    def test_varying_dimensions(self, tree_class, n_dims):
        """Test tree construction and query in varying dimensions."""
        rng = np.random.default_rng(42)
        n_points = 50
        points = rng.uniform(0, 10, (n_points, n_dims))

        tree = tree_class(points)
        assert tree.n_features == n_dims
        assert tree.n_samples == n_points

        result = tree.query(points[:3], k=5)
        assert result.indices.shape == (3, 5)
        assert result.distances.shape == (3, 5)

    @pytest.mark.parametrize("n_points", [1, 2, 5, 10, 100, 500])
    def test_varying_sizes(self, tree_class, n_points):
        """Test tree with varying dataset sizes."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (n_points, 3))

        tree = tree_class(points)
        assert tree.n_samples == n_points
        assert len(tree) == n_points

        k = min(5, n_points)
        result = tree.query(points[:1], k=k)
        assert result.indices.shape == (1, k)


# =============================================================================
# Parametrized: Edge Case Data Patterns
# =============================================================================


class TestDataPatterns:
    """Test spatial structures with various data patterns."""

    @pytest.mark.parametrize(
        "pattern,expected_shape",
        [
            ("grid", (16, 2)),
            ("collinear", (10, 2)),
            ("clustered", (20, 2)),
            ("uniform", (50, 2)),
        ],
    )
    def test_data_patterns(self, tree_class, pattern, expected_shape):
        """Test tree with different data distribution patterns."""
        rng = np.random.default_rng(42)

        if pattern == "grid":
            x, y = np.meshgrid(np.arange(4), np.arange(4))
            points = np.column_stack([x.ravel(), y.ravel()]).astype(float)
        elif pattern == "collinear":
            points = np.column_stack([np.arange(10), np.zeros(10)]).astype(float)
        elif pattern == "clustered":
            cluster1 = rng.normal(0, 0.1, (10, 2))
            cluster2 = rng.normal(10, 0.1, (10, 2))
            points = np.vstack([cluster1, cluster2])
        else:
            points = rng.uniform(0, 10, expected_shape)

        tree = tree_class(points)
        assert tree.n_samples == expected_shape[0]

        result = tree.query(points[:1], k=min(3, expected_shape[0]))
        assert result.indices.shape[0] == 1

    def test_duplicate_points(self, tree_class):
        """Test handling of duplicate points."""
        points = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [1, 1]])
        tree = tree_class(points)

        result = tree.query([[0, 0]], k=3)
        assert result.indices.shape == (1, 3)
        assert np.sum(result.distances[0] == 0) >= 1

    def test_near_duplicate_points(self, tree_class):
        """Test handling of nearly-duplicate points."""
        eps = 1e-10
        points = np.array(
            [
                [0, 0],
                [eps, 0],
                [0, eps],
                [eps, eps],
                [1, 1],
            ]
        )
        tree = tree_class(points)

        result = tree.query([[0, 0]], k=4)
        assert result.indices.shape == (1, 4)
        assert np.all(result.distances[0] < 0.1)

    def test_collinear_points_nearest(self):
        """Test nearest neighbor query with collinear points."""
        points = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        tree = KDTree(points)

        result = tree.query([[1.5, 0]], k=2)
        assert set(result.indices[0]) == {1, 2}


# =============================================================================
# Parametrized: Query Edge Cases
# =============================================================================


class TestQueryEdgeCases:
    """Test query edge cases across all tree types."""

    @pytest.mark.parametrize("k", [1, 2, 3, 5, 10])
    def test_k_values(self, tree_class, k):
        """Test various k values for k-NN query."""
        rng = np.random.default_rng(42)
        n_points = 20
        points = rng.uniform(0, 10, (n_points, 3))

        tree = tree_class(points)
        k_actual = min(k, n_points)
        result = tree.query(points[:1], k=k_actual)

        assert result.indices.shape == (1, k_actual)
        assert result.distances.shape == (1, k_actual)
        assert np.all(result.distances[0, :-1] <= result.distances[0, 1:])

    def test_k_equals_n(self, tree_class):
        """Test k equal to number of points."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        tree = tree_class(points)

        result = tree.query([[0.5, 0.5]], k=5)
        assert result.indices.shape == (1, 5)
        valid_indices = set(result.indices[0])
        assert all(0 <= idx < 5 for idx in valid_indices)
        assert 4 in valid_indices

    @pytest.mark.parametrize("n_queries", [1, 5, 10, 50])
    def test_batch_queries(self, tree_class, n_queries):
        """Test batch query with varying number of queries."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (100, 3))
        queries = rng.uniform(0, 10, (n_queries, 3))

        tree = tree_class(points)
        result = tree.query(queries, k=3)

        assert result.indices.shape == (n_queries, 3)
        assert result.distances.shape == (n_queries, 3)

    def test_query_point_from_dataset(self, tree_class):
        """Test querying with a point from the dataset."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = tree_class(points)

        result = tree.query(points[[0]], k=1)
        assert result.indices[0, 0] == 0
        assert_allclose(result.distances[0, 0], 0.0, atol=1e-10)

    def test_query_far_point(self, tree_class):
        """Test querying with a point far from all data."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = tree_class(points)

        result = tree.query([[1000, 1000]], k=1)
        assert result.indices.shape == (1, 1)
        assert result.distances[0, 0] > 1000

    def test_1d_query_vector(self):
        """Test handles 1D query vector."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        result = tree.query(np.array([0.5, 0.5]), k=1)
        assert result.indices.shape == (1, 1)

    def test_multiple_queries_indices(self):
        """Test multiple query points return correct indices."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = KDTree(points)

        queries = np.array([[0, 0], [1, 1]])
        result = tree.query(queries, k=1)

        assert result.indices.shape == (2, 1)
        assert result.indices[0, 0] == 0
        assert result.indices[1, 0] == 3


# =============================================================================
# Parametrized: Radius Query Tests
# =============================================================================


class TestRadiusQueryParametrized:
    """Test radius query edge cases."""

    @pytest.mark.parametrize("radius", [0.0, 0.1, 0.5, 1.0, 2.0, 10.0])
    def test_varying_radius(self, tree_class, radius):
        """Test radius query with varying radii."""
        points = np.array([[0, 0], [0.3, 0], [0.7, 0], [1.0, 0], [2.0, 0], [5.0, 0]])
        tree = tree_class(points)

        result = tree.query_radius([[0, 0]], r=radius)
        assert len(result) == 1

        expected = sum(1 for p in points if np.sqrt(np.sum(p**2)) <= radius)
        assert len(result[0]) == expected

    def test_zero_radius(self, tree_class):
        """Test radius query with zero radius."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)

        result = tree.query_radius([[0, 0]], r=0.0)
        assert len(result[0]) == 1
        assert result[0][0] == 0

    def test_large_radius(self, tree_class):
        """Test radius query with very large radius."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (50, 3))
        tree = tree_class(points)

        result = tree.query_radius([[5, 5, 5]], r=1000.0)
        assert len(result[0]) == 50

    def test_batch_radius_query(self, tree_class):
        """Test batch radius query."""
        points = np.array([[0, 0], [5, 0], [10, 0]])
        tree = tree_class(points)

        queries = np.array([[0, 0], [5, 0], [10, 0]])
        result = tree.query_radius(queries, r=1.0)

        assert len(result) == 3
        assert 0 in result[0]
        assert 1 in result[1]
        assert 2 in result[2]

    def test_empty_radius_query(self):
        """Test radius query with no results."""
        points = np.array([[0, 0], [1, 0]])
        tree = KDTree(points)

        indices = tree.query_radius([[10, 10]], r=1.0)
        assert len(indices[0]) == 0

    def test_query_ball_point_alias(self):
        """Test query_ball_point is alias for query_radius."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        result1 = tree.query_radius([[0, 0]], r=1.5)
        result2 = tree.query_ball_point([[0, 0]], r=1.5)

        assert result1 == result2

    def test_radius_boundary_inclusion(self):
        """Test points exactly on radius boundary are included."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = KDTree(points)

        result = tree.query_radius([[0, 0]], r=1.0)
        assert 1 in result[0]
        assert 2 in result[0]


# =============================================================================
# Custom Metric Tests (VPTree, CoverTree only)
# =============================================================================


class TestCustomMetrics:
    """Test metric trees with custom distance functions."""

    def test_manhattan_metric(self, metric_tree_class):
        """Test with Manhattan (L1) distance."""

        def manhattan(x, y):
            return float(np.sum(np.abs(x - y)))

        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        tree = metric_tree_class(points, metric=manhattan)

        result = tree.query([[0.5, 0.5]], k=2)
        assert result.indices.shape == (1, 2)

    def test_chebyshev_metric(self, metric_tree_class):
        """Test with Chebyshev (L-infinity) distance."""

        def chebyshev(x, y):
            return float(np.max(np.abs(x - y)))

        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        tree = metric_tree_class(points, metric=chebyshev)

        result = tree.query([[0.5, 0.5]], k=1)
        assert result.indices[0, 0] == 4

    def test_weighted_euclidean(self, metric_tree_class):
        """Test with weighted Euclidean distance."""
        weights = np.array([2.0, 1.0])

        def weighted_euclidean(x, y):
            return float(np.sqrt(np.sum(weights * (x - y) ** 2)))

        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = metric_tree_class(points, metric=weighted_euclidean)

        result = tree.query([[0.5, 0.5]], k=2)
        assert result.indices.shape == (1, 2)


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Test input validation edge cases."""

    def test_invalid_1d_data(self, tree_class):
        """Test that 1D data raises error."""
        with pytest.raises(ValueError, match="2-dimensional"):
            tree_class(np.array([1, 2, 3]))

    def test_invalid_3d_data(self, tree_class):
        """Test that 3D data raises error."""
        with pytest.raises(ValueError, match="2-dimensional"):
            tree_class(np.random.rand(2, 3, 4))

    def test_query_dimension_mismatch(self, tree_class):
        """Test query with wrong number of features."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)

        with pytest.raises(ValueError, match="features"):
            tree.query([[0, 0, 0]], k=1)

    def test_empty_query(self, tree_class):
        """Test empty query array."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)

        result = tree.query(np.empty((0, 2)), k=1)
        assert result.indices.shape == (0, 1)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Test numerical edge cases."""

    def test_very_small_values(self, tree_class):
        """Test with very small coordinate values."""
        eps = 1e-15
        points = np.array([[0, 0], [eps, 0], [0, eps], [eps, eps]])
        tree = tree_class(points)

        result = tree.query([[0, 0]], k=4)
        assert result.indices.shape == (1, 4)

    def test_very_large_values(self, tree_class):
        """Test with very large coordinate values."""
        large = 1e15
        points = np.array([[0, 0], [large, 0], [0, large], [large, large]])
        tree = tree_class(points)

        result = tree.query([[0, 0]], k=1)
        assert result.indices[0, 0] == 0

    def test_mixed_scale_values(self, tree_class):
        """Test with mixed scale coordinate values."""
        points = np.array([[1e-10, 1e10], [1e10, 1e-10], [1, 1], [0, 0], [1e5, 1e-5]])
        tree = tree_class(points)

        result = tree.query(points, k=2)
        assert result.indices.shape == (5, 2)
        assert_array_equal(result.indices[:, 0], np.arange(5))


# =============================================================================
# Abstract Base Class Contract Tests
# =============================================================================


class TestBaseSpatialIndexContract:
    """Test that all trees properly implement BaseSpatialIndex contract."""

    def test_inheritance(self, tree_class):
        """Test proper inheritance from BaseSpatialIndex."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)
        assert isinstance(tree, BaseSpatialIndex)

    def test_required_attributes(self, tree_class):
        """Test required attributes exist."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)

        assert hasattr(tree, "data")
        assert hasattr(tree, "n_samples")
        assert hasattr(tree, "n_features")

    def test_len_method(self, tree_class):
        """Test __len__ returns n_samples."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)
        assert len(tree) == 3

    def test_repr_method(self, tree_class):
        """Test __repr__ returns informative string."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = tree_class(points)
        repr_str = repr(tree)
        assert tree_class.__name__ in repr_str
        assert "n_samples=3" in repr_str
        assert "n_features=2" in repr_str


class TestMetricSpatialIndexContract:
    """Test that metric trees properly implement MetricSpatialIndex contract."""

    def test_inheritance(self, metric_tree_class):
        """Test proper inheritance from MetricSpatialIndex."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = metric_tree_class(points)
        assert isinstance(tree, MetricSpatialIndex)
        assert isinstance(tree, BaseSpatialIndex)

    def test_metric_attribute(self, metric_tree_class):
        """Test metric attribute exists."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = metric_tree_class(points)
        assert hasattr(tree, "metric")
        assert callable(tree.metric)

    def test_default_euclidean_metric(self, metric_tree_class):
        """Test default metric is Euclidean."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        tree = metric_tree_class(points)

        dist = tree.metric(np.array([0, 0]), np.array([3, 4]))
        assert_allclose(dist, 5.0)


# =============================================================================
# Correctness Verification Tests
# =============================================================================


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

        result = tree.query(points, k=1)

        assert_array_equal(result.indices[:, 0], np.arange(100))
        assert_allclose(result.distances[:, 0], 0.0)

    def test_distances_sorted(self):
        """Distances are returned in sorted order."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (20, 2))
        tree = KDTree(points)

        result = tree.query([[5, 5]], k=10)

        distances = result.distances[0]
        assert np.all(distances[:-1] <= distances[1:])

    def test_balltree_matches_kdtree(self):
        """BallTree results match KDTree."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 10, (50, 3))

        kd = KDTree(points)
        ball = BallTree(points)

        queries = rng.uniform(0, 10, (10, 3))

        kd_result = kd.query(queries, k=5)
        ball_result = ball.query(queries, k=5)

        for i in range(10):
            kd_set = set(kd_result.indices[i])
            ball_set = set(ball_result.indices[i])
            assert len(kd_set & ball_set) >= 4


# =============================================================================
# Result Type Tests
# =============================================================================


class TestNearestNeighborResultType:
    """Tests for NearestNeighborResult."""

    def test_namedtuple(self):
        """Is a proper named tuple."""
        result = NearestNeighborResult(
            indices=np.array([[0, 1]]),
            distances=np.array([[0.0, 1.0]]),
        )

        assert_array_equal(result.indices, [[0, 1]])
        assert_array_equal(result.distances, [[0.0, 1.0]])


# =============================================================================
# Performance Tests
# =============================================================================


class TestSpatialTreePerformance:
    """Performance-related tests."""

    def test_query_large_dataset(self):
        """Tree query works for larger datasets."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (500, 3))
        queries = rng.uniform(0, 100, (10, 3))

        tree = KDTree(points)
        result = tree.query(queries, k=10)

        assert result.indices.shape == (10, 10)

    def test_radius_query_efficiency(self):
        """Radius query is efficient."""
        rng = np.random.default_rng(42)
        points = rng.uniform(0, 100, (500, 2))

        tree = KDTree(points)
        result = tree.query_radius([[50, 50]], r=5)

        assert isinstance(result[0], list)


class TestKDTreeLeafSize:
    """``leaf_size`` must affect the tree and never the answers.

    ``KDTree._build_tree`` recursed to one point per node and never read
    ``self.leaf_size``, so the documented "Maximum number of points in a leaf
    node" did nothing -- while ``BallTree``, in the same module, honoured the
    identical parameter. The pair of properties below is what distinguishes
    an implemented parameter from an ignored one: the structure changes, the
    results do not.
    """

    @staticmethod
    def _depth(node):
        if node is None:
            return 0
        return 1 + max(
            TestKDTreeLeafSize._depth(node.left),
            TestKDTreeLeafSize._depth(node.right),
        )

    def test_leaf_size_changes_the_tree_shape(self):
        X = np.random.RandomState(0).randn(200, 3)
        depths = [self._depth(KDTree(X, leaf_size=ls).root) for ls in (1, 10, 100)]
        assert depths[0] > depths[1] > depths[2]

    def test_a_large_leaf_size_collapses_the_tree_to_one_bucket(self):
        X = np.random.RandomState(1).randn(50, 2)
        tree = KDTree(X, leaf_size=1000)
        assert tree.root.bucket is not None
        assert len(tree.root.bucket) == 50

    @pytest.mark.parametrize("leaf_size", [1, 2, 10, 1000])
    def test_results_are_identical_to_brute_force(self, leaf_size):
        rs = np.random.RandomState(2)
        X = rs.randn(80, 3)
        queries = rs.randn(10, 3)
        exact = np.sqrt(((queries[:, None, :] - X[None, :, :]) ** 2).sum(-1))

        tree = KDTree(X, leaf_size=leaf_size)

        result = tree.query(queries, k=5)
        assert_allclose(
            np.sort(result.distances, axis=1), np.sort(exact, axis=1)[:, :5]
        )

        for i, q in enumerate(queries):
            found = sorted(tree.query_radius(q.reshape(1, -1), 1.0)[0])
            expected = sorted(np.where(exact[i] <= 1.0)[0].tolist())
            assert found == expected
