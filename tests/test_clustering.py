"""Tests for K-means clustering."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.clustering import (
    KMeansResult,
    kmeans_plusplus_init,
    assign_clusters,
    update_centers,
    kmeans,
    kmeans_elbow,
)


class TestKMeansPlusPlusInit:
    """Tests for K-means++ initialization."""

    def test_correct_number_of_centers(self):
        """Returns correct number of centers."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))

        centers = kmeans_plusplus_init(X, n_clusters=5, rng=rng)

        assert centers.shape == (5, 3)

    def test_centers_from_data(self):
        """Centers are selected from data points."""
        rng = np.random.default_rng(42)
        X = np.array([[0, 0], [1, 0], [0, 1], [10, 10], [11, 10], [10, 11]])

        centers = kmeans_plusplus_init(X, n_clusters=2, rng=rng)

        # Each center should match a data point
        for center in centers:
            distances = np.sqrt(np.sum((X - center) ** 2, axis=1))
            assert np.min(distances) < 1e-10

    def test_spread_initialization(self):
        """Centers should be spread out."""
        rng = np.random.default_rng(42)
        # Two well-separated clusters
        X = np.vstack([
            rng.normal(0, 0.5, (50, 2)),
            rng.normal(10, 0.5, (50, 2))
        ])

        centers = kmeans_plusplus_init(X, n_clusters=2, rng=rng)

        # Centers should not be too close
        dist = np.linalg.norm(centers[0] - centers[1])
        assert dist > 5  # Should be roughly 10

    def test_error_on_too_many_clusters(self):
        """Raises error if n_clusters > n_samples."""
        X = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError):
            kmeans_plusplus_init(X, n_clusters=5)


class TestAssignClusters:
    """Tests for cluster assignment."""

    def test_correct_assignment(self):
        """Points are assigned to nearest center."""
        X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
        centers = np.array([[0.5, 0], [10.5, 10]])

        labels, inertia = assign_clusters(X, centers)

        assert_array_equal(labels, [0, 0, 1, 1])

    def test_inertia_calculation(self):
        """Inertia is sum of squared distances."""
        X = np.array([[0, 0], [2, 0]])
        centers = np.array([[1, 0]])  # Both points are 1 away

        labels, inertia = assign_clusters(X, centers)

        # Inertia = 1^2 + 1^2 = 2
        assert_allclose(inertia, 2.0)


class TestUpdateCenters:
    """Tests for center update."""

    def test_center_is_mean(self):
        """Updated center is mean of assigned points."""
        X = np.array([[0, 0], [2, 0], [4, 0], [10, 10]])
        labels = np.array([0, 0, 0, 1])

        centers = update_centers(X, labels, n_clusters=2)

        assert_allclose(centers[0], [2, 0])  # Mean of first 3 points
        assert_allclose(centers[1], [10, 10])


class TestKMeans:
    """Tests for K-means algorithm."""

    def test_well_separated_clusters(self):
        """Finds well-separated clusters."""
        rng = np.random.default_rng(42)
        X1 = rng.normal(0, 0.5, (50, 2))
        X2 = rng.normal(10, 0.5, (50, 2))
        X = np.vstack([X1, X2])

        result = kmeans(X, n_clusters=2, rng=rng)

        assert result.converged
        assert len(np.unique(result.labels)) == 2

        # Each cluster should contain mostly points from one group
        cluster_0_mean = X[result.labels == 0].mean(axis=0)
        cluster_1_mean = X[result.labels == 1].mean(axis=0)

        # One cluster should be near (0,0), other near (10,10)
        dist_to_origin = [np.linalg.norm(cluster_0_mean), np.linalg.norm(cluster_1_mean)]
        assert min(dist_to_origin) < 2
        assert max(dist_to_origin) > 8

    def test_convergence(self):
        """Algorithm converges within max_iter."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        result = kmeans(X, n_clusters=5, max_iter=300, rng=rng)

        assert result.n_iter <= 300

    def test_reproducibility(self):
        """Same seed produces same results."""
        X = np.random.default_rng(42).standard_normal((100, 2))

        result1 = kmeans(X, n_clusters=3, rng=np.random.default_rng(123))
        result2 = kmeans(X, n_clusters=3, rng=np.random.default_rng(123))

        assert_array_equal(result1.labels, result2.labels)
        assert_allclose(result1.centers, result2.centers)

    def test_with_initial_centers(self):
        """Works with user-provided initial centers."""
        X = np.array([[0, 0], [1, 0], [10, 10], [11, 10]])
        init_centers = np.array([[0, 0], [10, 10]])

        result = kmeans(X, n_clusters=2, init=init_centers)

        assert result.converged
        assert len(np.unique(result.labels)) == 2

    def test_random_init(self):
        """Random initialization works."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        result = kmeans(X, n_clusters=3, init="random", rng=rng)

        assert result.labels.shape == (100,)
        assert result.centers.shape == (3, 2)

    def test_n_init_multiple_runs(self):
        """Multiple initializations find better solution."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        result_1 = kmeans(X, n_clusters=3, n_init=1, rng=rng)
        result_10 = kmeans(X, n_clusters=3, n_init=10, rng=rng)

        # 10 initializations should find same or better solution
        assert result_10.inertia <= result_1.inertia * 1.1  # Allow small variance

    def test_error_on_invalid_clusters(self):
        """Raises error on invalid n_clusters."""
        X = np.array([[0, 0], [1, 1]])

        with pytest.raises(ValueError):
            kmeans(X, n_clusters=0)

        with pytest.raises(ValueError):
            kmeans(X, n_clusters=10)

    def test_result_type(self):
        """Returns KMeansResult."""
        X = np.random.default_rng(42).standard_normal((50, 2))

        result = kmeans(X, n_clusters=3)

        assert isinstance(result, KMeansResult)
        assert hasattr(result, 'labels')
        assert hasattr(result, 'centers')
        assert hasattr(result, 'inertia')
        assert hasattr(result, 'n_iter')
        assert hasattr(result, 'converged')


class TestKMeansElbow:
    """Tests for elbow method helper."""

    def test_range_of_k(self):
        """Computes inertia for range of k."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        results = kmeans_elbow(X, k_range=range(1, 6), rng=rng)

        assert len(results['k_values']) == 5
        assert len(results['inertias']) == 5
        assert results['k_values'] == [1, 2, 3, 4, 5]

    def test_decreasing_inertia(self):
        """Inertia should generally decrease with more clusters."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        results = kmeans_elbow(X, k_range=range(1, 6), rng=rng)

        # k=1 should have highest inertia
        assert results['inertias'][0] >= results['inertias'][-1]


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_point(self):
        """Works with single data point."""
        X = np.array([[1.0, 2.0]])

        result = kmeans(X, n_clusters=1)

        assert_allclose(result.centers[0], [1.0, 2.0])
        assert result.labels[0] == 0

    def test_n_clusters_equals_n_samples(self):
        """Works when n_clusters == n_samples."""
        X = np.array([[0, 0], [1, 1], [2, 2]])

        result = kmeans(X, n_clusters=3)

        # Each point is its own cluster
        assert len(np.unique(result.labels)) == 3

    def test_collinear_data(self):
        """Works with collinear data."""
        X = np.array([[i, 0] for i in range(10)])

        result = kmeans(X, n_clusters=2, rng=np.random.default_rng(42))

        assert result.converged

    def test_high_dimensional(self):
        """Works with higher dimensions."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 20))

        result = kmeans(X, n_clusters=5, rng=rng)

        assert result.centers.shape == (5, 20)
        assert result.labels.shape == (100,)
