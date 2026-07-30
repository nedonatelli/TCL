"""Tests for clustering algorithms (K-means, DBSCAN, Hierarchical)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.clustering import (  # K-means; DBSCAN; Hierarchical
    DBSCANResult,
    HierarchicalResult,
    KMeansResult,
    agglomerative_clustering,
    assign_clusters,
    compute_neighbors,
    cut_dendrogram,
    dbscan,
    dbscan_predict,
    fcluster,
    kmeans,
    kmeans_elbow,
    kmeans_plusplus_init,
    update_centers,
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
        X = np.vstack([rng.normal(0, 0.5, (50, 2)), rng.normal(10, 0.5, (50, 2))])

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
        dist_to_origin = [
            np.linalg.norm(cluster_0_mean),
            np.linalg.norm(cluster_1_mean),
        ]
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
        assert hasattr(result, "labels")
        assert hasattr(result, "centers")
        assert hasattr(result, "inertia")
        assert hasattr(result, "n_iter")
        assert hasattr(result, "converged")


class TestKMeansElbow:
    """Tests for elbow method helper."""

    def test_range_of_k(self):
        """Computes inertia for range of k."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        results = kmeans_elbow(X, k_range=range(1, 6), rng=rng)

        assert len(results["k_values"]) == 5
        assert len(results["inertias"]) == 5
        assert results["k_values"] == [1, 2, 3, 4, 5]

    def test_decreasing_inertia(self):
        """Inertia should generally decrease with more clusters."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 2))

        results = kmeans_elbow(X, k_range=range(1, 6), rng=rng)

        # k=1 should have highest inertia
        assert results["inertias"][0] >= results["inertias"][-1]


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


# =============================================================================
# DBSCAN Tests
# =============================================================================


class TestComputeNeighbors:
    """Tests for neighbor computation."""

    def test_self_is_neighbor(self):
        """Each point is its own neighbor."""
        X = np.array([[0, 0], [10, 10]])
        neighbors = compute_neighbors(X, eps=1.0)

        assert 0 in neighbors[0]
        assert 1 in neighbors[1]

    def test_close_points_are_neighbors(self):
        """Close points are neighbors."""
        X = np.array([[0, 0], [0.5, 0], [10, 10]])
        neighbors = compute_neighbors(X, eps=1.0)

        assert 1 in neighbors[0]
        assert 0 in neighbors[1]
        assert 2 not in neighbors[0]


class TestDBSCAN:
    """Tests for DBSCAN algorithm."""

    def test_two_clusters(self):
        """Finds two well-separated clusters."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(0, 0.3, (30, 2))
        cluster2 = rng.normal(5, 0.3, (30, 2))
        X = np.vstack([cluster1, cluster2])

        result = dbscan(X, eps=0.8, min_samples=5)

        assert result.n_clusters == 2
        assert len(np.unique(result.labels[result.labels >= 0])) == 2

    def test_noise_detection(self):
        """Detects noise points."""
        rng = np.random.default_rng(42)
        cluster = rng.normal(0, 0.3, (30, 2))
        noise = np.array([[10, 10], [15, 15], [-10, -10]])
        X = np.vstack([cluster, noise])

        result = dbscan(X, eps=0.8, min_samples=5)

        # Noise points should be labeled -1
        assert result.n_noise >= 1
        assert -1 in result.labels

    def test_core_samples(self):
        """Identifies core samples."""
        rng = np.random.default_rng(42)
        cluster = rng.normal(0, 0.2, (20, 2))
        X = cluster

        result = dbscan(X, eps=0.5, min_samples=3)

        # Most points in dense cluster should be core samples
        assert len(result.core_sample_indices) > 0

    def test_result_type(self):
        """Returns DBSCANResult."""
        X = np.random.default_rng(42).standard_normal((50, 2))

        result = dbscan(X, eps=0.5, min_samples=5)

        assert isinstance(result, DBSCANResult)
        assert hasattr(result, "labels")
        assert hasattr(result, "n_clusters")
        assert hasattr(result, "core_sample_indices")
        assert hasattr(result, "n_noise")

    def test_empty_data(self):
        """Handles empty data."""
        X = np.zeros((0, 2))

        result = dbscan(X, eps=0.5)

        assert result.n_clusters == 0
        assert len(result.labels) == 0

    def test_single_point(self):
        """Single point is noise with min_samples > 1."""
        X = np.array([[0.0, 0.0]])

        result = dbscan(X, eps=0.5, min_samples=2)

        assert result.n_clusters == 0
        assert result.labels[0] == -1

    def test_all_noise_sparse_data(self):
        """Sparse data produces all noise."""
        X = np.array([[0, 0], [100, 100], [200, 200]])

        result = dbscan(X, eps=1.0, min_samples=2)

        # All points too far apart
        assert result.n_clusters == 0
        assert result.n_noise == 3


class TestDBSCANPredict:
    """Tests for DBSCAN prediction."""

    def test_predict_nearby_point(self):
        """Predicts cluster for nearby point."""
        # Training data: one cluster
        X_train = np.array([[0, 0], [0.1, 0], [0, 0.1], [-0.1, 0]])
        labels_train = np.array([0, 0, 0, 0])

        X_new = np.array([[0.05, 0.05]])
        labels_new = dbscan_predict(X_new, X_train, labels_train, eps=0.5)

        assert labels_new[0] == 0

    def test_predict_far_point(self):
        """Far point gets no cluster."""
        X_train = np.array([[0, 0], [0.1, 0]])
        labels_train = np.array([0, 0])

        X_new = np.array([[100, 100]])
        labels_new = dbscan_predict(X_new, X_train, labels_train, eps=0.5)

        assert labels_new[0] == -1


# =============================================================================
# Hierarchical Clustering Tests
# =============================================================================


class TestAgglomerativeClustering:
    """Tests for agglomerative clustering."""

    def test_two_clusters(self):
        """Finds two well-separated clusters."""
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (20, 2)), rng.normal(5, 0.5, (20, 2))])

        result = agglomerative_clustering(X, n_clusters=2)

        assert result.n_clusters == 2
        assert len(np.unique(result.labels)) == 2

    def test_linkage_methods(self):
        """All linkage methods work."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 2))

        for linkage in ["single", "complete", "average", "ward"]:
            result = agglomerative_clustering(X, n_clusters=3, linkage=linkage)
            assert result.n_clusters == 3

    def test_distance_threshold(self):
        """Distance threshold controls number of clusters."""
        rng = np.random.default_rng(42)
        # Two tight clusters far apart
        X = np.vstack([rng.normal(0, 0.1, (10, 2)), rng.normal(10, 0.1, (10, 2))])

        result = agglomerative_clustering(X, distance_threshold=1.0)

        # Should find 2 clusters (gap is ~10)
        assert result.n_clusters == 2

    def test_dendrogram(self):
        """Builds valid dendrogram."""
        X = np.array([[0, 0], [1, 0], [5, 0], [6, 0]])

        result = agglomerative_clustering(X, n_clusters=1)

        # n-1 merges for n points
        assert len(result.dendrogram) == 3
        assert result.linkage_matrix.shape == (3, 4)

    def test_result_type(self):
        """Returns HierarchicalResult."""
        X = np.random.default_rng(42).standard_normal((20, 2))

        result = agglomerative_clustering(X, n_clusters=3)

        assert isinstance(result, HierarchicalResult)
        assert hasattr(result, "labels")
        assert hasattr(result, "n_clusters")
        assert hasattr(result, "linkage_matrix")
        assert hasattr(result, "dendrogram")

    def test_empty_data(self):
        """Handles empty data."""
        X = np.zeros((0, 2))

        result = agglomerative_clustering(X, n_clusters=2)

        assert result.n_clusters == 0

    def test_single_point(self):
        """Single point forms one cluster."""
        X = np.array([[0.0, 0.0]])

        result = agglomerative_clustering(X, n_clusters=1)

        assert result.n_clusters == 1
        assert result.labels[0] == 0


class TestCutDendrogram:
    """Tests for dendrogram cutting."""

    def test_cut_by_n_clusters(self):
        """Cut by number of clusters."""
        X = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])

        result = agglomerative_clustering(X, n_clusters=1)

        # Cut to get 2 clusters
        labels = cut_dendrogram(result.linkage_matrix, 4, n_clusters=2)

        assert len(np.unique(labels)) == 2


class TestFcluster:
    """Tests for scipy-compatible fcluster."""

    def test_maxclust(self):
        """maxclust criterion works."""
        X = np.array([[0, 0], [1, 0], [10, 0], [11, 0]])

        result = agglomerative_clustering(X, n_clusters=1)

        labels = fcluster(result.linkage_matrix, 4, 2, criterion="maxclust")

        # 1-indexed labels
        assert np.min(labels) >= 1
        assert len(np.unique(labels)) == 2

    def test_distance_criterion(self):
        """distance criterion works."""
        X = np.array([[0, 0], [0.1, 0], [10, 0], [10.1, 0]])

        result = agglomerative_clustering(X, n_clusters=1, linkage="single")

        labels = fcluster(result.linkage_matrix, 4, 1.0, criterion="distance")

        # Should have 2 clusters (points 0,1 and 2,3)
        assert len(np.unique(labels)) == 2


class TestClusteringIntegration:
    """Integration tests comparing clustering methods."""

    def test_all_methods_find_obvious_clusters(self):
        """All methods find obvious clusters."""
        rng = np.random.default_rng(42)
        # Three well-separated clusters
        X = np.vstack(
            [
                rng.normal([0, 0], 0.3, (30, 2)),
                rng.normal([5, 0], 0.3, (30, 2)),
                rng.normal([2.5, 5], 0.3, (30, 2)),
            ]
        )

        # K-means
        km_result = kmeans(X, n_clusters=3, rng=rng)
        assert len(np.unique(km_result.labels)) == 3

        # DBSCAN
        db_result = dbscan(X, eps=0.8, min_samples=5)
        assert db_result.n_clusters == 3

        # Hierarchical
        hc_result = agglomerative_clustering(X, n_clusters=3)
        assert hc_result.n_clusters == 3
