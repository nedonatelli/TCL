"""
Benchmarks for clustering algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest

from pytcl.clustering.dbscan import dbscan
from pytcl.clustering.kmeans import kmeans


class TestKMeansBenchmarks:
    """Benchmark K-means clustering."""

    @pytest.mark.full
    def test_kmeans_100_points(self, benchmark, clustering_test_data):
        """Benchmark K-means on 100 points."""
        data = clustering_test_data[100]
        k = 5

        result = benchmark(kmeans, data, k)

        assert len(result.labels) == len(data)
        assert len(result.centers) == k

    @pytest.mark.full
    def test_kmeans_500_points(self, benchmark, clustering_test_data):
        """Benchmark K-means on 500 points."""
        data = clustering_test_data[500]
        k = 5

        result = benchmark(kmeans, data, k)

        assert len(result.labels) == len(data)

    @pytest.mark.full
    def test_kmeans_1000_points(self, benchmark, clustering_test_data):
        """Benchmark K-means on 1000 points."""
        data = clustering_test_data[1000]
        k = 5

        result = benchmark(kmeans, data, k)

        assert len(result.labels) == len(data)


class TestDBSCANBenchmarks:
    """Benchmark DBSCAN clustering."""

    @pytest.mark.full
    def test_dbscan_100_points(self, benchmark, clustering_test_data):
        """Benchmark DBSCAN on 100 points."""
        data = clustering_test_data[100]
        eps = 1.0
        min_samples = 3

        # Warm up Numba JIT for distance matrix
        _ = dbscan(data, eps, min_samples)

        result = benchmark(dbscan, data, eps, min_samples)

        assert len(result.labels) == len(data)

    @pytest.mark.full
    def test_dbscan_500_points(self, benchmark, clustering_test_data):
        """Benchmark DBSCAN on 500 points."""
        data = clustering_test_data[500]
        eps = 1.0
        min_samples = 3

        # Warm up
        _ = dbscan(data, eps, min_samples)

        result = benchmark(dbscan, data, eps, min_samples)

        assert len(result.labels) == len(data)

    @pytest.mark.full
    def test_dbscan_1000_points(self, benchmark, clustering_test_data):
        """Benchmark DBSCAN on 1000 points (O(n^2) distance matrix)."""
        data = clustering_test_data[1000]
        eps = 1.0
        min_samples = 5

        # Warm up
        _ = dbscan(data, eps, min_samples)

        result = benchmark(dbscan, data, eps, min_samples)

        assert len(result.labels) == len(data)


class TestClusteringComparisonBenchmarks:
    """Compare clustering algorithm performance."""

    @pytest.mark.full
    def test_kmeans_vs_dbscan_500(self, benchmark, clustering_test_data):
        """Benchmark both algorithms on same data for comparison."""
        data = clustering_test_data[500]

        # This test just runs K-means; DBSCAN is in separate test
        result = benchmark(kmeans, data, 5)

        assert len(result.labels) == len(data)
