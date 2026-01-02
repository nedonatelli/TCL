"""
Benchmarks for gating and Mahalanobis distance computations.

These are light benchmarks that run on every PR to catch performance regressions
in the Numba-optimized distance calculations.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.gating import mahalanobis_distance


class TestMahalanobisDistanceBenchmarks:
    """Benchmark Mahalanobis distance computations."""

    @pytest.mark.light
    def test_mahalanobis_2d(self, benchmark, covariance_matrices):
        """Benchmark 2D Mahalanobis distance (optimized path)."""
        S = covariance_matrices[2]
        innovation = np.array([1.0, 0.5])

        # Warm up Numba JIT
        _ = mahalanobis_distance(innovation, S)

        result = benchmark(mahalanobis_distance, innovation, S)

        assert result >= 0.0

    @pytest.mark.light
    def test_mahalanobis_3d(self, benchmark, covariance_matrices):
        """Benchmark 3D Mahalanobis distance (optimized path)."""
        S = covariance_matrices[3]
        innovation = np.array([1.0, 0.5, -0.3])

        # Warm up Numba JIT
        _ = mahalanobis_distance(innovation, S)

        result = benchmark(mahalanobis_distance, innovation, S)

        assert result >= 0.0

    @pytest.mark.light
    def test_mahalanobis_6d(self, benchmark, covariance_matrices):
        """Benchmark 6D Mahalanobis distance (general path)."""
        S = covariance_matrices[6]
        np.random.seed(42)
        innovation = np.random.randn(6)

        # Warm up Numba JIT
        _ = mahalanobis_distance(innovation, S)

        result = benchmark(mahalanobis_distance, innovation, S)

        assert result >= 0.0


class TestBatchMahalanobisBenchmarks:
    """Benchmark batched Mahalanobis distance computations."""

    @pytest.mark.light
    def test_batch_100_3d(self, benchmark, random_point_clouds, covariance_matrices):
        """Benchmark 100 3D Mahalanobis distances."""
        points = random_point_clouds[100][3]
        S = covariance_matrices[3]
        center = np.zeros(3)

        def compute_all():
            distances = []
            for p in points:
                d = mahalanobis_distance(p - center, S)
                distances.append(d)
            return distances

        # Warm up
        _ = compute_all()

        result = benchmark(compute_all)

        assert len(result) == 100

    @pytest.mark.full
    def test_batch_1000_3d(self, benchmark, random_point_clouds, covariance_matrices):
        """Benchmark 1000 3D Mahalanobis distances."""
        points = random_point_clouds[1000][3]
        S = covariance_matrices[3]
        center = np.zeros(3)

        def compute_all():
            distances = []
            for p in points:
                d = mahalanobis_distance(p - center, S)
                distances.append(d)
            return distances

        # Warm up
        _ = compute_all()

        result = benchmark(compute_all)

        assert len(result) == 1000


class TestGatingScenarioBenchmarks:
    """Benchmark realistic gating scenarios."""

    @pytest.mark.light
    def test_gate_20_tracks_50_meas(self, benchmark, gating_test_data):
        """Benchmark gating 50 measurements against 20 tracks."""
        data = gating_test_data
        threshold = 9.21  # chi2(3, 0.99)

        def perform_gating():
            gated = []
            for i, (track_state, track_cov) in enumerate(
                zip(data["track_states"], data["track_covs"])
            ):
                for j, meas in enumerate(data["measurements"]):
                    innovation = meas - track_state
                    d2 = mahalanobis_distance(innovation, track_cov)
                    if d2 < threshold:
                        gated.append((i, j, d2))
            return gated

        # Warm up
        _ = perform_gating()

        result = benchmark(perform_gating)

        assert isinstance(result, list)
