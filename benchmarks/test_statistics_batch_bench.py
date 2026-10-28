"""
Benchmarks for the batch NEES and Mahalanobis paths (v2.12.0 Task 6.2).

Both batch paths are vectorised over the sample axis; a per-row Python loop
was ~200x slower at this size.
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.statistics.distributions import (
    MultivariateGaussian,
)
from pytcl.mathematical_functions.statistics.estimators import nees

M, N = 10000, 6


class TestBatchStatisticsBenchmarks:
    """Benchmark batch statistics at m=10000, n=6."""

    @pytest.mark.light
    def test_batch_nees(self, benchmark, covariance_matrices):
        """Benchmark estimators.nees on a (10000, 6) error batch."""
        P = covariance_matrices[N]
        errors = np.random.default_rng(42).standard_normal((M, N))

        result = benchmark(nees, errors, P)

        assert result.shape == (M,)

    @pytest.mark.light
    def test_multivariate_gaussian_mahalanobis_batch(
        self, benchmark, covariance_matrices
    ):
        """Benchmark MultivariateGaussian.mahalanobis on a (10000, 6) batch."""
        dist = MultivariateGaussian(np.zeros(N), covariance_matrices[N])
        x = np.random.default_rng(42).standard_normal((M, N))

        result = benchmark(dist.mahalanobis, x)

        assert result.shape == (M,)
