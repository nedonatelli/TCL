"""
Benchmarks for JPDA (Joint Probabilistic Data Association) algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest

from pytcl.assignment_algorithms.jpda import jpda_probabilities


class TestJPDAProbabilityBenchmarks:
    """Benchmark JPDA probability computations."""

    @pytest.mark.full
    def test_jpda_small(self, benchmark, jpda_test_data):
        """Benchmark JPDA with 5 tracks, 10 measurements."""
        scenario = jpda_test_data["small"]
        likelihood = scenario["likelihood_matrix"]
        gated = scenario["gated"]
        pd = scenario["detection_prob"]
        clutter = scenario["clutter_density"]

        # Warm up Numba JIT
        _ = jpda_probabilities(likelihood, gated, pd, clutter)

        result = benchmark(jpda_probabilities, likelihood, gated, pd, clutter)

        assert result.shape[0] == likelihood.shape[0]

    @pytest.mark.full
    def test_jpda_medium(self, benchmark, jpda_test_data):
        """Benchmark JPDA with 10 tracks, 20 measurements."""
        scenario = jpda_test_data["medium"]
        likelihood = scenario["likelihood_matrix"]
        gated = scenario["gated"]
        pd = scenario["detection_prob"]
        clutter = scenario["clutter_density"]

        # Warm up
        _ = jpda_probabilities(likelihood, gated, pd, clutter)

        result = benchmark(jpda_probabilities, likelihood, gated, pd, clutter)

        assert result.shape[0] == likelihood.shape[0]

    @pytest.mark.full
    def test_jpda_large(self, benchmark, jpda_test_data):
        """Benchmark JPDA with 20 tracks, 50 measurements."""
        scenario = jpda_test_data["large"]
        likelihood = scenario["likelihood_matrix"]
        gated = scenario["gated"]
        pd = scenario["detection_prob"]
        clutter = scenario["clutter_density"]

        # Warm up
        _ = jpda_probabilities(likelihood, gated, pd, clutter)

        result = benchmark(jpda_probabilities, likelihood, gated, pd, clutter)

        assert result.shape[0] == likelihood.shape[0]


class TestJPDAParameterBenchmarks:
    """Benchmark JPDA with different parameters."""

    @pytest.mark.full
    @pytest.mark.parametrize("pd", [0.7, 0.9, 0.99])
    def test_jpda_detection_prob(self, benchmark, jpda_test_data, pd):
        """Benchmark JPDA with varying detection probability."""
        scenario = jpda_test_data["medium"]
        likelihood = scenario["likelihood_matrix"]
        gated = scenario["gated"]
        clutter = scenario["clutter_density"]

        # Warm up
        _ = jpda_probabilities(likelihood, gated, pd, clutter)

        result = benchmark(jpda_probabilities, likelihood, gated, pd, clutter)

        assert result is not None

    @pytest.mark.full
    @pytest.mark.parametrize("clutter", [1e-8, 1e-6, 1e-4])
    def test_jpda_clutter_density(self, benchmark, jpda_test_data, clutter):
        """Benchmark JPDA with varying clutter density."""
        scenario = jpda_test_data["medium"]
        likelihood = scenario["likelihood_matrix"]
        gated = scenario["gated"]
        pd = scenario["detection_prob"]

        # Warm up
        _ = jpda_probabilities(likelihood, gated, pd, clutter)

        result = benchmark(jpda_probabilities, likelihood, gated, pd, clutter)

        assert result is not None
