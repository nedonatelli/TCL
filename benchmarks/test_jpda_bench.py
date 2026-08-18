"""
Benchmarks for JPDA (Joint Probabilistic Data Association) algorithms.

These are full benchmarks that run on main branch merges and nightly builds.
"""

import pytest

from pytcl.assignment_algorithms.jpda import jpda_probabilities, jpda_update


class TestJPDAFullPipelineBenchmarks:
    """Benchmark the full JPDA update pipeline at ROADMAP target scale."""

    @pytest.mark.full
    def test_jpda_update_100_targets_50_meas(
        self, benchmark, jpda_full_pipeline_100_50
    ):
        """Benchmark jpda_update: 100 targets, 50 measurements.

        This reproduces the ROADMAP.md "JPDA (100 targets, 50 meas)"
        performance target. It runs jpda_update end to end (likelihood
        matrix + gating, association probabilities, per-track Kalman
        update) rather than jpda_probabilities alone -- the raw numba
        kernel is microseconds-scale at this size and does not account
        for the target's measured cost.
        """
        data = jpda_full_pipeline_100_50

        # Warm up Numba JIT
        _ = jpda_update(
            data["track_states"],
            data["track_covariances"],
            data["measurements"],
            data["H"],
            data["R"],
        )

        result = benchmark(
            jpda_update,
            data["track_states"],
            data["track_covariances"],
            data["measurements"],
            data["H"],
            data["R"],
        )

        assert len(result.states) == 100


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
