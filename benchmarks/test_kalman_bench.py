"""
Benchmarks for Kalman filter operations.

These are light benchmarks that run on every PR to catch performance regressions
in the core filtering algorithms.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update


class TestKalmanPredictBenchmarks:
    """Benchmark Kalman filter prediction step."""

    @pytest.mark.light
    @pytest.mark.parametrize("state_dim", [4, 6, 9])
    def test_kf_predict(self, benchmark, kalman_test_matrices, state_dim):
        """Benchmark kf_predict for various state dimensions."""
        m = kalman_test_matrices[state_dim]

        result = benchmark(kf_predict, m["x"], m["P"], m["F"], m["Q"])

        assert result.x.shape == (state_dim,)
        assert result.P.shape == (state_dim, state_dim)

    @pytest.mark.light
    def test_kf_predict_12state(self, benchmark, kalman_test_matrices):
        """Benchmark kf_predict for 12-state (large IMM-style) filter."""
        m = kalman_test_matrices[12]

        result = benchmark(kf_predict, m["x"], m["P"], m["F"], m["Q"])

        assert result.x.shape == (12,)


class TestKalmanUpdateBenchmarks:
    """Benchmark Kalman filter update step."""

    @pytest.mark.light
    @pytest.mark.parametrize(
        "state_dim,meas_dim",
        [(4, 2), (6, 3), (9, 3)],
    )
    def test_kf_update(
        self, benchmark, kalman_test_matrices, measurement_matrices, state_dim, meas_dim
    ):
        """Benchmark kf_update for various dimensions."""
        km = kalman_test_matrices[state_dim]
        mm = measurement_matrices[(state_dim, meas_dim)]

        result = benchmark(kf_update, km["x"], km["P"], mm["z"], mm["H"], mm["R"])

        assert result.x.shape == (state_dim,)
        assert result.P.shape == (state_dim, state_dim)
        assert result.K.shape == (state_dim, meas_dim)

    @pytest.mark.light
    def test_kf_update_12state(
        self, benchmark, kalman_test_matrices, measurement_matrices
    ):
        """Benchmark kf_update for 12-state filter."""
        km = kalman_test_matrices[12]
        mm = measurement_matrices[(12, 4)]

        result = benchmark(kf_update, km["x"], km["P"], mm["z"], mm["H"], mm["R"])

        assert result.x.shape == (12,)


class TestKalmanCycleBenchmarks:
    """Benchmark full predict-update cycles."""

    @pytest.mark.light
    def test_10_cycles_state_4(
        self, benchmark, kalman_test_matrices, measurement_matrices
    ):
        """Benchmark 10 predict-update cycles with 4-state tracker."""
        km = kalman_test_matrices[4]
        mm = measurement_matrices[(4, 2)]
        np.random.seed(42)
        measurements = [np.random.randn(2) for _ in range(10)]

        def run_cycles():
            x, P = km["x"].copy(), km["P"].copy()
            for z in measurements:
                pred = kf_predict(x, P, km["F"], km["Q"])
                upd = kf_update(pred.x, pred.P, z, mm["H"], mm["R"])
                x, P = upd.x, upd.P
            return x

        result = benchmark(run_cycles)
        assert result.shape == (4,)

    @pytest.mark.full
    def test_100_cycles_state_6(
        self, benchmark, kalman_test_matrices, measurement_matrices
    ):
        """Benchmark 100 predict-update cycles with 6-state tracker."""
        km = kalman_test_matrices[6]
        mm = measurement_matrices[(6, 3)]
        np.random.seed(42)
        measurements = [np.random.randn(3) for _ in range(100)]

        def run_cycles():
            x, P = km["x"].copy(), km["P"].copy()
            for z in measurements:
                pred = kf_predict(x, P, km["F"], km["Q"])
                upd = kf_update(pred.x, pred.P, z, mm["H"], mm["R"])
                x, P = upd.x, upd.P
            return x

        result = benchmark(run_cycles)
        assert result.shape == (6,)
