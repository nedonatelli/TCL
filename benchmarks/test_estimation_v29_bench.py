"""
Benchmarks for the v2.9.0 dynamic-estimation ports.

All tests are @pytest.mark.full (nightly/on-main only): the file is new
and unmeasured against benchmark-light.yml's ~2-minute-per-file budget,
and none of these entries has an SLO yet -- per the gate-calibration
doctrine (see .benchmarks/slos.json derivations), SLOs are set from
observed CI history, so these run SLO-less first to accumulate it.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation import (
    batch_ls_lin_meas_lin_dyn,
    batch_ls_nonlin_meas_lin_dyn,
    fp_info_batch_smoother,
    kalman_batch_smoother,
    kalman_fir_smoother,
    pcrlb_pred_add,
    pcrlb_update_add_no_clutter,
    riccati_pred_no_clutter,
    sqrt_cub_kal_batch_smoother,
)
from pytcl.dynamic_estimation.kalman import (
    blue_polar_meas_update,
    enkf_update,
    qmc_kf_update,
    sqrt_ckf_predict,
    sqrt_ckf_update,
)

F4 = np.kron(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(2))
H4 = np.hstack([np.eye(2), np.zeros((2, 2))])
Q4 = np.kron(np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]), 0.01 * np.eye(2))
R2 = 0.1 * np.eye(2)
X0_4 = np.array([0.0, 0.0, 1.0, -0.5])
P0_4 = np.diag([1.0, 1.0, 0.25, 0.25])

_rng = np.random.default_rng(1234)
N_STEPS = 50
Z_BATCH = H4 @ (
    np.column_stack([np.linalg.matrix_power(F4, k) @ X0_4 for k in range(N_STEPS)])
) + 0.3 * _rng.standard_normal((2, N_STEPS))


class TestBatchSmootherBenchmarks:
    """The v2.9.0 batch and interval smoothers."""

    @pytest.mark.full
    def test_kalman_batch_smoother_rts_50_steps(self, benchmark):
        result = benchmark(
            kalman_batch_smoother,
            X0_4,
            P0_4,
            Z_BATCH,
            None,
            H4,
            F4,
            R2,
            Q4,
            None,
            False,
        )
        assert result.x.shape == (4, N_STEPS)

    @pytest.mark.full
    def test_fp_info_batch_smoother_50_steps(self, benchmark):
        result = benchmark(
            fp_info_batch_smoother, None, None, Z_BATCH, None, H4, F4, R2, Q4
        )
        assert result.x.shape == (4, N_STEPS)

    @pytest.mark.full
    def test_sqrt_cub_batch_smoother_20_steps(self, benchmark):
        s_r = np.linalg.cholesky(R2)
        s_q = np.linalg.cholesky(Q4)
        s0 = np.linalg.cholesky(P0_4)
        z20 = Z_BATCH[:, :20]
        result = benchmark(
            sqrt_cub_kal_batch_smoother,
            X0_4,
            s0,
            z20,
            lambda x: H4 @ x,
            lambda x: F4 @ x,
            s_r,
            s_q,
        )
        assert result.x.shape == (4, 20)

    @pytest.mark.full
    def test_kalman_fir_smoother_20_steps(self, benchmark):
        n = 20
        h_stack = np.repeat(H4[:, :, np.newaxis], n, axis=2)
        f_stack = np.repeat(F4[:, :, np.newaxis], n - 1, axis=2)
        r_stack = np.repeat(R2[:, :, np.newaxis], n, axis=2)
        q_stack = np.repeat(Q4[:, :, np.newaxis], n - 1, axis=2)
        result = benchmark(
            kalman_fir_smoother,
            Z_BATCH[:, :n],
            None,
            h_stack,
            f_stack,
            r_stack,
            q_stack,
            n // 2,
        )
        assert result.x.shape == (4,)


class TestBatchLSBenchmarks:
    """The v2.9.0 batch least-squares estimators."""

    @pytest.mark.full
    def test_batch_ls_lin_50_meas(self, benchmark):
        result = benchmark(batch_ls_lin_meas_lin_dyn, Z_BATCH, H4, F4, R2, 0, Q4)
        assert result.x.shape == (4,)

    @pytest.mark.full
    def test_batch_ls_gauss_newton_20_meas(self, benchmark):
        z20 = Z_BATCH[:, :20]
        result = benchmark(
            batch_ls_nonlin_meas_lin_dyn,
            X0_4 + 0.1,
            z20,
            lambda x: H4 @ x,
            F4,
            R2,
            0,
            lambda x: H4,
            10,
        )
        assert result.x.shape == (4,)


class TestSampledFilterBenchmarks:
    """The v2.9.0 ensemble and Monte-Carlo filter steps."""

    @pytest.mark.full
    def test_enkf_update_100_members(self, benchmark):
        rng = np.random.default_rng(7)
        x_ens = X0_4[:, np.newaxis] + 0.3 * rng.standard_normal((4, 100))
        s_r = np.linalg.cholesky(R2)
        z = H4 @ X0_4 + 0.1
        w_samp = s_r @ rng.standard_normal((2, 100))

        def run():
            return enkf_update(x_ens.copy(), z, s_r, lambda x: H4 @ x, 0, w_samp=w_samp)

        result = benchmark(run)
        assert result.x_update.shape == (4,)

    @pytest.mark.full
    def test_qmc_kf_update_1000_samples(self, benchmark):
        z = H4 @ X0_4 + 0.1

        def run():
            return qmc_kf_update(
                X0_4,
                P0_4,
                z,
                R2,
                lambda x: H4 @ x,
                1000,
                rng=np.random.default_rng(11),
            )

        result = benchmark(run)
        assert result.x.shape == (4,)

    @pytest.mark.full
    def test_blue_polar_meas_update(self, benchmark):
        x = np.array([1000.0, 10.0, 500.0, -5.0])
        p = np.diag([100.0, 4.0, 100.0, 4.0])
        z = np.array([np.hypot(1000.0, 500.0), np.arctan2(500.0, 1000.0)])
        r = np.diag([25.0, 1e-4])
        result = benchmark(blue_polar_meas_update, x, p, z, r)
        assert result.x.shape == (4,)

    @pytest.mark.full
    def test_sqrt_ckf_cycle(self, benchmark):
        s0 = np.linalg.cholesky(P0_4)
        s_q = np.linalg.cholesky(Q4)
        s_r = np.linalg.cholesky(R2)
        z = H4 @ X0_4 + 0.1

        def run():
            pred = sqrt_ckf_predict(X0_4, s0, lambda x: F4 @ x, s_q)
            return sqrt_ckf_update(pred.x, pred.S, z, s_r, lambda x: H4 @ x)

        result = benchmark(run)
        assert result.x.shape == (4,)


class TestPerformancePredictionBenchmarks:
    """The v2.9.0 Riccati/PCRLB performance-prediction tools."""

    F6 = np.kron(np.array([[1.0, 1.0], [0.0, 1.0]]), np.eye(3))
    H6 = np.hstack([np.eye(3), np.zeros((3, 3))])
    Q6 = np.kron(np.array([[1 / 3, 1 / 2], [1 / 2, 1.0]]), np.eye(3))
    R3 = np.diag([10.0, 10.0, 10.0])

    @pytest.mark.full
    def test_riccati_pred_pd05(self, benchmark):
        result = benchmark(
            riccati_pred_no_clutter, self.H6, self.F6, self.R3, self.Q6, 0.5
        )
        assert result.converged

    @pytest.mark.full
    def test_pcrlb_cubature_cycle(self, benchmark):
        x_prior = np.array([1.0, 0.5])
        p_prior = np.array([[0.09, 0.02], [0.02, 0.06]])
        q2 = np.array([[0.02, 0.01], [0.01, 0.03]])
        r1 = np.array([[4.0]])
        fj = lambda x: np.array([[1.0, 1.0], [-0.1 * x[0], 1.0]])  # noqa: E731
        hj = lambda x: np.array([[2.0 * x[0], 0.2]])  # noqa: E731

        def run():
            j = pcrlb_pred_add(0.5 * np.eye(2), x_prior, p_prior, q2, fj)
            return pcrlb_update_add_no_clutter(j, x_prior, p_prior, r1, 0.8, hj)

        result = benchmark(run)
        assert result.shape == (2, 2)
