"""Static localization estimators against MATLAB TCL reference fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_static_estimation.m.
Every input in this file mirrors that capture script verbatim; the
fixtures hold MATLAB's outputs.

REFERENCE-class tests, with PROPERTY-class exact-recovery checks (an
error-free geometry must reproduce the true target) alongside.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.coordinate_systems.rotations import rot_axis_to_vec
from pytcl.static_estimation import (
    ad_hoc_cart_cov,
    direction_only_static_loc_est,
    poly_meas_fim,
    range_only_static_loc_est_np,
    range_rate_ratio_to_static_pos_2d,
    range_rate_to_static_pos,
    rr_only_static_vel_est,
    tdoa_only_static_loc_est,
    tdoa_to_cart,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Literal transcriptions of closed-form linear algebra: measured max
# disagreement 1e-13 absolute on coordinates in the thousands.
RTOL_EXACT = 1e-9
# The minimal-measurement two-solution case takes a square root of a
# near-cancellation; measured max relative disagreement 1e-9.
RTOL_MINIMAL = 1e-7


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


def _assert_same_solution_set(actual, desired, rtol):
    """Order-independent one-to-one matching of solution columns
    (column order out of the polynomial solver is tie-unstable across
    platforms)."""
    assert actual.shape == desired.shape
    scale = max(1.0, float(np.max(np.abs(desired))))
    remaining = list(range(actual.shape[1]))
    for j in range(desired.shape[1]):
        dists = [np.max(np.abs(actual[:, i] - desired[:, j])) for i in remaining]
        best = int(np.argmin(dists))
        assert dists[best] <= rtol * scale, (
            f"no solution within {rtol * scale} of fixture column "
            f"{desired[:, j]}; nearest was {dists[best]} away"
        )
        remaining.pop(best)


# The five-sensor scene of TDOAOnlyStaticLocEst.m's docstring example.
S1 = np.array([9.0, 39.0, 100.0])
S2 = np.array([65.0, 10.0, -60.0])
S3 = np.array([64.0, 71.0, 43.0])
S4 = np.array([-128.0, 6.0, 12.0])
S5 = np.array([0.0, -20.0, 4.0])
C_SOUND = 341.0
T_TRUE = np.array([27.0, 0.0, -42.0])


def _tdoa(target, rx, ref, c):
    return (
        np.linalg.norm(target[:, None] - rx, axis=0) - np.linalg.norm(target - ref)
    ) / c


class TestTdoaOnlyStaticLocEst:
    RX = np.column_stack([S2, S3, S4, S5])

    def test_single_reference_matches_matlab(self):
        delays = _tdoa(T_TRUE, self.RX, S1, C_SOUND)
        loc = tdoa_only_static_loc_est(delays, S1, self.RX, C_SOUND)
        np.testing.assert_allclose(
            loc, _load("se_tdoa_only_case1.csv").ravel(), rtol=RTOL_EXACT, atol=1e-9
        )

    def test_error_free_delays_recover_the_emitter(self):
        delays = _tdoa(T_TRUE, self.RX, S1, C_SOUND)
        loc = tdoa_only_static_loc_est(delays, S1, self.RX, C_SOUND)
        np.testing.assert_allclose(loc, T_TRUE, atol=1e-9)

    def test_perturbed_delays_match_matlab(self):
        delays = _tdoa(T_TRUE, self.RX, S1, C_SOUND) + np.array(
            [2e-4, -1e-4, 3e-4, -2e-4]
        )
        loc = tdoa_only_static_loc_est(delays, S1, self.RX, C_SOUND)
        np.testing.assert_allclose(
            loc, _load("se_tdoa_only_case2.csv").ravel(), rtol=RTOL_EXACT
        )

    def test_two_references_match_matlab(self):
        non_ref1 = np.column_stack([S2, S3, S4])
        non_ref2 = np.column_stack([S2, S3])
        delays1 = _tdoa(T_TRUE, non_ref1, S1, C_SOUND)
        delays2 = _tdoa(T_TRUE, non_ref2, S5, C_SOUND)
        loc = tdoa_only_static_loc_est(
            [delays1, delays2],
            np.column_stack([S1, S5]),
            [non_ref1, non_ref2],
            C_SOUND,
        )
        np.testing.assert_allclose(
            loc, _load("se_tdoa_only_case3.csv").ravel(), rtol=RTOL_EXACT, atol=1e-9
        )
        np.testing.assert_allclose(loc, T_TRUE, atol=1e-8)

    def test_too_few_measurements_raise(self):
        with pytest.raises(ValueError, match="minimum"):
            tdoa_only_static_loc_est(
                np.zeros(3), S1, np.column_stack([S2, S3, S4]), C_SOUND
            )


# The bistatic range scene from the capture script.
T_LOC = np.array([4e3, -2e3, 3e3])
Z_RX = np.array([100.0, 200.0, -50.0])
Z_TX5 = np.array(
    [
        [0.0, 8e3, -6e3, 2e3, -3e3],
        [0.0, 1e3, 5e3, -7e3, 2e3],
        [0.0, -2e3, 1e3, 4e3, 9e3],
    ]
)
R_B5 = np.linalg.norm(T_LOC[:, None] - Z_TX5, axis=0) + np.linalg.norm(T_LOC - Z_RX)
# Central differences with a 1e-2 m step on ranges of ~1e4 m: truncation and
# roundoff each sit near 1e-9 relative; measured worst element 1.7e-9.
TAYLOR_FD_RTOL = 1e-6


def _bistatic_range_jacobian(x, z_tx, z_rx):
    """Analytic d r_i / d x of r_i = abs(x - z_rx) + abs(x - z_tx_i)."""
    to_tx = x[:, None] - z_tx
    return (x - z_rx) / np.linalg.norm(x - z_rx) + (
        to_tx / np.linalg.norm(to_tx, axis=0)
    ).T


def _estimator_jacobian_fd(r, z_tx, z_rx, step=1e-2):
    """Central-difference d x_est / d r of the spherical-intersection
    estimator itself: the oracle for its first-order (Taylor) covariance,
    independent of any closed-form derivative."""
    cols = []
    for i in range(len(r)):
        dr = np.zeros(len(r))
        dr[i] = step
        hi = range_only_static_loc_est_np(r + dr, z_tx, z_rx, 1).x_est
        lo = range_only_static_loc_est_np(r - dr, z_tx, z_rx, 1).x_est
        cols.append((hi - lo) / (2.0 * step))
    return np.column_stack(cols)


class TestRangeOnlyStaticLocEstNP:
    def test_spherical_intersection_matches_matlab(self):
        result = range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 1)
        np.testing.assert_allclose(
            result.x_est, _load("se_range_only_case1.csv").ravel(), rtol=RTOL_EXACT
        )
        np.testing.assert_allclose(result.x_est, T_LOC, rtol=1e-9)
        assert result.p_taylor is None and result.p_crlb is None

    def test_spherical_interpolation_matches_matlab(self):
        result = range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 0)
        np.testing.assert_allclose(
            result.x_est, _load("se_range_only_case2.csv").ravel(), rtol=RTOL_EXACT
        )

    def test_minimal_system_returns_both_solutions(self):
        result = range_only_static_loc_est_np(R_B5[:3], Z_TX5[:, :3], Z_RX, 1)
        ref = _load("se_range_only_case3.csv")
        assert result.x_est.shape == (3, 2)
        np.testing.assert_allclose(result.x_est, ref, rtol=RTOL_MINIMAL)
        # The true target is the second solution in this geometry.
        np.testing.assert_allclose(result.x_est[:, 1], T_LOC, rtol=1e-6)

    def test_noisy_estimate_matches_matlab_covariances_diverge_by_design(self):
        r_noisy = R_B5 + np.array([3.0, -5.0, 2.0, -1.0, 4.0])
        r_cov = np.diag([9.0, 25.0, 4.0, 1.0, 16.0])
        result = range_only_static_loc_est_np(r_noisy, Z_TX5, Z_RX, 1, r_cov)
        ref = _load("se_range_only_case4.csv").ravel()
        np.testing.assert_allclose(result.x_est, ref[:3], rtol=RTOL_EXACT)
        g = _estimator_jacobian_fd(r_noisy, Z_TX5, Z_RX)
        np.testing.assert_allclose(
            result.p_taylor[:, :, 0], g @ r_cov @ g.T, rtol=TAYLOR_FD_RTOL, atol=0
        )
        j = _bistatic_range_jacobian(result.x_est, Z_TX5, Z_RX)
        np.testing.assert_allclose(
            result.p_crlb[:, :, 0],
            np.linalg.inv(j.T @ np.linalg.inv(r_cov) @ j),
            rtol=1e-9,
        )

    def test_matlab_covariances_diverge_by_the_documented_factors(self):
        # The fixture's PTaylor (ref[3:12]) and PCRLB (ref[12:21]) carry the
        # upstream defects. Pin the ratios quoted in the docstring/CHANGELOG
        # so the divergence is explicit; rtol 1e-3 covers their rounding.
        r_noisy = R_B5 + np.array([3.0, -5.0, 2.0, -1.0, 4.0])
        r_cov = np.diag([9.0, 25.0, 4.0, 1.0, 16.0])
        result = range_only_static_loc_est_np(r_noisy, Z_TX5, Z_RX, 1, r_cov)
        ref = _load("se_range_only_case4.csv").ravel()
        matlab_taylor = ref[3:12].reshape(3, 3, order="F")
        matlab_crlb = ref[12:21].reshape(3, 3, order="F")
        np.testing.assert_allclose(
            np.diag(result.p_taylor[:, :, 0]) / np.diag(matlab_taylor),
            [2.271, 3.904, 7.881],
            rtol=1e-3,
        )
        np.testing.assert_allclose(
            np.diag(matlab_crlb) / np.diag(result.p_crlb[:, :, 0]),
            [17.08, 19.61, 1.106],
            rtol=1e-3,
        )

    def test_clipped_discriminant_taylor_is_the_estimator_derivative(self):
        # This draw makes Equation 21's discriminant negative (about -2.4e7),
        # so the root term is clipped to 0 and abs(x_est) != Rt. Linearizing
        # with the abs(x) constraint there gave diag [12040, 20746, 59357].
        r_clip = R_B5 + np.array([1.0, -28.0, 129.0, 101.0, -271.0])
        s_rel = (Z_TX5 - Z_RX[:, None]).T
        s_star = np.linalg.pinv(s_rel)
        a = s_star @ (0.5 * (np.sum(s_rel**2, axis=1) - r_clip**2))
        b = s_star @ r_clip
        assert (a @ b) ** 2 - ((b @ b) - 1.0) * (a @ a) < -1e6
        r_cov = 4.0 * np.eye(5)
        result = range_only_static_loc_est_np(r_clip, Z_TX5, Z_RX, 1, r_cov)
        g = _estimator_jacobian_fd(r_clip, Z_TX5, Z_RX)
        np.testing.assert_allclose(
            result.p_taylor[:, :, 0], g @ r_cov @ g.T, rtol=TAYLOR_FD_RTOL, atol=0
        )

    def test_taylor_covariance_is_the_linearization_of_the_returned_root(self):
        # rangeOnlyStaticLocEstNP linearizes about xEst1 even when xEst2 is
        # returned (it is here), and uses lsqminnorm(Delta, ...), which is not
        # the derivative of the SStar-based estimate when num_meas > 3. With
        # R = 4 I the MATLAB diagonal was [3.543, 4.434, 8.407].
        r_cov = 4.0 * np.eye(5)
        result = range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 1, r_cov)
        np.testing.assert_allclose(result.x_est, T_LOC, rtol=1e-9)
        g = _estimator_jacobian_fd(R_B5, Z_TX5, Z_RX)
        np.testing.assert_allclose(
            g @ _bistatic_range_jacobian(T_LOC, Z_TX5, Z_RX), np.eye(3), atol=1e-6
        )
        np.testing.assert_allclose(
            result.p_taylor[:, :, 0], g @ r_cov @ g.T, rtol=TAYLOR_FD_RTOL, atol=0
        )

    def test_crlb_is_the_fisher_inverse(self):
        # The original inverts dxdr R^-1 dxdr' (estimator Jacobian, wrong
        # orientation); the diagonal came out 17x-20x the true bound here.
        r_cov = np.diag([9.0, 25.0, 4.0, 1.0, 16.0])
        result = range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 1, r_cov)
        j = _bistatic_range_jacobian(T_LOC, Z_TX5, Z_RX)
        expected = np.linalg.inv(j.T @ np.linalg.inv(r_cov) @ j)
        np.testing.assert_allclose(result.p_crlb[:, :, 0], expected, rtol=1e-9)

    def test_minimal_system_covariances_per_solution(self):
        r_cov = np.diag([9.0, 25.0, 4.0])
        tx = Z_TX5[:, :3]
        result = range_only_static_loc_est_np(R_B5[:3], tx, Z_RX, 1, r_cov)
        for k in range(2):
            x = result.x_est[:, k]
            j_inv = np.linalg.inv(_bistatic_range_jacobian(x, tx, Z_RX))
            np.testing.assert_allclose(
                result.p_taylor[:, :, k], j_inv @ r_cov @ j_inv.T, rtol=1e-6
            )
            np.testing.assert_allclose(
                result.p_crlb[:, :, k], j_inv @ r_cov @ j_inv.T, rtol=1e-6
            )

    def test_taylor_covariance_matches_monte_carlo_spread(self):
        # PROPERTY: the sample covariance of the returned estimate over noisy
        # draws. Relative std of a 2000-draw variance is sqrt(2/2000) = 3.2%;
        # measured disagreement at this seed is 2.1%. The pre-fix p_taylor
        # was 2.4x-8.1x too small on these diagonals.
        r_cov = 4.0 * np.eye(5)
        p_taylor = range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 1, r_cov).p_taylor[
            :, :, 0
        ]
        rng = np.random.default_rng(20260914)
        noise = rng.standard_normal((2000, 5)) @ np.linalg.cholesky(r_cov).T
        draws = np.array(
            [
                range_only_static_loc_est_np(R_B5 + n, Z_TX5, Z_RX, 1).x_est
                for n in noise
            ]
        )
        np.testing.assert_allclose(
            np.diag(np.cov(draws.T)), np.diag(p_taylor), rtol=0.12
        )

    def test_exactly_coplanar_sensors_with_r_cov_warn(self):
        # Receiver and transmitters all on the ground plane z = 0, target
        # above it: z is unidentified, so both covariances are infinite
        # there, and the estimate lies in the sensors' plane. v2.11.0
        # returned a covariance with no error; v2.11.1 raised LinAlgError.
        # The identified block is checked against a planar oracle in
        # test_unidentifiable_covariance.py.
        z_rx = np.zeros(3)
        z_tx = np.array(
            [
                [1e3, 8e3, -6e3, 2e3, -3e3],
                [0.0, 1e3, 5e3, -7e3, 2e3],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        r = np.linalg.norm(T_LOC[:, None] - z_tx, axis=0) + np.linalg.norm(T_LOC)
        with pytest.warns(RuntimeWarning, match=r"components \[2\] unidentified"):
            res = range_only_static_loc_est_np(r, z_tx, z_rx, 1, 4.0 * np.eye(5))
        assert res.x_est[2] == 0.0
        for p in (res.p_taylor[:, :, 0], res.p_crlb[:, :, 0]):
            assert np.isposinf(p[2, 2])
            assert np.all(np.isnan(p[2, :2]))
            assert np.all(np.isnan(p[:2, 2]))
            assert np.all(np.isfinite(p[:2, :2]))

    def test_r_cov_with_method_zero_raises(self):
        with pytest.raises(ValueError, match="only for method 1"):
            range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 0, np.eye(5))

    def test_too_few_measurements_raise(self):
        with pytest.raises(ValueError, match="three"):
            range_only_static_loc_est_np(R_B5[:2], Z_TX5[:, :2], Z_RX)

    def test_method_zero_refuses_minimal_systems(self):
        with pytest.raises(ValueError, match="num_meas == 3"):
            range_only_static_loc_est_np(R_B5[:3], Z_TX5[:, :3], Z_RX, 0)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            range_only_static_loc_est_np(R_B5, Z_TX5, Z_RX, 2)


class TestRROnlyStaticVelEst:
    # The three-channel bistatic scene of RROnlyStaticVelEst.m example 1.
    Z_TAR = np.array([0.0, 40e3, 40e3])
    V_TAR = np.array([400.0, -200.0, 100.0])
    STATES_TX = np.column_stack(
        [
            [100.0, 10e3, 3e3, 50.0, 50.0, -50.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, -20.0],
            [10e3, 10e3, 3e3, 100.0, -100.0, 100.0],
        ]
    )
    STATES_RX = np.column_stack(
        [
            [-10e3, 0.0, 3e3, 100.0, 100.0, 100.0],
            [0.0, 10e3, 30.0, -80.0, -200.0, -20.0],
            [0.0, 10e3, 30.0, -80.0, -200.0, -20.0],
        ]
    )

    def test_3d_bistatic_matches_matlab_and_recovers_velocity(self):
        row = _load("se_rr_only_case1.csv").ravel()
        rr, ref = row[:3], row[3:]
        v = rr_only_static_vel_est(rr, self.STATES_TX, self.STATES_RX, self.Z_TAR)
        np.testing.assert_allclose(v, ref, rtol=RTOL_EXACT)
        np.testing.assert_allclose(v, self.V_TAR, rtol=1e-9)

    def test_half_range_doubles_the_rates(self):
        row = _load("se_rr_only_case1.csv").ravel()
        ref = _load("se_rr_only_case2.csv").ravel()
        v = rr_only_static_vel_est(
            row[:3] / 2.0,
            self.STATES_TX,
            self.STATES_RX,
            self.Z_TAR,
            use_half_range=True,
        )
        np.testing.assert_allclose(v, ref, rtol=RTOL_EXACT)
        np.testing.assert_allclose(v, row[3:], rtol=RTOL_EXACT)

    def test_2d_matches_matlab(self):
        row = _load("se_rr_only_case3.csv").ravel()
        tx = np.column_stack(
            [
                [100.0, 10e3, 50.0, 50.0],
                [0.0, 0.0, 0.0, 0.0],
                [10e3, 10e3, 100.0, -100.0],
            ]
        )
        rx = np.column_stack(
            [
                [-10e3, 0.0, 100.0, 100.0],
                [0.0, 10e3, -80.0, -200.0],
                [0.0, 10e3, -80.0, -200.0],
            ]
        )
        v = rr_only_static_vel_est(row[:3], tx, rx, np.array([0.0, 40e3]))
        np.testing.assert_allclose(v, row[3:], rtol=RTOL_EXACT)
        np.testing.assert_allclose(v, [400.0, -200.0], rtol=1e-9)

    def test_target_is_transmitter_matches_matlab(self):
        row = _load("se_rr_only_case4.csv").ravel()
        x_rx = np.array(
            [
                [0.5, -1.2, 2.0, 0.0, -0.8],
                [1.0, 0.3, -1.5, 2.2, 0.7],
                [-0.6, 1.8, 0.4, -1.0, 1.3],
                [0.1, -0.5, 0.7, 0.2, -0.3],
                [-0.2, 0.4, 0.1, -0.6, 0.5],
                [0.3, 0.2, -0.4, 0.1, -0.1],
            ]
        )
        v = rr_only_static_vel_est(row[:5], None, x_rx, np.array([1.5, -0.4, 2.2]))
        np.testing.assert_allclose(v, row[5:], rtol=RTOL_EXACT)
        np.testing.assert_allclose(v, [0.3, 1.1, -0.7], rtol=1e-9)

    def test_shared_sensor_state_broadcasts(self):
        # A single (6,) receiver state must behave as if repeated per
        # measurement; verified by comparing against explicit tiling.
        rr = np.array([5.0, 5.0, 5.0])
        rx_one = np.array([1e3, 2e3, 3e3, 10.0, -20.0, 5.0])
        rx_tiled = np.tile(rx_one[:, None], (1, 3))
        v_one = rr_only_static_vel_est(rr, self.STATES_TX, rx_one, self.Z_TAR)
        v_tiled = rr_only_static_vel_est(rr, self.STATES_TX, rx_tiled, self.Z_TAR)
        np.testing.assert_allclose(v_one, v_tiled, rtol=1e-12)


class TestAdHocCartCov:
    def test_3d_two_beamwidths_matches_matlab(self):
        V = ad_hoc_cart_cov(
            5e6, [np.deg2rad(2.0), np.deg2rad(10.0)], 10.0, [1e3, 1e3, 1e3]
        )
        np.testing.assert_allclose(V, _load("se_adhoc_cov_case1.csv"), rtol=RTOL_EXACT)

    def test_3d_scalar_beamwidth_matches_matlab(self):
        V = ad_hoc_cart_cov(2e6, np.deg2rad(3.0), 15.0, [-2e3, 500.0, 1e3])
        np.testing.assert_allclose(V, _load("se_adhoc_cov_case2.csv"), rtol=RTOL_EXACT)

    def test_2d_matches_matlab(self):
        V = ad_hoc_cart_cov(5e6, np.deg2rad(2.0), 10.0, [3e3, -4e3])
        np.testing.assert_allclose(V, _load("se_adhoc_cov_case3.csv"), rtol=RTOL_EXACT)

    def test_result_is_symmetric_positive_definite(self):
        V = ad_hoc_cart_cov(
            5e6, [np.deg2rad(2.0), np.deg2rad(10.0)], 10.0, [1e3, -2e3, 3e3]
        )
        np.testing.assert_allclose(V, V.T, rtol=1e-12)
        assert np.all(np.linalg.eigvalsh(V) > 0)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError, match="dim"):
            ad_hoc_cart_cov(5e6, 0.03, 10.0, [1.0, 2.0, 3.0], dim=4)


class TestRotAxisToVec:
    U3 = np.array([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_3d_matches_matlab(self, axis):
        R = rot_axis_to_vec(self.U3, axis)
        ref = _load(f"rot_axis2vec_3d_ax{axis + 1}.csv")
        np.testing.assert_allclose(R, ref, rtol=0, atol=1e-14)

    def test_negative_leading_component_matches_matlab(self):
        R = rot_axis_to_vec(np.array([-1.0, 0.5, -2.0]), 0)
        np.testing.assert_allclose(
            R, _load("rot_axis2vec_3d_neg.csv"), rtol=0, atol=1e-14
        )

    def test_2d_matches_matlab(self):
        R = rot_axis_to_vec(np.array([3.0, -4.0]), 1)
        np.testing.assert_allclose(R, _load("rot_axis2vec_2d.csv"), rtol=0, atol=1e-14)

    def test_8d_matches_matlab(self):
        u = np.array([53.0, 183.0, -225.0, 86.0, 31.0, -130.0, -43.0, 34.0])
        R = rot_axis_to_vec(u, 4)
        np.testing.assert_allclose(R, _load("rot_axis2vec_8d.csv"), rtol=0, atol=1e-14)

    @pytest.mark.parametrize("axis", ["x", "y", "z"])
    def test_axis_maps_onto_the_direction(self, axis):
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        e = np.zeros(3)
        e[idx] = 1.0
        R = rot_axis_to_vec(self.U3, axis)
        np.testing.assert_allclose(R @ e, self.U3 / np.linalg.norm(self.U3), atol=1e-14)

    def test_it_is_a_proper_rotation(self):
        for u in ([1.0, 2.0, 3.0], [-5.0, 0.1, 0.0], [0.0, 0.0, -1.0]):
            R = rot_axis_to_vec(np.array(u))
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-13)
            assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)

    def test_batched_input_stacks_matrices(self):
        u = np.column_stack([self.U3, [-1.0, 0.5, -2.0]])
        R = rot_axis_to_vec(u, 0)
        assert R.shape == (3, 3, 2)
        np.testing.assert_allclose(R[:, :, 0], rot_axis_to_vec(self.U3, 0))
        np.testing.assert_allclose(
            R[:, :, 1], rot_axis_to_vec(np.array([-1.0, 0.5, -2.0]), 0)
        )

    def test_default_axis_is_z_in_3d_and_x_otherwise(self):
        np.testing.assert_allclose(
            rot_axis_to_vec(self.U3), rot_axis_to_vec(self.U3, 2)
        )
        u2 = np.array([3.0, -4.0])
        np.testing.assert_allclose(rot_axis_to_vec(u2), rot_axis_to_vec(u2, 0))

    def test_scalar_dimension_returns_identity(self):
        np.testing.assert_allclose(rot_axis_to_vec(np.array([5.0])), np.ones((1, 1)))

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            rot_axis_to_vec(self.U3, "w")
        with pytest.raises(ValueError):
            rot_axis_to_vec(self.U3, 3)


class TestTdoaToCart:
    """TDOA2Cart: minimal TDOA sets through the polynomial solver."""

    def test_3d_matches_matlab_and_recovers_target(self):
        c = 341.0

        def d(a, b):
            return np.linalg.norm(T_TRUE - a) - np.linalg.norm(T_TRUE - b)

        tdoa = np.array([d(S2, S1), d(S3, S1), d(S3, S4)]) / c
        res = tdoa_to_cart(
            tdoa, np.column_stack([S1, S1, S4]), np.column_stack([S2, S3, S3]), c
        )
        assert res.exit_code == 0
        _assert_same_solution_set(res.z_cart, _load("se_tdoa2cart_3d.csv"), 1e-9)
        assert np.min(np.linalg.norm(res.z_cart - T_TRUE[:, None], axis=0)) < 1e-6

    def test_2d_shared_reference_matches_matlab(self):
        c = 341.0
        t2 = np.array([350.0, -125.0])
        r1 = np.array([0.0, 0.0])
        r2 = np.array([800.0, 200.0])
        r3 = np.array([-300.0, 700.0])

        def d(a, b):
            return np.linalg.norm(t2 - a) - np.linalg.norm(t2 - b)

        tdoa = np.array([d(r2, r1), d(r3, r1)]) / c
        # A single reference column must broadcast to all measurements.
        res = tdoa_to_cart(tdoa, r1, np.column_stack([r2, r3]), c)
        assert res.exit_code == 0
        _assert_same_solution_set(res.z_cart, _load("se_tdoa2cart_2d.csv"), 1e-9)
        assert np.min(np.linalg.norm(res.z_cart - t2[:, None], axis=0)) < 1e-6

    def test_invalid_dimensionality_raises(self):
        with pytest.raises(ValueError, match="dimensionality"):
            tdoa_to_cart(np.zeros(4), np.zeros((4, 4)), np.zeros((4, 4)))


class TestRangeRateToStaticPos:
    """rangeRate2StaticPos: emitter from minimal range rates."""

    U2 = np.array([1e3, 5e3])
    S2D = np.array([[500.0, 1100.0], [2500.0, 2500.0]])
    SD2D = np.array([[300.0, 300.0], [0.0, 0.0]])

    def _rr(self, u, s, sd):
        return np.array(
            [
                -sd[:, k] @ (u - s[:, k]) / np.linalg.norm(u - s[:, k])
                for k in range(s.shape[1])
            ]
        )

    def test_2d_matches_matlab_and_recovers_emitter(self):
        rr = self._rr(self.U2, self.S2D, self.SD2D)
        res = range_rate_to_static_pos(rr, np.vstack([self.S2D, self.SD2D]))
        assert res.exit_code == 0
        _assert_same_solution_set(res.z_cart, _load("se_rr2pos_2d.csv"), 1e-6)
        assert np.min(np.linalg.norm(res.z_cart - self.U2[:, None], axis=0)) < 1e-3

    def test_3d_matches_matlab_and_recovers_emitter(self):
        u3 = np.array([1e3, 5e3, 2e3])
        s3 = np.array(
            [[500.0, 1100.0, 5000.0], [3000.0, 1000.0, -1000.0], [0.0, 2000.0, 1000.0]]
        )
        sd3 = np.array([[300.0, 100.0, 0.0], [0.0, 100.0, 300.0], [0.0, 0.0, 0.0]])
        rr = self._rr(u3, s3, sd3)
        res = range_rate_to_static_pos(rr, np.vstack([s3, sd3]))
        assert res.exit_code == 0
        # The 3D solutions pass through the eigendecomposition of a
        # heavier lifted system; measured set agreement 2.2e-3 absolute
        # on kilometre-scale coordinates (~1e-6 relative).
        _assert_same_solution_set(res.z_cart, _load("se_rr2pos_3d.csv"), 1e-5)
        assert np.min(np.linalg.norm(res.z_cart - u3[:, None], axis=0)) < 1e-3

    def test_invalid_dimensionality_raises(self):
        with pytest.raises(ValueError, match="dimensionality"):
            range_rate_to_static_pos(np.zeros(4), np.zeros((8, 4)))


class TestRangeRateRatioToStaticPos2D:
    """rangeRateRatio2StaticPos2D: Doppler-ratio-only localization."""

    def test_matches_matlab_and_recovers_emitter(self):
        u = np.array([1e3, 5e3])
        ref = np.array([1000.0, 3000.0, 150.0, -150.0])
        s = np.array([[500.0, 1100.0], [2500.0, 2500.0]])
        sd = np.array([[300.0, 300.0], [0.0, 0.0]])
        c = 299792458.0

        def rrate(p, v):
            return -v @ (u - p) / np.linalg.norm(u - p)

        rr_ref = rrate(ref[:2], ref[2:])
        f_rat = np.array(
            [(1 - rr_ref / c) / (1 - rrate(s[:, k], sd[:, k]) / c) for k in (0, 1)]
        )
        res = range_rate_ratio_to_static_pos_2d(f_rat, ref, np.vstack([s, sd]))
        assert res.exit_code == 0
        _assert_same_solution_set(res.z_cart, _load("se_rrratio_2d.csv"), 1e-6)
        assert np.min(np.linalg.norm(res.z_cart - u[:, None], axis=0)) < 1e-3


class TestPolyMeasFim:
    """computePolyMeasFIM: CRLB via cubature for mixed measurement sets."""

    X = np.array([1e3, 2e3, 3e3])
    SENSORS = np.array(
        [
            [0.0, 8e3, -6e3, 2e3],
            [0.0, 1e3, 5e3, -7e3],
            [0.0, -2e3, 1e3, 4e3],
        ]
    )
    C = 299792458.0

    def test_tdoa_and_range_matches_matlab(self):
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        xi, w = fifth_order_cubature_points(3)
        fim = poly_meas_fim(
            self.X,
            [1e-14, 1e-14, 100.0],
            None,
            [0, 0, 1],
            np.array([[0, 0, 2], [1, 2, 3]]),
            self.SENSORS,
            self.C,
            xi,
            w,
        )
        ref = _load("se_fim_tdoa_range.csv")
        np.testing.assert_allclose(fim, ref, rtol=1e-12)
        assert np.all(np.linalg.eigvalsh(fim) > 0)

    def test_range_rate_and_frequency_matches_matlab(self):
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        vel = np.array(
            [
                [50.0, -80.0, 20.0, 90.0],
                [30.0, 60.0, -70.0, -40.0],
                [-20.0, 10.0, 50.0, 25.0],
            ]
        )
        xi, w = fifth_order_cubature_points(4)
        fim = poly_meas_fim(
            self.X,
            [0.25, 0.25, 400.0, 400.0],
            1e9,
            [2, 2, 3, 3],
            np.array([[0, 1, 2, 3], [0, 0, 0, 0]]),
            np.vstack([self.SENSORS, vel]),
            self.C,
            xi,
            w,
        )
        ref = _load("se_fim_rr_freq.csv")
        np.testing.assert_allclose(fim, ref, rtol=1e-12)
        # Frequency measurements append an f_tx row/column.
        assert fim.shape == (4, 4)

    def test_frequency_without_f_tx_raises(self):
        with pytest.raises(ValueError, match="f_tx"):
            poly_meas_fim(
                self.X,
                [1.0],
                None,
                [3],
                np.array([[0], [0]]),
                np.vstack([self.SENSORS, np.zeros((3, 4))]),
            )

    def test_unknown_measurement_type_raises(self):
        with pytest.raises(ValueError, match="measurement type"):
            poly_meas_fim(self.X, [1.0], None, [7], np.array([[0], [1]]), self.SENSORS)


class TestDirectionOnlyStaticLocEst:
    """directionOnlyStaticLocEst. All MATLAB fixtures use identity RInv,
    where the original's RInv-overwrite bug is a no-op and MATLAB
    remains a valid oracle; the port's fixed weighting is checked by a
    property test instead."""

    T_TRUE = np.array([500.0, 800.0, 1200.0])
    L_RX = np.array(
        [
            [0.0, 1000.0, -200.0, 400.0],
            [0.0, 100.0, 900.0, -300.0],
            [0.0, -200.0, 300.0, 700.0],
        ]
    )

    @property
    def u(self):
        u = self.T_TRUE[:, None] - self.L_RX
        return u / np.linalg.norm(u, axis=0)

    def test_alg0_matches_matlab_and_recovers_target(self):
        res = direction_only_static_loc_est(self.u, self.L_RX, 0)
        assert res.exit_code == 0
        np.testing.assert_allclose(
            res.t, _load("se_dironly_alg0.csv").ravel(), atol=1e-9
        )
        np.testing.assert_allclose(res.t, self.T_TRUE, atol=1e-8)

    def test_alg0_weighted_ls_matches_matlab(self):
        w = np.stack([np.diag([1.0, 2.0, 4.0]) * (k + 1) for k in range(4)], axis=2)
        res = direction_only_static_loc_est(self.u, self.L_RX, 0, w=w, num_iter=0)
        np.testing.assert_allclose(
            res.t, _load("se_dironly_alg0_weighted.csv").ravel(), atol=1e-9
        )

    def test_alg0_constrained_matches_matlab(self):
        # The port substitutes a bounds-constrained SciPy solve for
        # convexQuadProg; the convex objective has one minimum.
        res = direction_only_static_loc_est(
            self.u, self.L_RX, 0, use_const_alg=True, num_iter=0
        )
        np.testing.assert_allclose(
            res.t, _load("se_dironly_alg0_const.csv").ravel(), atol=1e-6
        )

    def test_alg2_known_ranges_matches_matlab(self):
        r = np.linalg.norm(self.T_TRUE[:, None] - self.L_RX, axis=0)
        res = direction_only_static_loc_est(self.u, self.L_RX, 2, r=r)
        np.testing.assert_allclose(
            res.t, _load("se_dironly_alg2.csv").ravel(), atol=1e-9
        )

    def test_alg1_converges_at_least_as_deep_as_matlab(self):
        # SciPy BFGS replaces quasiNetwonBFGS: same cost surface, so
        # the port must land within the same shallow bowl (metres on a
        # km scale) at a cost no worse than the MATLAB optimum.
        from pytcl.static_estimation.localization import _direction_cost

        u_noisy = _load("se_dironly_unoisy.csv")
        res = direction_only_static_loc_est(u_noisy, self.L_RX, 1)
        assert res.exit_code == 0
        t_ml = _load("se_dironly_alg1.csv").ravel()
        assert np.max(np.abs(res.t - t_ml)) < 10.0
        r_inv = np.tile(np.eye(3)[:, :, None], (1, 1, 4))
        cost_py, _ = _direction_cost(res.t, u_noisy, self.L_RX, r_inv)
        cost_ml, _ = _direction_cost(t_ml, u_noisy, self.L_RX, r_inv)
        assert cost_py <= cost_ml + 1e-12

    def test_alg3_from_offset_start_recovers_target(self):
        res = direction_only_static_loc_est(
            self.u, self.L_RX, 3, t_init=self.T_TRUE + [50.0, -80.0, 30.0]
        )
        np.testing.assert_allclose(res.t, self.T_TRUE, atol=1e-5)

    def test_known_r_honors_weights_unlike_the_original(self):
        # The MATLAB triangulateKnownR overwrites RInv with eye(3);
        # the port honors it, so unequal weights must move the estimate
        # off the identity-weight solution when directions are noisy.
        u_noisy = _load("se_dironly_unoisy.csv")
        r = np.linalg.norm(self.T_TRUE[:, None] - self.L_RX, axis=0)
        r_inv = np.stack([np.eye(3) * (10.0**k) for k in range(4)], axis=2)
        res_w = direction_only_static_loc_est(u_noisy, self.L_RX, 2, r=r, r_inv=r_inv)
        res_i = direction_only_static_loc_est(u_noisy, self.L_RX, 2, r=r)
        assert np.max(np.abs(res_w.t - res_i.t)) > 1e-3

    def test_2d_works(self):
        # The original crashes every 2D call through triangulateKnownR
        # (a 3x3 assigned into a 2x2 slot); the port must not.
        t2 = np.array([300.0, 400.0])
        l2 = np.array([[0.0, 500.0, -100.0], [0.0, 50.0, 600.0]])
        u2 = t2[:, None] - l2
        u2 = u2 / np.linalg.norm(u2, axis=0)
        res = direction_only_static_loc_est(u2, l2, 0)
        np.testing.assert_allclose(res.t, t2, atol=1e-8)

    def test_missing_required_params_raise(self):
        with pytest.raises(ValueError, match="ranges r"):
            direction_only_static_loc_est(self.u, self.L_RX, 2)
        with pytest.raises(ValueError, match="t_init"):
            direction_only_static_loc_est(self.u, self.L_RX, 3)
        with pytest.raises(ValueError, match="algorithm"):
            direction_only_static_loc_est(self.u, self.L_RX, 9)
