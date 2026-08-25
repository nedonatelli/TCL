"""Tests for the standard-exponential-model refraction functions.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_exp_refraction.m.
The receiver/transmitter geometry mirrors that script: an observer at
(0.61 rad, 0.25 rad, 100 m) and a bistatic transmitter at
(0.62 rad, 0.27 rad, 50 m) on the WGS-84 ellipsoid.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.refraction import (
    cart2ruv_std_refrac,
    cart2ruv_std_refrac_cubature,
    reduce_std_refrac_to_sphere,
    ruv2cart_std_refrac,
    ruv2cart_std_refrac_cubature,
    std_refrac_bias_approx,
)
from pytcl.navigation.geodesy import geodetic_to_ecef, osculating_sphere

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Tolerances are pinned from measured MATLAB agreement with ~50x headroom:
# osculating sphere is pure closed-form geometry (measured 7e-16 rel);
# the ray tracers differ only through solve_bvp/solve_ivp vs bvp5c/ode45
# and quad vs integral (measured: range 5e-7 m, direction cosines 8e-8,
# inverse conversion 7e-6 m); bias algorithm 0 is closed-form (6e-14),
# algorithm 1's endpoint slope carries the BVP difference (1.6e-5 rel on
# a ~1e-3 rad value).
ABS_RANGE = 1e-4  # meters
ABS_UV = 1e-5
ABS_CART = 1e-3  # meters
RTOL_EXACT = 1e-12


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


@pytest.fixture(scope="module")
def z_rx():
    return np.array(geodetic_to_ecef(0.61, 0.25, 100.0)).ravel()


@pytest.fixture(scope="module")
def z_tx():
    return np.array(geodetic_to_ecef(0.62, 0.27, 50.0)).ravel()


class TestOsculatingSphereMatlab:
    def test_matches_matlab(self):
        rows = _load("exp_osculating_sphere.csv")
        for lat, lon, r_ref, cx, cy, cz in rows:
            sph = osculating_sphere(lat, lon)
            assert sph.radius == pytest.approx(r_ref, rel=RTOL_EXACT)
            np.testing.assert_allclose(sph.center, [cx, cy, cz], atol=1e-6)

    def test_sphere_touches_ellipsoid_at_the_point(self):
        # The defining property: the surface point lies on the sphere.
        for lat, lon in [(0.0, 0.0), (0.7, 1.2), (-1.2, -2.0)]:
            sph = osculating_sphere(lat, lon)
            surface = np.array(geodetic_to_ecef(lat, lon, 0.0)).ravel()
            dist = np.linalg.norm(surface - sph.center)
            assert dist == pytest.approx(sph.radius, rel=1e-12)
        # Gaussian radius grows toward the poles.
        assert (
            osculating_sphere(0.0, 0.0).radius
            < osculating_sphere(np.pi / 2, 0.0).radius
        )


class TestCart2RuvStdRefracMatlab:
    def test_monostatic_matches_matlab(self, z_rx):
        rows = _load("exp_cart2ruv_mono.csv")
        for row in rows:
            ns = row[0]
            z_tar, z_ref = row[3:6], row[6:9]
            u_tx, u_tar_rx, u_tar_tx = row[9:12], row[12:15], row[15:18]
            res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx, np.eye(3), ns)
            assert res.z[0, 0] == pytest.approx(z_ref[0], abs=ABS_RANGE)
            np.testing.assert_allclose(res.z[1:, 0], z_ref[1:], atol=ABS_UV)
            np.testing.assert_allclose(res.u_tx[:, 0], u_tx, atol=ABS_UV)
            np.testing.assert_allclose(res.u_tar_rx[:, 0], u_tar_rx, atol=ABS_UV)
            np.testing.assert_allclose(res.u_tar_tx[:, 0], u_tar_tx, atol=ABS_UV)

    def test_bistatic_matches_matlab(self, z_rx, z_tx):
        rows = _load("exp_cart2ruv_bistatic.csv")
        for row in rows:
            z_tar, z_ref = row[2:5], row[5:8]
            res = cart2ruv_std_refrac(z_tar, False, z_tx, z_rx, np.eye(3), 313.0)
            assert res.z[0, 0] == pytest.approx(z_ref[0], abs=ABS_RANGE)
            np.testing.assert_allclose(res.z[1:, 0], z_ref[1:], atol=ABS_UV)

    def test_apparent_range_exceeds_geometric(self, z_rx):
        # The optical path through the atmosphere is longer than vacuum.
        z_tar = z_rx + np.array([50e3, 50e3, 20e3])
        res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx)
        assert res.z[0, 0] > np.linalg.norm(z_tar - z_rx)

    def test_half_range_halves(self, z_rx):
        z_tar = z_rx + np.array([50e3, 50e3, 20e3])
        full = cart2ruv_std_refrac(z_tar, False, z_rx, z_rx)
        half = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx)
        assert full.z[0, 0] == pytest.approx(2 * half.z[0, 0], rel=1e-12)

    def test_include_w_completes_unit_vector(self, z_rx):
        z_tar = z_rx + np.array([50e3, 50e3, 20e3])
        res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx, include_w=True)
        assert res.z.shape[0] == 4
        assert np.linalg.norm(res.z[1:, 0]) == pytest.approx(1.0, rel=1e-9)


class TestRuv2CartStdRefracMatlab:
    def test_monostatic_matches_matlab(self, z_rx):
        rows = _load("exp_ruv2cart_mono.csv")
        for row in rows:
            z_meas, z_cart_ref = row[2:5], row[5:8]
            back = ruv2cart_std_refrac(z_meas, True, z_rx, z_rx, np.eye(3), 313.0)
            np.testing.assert_allclose(back[:, 0], z_cart_ref, atol=ABS_CART)

    def test_bistatic_matches_matlab(self, z_rx, z_tx):
        rows = _load("exp_ruv2cart_bistatic.csv")
        for row in rows:
            z_meas, z_cart_ref = row[2:5], row[5:8]
            back = ruv2cart_std_refrac(z_meas, False, z_tx, z_rx, np.eye(3), 313.0)
            np.testing.assert_allclose(back[:, 0], z_cart_ref, atol=ABS_CART)

    def test_round_trip(self, z_rx):
        z_tar = z_rx + np.array([30e3, 60e3, 15e3])
        ruv = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx).z
        back = ruv2cart_std_refrac(ruv, True, z_rx, z_rx)
        np.testing.assert_allclose(back[:, 0], z_tar, atol=1e-2)

    def test_near_vertical_round_trip(self, z_rx):
        # The closed-form branch: target almost straight up.
        up = z_rx / np.linalg.norm(z_rx)
        z_tar = z_rx + 40e3 * up
        ruv = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx).z
        back = ruv2cart_std_refrac(ruv, True, z_rx, z_rx)
        np.testing.assert_allclose(back[:, 0], z_tar, atol=1.0)


class TestStdRefracBiasApproxMatlab:
    def test_matches_matlab(self):
        rows = _load("exp_bias_approx.csv")
        for alg, path_len, el, h, dr_ref, dt_ref in rows:
            res = std_refrac_bias_approx(path_len, el, h, 313.0, None, None, int(alg))
            if int(alg) == 0:
                assert res.delta_r_one_way == pytest.approx(dr_ref, rel=RTOL_EXACT)
                assert res.delta_theta == pytest.approx(dt_ref, rel=RTOL_EXACT)
            else:
                assert res.delta_r_one_way == pytest.approx(dr_ref, rel=1e-6)
                assert res.delta_theta == pytest.approx(dt_ref, rel=1e-3, abs=1e-9)

    def test_algorithms_agree_roughly(self):
        r0 = std_refrac_bias_approx(100e3, 0.1, 100.0, algorithm=0)
        r1 = std_refrac_bias_approx(100e3, 0.1, 100.0, algorithm=1)
        assert r0.delta_r_one_way == pytest.approx(r1.delta_r_one_way, rel=0.05)

    def test_near_vertical_has_zero_angle_bias(self):
        res = std_refrac_bias_approx(100e3, np.pi / 2 - 1e-4, 0.0)
        assert res.delta_theta == 0.0
        assert res.delta_r_one_way > 0

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError, match="49"):
            std_refrac_bias_approx(100e3, np.deg2rad(50), 0.0, algorithm=0)
        with pytest.raises(ValueError, match="algorithm"):
            std_refrac_bias_approx(100e3, 0.1, 0.0, algorithm=2)


class TestReduceStdRefracMatlab:
    def test_matches_matlab(self):
        rows = np.vstack(
            [
                _load("exp_reduce_refrac.csv"),
                # High-altitude, low-refractivity rows where the model
                # genuinely has two sea-level solutions.
                _load("exp_reduce_refrac_two_sol.csv"),
            ]
        )
        saw_two = False
        for n_meas, height, n_sol, s1, s2 in rows:
            vals = reduce_std_refrac_to_sphere(n_meas, height)
            assert len(vals) == int(n_sol)
            ref = sorted(s for s in (s1, s2) if not np.isnan(s))
            np.testing.assert_allclose(sorted(vals), ref, atol=1e-5)
            saw_two = saw_two or int(n_sol) == 2
        # The fixtures must actually exercise the two-solution case.
        assert saw_two

    def test_zero_height_is_identity(self):
        vals = reduce_std_refrac_to_sphere(300.0, 0.0)
        assert vals[0] == pytest.approx(300.0, abs=1e-3)


class TestCubatureConversionsMatlab:
    def test_cart2ruv_cubature_matches_matlab(self, z_rx):
        row = _load("exp_cubature_c2r.csv")[0]
        z_tar = row[0:3]
        z_ref = row[3:6]
        cov_ref = row[6:15].reshape(3, 3, order="F")
        res = cart2ruv_std_refrac_cubature(
            z_tar, np.diag([100.0] * 3), True, z_rx, z_rx, np.eye(3), 313.0
        )
        np.testing.assert_allclose(res.mean[:, 0], z_ref, atol=1e-6)
        np.testing.assert_allclose(
            res.covariance[:, :, 0],
            cov_ref,
            atol=1e-8 * np.abs(cov_ref).max(),
        )

    def test_ruv2cart_cubature_matches_matlab(self, z_rx):
        row = _load("exp_cubature_r2c.csv")[0]
        z_meas = row[0:3]
        z_ref = row[3:6]
        cov_ref = row[6:15].reshape(3, 3, order="F")
        res = ruv2cart_std_refrac_cubature(
            z_meas,
            np.diag([10.0, 1e-4, 1e-4]),
            True,
            z_rx,
            z_rx,
            np.eye(3),
            313.0,
        )
        np.testing.assert_allclose(res.mean[:, 0], z_ref, atol=1e-4)
        np.testing.assert_allclose(
            res.covariance[:, :, 0],
            cov_ref,
            atol=1e-7 * np.abs(cov_ref).max(),
        )


class TestSphereFrameAndDegenerateInputs:
    """Explicit-sphere geometries that hit the closed-form vertical
    branches and the degenerate-input guards of the r-u-v seed."""

    R_E = 6378137.0

    def test_radial_target_round_trip_closed_form(self):
        # Receiver and target exactly radial in the sphere frame: both
        # converters take their closed-form vertical branches.
        z_rx = np.array([self.R_E + 100.0, 0.0, 0.0])
        z_tar = np.array([self.R_E + 40100.0, 0.0, 0.0])
        res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx, r_e=self.R_E)
        assert res.z[0, 0] > 40000.0  # vacuum range plus positive excess
        np.testing.assert_allclose(res.u_tx[:, 0], [1.0, 0.0, 0.0])
        back = ruv2cart_std_refrac(res.z, True, z_rx, z_rx, r_e=self.R_E)
        np.testing.assert_allclose(back[:, 0], z_tar, atol=1.0)

    def test_radial_target_four_row_measurement(self):
        z_rx = np.array([self.R_E + 100.0, 0.0, 0.0])
        z_tar = np.array([self.R_E + 40100.0, 0.0, 0.0])
        res = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx, r_e=self.R_E, include_w=True)
        assert res.z.shape[0] == 4
        back = ruv2cart_std_refrac(res.z, True, z_rx, z_rx, r_e=self.R_E)
        np.testing.assert_allclose(back[:, 0], z_tar, atol=1.0)

    def test_bistatic_vertical_branch_runs(self):
        # Covers the bistatic near-vertical search; the MATLAB original
        # treats u*range as an absolute position there (see the code
        # comment), so only finiteness is asserted.
        z_rx = np.array([self.R_E, 0.0, 0.0])
        z_tx = np.array([self.R_E, 100e3, 0.0])
        z_tar = np.array([self.R_E + 40e3, 0.0, 0.0])
        meas = cart2ruv_std_refrac(z_tar, False, z_tx, z_rx, r_e=self.R_E)
        back = ruv2cart_std_refrac(meas.z, False, z_tx, z_rx, r_e=self.R_E)
        assert np.all(np.isfinite(back))

    def test_overlong_direction_cosines_are_normalized(self):
        # u^2 + v^2 > 1 must be clamped, not produce NaN.
        z_rx = np.array([self.R_E, 0.0, 0.0])
        meas = np.array([50e3, 0.8, 0.7])
        back = ruv2cart_std_refrac(meas, True, z_rx, z_rx, r_e=self.R_E)
        assert np.all(np.isfinite(back))

    def test_zero_range_measurement(self):
        z_rx = np.array([self.R_E, 0.0, 0.0])
        meas = np.array([0.0, 0.1, 0.1])
        back = ruv2cart_std_refrac(meas, True, z_rx, z_rx, r_e=self.R_E)
        np.testing.assert_allclose(back[:, 0], z_rx, atol=1e-6)


class TestCubatureNonDefaultArguments:
    """The cubature wrappers with explicitly supplied points/weights and
    per-measurement covariance stacks, asserted against the default path."""

    def test_explicit_points_match_default(self, z_rx):
        from pytcl.mathematical_functions.numerical_integration import (
            fifth_order_cubature_points,
        )

        z_tar = z_rx + np.array([30e3, 40e3, 10e3])
        sr = np.diag([50.0, 50.0, 50.0])
        xi, w = fifth_order_cubature_points(3)
        default = cart2ruv_std_refrac_cubature(
            z_tar, sr, True, z_rx, z_rx, np.eye(3), 313.0
        )
        explicit = cart2ruv_std_refrac_cubature(
            z_tar,
            sr,
            True,
            z_rx,
            z_rx,
            np.eye(3),
            313.0,
            points=xi,
            weights=w,
        )
        np.testing.assert_allclose(explicit.mean, default.mean, rtol=1e-12)
        np.testing.assert_allclose(explicit.covariance, default.covariance, rtol=1e-10)

    def test_stacked_sqrt_cov_matches_per_measurement_calls(self, z_rx):
        z_tar = np.column_stack(
            [
                z_rx + np.array([30e3, 40e3, 10e3]),
                z_rx + np.array([60e3, 20e3, 15e3]),
            ]
        )
        srs = np.stack(
            [np.diag([50.0, 50.0, 50.0]), np.diag([120.0, 80.0, 40.0])],
            axis=2,
        )
        stacked = cart2ruv_std_refrac_cubature(
            z_tar, srs, True, z_rx, z_rx, np.eye(3), 313.0
        )
        for i in range(2):
            single = cart2ruv_std_refrac_cubature(
                z_tar[:, i], srs[:, :, i], True, z_rx, z_rx, np.eye(3), 313.0
            )
            np.testing.assert_allclose(
                stacked.mean[:, i], single.mean[:, 0], rtol=1e-12
            )
            np.testing.assert_allclose(
                stacked.covariance[:, :, i],
                single.covariance[:, :, 0],
                rtol=1e-10,
            )

    def test_ruv2cart_cubature_explicit_points(self, z_rx):
        from pytcl.mathematical_functions.numerical_integration import (
            fifth_order_cubature_points,
        )

        z_tar = z_rx + np.array([1e3, 5e3, 50e3])
        ruv = cart2ruv_std_refrac(z_tar, True, z_rx, z_rx).z[:, 0]
        sr = np.diag([10.0, 1e-4, 1e-4])
        xi, w = fifth_order_cubature_points(3)
        default = ruv2cart_std_refrac_cubature(ruv, sr, True, z_rx, z_rx)
        explicit = ruv2cart_std_refrac_cubature(
            ruv, sr, True, z_rx, z_rx, points=xi, weights=w
        )
        np.testing.assert_allclose(explicit.mean, default.mean, rtol=1e-12)
        np.testing.assert_allclose(explicit.covariance, default.covariance, rtol=1e-10)
