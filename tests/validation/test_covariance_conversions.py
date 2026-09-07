"""Measurement conversions with covariances against MATLAB TCL fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via
scripts/matlab_capture/capture_covariance_conversions.m; inputs
mirrored verbatim. MATLAB's fifthOrderCubPoints and pytcl's
fifth_order_cubature_points implement DIFFERENT degree-5 rules (they
agree only to the rules' truncation error on non-polynomial
conversions), so the cubature fixtures pass one shared point set --
pytcl's, exported as cc_xi*/cc_w* -- explicitly on both sides, making
them machine-precision. Default-point calls remain degree-5 exact but
differ across the libraries at that truncation level by design.

One port deliberately diverges from upstream:
monostat_ruv2cart_taylor with several measurements and
use_half_range=False, where the original halves only the FIRST
measurement's range while scaling every covariance; those fixtures
are captured one measurement at a time (where the original is
correct) and the port's batched path is checked against the stacked
single calls.
"""

from pathlib import Path

import numpy as np

from pytcl.coordinate_systems import (
    camera_coords2uv_cubature,
    cart2ruv_bistatic,
    monostat_ruv2cart_taylor,
    ruv2ruv_cubature,
    uv2spher_ang_cubature,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Deterministic transcriptions: measured max disagreement 4.4e-10,
# on covariance entries of magnitude ~1e5 (relative ~1e-15).
ATOL = 1e-8
ATOL_ANG = 1e-12


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


def _load_stack(name, d):
    flat = _load(name)
    n = flat.shape[0] // d
    return np.stack([flat[k * d : (k + 1) * d, :] for k in range(n)], axis=2)


def _shared_points(dim):
    xi = _load(f"cc_xi{dim}.csv").T
    w = _load(f"cc_w{dim}.csv").ravel()
    return xi, w


Z_UV = np.array([[0.3, -0.2, 0.05], [0.4, 0.1, -0.35]])
R_UV = np.array([[2e-3, 5e-4], [5e-4, 3e-3]])
S_R_UV = np.linalg.cholesky(R_UV)
M_S = np.array(
    [
        [np.cos(0.3), -np.sin(0.3), 0.0],
        [np.sin(0.3), np.cos(0.3), 0.0],
        [0.0, 0.0, 1.0],
    ]
)
M_UV = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, np.cos(-0.2), -np.sin(-0.2)],
        [0.0, np.sin(-0.2), np.cos(-0.2)],
    ]
)


class TestUv2SpherAngCubature:
    def test_matches_matlab_all_system_types(self):
        for sys_type in range(4):
            xi, w = _shared_points(2)
            res = uv2spher_ang_cubature(Z_UV, S_R_UV, sys_type, M_S, M_UV, xi, w)
            np.testing.assert_allclose(
                res.z, _load(f"cc_uv2sph_z{sys_type}.csv"), atol=ATOL_ANG
            )
            np.testing.assert_allclose(
                res.R, _load_stack(f"cc_uv2sph_R{sys_type}.csv", 2), atol=ATOL_ANG
            )

    def test_mean_is_consistent_direction(self):
        # For tight noise the cubature mean approaches the noise-free
        # conversion.
        from pytcl.coordinate_systems import uv2spher_ang

        res = uv2spher_ang_cubature(Z_UV[:, :1], 1e-10 * np.eye(2), 0, M_S, M_UV)
        direct = uv2spher_ang(Z_UV[:, :1], 0, M_S, M_UV)
        np.testing.assert_allclose(res.z, direct, atol=1e-6)


POINT_CART = 1e4 * np.array([1.173024396049598, -4.844345843776918, 3.969150579064054])
S_RUV = np.linalg.cholesky(np.diag([50.0**2, 1e-6, 1e-6]))
X_TX1 = np.zeros(3)
X_RX1 = 1e4 * np.array([1.0, 2.0, 0.0])
M2_ROT = np.array(
    [
        [np.cos(0.4), np.sin(0.4), 0.0],
        [-np.sin(0.4), np.cos(0.4), 0.0],
        [0.0, 0.0, 1.0],
    ]
)


class TestRuv2RuvCubature:
    Z_RUV1 = cart2ruv_bistatic(POINT_CART, False, X_TX1, X_RX1, None).ravel()

    def test_matches_matlab_all_include_w_modes(self):
        for include_w in range(3):
            d = 4 if include_w else 3
            xi, w = _shared_points(3)
            res = ruv2ruv_cubature(
                self.Z_RUV1,
                S_RUV,
                False,
                X_TX1,
                X_RX1,
                None,
                X_TX1,
                X_TX1,
                M2_ROT,
                include_w,
                xi,
                w,
            )
            np.testing.assert_allclose(
                res.z, _load(f"cc_ruv2ruv_z{include_w}.csv"), atol=ATOL
            )
            np.testing.assert_allclose(
                res.R[:, :, 0], _load(f"cc_ruv2ruv_R{include_w}.csv"), atol=ATOL
            )
            assert res.z.shape == (d, 1)

    def test_batch_with_channel_range_conventions_matches_matlab(self):
        z2 = cart2ruv_bistatic(
            1e4 * np.array([0.5, -2.1, 1.7]), False, X_TX1, X_RX1, None
        ).ravel()
        z_both = np.column_stack([self.Z_RUV1, z2])
        s2 = np.linalg.cholesky(np.diag([30.0**2, 4e-6, 4e-6]))
        s_both = np.stack([S_RUV, s2], axis=2)
        xi, w = _shared_points(3)
        res = ruv2ruv_cubature(
            z_both,
            s_both,
            (False, True),
            X_TX1,
            X_RX1,
            None,
            X_TX1,
            X_TX1,
            M2_ROT,
            0,
            xi,
            w,
        )
        np.testing.assert_allclose(res.z, _load("cc_ruv2ruv_zmix.csv"), atol=ATOL)
        np.testing.assert_allclose(
            res.R, _load_stack("cc_ruv2ruv_Rmix.csv", 3), atol=ATOL
        )

    def test_unit_direction_in_mean_direction_mode(self):
        res = ruv2ruv_cubature(
            self.Z_RUV1, S_RUV, False, X_TX1, X_RX1, None, X_TX1, X_TX1, M2_ROT, 1
        )
        np.testing.assert_allclose(np.linalg.norm(res.z[1:, 0]), 1.0, atol=1e-12)


class TestCameraCoords2UvCubature:
    def test_matches_matlab(self):
        a = np.diag([35e-3, 35e-3, 1.0])
        z_cam = np.array([[1e-2, -7e-3], [-2e-2, 1.4e-2]])
        s_r = np.diag([1e-4, 1e-4])
        xi, w = _shared_points(2)
        res = camera_coords2uv_cubature(z_cam, s_r, a, xi, w)
        np.testing.assert_allclose(res.z, _load("cc_cam2uv_z.csv"), atol=ATOL_ANG)
        np.testing.assert_allclose(
            res.R, _load_stack("cc_cam2uv_R.csv", 2), atol=ATOL_ANG
        )


Z_RUV_M = np.array([[100e3, 80e3], [0.5, -0.3], [0.2, 0.4]])
R_M = np.diag([1.0, 1e-6, 25e-6])
M_ROT = np.array(
    [
        [np.cos(0.25), 0.0, -np.sin(0.25)],
        [0.0, 1.0, 0.0],
        [np.sin(0.25), 0.0, np.cos(0.25)],
    ]
)
Z_RX = np.array([100.0, -200.0, 50.0])


class TestMonostatRuv2CartTaylor:
    def test_matches_matlab_all_algorithms_one_way(self):
        for alg in range(4):
            res = monostat_ruv2cart_taylor(Z_RUV_M, R_M, True, Z_RX, M_ROT, alg)
            np.testing.assert_allclose(
                res.z, _load(f"cc_ruv2cart_z{alg}.csv"), atol=ATOL
            )
            np.testing.assert_allclose(
                res.R, _load_stack(f"cc_ruv2cart_R{alg}.csv", 3), atol=ATOL
            )

    def test_two_way_batch_matches_stacked_single_matlab_calls(self):
        # The original corrupts measurements 2..N when
        # useHalfRange=false (it halves only the first range); the
        # port's batched result must equal the original applied one
        # measurement at a time.
        z_tw = Z_RUV_M.copy()
        z_tw[0, :] *= 2.0
        for alg in range(4):
            res = monostat_ruv2cart_taylor(z_tw, R_M, False, Z_RX, M_ROT, alg)
            np.testing.assert_allclose(
                res.z, _load(f"cc_ruv2cart_tw_z{alg}.csv"), atol=ATOL
            )
            np.testing.assert_allclose(
                res.R, _load_stack(f"cc_ruv2cart_tw_R{alg}.csv", 3), atol=ATOL
            )

    def test_two_way_equals_prehalved_one_way(self):
        z_tw = Z_RUV_M.copy()
        z_tw[0, :] *= 2.0
        d = np.diag([0.5, 1.0, 1.0])
        halved = monostat_ruv2cart_taylor(Z_RUV_M, d @ R_M @ d.T, True, Z_RX, M_ROT, 1)
        two_way = monostat_ruv2cart_taylor(z_tw, R_M, False, Z_RX, M_ROT, 1)
        np.testing.assert_allclose(two_way.z, halved.z, atol=1e-12)
        np.testing.assert_allclose(two_way.R, halved.R, atol=1e-12)

    def test_unknown_algorithm_raises(self):
        import pytest

        with pytest.raises(ValueError):
            monostat_ruv2cart_taylor(Z_RUV_M, R_M, True, Z_RX, M_ROT, 7)
