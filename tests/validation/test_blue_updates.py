"""BLUE polar/spherical measurement updates against MATLAB fixtures.

Reference values captured from the Tracker Component Library (commit
a9acd8f) via the capture script recorded in
scripts/matlab_capture/capture_blue.m's inputs, mirrored verbatim
here. Both updates are deterministic closed forms.
"""

from pathlib import Path

import numpy as np

from pytcl.dynamic_estimation.kalman import (
    blue_polar_meas_update,
    blue_spher_meas_update,
)
from pytcl.dynamic_estimation.kalman.linear import kf_update

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Literal transcriptions: measured max disagreement 4.4e-12 on
# km-scale states.
ATOL = 1e-9


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


X_POLAR = np.array([1000.0, 500.0, 10.0, -5.0])
P_POLAR = np.array(
    [
        [100.0, 10.0, 5.0, 0.0],
        [10.0, 120.0, 0.0, 8.0],
        [5.0, 0.0, 25.0, 3.0],
        [0.0, 8.0, 3.0, 30.0],
    ]
)
Z_POLAR = np.array([np.hypot(1010.0, 495.0), np.arctan2(495.0, 1010.0)])
R_POLAR = np.diag([25.0, 1e-4])

X_SPHER = np.array([2000.0, 1000.0, 500.0, 10.0, -5.0, 2.0])
P_SPHER = np.diag([100.0, 110.0, 90.0, 25.0, 30.0, 20.0])
P_SPHER[0, 1] = P_SPHER[1, 0] = 15.0
P_SPHER[0, 3] = P_SPHER[3, 0] = 5.0
P_SPHER[2, 5] = P_SPHER[5, 2] = 4.0
_TRUE = np.array([2010.0, 995.0, 505.0])
_R = np.linalg.norm(_TRUE)
Z_SPHER = np.array([_R, np.arctan2(995.0, 2010.0), np.arcsin(505.0 / _R)])
R_SPHER = np.diag([25.0, 1e-4, 1e-4])


class TestBluePolarAgainstMatlab:
    def test_matches_matlab(self):
        res = blue_polar_meas_update(X_POLAR, P_POLAR, Z_POLAR, R_POLAR)
        np.testing.assert_allclose(res.x, _load("blue_polar_x.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("blue_polar_P.csv"), atol=ATOL)
        np.testing.assert_allclose(
            res.innov, _load("blue_polar_innov.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.pzz, _load("blue_polar_Pzz.csv"), atol=ATOL)
        np.testing.assert_allclose(res.gain, _load("blue_polar_W.csv"), atol=ATOL)

    def test_update_reduces_position_uncertainty(self):
        res = blue_polar_meas_update(X_POLAR, P_POLAR, Z_POLAR, R_POLAR)
        assert np.all(np.diag(res.P)[:2] < np.diag(P_POLAR)[:2])
        np.testing.assert_allclose(res.P, res.P.T, atol=1e-10)

    def test_small_angle_noise_approaches_converted_measurement_kf(self):
        # As the angle variance shrinks, the debiased BLUE update
        # approaches a plain KF update with the Cartesian-converted
        # measurement and its linearized covariance.
        r_small = np.diag([25.0, 1e-10])
        res = blue_polar_meas_update(X_POLAR, P_POLAR, Z_POLAR, r_small)
        z_cart = np.array(
            [Z_POLAR[0] * np.cos(Z_POLAR[1]), Z_POLAR[0] * np.sin(Z_POLAR[1])]
        )
        h_mat = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # Converted-measurement covariance for tiny angle noise.
        theta = Z_POLAR[1]
        j = np.array(
            [
                [np.cos(theta), -Z_POLAR[0] * np.sin(theta)],
                [np.sin(theta), Z_POLAR[0] * np.cos(theta)],
            ]
        )
        r_cart = j @ r_small @ j.T
        kf = kf_update(X_POLAR, P_POLAR, z_cart, h_mat, r_cart)
        # Not exact even in the limit: the BLUE Pzz spreads the range
        # variance isotropically rather than through the measurement
        # Jacobian, so a few-percent gap to the converted-measurement
        # KF remains. Measured 2.2% here; the fixtures carry the exact
        # oracle.
        np.testing.assert_allclose(res.x, kf.x, rtol=0.05)


class TestBlueSpherAgainstMatlab:
    def test_matches_matlab(self):
        res = blue_spher_meas_update(X_SPHER, P_SPHER, Z_SPHER, R_SPHER)
        np.testing.assert_allclose(res.x, _load("blue_spher_x.csv").ravel(), atol=ATOL)
        np.testing.assert_allclose(res.P, _load("blue_spher_P.csv"), atol=ATOL)
        np.testing.assert_allclose(
            res.innov, _load("blue_spher_innov.csv").ravel(), atol=ATOL
        )
        np.testing.assert_allclose(res.pzz, _load("blue_spher_S.csv"), atol=ATOL)
        np.testing.assert_allclose(res.gain, _load("blue_spher_W.csv"), atol=ATOL)

    def test_update_reduces_position_uncertainty(self):
        res = blue_spher_meas_update(X_SPHER, P_SPHER, Z_SPHER, R_SPHER)
        assert np.all(np.diag(res.P)[:3] < np.diag(P_SPHER)[:3])
        np.testing.assert_allclose(res.P, res.P.T, atol=1e-10)

    def test_state_ordering_round_trip(self):
        # The internal interleaved ordering must be undone: velocity
        # components stay in place for a measurement that only carries
        # position information weakly coupled to velocity.
        p_diag = np.diag([100.0, 110.0, 90.0, 25.0, 30.0, 20.0])
        res = blue_spher_meas_update(X_SPHER, p_diag, Z_SPHER, R_SPHER)
        # With a block-diagonal P, velocities are untouched.
        np.testing.assert_allclose(res.x[3:], X_SPHER[3:], atol=1e-9)
        np.testing.assert_allclose(np.diag(res.P)[3:], np.diag(p_diag)[3:], atol=1e-9)
