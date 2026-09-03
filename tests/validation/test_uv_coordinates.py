"""u-v direction-cosine conversions against MATLAB TCL fixtures.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_uv_coords.m. Every
input here mirrors that capture script verbatim.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.coordinate_systems.conversions.uv import (
    camera_coords2uv,
    cart2ruv_bistatic,
    ruv2cart_bistatic,
    ruv2ruv,
    spher_ang2uv,
    state_ruv2cart,
    uv2spher_ang,
)
from pytcl.coordinate_systems.rotations import rot_axis_to_vec

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Literal transcriptions; measured max disagreement 7.3e-12 on
# kilometre-scale coordinates.
ATOL = 1e-9


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


UV_IN = np.array([[0.3, -0.5, 0.1, 0.62], [0.4, 0.2, -0.7, -0.05]])
_n1 = np.array([1.0, 2.0, 3.0])
_n2 = np.array([-1.0, 1.0, 4.0])
MS = rot_axis_to_vec(_n1 / np.linalg.norm(_n1), "z")
MUV = rot_axis_to_vec(_n2 / np.linalg.norm(_n2), "z")


class TestUvSpherAng:
    @pytest.mark.parametrize("system_type", [0, 1, 2, 3])
    def test_uv2spher_matches_matlab(self, system_type):
        az_el = uv2spher_ang(UV_IN, system_type, MS, MUV)
        np.testing.assert_allclose(
            az_el, _load(f"uv2spher_st{system_type}.csv"), atol=ATOL
        )

    @pytest.mark.parametrize("system_type", [0, 1, 2, 3])
    def test_spher2uv_matches_matlab(self, system_type):
        az_el = _load(f"uv2spher_st{system_type}.csv")
        uvw = spher_ang2uv(az_el, system_type, True, MS, MUV)
        np.testing.assert_allclose(
            uvw, _load(f"spher2uv_st{system_type}.csv"), atol=ATOL
        )

    @pytest.mark.parametrize("system_type", [0, 1, 2, 3])
    def test_round_trip(self, system_type):
        az_el = uv2spher_ang(UV_IN, system_type, MS, MUV)
        uvw = spher_ang2uv(az_el, system_type, True, MS, MUV)
        np.testing.assert_allclose(uvw[:2, :], UV_IN, atol=1e-12)

    def test_invalid_system_type_raises(self):
        with pytest.raises(ValueError):
            uv2spher_ang(UV_IN, 4)
        with pytest.raises(ValueError):
            spher_ang2uv(np.zeros((2, 1)), 4)


M_RX = rot_axis_to_vec(np.array([0.2, -0.1, 0.97]), "z")
Z_CART = np.array(
    [[100.0, 2000.0, -500.0], [50.0, -300.0, 900.0], [800.0, 1200.0, 300.0]]
)
Z_TX = np.array([10.0, 20.0, 5.0])
Z_RX = np.array([-15.0, 8.0, 2.0])


class TestBistaticRuv:
    def test_cart2ruv_matches_matlab(self):
        z = cart2ruv_bistatic(Z_CART, False, Z_TX, Z_RX, M_RX, include_w=True)
        np.testing.assert_allclose(z, _load("cart2ruv_bistatic.csv"), atol=ATOL)

    def test_ruv2cart_matches_matlab(self):
        z_c = ruv2cart_bistatic(_load("cart2ruv_bistatic.csv"), False, Z_TX, Z_RX, M_RX)
        np.testing.assert_allclose(z_c, _load("ruv2cart_bistatic.csv"), atol=ATOL)
        np.testing.assert_allclose(z_c, Z_CART, atol=1e-9)

    def test_half_range_monostatic_matches_matlab(self):
        z = cart2ruv_bistatic(Z_CART, True, Z_RX, Z_RX, M_RX)
        np.testing.assert_allclose(z, _load("cart2ruv_half.csv"), atol=ATOL)
        z_c = ruv2cart_bistatic(_load("cart2ruv_half.csv"), True, Z_RX, Z_RX, M_RX)
        np.testing.assert_allclose(z_c, _load("ruv2cart_half.csv"), atol=ATOL)

    def test_ruv2ruv_matches_matlab(self):
        m2 = rot_axis_to_vec(np.array([0.5, 0.5, 0.7071]), "z")
        z_tx2 = np.array([-30.0, 12.0, 9.0])
        z_rx2 = np.array([25.0, -18.0, 4.0])
        z_new = ruv2ruv(
            _load("cart2ruv_bistatic.csv"),
            False,
            Z_TX,
            Z_RX,
            M_RX,
            z_tx2,
            z_rx2,
            m2,
        )
        np.testing.assert_allclose(z_new, _load("ruv2ruv_pair.csv"), atol=ATOL)
        # w must be carried through because the input had four rows.
        assert z_new.shape[0] == 4

    def test_out_of_unit_disc_input_is_normalized(self):
        # |uv| > 1 (noise) must not produce NaN; the original normalizes.
        z = np.array([100.0, 0.9, 0.6])
        z_c = ruv2cart_bistatic(z)
        assert np.all(np.isfinite(z_c))


class TestStateRuv2Cart:
    def test_6_state_matches_matlab(self):
        x = np.array([1000.0, 0.3, -0.4, 12.0, 1e-3, -2e-3])
        np.testing.assert_allclose(
            state_ruv2cart(x), _load("state_ruv2cart_6.csv"), atol=ATOL
        )

    def test_9_state_matches_matlab(self):
        x = np.array([1000.0, 0.3, -0.4, 12.0, 1e-3, -2e-3, 0.5, 1e-5, 2e-5])
        np.testing.assert_allclose(
            state_ruv2cart(x), _load("state_ruv2cart_9.csv"), atol=ATOL
        )

    def test_velocity_matches_numeric_differentiation(self):
        x = np.array([1000.0, 0.3, -0.4, 12.0, 1e-3, -2e-3])
        dt = 1e-6
        xp = x.copy()
        xp[:3] += x[3:6] * dt
        cs, cs2 = state_ruv2cart(x), state_ruv2cart(xp)
        vel_num = (cs2[:3] - cs[:3]).ravel() / dt
        np.testing.assert_allclose(vel_num, cs[3:6].ravel(), rtol=1e-5)


class TestCameraCoords2Uv:
    A = np.array([[500.0, 2.0, 320.0], [0.0, 480.0, 240.0], [0.0, 0.0, 1.0]])
    Z_CAM = np.array([[320.0, 100.0, 500.0], [240.0, 50.0, 400.0]])

    def test_matches_matlab(self):
        m_cam = rot_axis_to_vec(np.array([0.1, 0.2, 0.97]), "z")
        d = camera_coords2uv(self.Z_CAM, self.A, m_cam, True)
        np.testing.assert_allclose(d, _load("camera2uv.csv"), atol=ATOL)

    def test_outputs_are_unit_vectors(self):
        d = camera_coords2uv(self.Z_CAM, self.A)
        np.testing.assert_allclose(np.linalg.norm(d, axis=0), 1.0, atol=1e-12)

    def test_without_w(self):
        d = camera_coords2uv(self.Z_CAM, self.A, include_w=False)
        assert d.shape == (2, 3)

    def test_bad_intrinsics_row_raises(self):
        bad = self.A.copy()
        bad[2, 0] = 1.0
        with pytest.raises(ValueError, match="third row"):
            camera_coords2uv(self.Z_CAM, bad)
