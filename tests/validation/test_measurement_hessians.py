"""Measurement Hessians and cross derivatives against MATLAB.

Fixtures captured by scripts/matlab_capture/
capture_measurement_hessians.m over the same bistatic geometry as the
Jacobian tier. MATLAB flattens column-major, hence order="F".

Central finite differences of the underlying measurement functions
serve as the independent second oracle. Two paths cannot be captured
from MATLAB and are validated numerically instead: the 2-D cross
gradient's rotation arguments (its MATLAB source calls the undefined
rotMat2D2Angle and errors upstream) and rectangular output maps in
hessian_of_affine_trans_fun (the MATLAB in-place assignment requires
square maps).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pytcl.coordinate_systems.hessians import (
    calc_spher_conv_hessian,
    calc_spher_hessian,
    calc_spher_inv_hessian,
    hessian_chain_rule,
    hessian_of_affine_trans_fun,
    polar_u_2d_cross_grad,
    polar_u_2d_cross_hessian,
    range_hessian,
    spher_ang_hessian,
    spher_ang_uv_cross_grad,
    spher_ang_uv_cross_hessian,
    u_hessian_2d,
    u_hessian_3d,
    u_polar_2d_cross_grad,
    u_polar_2d_cross_hessian,
    uv_hessian,
    uv_spher_ang_cross_grad,
    uv_spher_ang_cross_hessian,
)
from pytcl.coordinate_systems.jacobians.component_gradients import uv_gradient
from pytcl.coordinate_systems.jacobians.measurement_jacobians import (
    calc_spher_jacob,
)

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "matlab" / "measurement_hessians.csv"
)

T = np.array([-3e3, -2e3, -1e3])
L_TX = np.array([-12e3, 8e3, 5e3])
L_RX = np.array([4e3, -6e3, 12.0])
M = expm(np.array([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]]))
P2 = np.array([3e3, 4e3])
L_TX2 = np.array([-2e3, 1e3])
L_RX2 = np.array([500.0, -300.0])
M2 = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
MS = M
MUV = expm(np.array([[0.0, 0.1, -0.2], [-0.1, 0.0, 0.15], [0.2, -0.15, 0.0]]))
AZ_EL = np.array([0.5, 0.2])
UV_PT = np.array([0.1, 0.2])
UVW_PT = np.array([0.1, 0.2, np.sqrt(1 - 0.1**2 - 0.2**2)])
Z_SPH = np.array([9e3, 0.5, 0.2])
AZ_LIST = np.array([0.3, 0.7, 1.4])
U_LIST = np.array([0.2, 0.5, 0.8])
UV_COLS = np.array([[0.2, 0.5], [0.6, 0.3]])

RTOL = 1e-12


def _fixtures() -> dict:
    out = {}
    with open(FIXTURE) as f:
        for row in csv.reader(f):
            if row[0] == "label":
                continue
            out[row[0]] = np.array([float(v) for v in row[1:]])
    return out


FIX = _fixtures()


def _check(label: str, got: np.ndarray) -> None:
    ref = FIX[label].reshape(got.shape, order="F")
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=1e-300, err_msg=label)


class TestComponentHessiansAgainstMatlab:
    @pytest.mark.parametrize("uhr", [False, True])
    def test_range_hessian(self, uhr):
        _check(f"rangeHessian_uhr{int(uhr)}_bi", range_hessian(T, uhr, L_TX, L_RX))
        _check(f"rangeHessian_uhr{int(uhr)}_mono", range_hessian(T, uhr))

    def test_range_hessian_2d(self):
        _check("rangeHessian_2d", range_hessian(P2, False, L_TX2, L_RX2))

    @pytest.mark.parametrize("st", [0, 1, 2, 3])
    def test_spher_ang_hessian(self, st):
        _check(f"spherAngHessian_st{st}", spher_ang_hessian(T, st, L_RX, M))
        _check(f"spherAngHessian_st{st}_plain", spher_ang_hessian(T, st))

    def test_uv_hessians(self):
        _check("uvHessian", uv_hessian(T, L_RX, M))
        _check("uvHessian_w", uv_hessian(T, L_RX, M, include_w=True))
        _check("uHessian2D", u_hessian_2d(P2, L_RX2, M2))
        _check("uHessian2D_v", u_hessian_2d(P2, L_RX2, M2, include_v=True))
        _check("uHessian3D", u_hessian_3d(T, L_RX, M))


class TestMeasurementHessiansAgainstMatlab:
    @pytest.mark.parametrize("st", [0, 1, 2, 3])
    def test_spher_family(self, st):
        _check(
            f"calcSpherHessian_st{st}_bi",
            calc_spher_hessian(T, st, False, L_TX, L_RX, M),
        )
        _check(f"calcSpherHessian_st{st}_mono", calc_spher_hessian(T, st))
        _check(f"calcSpherInvHessian_st{st}", calc_spher_inv_hessian(Z_SPH, st))

    def test_converted(self):
        rb = np.linalg.norm(T - L_RX) + np.linalg.norm(T - L_TX)
        xl = M @ (T - L_RX)
        z = np.array(
            [rb, np.arctan2(xl[1], xl[0]), np.arcsin(xl[2] / np.linalg.norm(xl))]
        )
        _check(
            "calcSpherConvHessian_st0_bi",
            calc_spher_conv_hessian(z, 0, False, L_TX, L_RX, M),
        )

    def test_helpers(self):
        h_uvw = uv_hessian(T, L_RX, M, include_w=True)
        mm = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0], [0.5, 0.0, 1.0]])
        _check("HessianOfAffineTransFun", hessian_of_affine_trans_fun(h_uvw, mm))
        hf = calc_spher_hessian(T, 0, False, L_TX, L_RX, M)
        jf = calc_spher_jacob(T, 0, False, L_TX, L_RX, M)
        hg = uv_hessian(T, L_RX, M, include_w=True)
        jg = uv_gradient(T, L_RX, M, include_w=True)
        _check("HessianChainRule", hessian_chain_rule(hf, hg, jf, jg))


class TestCrossDerivativesAgainstMatlab:
    @pytest.mark.parametrize("st", [0, 1, 2, 3])
    def test_uv_spher_family(self, st):
        _check(
            f"uvSpherAngCrossGrad_st{st}",
            uv_spher_ang_cross_grad(AZ_EL, st, False, MS, MUV),
        )
        _check(
            f"uvSpherAngCrossGrad_st{st}_w",
            uv_spher_ang_cross_grad(AZ_EL, st, True, MS, MUV),
        )
        _check(
            f"spherAngUvCrossGrad_st{st}",
            spher_ang_uv_cross_grad(UV_PT, st, MS, MUV),
        )
        _check(
            f"spherAngUvCrossGrad_st{st}_w",
            spher_ang_uv_cross_grad(UVW_PT, st, MS, MUV),
        )
        _check(
            f"uvSpherAngCrossHessian_st{st}",
            uv_spher_ang_cross_hessian(AZ_EL, st, False, MS, MUV),
        )
        _check(
            f"uvSpherAngCrossHessian_st{st}_w",
            uv_spher_ang_cross_hessian(AZ_EL, st, True, MS, MUV),
        )
        _check(
            f"spherAngUvCrossHessian_st{st}",
            spher_ang_uv_cross_hessian(UV_PT, st, MS, MUV),
        )

    @pytest.mark.parametrize("st", [0, 1])
    def test_polar_u_family(self, st):
        _check(f"uPolar2DCrossGrad_st{st}", u_polar_2d_cross_grad(AZ_LIST, st))
        _check(
            f"uPolar2DCrossGrad_st{st}_v",
            u_polar_2d_cross_grad(AZ_LIST, st, include_v=True),
        )
        _check(f"polarU2DCrossGrad_st{st}_u", polar_u_2d_cross_grad(U_LIST, st))
        _check(f"polarU2DCrossGrad_st{st}_uv", polar_u_2d_cross_grad(UV_COLS, st))
        _check(f"uPolar2DCrossHessian_st{st}", u_polar_2d_cross_hessian(AZ_LIST, st))
        _check(
            f"uPolar2DCrossHessian_st{st}_v",
            u_polar_2d_cross_hessian(AZ_LIST, st, include_v=True),
        )
        _check(f"polarU2DCrossHessian_st{st}_u", polar_u_2d_cross_hessian(U_LIST, st))
        _check(f"polarU2DCrossHessian_st{st}_uv", polar_u_2d_cross_hessian(UV_COLS, st))


def _num_hess(f, x, h):
    x = np.asarray(x, dtype=float)
    n = x.size
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            for si, sj, s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
                xx = x.copy()
                xx[i] += si * h
                xx[j] += sj * h
                out[i, j] += s * f(xx)
            out[i, j] /= 4 * h * h
    return out


class TestAgainstNumericalDifferentiation:
    """Independent oracle, and the paths MATLAB cannot capture."""

    def test_uv_hessian_is_the_direction_cosine_curvature(self):
        def comp(p, c):
            xl = M @ (p - L_RX)
            return (xl / np.linalg.norm(xl))[c]

        h = uv_hessian(T, L_RX, M, include_w=True)
        for c in range(3):
            np.testing.assert_allclose(
                _num_hess(lambda p, c=c: comp(p, c), T, 1.0),
                h[:, :, c],
                rtol=0,
                atol=1e-13,
            )

    def test_spher_inv_hessian_is_the_conversion_curvature(self):
        for st in (0, 1, 2, 3):

            def cart(zz, comp, st=st):
                r, az, el = zz
                if st == 0:
                    v = [
                        r * np.cos(az) * np.cos(el),
                        r * np.sin(az) * np.cos(el),
                        r * np.sin(el),
                    ]
                elif st == 1:
                    v = [
                        r * np.sin(az) * np.cos(el),
                        r * np.sin(el),
                        r * np.cos(az) * np.cos(el),
                    ]
                elif st == 2:
                    v = [
                        r * np.cos(az) * np.sin(el),
                        r * np.sin(az) * np.sin(el),
                        r * np.cos(el),
                    ]
                else:
                    v = [
                        r * np.sin(az) * np.cos(el),
                        r * np.cos(az) * np.cos(el),
                        r * np.sin(el),
                    ]
                return v[comp]

            h = calc_spher_inv_hessian(Z_SPH, st)
            for c in range(3):
                # Entries scale with r = 9e3, so finite-difference
                # round-off sits near 1e-3 absolute (~1e-7 relative);
                # exactness is pinned by the MATLAB fixtures above.
                np.testing.assert_allclose(
                    _num_hess(lambda zz, c=c: cart(zz, c), Z_SPH, 1e-3),
                    h[:, :, c],
                    rtol=0,
                    atol=1e-2,
                )

    def test_2d_cross_grad_rotations_work_despite_upstream(self):
        # The MATLAB source errors when rotations are passed (its
        # rotMat2D2Angle helper does not exist); the intended
        # semantics are validated numerically here.
        mp = np.array([[np.cos(0.4), -np.sin(0.4)], [np.sin(0.4), np.cos(0.4)]])
        mu = np.array([[np.cos(-0.25), -np.sin(-0.25)], [np.sin(-0.25), np.cos(-0.25)]])
        az0 = 0.7

        def u_of(a, c):
            u_p = np.array([np.cos(a), np.sin(a)])
            return (mu @ (mp.T @ u_p))[c]

        got = u_polar_2d_cross_grad(az0, 0, mp, mu, include_v=True)
        num = np.array(
            [[(u_of(az0 + 1e-7, c) - u_of(az0 - 1e-7, c)) / 2e-7] for c in range(2)]
        )
        np.testing.assert_allclose(got, num, rtol=1e-6)

    def test_affine_trans_supports_rectangular_maps(self):
        # The MATLAB original assigns M*H in place and so requires a
        # square M; the port accepts rectangular output maps.
        mm = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, -1.0]])
        h_uvw = uv_hessian(T, L_RX, M, include_w=True)
        got = hessian_of_affine_trans_fun(h_uvw, mm)
        assert got.shape == (3, 3, 2)

        def mixed(p, c):
            xl = M @ (p - L_RX)
            return (mm @ (xl / np.linalg.norm(xl)))[c]

        for c in range(2):
            np.testing.assert_allclose(
                _num_hess(lambda p, c=c: mixed(p, c), T, 1.0),
                got[:, :, c],
                rtol=0,
                atol=1e-12,
            )

    def test_cross_grad_pair_are_inverse_maps(self):
        # d(az,el)/d(u,v) composed with d(u,v)/d(az,el) is the identity
        # when both use the same frames.
        j_fwd = uv_spher_ang_cross_grad(AZ_EL, 0, False, MS, MUV)
        sa, ca = np.sin(AZ_EL[0]), np.cos(AZ_EL[0])
        se, ce = np.sin(AZ_EL[1]), np.cos(AZ_EL[1])
        u_s = np.array([ca * ce, sa * ce, se])
        uvw = MUV @ (MS.T @ u_s)
        j_back = spher_ang_uv_cross_grad(uvw, 0, MS, MUV)
        np.testing.assert_allclose(j_back @ j_fwd, np.eye(2), atol=1e-12)

    def test_cross_hessian_accepts_explicit_w(self):
        # A 3-element [u, v, w] input must match the 2-element form on
        # the +w hemisphere.
        np.testing.assert_allclose(
            spher_ang_uv_cross_hessian(UVW_PT, 0, MS, MUV),
            spher_ang_uv_cross_hessian(UV_PT, 0, MS, MUV),
            rtol=1e-14,
        )

    def test_collocated_transmitter_range_hessian(self):
        # As in the gradient, a collocated transmitter contributes
        # nothing.
        got = range_hessian(T, False, T, L_RX)
        d = T - L_RX
        n = np.linalg.norm(d)
        np.testing.assert_allclose(
            got, -np.outer(d, d) / n**3 + np.eye(3) / n, rtol=1e-14
        )

    def test_invalid_conventions_are_rejected(self):
        for fn, arg in (
            (lambda: uv_spher_ang_cross_grad(AZ_EL, 9), None),
            (lambda: spher_ang_uv_cross_grad(UV_PT, 9), None),
            (lambda: u_polar_2d_cross_grad(0.3, 9), None),
            (lambda: polar_u_2d_cross_grad(0.3, 9), None),
            (lambda: uv_spher_ang_cross_hessian(AZ_EL, 9), None),
            (lambda: spher_ang_uv_cross_hessian(UV_PT, 9), None),
            (lambda: u_polar_2d_cross_hessian(0.3, 9), None),
            (lambda: polar_u_2d_cross_hessian(np.array([[0.3], [0.4]]), 9), None),
            (lambda: spher_ang_hessian(T, 9), None),
            (lambda: calc_spher_inv_hessian(Z_SPH, 9), None),
        ):
            with pytest.raises(ValueError):
                fn()
