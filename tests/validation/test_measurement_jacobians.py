"""Measurement Jacobians and component gradients against MATLAB.

Fixtures captured by scripts/matlab_capture/
capture_measurement_jacobians.m from headless R2026a over a fixed
bistatic geometry (rotated receiver frame, moving transmitter and
receiver states, all system types, both half-range conventions).
MATLAB flattens column-major, hence the order="F" reshapes.

A second, independent oracle -- central finite differences of the
measurement functions themselves -- guards the cases fixtures cannot
reach (arbitrary geometry via randomized points).
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pytcl.coordinate_systems.jacobians.component_gradients import (
    pol_ang_gradient,
    range_gradient,
    range_rate_gradient,
    spher_ang_gradient,
    tdoa_gradient,
    u_gradient_2d,
    u_gradient_3d,
    uv_gradient,
)
from pytcl.coordinate_systems.jacobians.measurement_jacobians import (
    calc_cart_rr_jacob,
    calc_polar_conv_jacob,
    calc_polar_jacob,
    calc_polar_rr_conv_jacob,
    calc_polar_rr_jacob,
    calc_ruv_conv_jacob,
    calc_ruv_jacob,
    calc_ruv_rr_conv_jacob,
    calc_ruv_rr_jacob,
    calc_spher_conv_jacob,
    calc_spher_inv_jacob,
    calc_spher_jacob,
    calc_spher_rr_jacob,
    norm_vec_jacob,
)

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "matlab" / "measurement_jacobians.csv"
)

# The capture driver's geometry, verbatim.
T = np.array([-3e3, -2e3, -1e3])
L_TX = np.array([-12e3, 8e3, 5e3])
L_RX = np.array([4e3, -6e3, 12.0])
M = expm(np.array([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]]))
XS = np.array([-3e3, -2e3, -1e3, 30.0, -20.0, 10.0])
S_TX = np.concatenate([L_TX, [5.0, 3.0, -2.0]])
S_RX = np.concatenate([L_RX, [-4.0, 2.0, 6.0]])
P2 = np.array([3e3, 4e3])
L_TX2 = np.array([-2e3, 1e3])
L_RX2 = np.array([500.0, -300.0])
M2 = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
XS2 = np.array([3e3, 4e3, 10.0, -5.0])
S_TX2 = np.concatenate([L_TX2, [2.0, -1.0]])
S_RX2 = np.concatenate([L_RX2, [-3.0, 4.0]])
L_REF = np.array([1e3, -2e3, 300.0])

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


def _spher_meas(x, st, l_rx, m):
    xl = (np.eye(3) if m is None else m) @ (x - (np.zeros(3) if l_rx is None else l_rx))
    r = np.linalg.norm(xl)
    if st == 0:
        return np.array([np.arctan2(xl[1], xl[0]), np.arcsin(xl[2] / r)])
    if st == 1:
        return np.array([np.arctan2(xl[0], xl[2]), np.arcsin(xl[1] / r)])
    if st == 2:
        return np.array([np.arctan2(xl[1], xl[0]), np.arccos(xl[2] / r)])
    return np.array([np.arctan2(xl[0], xl[1]), np.arcsin(xl[2] / r)])


def _num_jac(f, x, h=1e-4):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(f(x))
    j = np.zeros((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        j[:, i] = (np.asarray(f(xp)) - np.asarray(f(xm))) / (2 * h)
    return j


class TestComponentGradientsAgainstMatlab:
    @pytest.mark.parametrize("uhr", [False, True])
    def test_range_gradient(self, uhr):
        _check(f"rangeGradient_uhr{int(uhr)}_bi", range_gradient(T, uhr, L_TX, L_RX))
        _check(f"rangeGradient_uhr{int(uhr)}_mono", range_gradient(T, uhr))

    @pytest.mark.parametrize("uhr", [False, True])
    def test_range_rate_gradient(self, uhr):
        _check(
            f"rangeRateGradient_uhr{int(uhr)}_bi",
            range_rate_gradient(XS, uhr, S_TX, S_RX),
        )
        _check(f"rangeRateGradient_uhr{int(uhr)}_mono", range_rate_gradient(XS, uhr))
        _check(
            f"rangeRateGradient2D_uhr{int(uhr)}_bi",
            range_rate_gradient(XS2, uhr, S_TX2, S_RX2),
        )

    @pytest.mark.parametrize("st", [0, 1, 2, 3])
    def test_spher_ang_gradient(self, st):
        _check(f"spherAngGradient_st{st}", spher_ang_gradient(T, st, L_RX, M))
        _check(f"spherAngGradient_st{st}_plain", spher_ang_gradient(T, st))

    @pytest.mark.parametrize("st", [0, 1])
    def test_pol_ang_gradient(self, st):
        _check(f"polAngGradient_st{st}", pol_ang_gradient(P2, st, L_RX2))

    def test_uv_gradients(self):
        _check("uvGradient", uv_gradient(T, L_RX, M))
        _check("uvGradient_w", uv_gradient(T, L_RX, M, include_w=True))
        _check("uGradient2D", u_gradient_2d(P2, L_RX2, M2))
        _check("uGradient2D_v", u_gradient_2d(P2, L_RX2, M2, include_v=True))
        _check("uGradient3D", u_gradient_3d(T, L_RX, M))

    def test_tdoa_and_norm_vec(self):
        _check("TDOAGradient", tdoa_gradient(T, L_REF, L_RX))
        _check("normVecJacob", norm_vec_jacob(T))


class TestMeasurementJacobiansAgainstMatlab:
    @pytest.mark.parametrize("st", [0, 1, 2, 3])
    def test_spher_family(self, st):
        _check(
            f"calcSpherJacob_st{st}_bi", calc_spher_jacob(T, st, False, L_TX, L_RX, M)
        )
        _check(f"calcSpherJacob_st{st}_mono", calc_spher_jacob(T, st))
        _check(f"calcSpherInvJacob_st{st}", calc_spher_inv_jacob([9e3, 0.5, 0.2], st))
        _check(
            f"calcSpherRRJacob_st{st}",
            calc_spher_rr_jacob(XS, st, False, S_TX, S_RX, M),
        )

    @pytest.mark.parametrize("st", [0, 1])
    def test_polar_family(self, st):
        _check(
            f"calcPolarJacob_st{st}_bi", calc_polar_jacob(P2, st, False, L_TX2, L_RX2)
        )
        _check(f"calcPolarJacob_st{st}_mono", calc_polar_jacob(P2, st))
        _check(
            f"calcPolarRRJacob_st{st}",
            calc_polar_rr_jacob(XS2, st, False, S_TX2, S_RX2),
        )

    def test_ruv_family(self):
        _check("calcRuvJacob", calc_ruv_jacob(T, False, L_TX, L_RX, M))
        _check(
            "calcRuvJacob_w", calc_ruv_jacob(T, False, L_TX, L_RX, M, include_w=True)
        )
        _check("calcRuvRRJacob", calc_ruv_rr_jacob(XS, False, S_TX, S_RX, M))
        _check(
            "calcRuvRRJacob_w",
            calc_ruv_rr_jacob(XS, False, S_TX, S_RX, M, include_w=True),
        )

    def test_cart_rr(self):
        _check("calcCartRRJacob_c0", calc_cart_rr_jacob(XS, 0, False, S_TX, S_RX))
        _check("calcCartRRJacob_c1", calc_cart_rr_jacob(XS, 1))

    def test_converted_family(self):
        # The same geometry's measurements, computed in Python: bistatic
        # range plus the rotated-frame angles/direction cosines.
        rb = np.linalg.norm(T - L_RX) + np.linalg.norm(T - L_TX)
        z_spher = np.concatenate([[rb], _spher_meas(T, 0, L_RX, M)])
        _check(
            "calcSpherConvJacob_st0_bi",
            calc_spher_conv_jacob(z_spher, 0, False, L_TX, L_RX, M),
        )
        r = np.linalg.norm(T)
        z_mono = np.concatenate([[r], _spher_meas(T, 0, None, None)])
        _check("calcSpherConvJacob_st0_mono", calc_spher_conv_jacob(z_mono, 0))

        rb2 = np.linalg.norm(P2 - L_RX2) + np.linalg.norm(P2 - L_TX2)
        pl = M2 @ (P2 - L_RX2)
        z_pol = np.array([rb2, np.arctan2(pl[1], pl[0])])
        _check(
            "calcPolarConvJacob_st0_bi",
            calc_polar_conv_jacob(z_pol, 0, False, L_TX2, L_RX2, M2),
        )
        _check(
            "calcPolarRRConvJacob_st0_bi",
            calc_polar_rr_conv_jacob(
                np.append(z_pol, 12.0), 0, False, L_TX2, L_RX2, M2
            ),
        )

        xl = M @ (T - L_RX)
        z_ruv = np.array([rb, xl[0] / np.linalg.norm(xl), xl[1] / np.linalg.norm(xl)])
        _check("calcRuvConvJacob_bi", calc_ruv_conv_jacob(z_ruv, False, L_TX, L_RX, M))
        _check(
            "calcRuvRRConvJacob_bi",
            calc_ruv_rr_conv_jacob(np.append(z_ruv, 25.0), False, L_TX, L_RX, M),
        )


class TestAgainstNumericalDifferentiation:
    """Independent oracle: finite differences of the measurements."""

    def test_spher_jacob_all_types_random_geometry(self):
        rng = np.random.default_rng(11)
        for st in (0, 1, 2, 3):
            x = rng.uniform(-1e4, 1e4, 3)
            l_rx = rng.uniform(-5e3, 5e3, 3)
            l_tx = rng.uniform(-5e3, 5e3, 3)
            m = expm(0.2 * (lambda a: a - a.T)(rng.standard_normal((3, 3))))

            def meas(p):
                rb = np.linalg.norm(p - l_rx) + np.linalg.norm(p - l_tx)
                return np.concatenate([[rb], _spher_meas(p, st, l_rx, m)])

            got = calc_spher_jacob(x, st, False, l_tx, l_rx, m)
            np.testing.assert_allclose(got, _num_jac(meas, x), rtol=2e-5, atol=1e-12)

    def test_spher_inv_is_the_inverse(self):
        rng = np.random.default_rng(13)
        for st in (0, 1, 2, 3):
            x = rng.uniform(-1e4, 1e4, 3)
            r = np.linalg.norm(x)
            ang = _spher_meas(x, st, None, None)
            z = np.array([r, ang[0], ang[1]])
            fwd = calc_spher_jacob(x, st, True)
            inv = calc_spher_inv_jacob(z, st)
            np.testing.assert_allclose(inv @ fwd, np.eye(3), atol=1e-11)

    def test_range_rate_gradient_moving_platforms(self):
        s = np.array([2e3, -4e3, 1e3, 25.0, 12.0, -8.0])
        s_tx = np.array([-1e3, 2e3, 500.0, 3.0, -6.0, 1.0])
        s_rx = np.array([4e3, 1e3, -2e3, -2.0, 4.0, 7.0])

        def rr(state):
            dtr = state[:3] - s_rx[:3]
            dtl = state[:3] - s_tx[:3]
            dvr = state[3:] - s_rx[3:]
            dvl = state[3:] - s_tx[3:]
            return [dtr @ dvr / np.linalg.norm(dtr) + dtl @ dvl / np.linalg.norm(dtl)]

        got = range_rate_gradient(s, False, s_tx, s_rx)
        np.testing.assert_allclose(got, _num_jac(rr, s), rtol=1e-6, atol=1e-12)

    def test_conv_jacobians_match_direct_at_the_recovered_point(self):
        # Inverting the measurement and differentiating there must give
        # the same matrix as differentiating at the original point.
        rb = np.linalg.norm(T - L_RX) + np.linalg.norm(T - L_TX)
        z = np.concatenate([[rb], _spher_meas(T, 0, L_RX, M)])
        np.testing.assert_allclose(
            calc_spher_conv_jacob(z, 0, False, L_TX, L_RX, M),
            calc_spher_jacob(T, 0, False, L_TX, L_RX, M),
            rtol=1e-9,
        )

    def test_collocated_transmitter_drops_its_term(self):
        g = range_gradient(T, False, T, L_RX)
        expected = (T - L_RX) / np.linalg.norm(T - L_RX)
        np.testing.assert_allclose(g[0], expected, rtol=1e-14)
        # Same in 3-D range rate: only the receiver term survives.
        g_rr = range_rate_gradient(XS, False, XS, S_RX)
        dtr = XS[:3] - S_RX[:3]
        dvr = XS[3:] - S_RX[3:]
        a = np.dot(dtr, dtr) * np.eye(3) - np.outer(dtr, dtr)
        np.testing.assert_allclose(
            g_rr[0, :3], a @ dvr / np.linalg.norm(dtr) ** 3, rtol=1e-14
        )
        np.testing.assert_allclose(g_rr[0, 3:], dtr / np.linalg.norm(dtr), rtol=1e-14)

    def test_one_dimensional_range_rate(self):
        s = np.array([5e3, 40.0])
        s_tx = np.array([-2e3, 3.0])
        s_rx = np.array([1e3, -7.0])

        def rr(state):
            dtr = state[:1] - s_rx[:1]
            dtl = state[:1] - s_tx[:1]
            dvr = state[1:] - s_rx[1:]
            dvl = state[1:] - s_tx[1:]
            return [dtr @ dvr / np.linalg.norm(dtr) + dtl @ dvl / np.linalg.norm(dtl)]

        got = range_rate_gradient(s, False, s_tx, s_rx)
        np.testing.assert_allclose(got, _num_jac(rr, s), rtol=1e-7, atol=1e-12)
        # Collocated transmitter in 1-D drops its velocity term.
        got_col = range_rate_gradient(s, False, s, s_rx)
        np.testing.assert_allclose(got_col[0, 1], 1.0, rtol=1e-14)

    def test_conv_jacobians_all_system_types_and_half_range(self):
        # Each converted Jacobian must equal the direct Jacobian at the
        # point the measurement inverts to, for every convention.
        for st in (1, 2, 3):
            rb = np.linalg.norm(T - L_RX) + np.linalg.norm(T - L_TX)
            z = np.concatenate([[rb], _spher_meas(T, st, L_RX, M)])
            np.testing.assert_allclose(
                calc_spher_conv_jacob(z, st, False, L_TX, L_RX, M),
                calc_spher_jacob(T, st, False, L_TX, L_RX, M),
                rtol=1e-9,
            )
        # Half-range bistatic spherical, and the monostatic full-range
        # arm of the inversion.
        rb = np.linalg.norm(T - L_RX) + np.linalg.norm(T - L_TX)
        z_half = np.concatenate([[rb / 2.0], _spher_meas(T, 0, L_RX, M)])
        np.testing.assert_allclose(
            calc_spher_conv_jacob(z_half, 0, True, L_TX, L_RX, M),
            calc_spher_jacob(T, 0, True, L_TX, L_RX, M),
            rtol=1e-9,
        )
        r = np.linalg.norm(T)
        z_full = np.concatenate([[2.0 * r], _spher_meas(T, 0, None, None)])
        np.testing.assert_allclose(
            calc_spher_conv_jacob(z_full, 0, False),
            calc_spher_jacob(T, 0, False),
            rtol=1e-9,
        )
        # Polar: the other angle convention, the half-range arm, and
        # the monostatic (collocated) inversion branch.
        pl = P2 - L_RX2
        for st, ang in ((0, np.arctan2(pl[1], pl[0])), (1, np.arctan2(pl[0], pl[1]))):
            rb2 = np.linalg.norm(P2 - L_RX2) + np.linalg.norm(P2 - L_TX2)
            np.testing.assert_allclose(
                calc_polar_conv_jacob([rb2, ang], st, False, L_TX2, L_RX2),
                calc_polar_jacob(P2, st, False, L_TX2, L_RX2),
                rtol=1e-9,
            )
        r2 = np.linalg.norm(P2 - L_RX2)
        ang0 = np.arctan2(pl[1], pl[0])
        np.testing.assert_allclose(
            calc_polar_conv_jacob([r2, ang0], 0, True, L_RX2, L_RX2),
            calc_polar_jacob(P2, 0, True, L_RX2, L_RX2),
            rtol=1e-9,
        )

    def test_invalid_conventions_are_rejected(self):
        with pytest.raises(ValueError):
            spher_ang_gradient(T, 9)
        with pytest.raises(ValueError):
            pol_ang_gradient(P2, 9)
        with pytest.raises(ValueError):
            calc_spher_inv_jacob([1e3, 0.1, 0.2], 9)
        with pytest.raises(ValueError):
            calc_spher_conv_jacob([1e3, 0.1, 0.2], 9)
        with pytest.raises(ValueError):
            calc_polar_conv_jacob([1e3, 0.1], 9)
        with pytest.raises(ValueError):
            range_rate_gradient(np.zeros(8))

    def test_batch_matches_single(self):
        pts = np.column_stack([T, T * 0.5 + 100.0])
        batch = spher_ang_gradient(pts, 0, L_RX, M)
        assert batch.shape == (2, 3, 2)
        # BLAS uses different kernels for matrix-matrix and
        # matrix-vector products, so agreement is to the last ULP, not
        # bitwise.
        np.testing.assert_allclose(
            batch[:, :, 0], spher_ang_gradient(T, 0, L_RX, M), rtol=1e-14
        )
