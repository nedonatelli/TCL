"""Magnetic coordinate systems against MATLAB and dipole theory.

Fixtures captured by scripts/matlab_capture/capture_magnetic_coords.m
(headless R2026a; the CD functions' default-model call form errors
upstream there -- ClusterSet 2-argument indexing -- so the fixture
driver extracts the degree-1 coefficients linearly, as
``spherITRS2SpherCD`` itself does).

The traced apex/QD fixtures compare the SciPy event-based tracer
against MATLAB's adaptive-RK-plus-fminbnd implementation, so their
tolerance reflects the two integrators' 1e-6 relative settings, not
transcription error. A pure-dipole closed form (field lines satisfy
r = L cos^2 of the magnetic latitude) provides an independent
mathematical oracle for the tracer.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from pytcl.magnetism.coordinates import (
    _dipole_terms,
    cart_cd2itrs,
    cd_rotation_matrix,
    geog_heading2mag,
    itrs2cart_cd,
    itrs2magnetic_apex,
    itrs2qd,
    mag_heading2geog,
    spher_cd2spher_itrs,
    spher_itrs2spher_cd,
    trace2earth_mag_apex,
)
from pytcl.magnetism.igrf import IGRF14
from pytcl.magnetism.wmm import WMM2025, MagneticCoefficients

FIXTURE = Path(__file__).parent.parent / "fixtures" / "matlab" / "magnetic_coords.csv"

G10, G11, H11 = -29350.0, -1410.3, 4545.5
Z_TEST = np.array([6.4e6, 1e5, 2e6])
Z_SPH = np.array([6.4e6, 0.5, 0.2])
POINTS = np.array(
    [[0.7, -1.2, 0.0], [-0.4, 2.0, 2.5], [100.0, 5e3, 10e3]]
)  # rows: lat / lon / h
HEADINGS = np.array([0.5, -1.0, 2.5])
TRACE_PTS = np.array(
    [[6.4e6, 5.8e6, 6.2e6], [1e5, -2.5e6, 1.9e6], [2e6, 1.5e6, -2.2e6]]
)


def _fixtures() -> dict:
    out = {}
    with open(FIXTURE) as f:
        for row in csv.reader(f):
            if row[0] == "label":
                continue
            out[row[0]] = np.array([float(v) for v in row[1:]])
    return out


FIX = _fixtures()


class TestCenteredDipoleAgainstMatlab:
    def test_cart_transforms_explicit(self):
        np.testing.assert_allclose(
            itrs2cart_cd(Z_TEST, G10, G11, H11),
            FIX["itrs2cart_cd_explicit"],
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            cart_cd2itrs(Z_TEST, G10, G11, H11),
            FIX["cart_cd2itrs_explicit"],
            rtol=1e-13,
        )

    def test_spher_transforms_explicit(self):
        np.testing.assert_allclose(
            spher_itrs2spher_cd(Z_SPH, G10, G11, H11),
            FIX["spher_itrs2cd_explicit"],
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            spher_cd2spher_itrs(Z_SPH, G10, G11, H11),
            FIX["spher_cd2itrs_explicit"],
            rtol=1e-13,
        )
        np.testing.assert_allclose(
            spher_itrs2spher_cd([0.5, 0.2], G10, G11, H11),
            FIX["spher_itrs2cd_2elem"],
            rtol=1e-13,
        )

    def test_igrf_default_matches_matlab_coefficients(self):
        # MATLAB works in Tesla; pytcl's IGRF14 tables are nT. The
        # degree-1 terms must agree after scaling, and the resulting
        # rotation exactly (it is scale-invariant).
        g10, g11, h11 = _dipole_terms(IGRF14, None)
        np.testing.assert_allclose(
            np.array([g10, g11, h11]) * 1e-9, FIX["igrf_degree1"], rtol=1e-12
        )
        np.testing.assert_allclose(
            itrs2cart_cd(Z_TEST), FIX["itrs2cart_cd_igrf"], rtol=1e-12
        )

    def test_rotation_sends_pole_to_z(self):
        g10, g11, h11 = _dipole_terms(IGRF14, None)
        pole = np.array([-g11, -h11, -g10])
        pole /= np.linalg.norm(pole)
        np.testing.assert_allclose(
            cd_rotation_matrix() @ pole, [0.0, 0.0, 1.0], atol=1e-14
        )


class TestHeadingsAgainstMatlab:
    def test_geog_to_mag(self):
        # With the IGRF the two chains agree to 1e-6 rad (worst at the
        # weak-horizontal-field Antarctic point), proving the pipeline.
        # The WMM rows carry a looser tolerance: the MATLAB tree ships
        # a 2024-11-13 pre-release WMM2025 COF while pytcl carries the
        # final release (validated against the official test values),
        # so the coefficient tables themselves differ slightly.
        for k in range(3):
            got = geog_heading2mag(POINTS[:, k], HEADINGS[k], WMM2025)
            np.testing.assert_allclose(got, FIX["geog_heading2mag_wmm"][k], atol=1e-2)
            got_i = geog_heading2mag(POINTS[:, k], HEADINGS[k], IGRF14)
            np.testing.assert_allclose(
                got_i, FIX["geog_heading2mag_igrf"][k], atol=2e-6
            )

    def test_mag_to_geog(self):
        for k in range(3):
            got = mag_heading2geog(POINTS[:, k], HEADINGS[k], WMM2025)
            np.testing.assert_allclose(got, FIX["mag_heading2geog_wmm"][k], atol=1e-2)

    def test_round_trip(self):
        p = POINTS[:, 0]
        back = mag_heading2geog(p, geog_heading2mag(p, 0.5))
        np.testing.assert_allclose(back, 0.5, atol=1e-14)


class TestApexTracingAgainstMatlab:
    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_traced_apex_points(self, k):
        apex, sign = trace2earth_mag_apex(TRACE_PTS[:, k])
        ref = FIX[f"trace_apex_{k + 1}"]
        assert sign == FIX[f"trace_sign_{k + 1}"][0]
        # Both tracers run at 1e-6 relative tolerance; agreement is
        # bounded by the integrators, not the transcription.
        np.testing.assert_allclose(apex, ref, rtol=2e-5)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_qd_coordinates(self, k):
        z, _ = itrs2qd(TRACE_PTS[:, k])
        ref = FIX[f"qd_{k + 1}"]
        np.testing.assert_allclose(z[0], ref[0], atol=2e-6)  # latitude, rad
        np.testing.assert_allclose(z[1], ref[1], atol=2e-6)  # longitude, rad
        np.testing.assert_allclose(z[2], ref[2], rtol=1e-9)  # height, m

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_apex_coordinates(self, k):
        z, _ = itrs2magnetic_apex(TRACE_PTS[:, k])
        ref = FIX[f"apex_{k + 1}"]
        np.testing.assert_allclose(z[0], ref[0], atol=2e-6)
        np.testing.assert_allclose(z[1], ref[1], atol=2e-6)
        np.testing.assert_allclose(z[2], ref[2], rtol=1e-9)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_modified_apex_coordinates(self, k):
        z, _ = itrs2magnetic_apex(TRACE_PTS[:, k], h_r=110e3)
        ref = FIX[f"apex_mod_{k + 1}"]
        np.testing.assert_allclose(z[0], ref[0], atol=2e-6)


class TestDipoleClosedForm:
    """Independent oracle: pure-dipole field lines are analytic."""

    def _dipole_coeffs(self) -> MagneticCoefficients:
        g = np.zeros_like(IGRF14.g)
        h = np.zeros_like(IGRF14.h)
        g[1, 0] = IGRF14.g[1, 0]
        g[1, 1] = IGRF14.g[1, 1]
        h[1, 1] = IGRF14.h[1, 1]
        return IGRF14._replace(g=g, h=h, g_dot=np.zeros_like(g), h_dot=np.zeros_like(h))

    def test_apex_matches_the_l_shell(self):
        # For a centered dipole, the field line through a point at CD
        # latitude lambda and radius r reaches its maximum radius
        # (the L-shell) at the CD equator: r_apex = r / cos^2 lambda,
        # with the apex on the CD equatorial plane.
        coeffs = self._dipole_coeffs()
        for x0 in ([6.9e6, 2e5, 2.4e6], [6.5e6, -3e6, -1.8e6]):
            x0 = np.array(x0)
            apex, _ = trace2earth_mag_apex(x0, coeffs)
            cd0 = itrs2cart_cd(x0, coeffs=coeffs)
            cd_apex = itrs2cart_cd(apex, coeffs=coeffs)
            r0 = np.linalg.norm(cd0)
            lam0 = np.arcsin(cd0[2] / r0)
            l_shell = r0 / np.cos(lam0) ** 2
            # The tracer stops at the maximum of *ellipsoidal* height,
            # not geocentric radius, so the flattening (~1/298) bounds
            # the residual mismatch of the stopping point; the radius
            # itself is far less sensitive there (the line is flat at
            # its apex).
            np.testing.assert_allclose(np.linalg.norm(cd_apex), l_shell, rtol=1e-4)
            assert abs(cd_apex[2]) / np.linalg.norm(cd_apex) < 0.06

    def test_polar_escape_reaches_infinity(self):
        # On the axis of a pure dipole the field line is exactly
        # radial and never comes back: the apex is declared infinite,
        # the QD latitude saturates at pi/2 and the longitude is NaN.
        coeffs = self._dipole_coeffs()
        x_axis = cart_cd2itrs([0.0, 0.0, 6.67e6], coeffs=coeffs)
        z, apex = itrs2qd(x_axis, coeffs=coeffs)
        assert np.all(np.isinf(apex))
        np.testing.assert_allclose(z[0], np.pi / 2, rtol=1e-12)
        assert np.isnan(z[1])
        za, _ = itrs2magnetic_apex(x_axis, coeffs=coeffs)
        np.testing.assert_allclose(za[0], np.pi / 2, rtol=1e-12)
        assert np.isnan(za[1])

    def test_apex_height_bounds_the_point(self):
        from pytcl.magnetism.coordinates import _height_of

        for k in range(3):
            z, apex = itrs2qd(TRACE_PTS[:, k])
            # The apex is the height maximum of the field line, so it
            # cannot sit below the starting point.
            assert _height_of(apex) >= z[2] - 1.0
            # The QD latitude sign matches the magnetic hemisphere
            # given by the trace direction.
            _, sign = trace2earth_mag_apex(TRACE_PTS[:, k])
            assert np.sign(z[0]) == -sign
