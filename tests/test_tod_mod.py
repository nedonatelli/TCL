"""
Tests for TOD/MOD (True of Date / Mean of Date) reference frame transformations.

These are legacy reference frame conventions used in older astrodynamics software.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.astronomical.reference_frames import (
    gcrf_to_itrf,
    gcrf_to_mod,
    gcrf_to_tod,
    itrf_to_gcrf,
    itrf_to_tod,
    mod_to_gcrf,
    mod_to_tod,
    nutation_matrix,
    precession_matrix_iau76,
    tod_to_gcrf,
    tod_to_itrf,
    tod_to_mod,
)

# Test epoch: J2000.0 + 10 years (approximately 2010)
JD_TT_TEST = 2455197.5  # 2010-01-01 12:00:00 TT
JD_UT1_TEST = 2455197.5  # Assume UT1 ≈ TT for testing


class TestGCRFToMOD:
    """Tests for GCRF to MOD (Mean of Date) transformation."""

    def test_gcrf_to_mod_basic(self):
        """Test basic GCRF to MOD transformation."""
        r_gcrf = np.array([6378.0, 0.0, 0.0])

        r_mod = gcrf_to_mod(r_gcrf, JD_TT_TEST)

        # Result should be a 3-vector
        assert r_mod.shape == (3,)
        # Magnitude should be preserved (rotation only)
        assert_allclose(np.linalg.norm(r_mod), np.linalg.norm(r_gcrf), rtol=1e-10)

    def test_gcrf_mod_roundtrip(self):
        """Test GCRF -> MOD -> GCRF roundtrip."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        r_mod = gcrf_to_mod(r_gcrf, JD_TT_TEST)
        r_gcrf_back = mod_to_gcrf(r_mod, JD_TT_TEST)

        assert_allclose(r_gcrf_back, r_gcrf, rtol=1e-10)

    def test_gcrf_to_mod_uses_precession(self):
        """Verify MOD transformation uses precession matrix."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        r_mod = gcrf_to_mod(r_gcrf, JD_TT_TEST)

        # Should equal precession matrix times r_gcrf
        P = precession_matrix_iau76(JD_TT_TEST)
        expected = P @ r_gcrf

        assert_allclose(r_mod, expected, rtol=1e-14)

    def test_mod_at_j2000_equals_gcrf(self):
        """At J2000.0, MOD should equal GCRF (no precession)."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])
        jd_j2000 = 2451545.0

        r_mod = gcrf_to_mod(r_gcrf, jd_j2000)

        # Should be nearly identical at J2000
        assert_allclose(r_mod, r_gcrf, rtol=1e-10)


class TestGCRFToTOD:
    """Tests for GCRF to TOD (True of Date) transformation."""

    def test_gcrf_to_tod_basic(self):
        """Test basic GCRF to TOD transformation."""
        r_gcrf = np.array([6378.0, 0.0, 0.0])

        r_tod = gcrf_to_tod(r_gcrf, JD_TT_TEST)

        assert r_tod.shape == (3,)
        # Magnitude should be preserved
        assert_allclose(np.linalg.norm(r_tod), np.linalg.norm(r_gcrf), rtol=1e-10)

    def test_gcrf_tod_roundtrip(self):
        """Test GCRF -> TOD -> GCRF roundtrip."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        r_tod = gcrf_to_tod(r_gcrf, JD_TT_TEST)
        r_gcrf_back = tod_to_gcrf(r_tod, JD_TT_TEST)

        assert_allclose(r_gcrf_back, r_gcrf, rtol=1e-10)

    def test_gcrf_to_tod_uses_precession_and_nutation(self):
        """Verify TOD uses both precession and nutation."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        r_tod = gcrf_to_tod(r_gcrf, JD_TT_TEST)

        # Should equal N @ P @ r_gcrf
        P = precession_matrix_iau76(JD_TT_TEST)
        N = nutation_matrix(JD_TT_TEST)
        expected = N @ (P @ r_gcrf)

        assert_allclose(r_tod, expected, rtol=1e-14)


class TestMODToTOD:
    """Tests for MOD to TOD transformation (nutation only)."""

    def test_mod_to_tod_basic(self):
        """Test MOD to TOD transformation."""
        r_mod = np.array([6378.0, 1000.0, 0.0])

        r_tod = mod_to_tod(r_mod, JD_TT_TEST)

        assert r_tod.shape == (3,)
        # Magnitude should be preserved
        assert_allclose(np.linalg.norm(r_tod), np.linalg.norm(r_mod), rtol=1e-10)

    def test_mod_tod_roundtrip(self):
        """Test MOD -> TOD -> MOD roundtrip."""
        r_mod = np.array([6378.0, 1000.0, 2000.0])

        r_tod = mod_to_tod(r_mod, JD_TT_TEST)
        r_mod_back = tod_to_mod(r_tod, JD_TT_TEST)

        assert_allclose(r_mod_back, r_mod, rtol=1e-10)

    def test_mod_to_tod_uses_nutation(self):
        """Verify MOD->TOD uses nutation matrix."""
        r_mod = np.array([6378.0, 1000.0, 2000.0])

        r_tod = mod_to_tod(r_mod, JD_TT_TEST)

        N = nutation_matrix(JD_TT_TEST)
        expected = N @ r_mod

        assert_allclose(r_tod, expected, rtol=1e-14)


class TestTODToITRF:
    """Tests for TOD to ITRF transformation."""

    def test_tod_to_itrf_basic(self):
        """Test TOD to ITRF transformation."""
        r_tod = np.array([6378.0, 0.0, 0.0])

        r_itrf = tod_to_itrf(r_tod, JD_UT1_TEST)

        assert r_itrf.shape == (3,)
        # Magnitude should be preserved
        assert_allclose(np.linalg.norm(r_itrf), np.linalg.norm(r_tod), rtol=1e-10)

    def test_tod_itrf_roundtrip(self):
        """Test TOD -> ITRF -> TOD roundtrip."""
        r_tod = np.array([6378.0, 1000.0, 2000.0])

        r_itrf = tod_to_itrf(r_tod, JD_UT1_TEST)
        r_tod_back = itrf_to_tod(r_itrf, JD_UT1_TEST)

        assert_allclose(r_tod_back, r_tod, rtol=1e-10)

    def test_tod_itrf_with_polar_motion(self):
        """Test TOD to ITRF with polar motion correction."""
        r_tod = np.array([6378.0, 1000.0, 2000.0])
        xp = 0.1 * np.pi / 180 / 3600  # 0.1 arcsec
        yp = 0.2 * np.pi / 180 / 3600  # 0.2 arcsec

        r_itrf = tod_to_itrf(r_tod, JD_UT1_TEST, xp=xp, yp=yp)
        r_tod_back = itrf_to_tod(r_itrf, JD_UT1_TEST, xp=xp, yp=yp)

        assert_allclose(r_tod_back, r_tod, rtol=1e-10)


class TestTransformationChains:
    """Tests for complete transformation chains."""

    def test_gcrf_itrf_via_tod(self):
        """Test GCRF -> TOD -> ITRF matches GCRF -> ITRF."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        # Via TOD
        r_tod = gcrf_to_tod(r_gcrf, JD_TT_TEST)
        r_itrf_via_tod = tod_to_itrf(r_tod, JD_UT1_TEST)

        # Direct (using existing function)
        r_itrf_direct = gcrf_to_itrf(r_gcrf, JD_TT_TEST, JD_UT1_TEST)

        # Should be very close (may differ slightly due to implementation)
        assert_allclose(r_itrf_via_tod, r_itrf_direct, rtol=1e-8)

    def test_itrf_gcrf_via_tod(self):
        """Test ITRF -> TOD -> GCRF matches ITRF -> GCRF."""
        r_itrf = np.array([6378.0, 1000.0, 2000.0])

        # Via TOD
        r_tod = itrf_to_tod(r_itrf, JD_UT1_TEST)
        r_gcrf_via_tod = tod_to_gcrf(r_tod, JD_TT_TEST)

        # Direct
        r_gcrf_direct = itrf_to_gcrf(r_itrf, JD_TT_TEST, JD_UT1_TEST)

        assert_allclose(r_gcrf_via_tod, r_gcrf_direct, rtol=1e-8)

    def test_gcrf_to_mod_to_tod_to_gcrf(self):
        """Test complete chain: GCRF -> MOD -> TOD -> GCRF."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        r_mod = gcrf_to_mod(r_gcrf, JD_TT_TEST)
        r_tod = mod_to_tod(r_mod, JD_TT_TEST)
        r_gcrf_back = tod_to_gcrf(r_tod, JD_TT_TEST)

        assert_allclose(r_gcrf_back, r_gcrf, rtol=1e-10)


class TestMagnitudePreservation:
    """Tests that all transformations preserve vector magnitude."""

    def test_all_transformations_preserve_magnitude(self):
        """All rotations should preserve magnitude."""
        r = np.array([6378.0, 1000.0, 2000.0])
        mag_original = np.linalg.norm(r)

        transformations = [
            ("gcrf_to_mod", gcrf_to_mod(r, JD_TT_TEST)),
            ("mod_to_gcrf", mod_to_gcrf(r, JD_TT_TEST)),
            ("gcrf_to_tod", gcrf_to_tod(r, JD_TT_TEST)),
            ("tod_to_gcrf", tod_to_gcrf(r, JD_TT_TEST)),
            ("mod_to_tod", mod_to_tod(r, JD_TT_TEST)),
            ("tod_to_mod", tod_to_mod(r, JD_TT_TEST)),
            ("tod_to_itrf", tod_to_itrf(r, JD_UT1_TEST)),
            ("itrf_to_tod", itrf_to_tod(r, JD_UT1_TEST)),
        ]

        for name, result in transformations:
            mag_result = np.linalg.norm(result)
            assert_allclose(
                mag_result,
                mag_original,
                rtol=1e-10,
                err_msg=f"{name} changed magnitude",
            )


class TestOrthogonality:
    """Tests that transformation matrices are orthogonal."""

    def test_precession_is_orthogonal(self):
        """Precession matrix should be orthogonal."""
        P = precession_matrix_iau76(JD_TT_TEST)

        # P @ P.T should equal identity (use atol for near-zero elements)
        assert_allclose(P @ P.T, np.eye(3), atol=1e-14)
        # det(P) should be 1 (proper rotation)
        assert_allclose(np.linalg.det(P), 1.0, rtol=1e-10)

    def test_nutation_is_orthogonal(self):
        """Nutation matrix should be orthogonal."""
        N = nutation_matrix(JD_TT_TEST)

        assert_allclose(N @ N.T, np.eye(3), atol=1e-14)
        assert_allclose(np.linalg.det(N), 1.0, rtol=1e-10)


class TestDifferentEpochs:
    """Tests at various epochs to ensure time-dependence works."""

    @pytest.mark.parametrize(
        "jd_tt",
        [
            2451545.0,  # J2000.0
            2455197.5,  # 2010
            2458849.5,  # 2020
            2462502.5,  # 2030
        ],
    )
    def test_roundtrip_at_various_epochs(self, jd_tt):
        """Test roundtrip transformations at various epochs."""
        r_gcrf = np.array([6378.0, 1000.0, 2000.0])

        # GCRF -> MOD -> GCRF
        r_mod = gcrf_to_mod(r_gcrf, jd_tt)
        r_back = mod_to_gcrf(r_mod, jd_tt)
        assert_allclose(r_back, r_gcrf, rtol=1e-10)

        # GCRF -> TOD -> GCRF
        r_tod = gcrf_to_tod(r_gcrf, jd_tt)
        r_back = tod_to_gcrf(r_tod, jd_tt)
        assert_allclose(r_back, r_gcrf, rtol=1e-10)

    def test_precession_increases_with_time(self):
        """Precession should increase with time from J2000."""
        r_gcrf = np.array([6378.0, 0.0, 0.0])

        r_mod_2010 = gcrf_to_mod(r_gcrf, 2455197.5)
        r_mod_2020 = gcrf_to_mod(r_gcrf, 2458849.5)

        # Difference from GCRF should be larger at later epoch
        diff_2010 = np.linalg.norm(r_mod_2010 - r_gcrf)
        diff_2020 = np.linalg.norm(r_mod_2020 - r_gcrf)

        assert diff_2020 > diff_2010
