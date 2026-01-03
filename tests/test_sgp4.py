"""
Tests for SGP4/SDP4 satellite propagation.

Test cases include validation against known TLE propagation results
and verification of TLE parsing.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.astronomical.reference_frames import (
    gcrf_to_teme,
    itrf_to_teme,
    itrf_to_teme_with_velocity,
    teme_to_gcrf,
    teme_to_itrf,
    teme_to_itrf_with_velocity,
)
from pytcl.astronomical.sgp4 import (
    SGP4Satellite,
    sgp4_propagate,
    sgp4_propagate_batch,
)
from pytcl.astronomical.tle import (
    is_deep_space,
    orbital_period_from_tle,
    parse_tle,
    parse_tle_3line,
    semi_major_axis_from_mean_motion,
    tle_epoch_to_jd,
)

# ISS TLE for testing - synthetic TLE with correct checksums
ISS_TLE_LINE1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
ISS_TLE_LINE2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"

# Vanguard 1 TLE (high eccentricity test)
VANGUARD_LINE1 = "1 00005U 58002B   24001.00000000  .00000023  00000-0  28605-4 0  9997"
VANGUARD_LINE2 = "2 00005  34.2500  40.4560 1845947 262.5280  75.7980 10.84861856344568"

# GPS satellite TLE (deep space - medium Earth orbit, period > 225 min)
GPS_LINE1 = "1 28474U 04045A   24001.00000000 -.00000037  00000-0  00000-0 0  9996"
GPS_LINE2 = "2 28474  55.4330 143.3940 0056500 248.5560 110.9890  2.00563774144569"


class TestTLEParsing:
    """Tests for TLE parsing functionality."""

    def test_parse_iss_tle(self):
        """Test parsing ISS TLE."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2, name="ISS (ZARYA)")

        assert tle.name == "ISS (ZARYA)"
        assert tle.catalog_number == 25544
        assert tle.classification == "U"
        assert tle.int_designator == "98067A"
        assert tle.epoch_year == 2024
        assert_allclose(tle.epoch_day, 1.5, rtol=1e-6)
        assert_allclose(np.degrees(tle.inclination), 51.64, rtol=1e-4)
        assert_allclose(np.degrees(tle.raan), 247.4627, rtol=1e-4)
        assert_allclose(tle.eccentricity, 0.0006703, rtol=1e-4)

    def test_parse_3line_tle(self):
        """Test parsing 3-line TLE format."""
        tle_str = f"ISS (ZARYA)\n{ISS_TLE_LINE1}\n{ISS_TLE_LINE2}"
        tle = parse_tle_3line(tle_str)

        assert tle.name == "ISS (ZARYA)"
        assert tle.catalog_number == 25544

    def test_checksum_verification(self):
        """Test that invalid checksums are detected."""
        # Modify checksum to be wrong
        bad_line1 = ISS_TLE_LINE1[:-1] + "0"  # Change last digit

        with pytest.raises(ValueError, match="checksum"):
            parse_tle(bad_line1, ISS_TLE_LINE2)

    def test_checksum_skip(self):
        """Test that checksum verification can be disabled."""
        bad_line1 = ISS_TLE_LINE1[:-1] + "0"

        # Should not raise when verification is disabled
        tle = parse_tle(bad_line1, ISS_TLE_LINE2, verify_checksum=False)
        assert tle.catalog_number == 25544

    def test_catalog_number_mismatch(self):
        """Test that mismatched catalog numbers are detected."""
        # Modify catalog number in line 2
        bad_line2 = ISS_TLE_LINE2[:2] + "99999" + ISS_TLE_LINE2[7:]

        with pytest.raises(ValueError, match="mismatch"):
            parse_tle(ISS_TLE_LINE1, bad_line2, verify_checksum=False)

    def test_mean_motion_conversion(self):
        """Test mean motion is correctly converted to rad/min."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)

        # Mean motion from TLE: 15.49815350 rev/day
        # Convert to rad/min: 15.49815350 * 2*pi / 1440
        expected_n = 15.49815350 * 2 * np.pi / 1440.0
        assert_allclose(tle.mean_motion, expected_n, rtol=1e-6)

    def test_tle_epoch_to_jd(self):
        """Test TLE epoch conversion to Julian date."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        jd = tle_epoch_to_jd(tle)

        # January 1, 2024 at 12:00 UTC
        # JD 2460310.5 is midnight Jan 1 2024
        # epoch_day 1.5 means Jan 1 at noon
        expected_jd = 2460310.5 + 0.5  # 2460311.0
        assert_allclose(jd, expected_jd, atol=0.001)


class TestDeepSpaceDetection:
    """Tests for deep-space satellite detection."""

    def test_iss_is_not_deep_space(self):
        """ISS is a near-Earth satellite (period < 225 min)."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        assert not is_deep_space(tle)

        # ISS orbital period is about 92 minutes
        period = orbital_period_from_tle(tle)
        assert period < 225 * 60  # Less than 225 minutes

    def test_gps_is_deep_space(self):
        """GPS satellites are deep-space (period > 225 min)."""
        tle = parse_tle(GPS_LINE1, GPS_LINE2, verify_checksum=False)
        assert is_deep_space(tle)

        # GPS orbital period is about 12 hours (720 minutes)
        period = orbital_period_from_tle(tle)
        assert period > 225 * 60


class TestSemiMajorAxis:
    """Tests for semi-major axis computation."""

    def test_iss_semi_major_axis(self):
        """Test ISS semi-major axis computation."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        a = semi_major_axis_from_mean_motion(tle.mean_motion)

        # ISS altitude is about 420 km, so a ≈ 6378 + 420 = 6798 km
        # Allow some tolerance for drag effects
        assert 6700 < a < 6900

    def test_gps_semi_major_axis(self):
        """Test GPS semi-major axis computation."""
        tle = parse_tle(GPS_LINE1, GPS_LINE2, verify_checksum=False)
        a = semi_major_axis_from_mean_motion(tle.mean_motion)

        # GPS orbit is about 26,560 km altitude (a ≈ 26,560 km)
        assert 25000 < a < 28000


class TestSGP4Propagation:
    """Tests for SGP4 propagation."""

    def test_sgp4_at_epoch(self):
        """Test SGP4 propagation at TLE epoch (t=0)."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        state = sgp4_propagate(tle, 0.0)

        # At epoch, position should be reasonable for ISS
        r_mag = np.linalg.norm(state.r)
        v_mag = np.linalg.norm(state.v)

        # ISS altitude ~420 km, so r ≈ 6800 km
        assert 6400 < r_mag < 7000

        # ISS velocity ~7.66 km/s
        assert 7.0 < v_mag < 8.0

        # No error
        assert state.error == 0

    def test_sgp4_propagation_forward(self):
        """Test SGP4 propagation forward in time."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat = SGP4Satellite(tle)

        state0 = sat.propagate(0.0)
        state60 = sat.propagate(60.0)  # 60 minutes later

        # Position should change
        assert not np.allclose(state0.r, state60.r)

        # But orbital radius should be similar (circular orbit)
        r0 = np.linalg.norm(state0.r)
        r60 = np.linalg.norm(state60.r)
        assert_allclose(r0, r60, rtol=0.01)

    def test_sgp4_propagation_backward(self):
        """Test SGP4 propagation backward in time."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat = SGP4Satellite(tle)

        state0 = sat.propagate(0.0)
        state_neg = sat.propagate(-60.0)  # 60 minutes earlier

        # Position should change
        assert not np.allclose(state0.r, state_neg.r)

    def test_sgp4_full_orbit(self):
        """Test SGP4 propagation for one full orbit."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat = SGP4Satellite(tle)

        # ISS orbital period ~92 minutes
        period = orbital_period_from_tle(tle) / 60.0  # Convert to minutes

        state0 = sat.propagate(0.0)
        state_orbit = sat.propagate(period)

        # After one orbit, position should be close to original
        # (not exact due to J2/J3/J4 perturbations and drag)
        # Allow 5% tolerance due to secular drift from perturbations
        assert_allclose(state0.r, state_orbit.r, rtol=0.05)

    def test_sgp4_batch_propagation(self):
        """Test batch propagation."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        times = np.array([0.0, 30.0, 60.0, 90.0])

        r, v = sgp4_propagate_batch(tle, times)

        assert r.shape == (4, 3)
        assert v.shape == (4, 3)

        # All positions should be valid
        for i in range(4):
            r_mag = np.linalg.norm(r[i])
            assert 6400 < r_mag < 7000

    def test_sgp4_satellite_class(self):
        """Test SGP4Satellite class interface."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2, name="ISS")
        sat = SGP4Satellite(tle)

        assert sat.tle.name == "ISS"
        assert not sat.is_deep_space
        assert sat.epoch_jd > 2460000  # After 2023

    def test_propagate_by_jd(self):
        """Test propagation to Julian date."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat = SGP4Satellite(tle)

        jd_epoch = sat.epoch_jd
        state0 = sat.propagate(0.0)
        state_jd = sat.propagate_jd(jd_epoch)

        # Should give same result
        assert_allclose(state0.r, state_jd.r, rtol=1e-10)
        assert_allclose(state0.v, state_jd.v, rtol=1e-10)


class TestSDP4Propagation:
    """Tests for SDP4 (deep-space) propagation."""

    def test_sdp4_gps(self):
        """Test SDP4 propagation for GPS satellite."""
        tle = parse_tle(GPS_LINE1, GPS_LINE2, verify_checksum=False)
        sat = SGP4Satellite(tle)

        assert sat.is_deep_space

        state = sat.propagate(0.0)

        # GPS altitude ~20,200 km, so r ≈ 26,560 km
        r_mag = np.linalg.norm(state.r)
        assert 25000 < r_mag < 28000

        # GPS velocity ~3.87 km/s
        v_mag = np.linalg.norm(state.v)
        assert 3.0 < v_mag < 5.0


class TestTEMETransformations:
    """Tests for TEME reference frame transformations."""

    def test_teme_to_itrf_roundtrip(self):
        """Test TEME <-> ITRF roundtrip."""
        r_teme = np.array([5000.0, 1000.0, 3000.0])
        jd_ut1 = 2460311.0

        r_itrf = teme_to_itrf(r_teme, jd_ut1)
        r_back = itrf_to_teme(r_itrf, jd_ut1)

        assert_allclose(r_back, r_teme, rtol=1e-10)

    def test_teme_to_gcrf_roundtrip(self):
        """Test TEME <-> GCRF roundtrip."""
        r_teme = np.array([5000.0, 1000.0, 3000.0])
        jd_tt = 2460311.0

        r_gcrf = teme_to_gcrf(r_teme, jd_tt)
        r_back = gcrf_to_teme(r_gcrf, jd_tt)

        assert_allclose(r_back, r_teme, rtol=1e-10)

    def test_teme_itrf_velocity_roundtrip(self):
        """Test TEME <-> ITRF with velocity roundtrip."""
        r_teme = np.array([6800.0, 0.0, 0.0])
        v_teme = np.array([0.0, 7.5, 0.0])
        jd_ut1 = 2460311.0

        r_itrf, v_itrf = teme_to_itrf_with_velocity(r_teme, v_teme, jd_ut1)
        r_back, v_back = itrf_to_teme_with_velocity(r_itrf, v_itrf, jd_ut1)

        assert_allclose(r_back, r_teme, rtol=1e-10)
        # Use both atol and rtol for velocity due to floating point precision
        # especially across platforms and Python versions (Windows Python 3.12 may have differences)
        assert_allclose(v_back, v_teme, atol=1e-14, rtol=1e-12)

    def test_sgp4_to_itrf_integration(self):
        """Test full SGP4 -> ITRF pipeline."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        state = sgp4_propagate(tle, 0.0)
        jd_ut1 = tle_epoch_to_jd(tle)

        # Transform to ITRF
        r_itrf, v_itrf = teme_to_itrf_with_velocity(state.r, state.v, jd_ut1)

        # Magnitudes should be preserved
        assert_allclose(np.linalg.norm(r_itrf), np.linalg.norm(state.r), rtol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_small_eccentricity(self):
        """Test handling of nearly circular orbit."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        # ISS has e ≈ 0.0007, which is very small
        assert tle.eccentricity < 0.001

        # Should still propagate correctly
        state = sgp4_propagate(tle, 60.0)
        assert state.error == 0

    def test_long_propagation(self):
        """Test propagation over extended time period."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        sat = SGP4Satellite(tle)

        # Propagate for 24 hours (1440 minutes)
        state = sat.propagate(1440.0)

        # Should still be in valid orbit
        r_mag = np.linalg.norm(state.r)
        assert 6000 < r_mag < 8000

    def test_high_eccentricity(self):
        """Test propagation with higher eccentricity orbit."""
        tle = parse_tle(VANGUARD_LINE1, VANGUARD_LINE2, verify_checksum=False)

        # Vanguard 1 has e ≈ 0.185
        assert tle.eccentricity > 0.1

        state = sgp4_propagate(tle, 0.0)

        # Should return valid position
        r_mag = np.linalg.norm(state.r)
        assert r_mag > 0


class TestOrbitalPeriod:
    """Tests for orbital period computation."""

    def test_iss_period(self):
        """Test ISS orbital period calculation."""
        tle = parse_tle(ISS_TLE_LINE1, ISS_TLE_LINE2)
        period = orbital_period_from_tle(tle)

        # ISS period is about 92 minutes = 5520 seconds
        assert 5400 < period < 5700

    def test_gps_period(self):
        """Test GPS orbital period calculation."""
        tle = parse_tle(GPS_LINE1, GPS_LINE2, verify_checksum=False)
        period = orbital_period_from_tle(tle)

        # GPS period is about 12 hours = 43200 seconds
        assert 40000 < period < 46000
