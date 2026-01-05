"""
Tests for ionospheric models.

Tests cover:
- Klobuchar ionospheric delay model
- Dual-frequency TEC computation
- Ionospheric delay from TEC
- Simplified IRI model
- Magnetic latitude calculation
- Scintillation index estimation
"""

import numpy as np
import pytest

from pytcl.atmosphere.ionosphere import (
    DEFAULT_KLOBUCHAR,
    F_L1,
    F_L2,
    SPEED_OF_LIGHT,
    IonosphereState,
    KlobucharCoefficients,
    dual_frequency_tec,
    ionospheric_delay_from_tec,
    klobuchar_delay,
    magnetic_latitude,
    scintillation_index,
    simple_iri,
)

# =============================================================================
# Tests for constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_speed_of_light(self):
        """Test speed of light constant."""
        assert SPEED_OF_LIGHT == pytest.approx(299792458.0)

    def test_gps_frequencies(self):
        """Test GPS frequency constants."""
        assert F_L1 == pytest.approx(1575.42e6)
        assert F_L2 == pytest.approx(1227.60e6)
        assert F_L1 > F_L2  # L1 is higher frequency


# =============================================================================
# Tests for Klobuchar coefficients
# =============================================================================


class TestKlobucharCoefficients:
    """Tests for Klobuchar coefficient handling."""

    def test_default_coefficients_shape(self):
        """Test default Klobuchar coefficients shape."""
        assert len(DEFAULT_KLOBUCHAR.alpha) == 4
        assert len(DEFAULT_KLOBUCHAR.beta) == 4

    def test_custom_coefficients(self):
        """Test creating custom coefficients."""
        alpha = np.array([1e-8, 2e-8, 3e-8, 4e-8])
        beta = np.array([1e5, 2e5, 3e5, 4e5])

        coef = KlobucharCoefficients(alpha=alpha, beta=beta)

        np.testing.assert_array_equal(coef.alpha, alpha)
        np.testing.assert_array_equal(coef.beta, beta)


# =============================================================================
# Tests for klobuchar_delay
# =============================================================================


class TestKlobucharDelay:
    """Tests for Klobuchar ionospheric delay model."""

    def test_klobuchar_delay_basic(self):
        """Test basic Klobuchar delay computation."""
        delay = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            elevation=np.radians(45),
            azimuth=np.radians(180),
            gps_time=43200,  # Noon
        )

        assert delay > 0  # Delay should be positive
        assert delay < 50  # Typical delays are < 50 meters

    def test_klobuchar_delay_noon_vs_midnight(self):
        """Test that delay is larger at noon than midnight."""
        # Use a longitude where noon GPS time corresponds to local noon
        delay_noon = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(0),  # Prime meridian for consistent local time
            elevation=np.radians(45),
            azimuth=np.radians(180),
            gps_time=50400,  # 14:00 UTC (peak ionospheric time)
        )

        delay_early_morning = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(0),
            elevation=np.radians(45),
            azimuth=np.radians(180),
            gps_time=18000,  # 05:00 UTC (early morning, low ionosphere)
        )

        assert delay_noon > delay_early_morning

    def test_klobuchar_delay_low_elevation(self):
        """Test that delay increases at low elevation angles."""
        delay_high = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            elevation=np.radians(80),
            azimuth=np.radians(180),
            gps_time=43200,
        )

        delay_low = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            elevation=np.radians(15),
            azimuth=np.radians(180),
            gps_time=43200,
        )

        assert delay_low > delay_high  # More atmosphere at low elevation

    def test_klobuchar_delay_with_custom_coefficients(self):
        """Test Klobuchar delay with custom coefficients."""
        alpha = np.array([5e-8, 1e-8, -5e-8, 0])
        beta = np.array([1.5e5, 0, -3e5, 1e5])
        coef = KlobucharCoefficients(alpha=alpha, beta=beta)

        delay = klobuchar_delay(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            elevation=np.radians(45),
            azimuth=np.radians(180),
            gps_time=43200,
            coefficients=coef,
        )

        assert delay > 0

    def test_klobuchar_delay_array_input(self):
        """Test Klobuchar delay with array inputs."""
        latitudes = np.radians([30, 40, 50])
        longitudes = np.radians([-105, -105, -105])
        elevations = np.radians([45, 45, 45])
        azimuths = np.radians([180, 180, 180])
        gps_times = np.array([43200, 43200, 43200])

        delays = klobuchar_delay(latitudes, longitudes, elevations, azimuths, gps_times)

        assert len(delays) == 3
        assert np.all(delays > 0)

    def test_klobuchar_delay_polar_latitude_clipping(self):
        """Test that extreme latitudes are handled (clipped at IPP)."""
        # High latitude (should be clipped internally)
        delay = klobuchar_delay(
            latitude=np.radians(80),
            longitude=np.radians(0),
            elevation=np.radians(45),
            azimuth=np.radians(0),  # Looking north
            gps_time=43200,
        )

        assert delay > 0
        assert np.isfinite(delay)


# =============================================================================
# Tests for dual_frequency_tec
# =============================================================================


class TestDualFrequencyTec:
    """Tests for dual-frequency TEC computation."""

    def test_dual_frequency_tec_basic(self):
        """Test basic TEC computation from dual-frequency pseudoranges."""
        # L2 pseudorange is slightly larger due to ionospheric delay
        p_l1 = 22000000.0
        p_l2 = 22000002.5  # About 2.5m more delay

        tec = dual_frequency_tec(p_l1, p_l2)

        assert tec > 0  # TEC should be positive

    def test_dual_frequency_tec_zero_difference(self):
        """Test TEC is zero when pseudoranges are equal."""
        p_l1 = 22000000.0
        p_l2 = 22000000.0

        tec = dual_frequency_tec(p_l1, p_l2)

        assert tec == pytest.approx(0.0)

    def test_dual_frequency_tec_array_input(self):
        """Test TEC computation with array inputs."""
        p_l1 = np.array([22000000.0, 22500000.0, 23000000.0])
        p_l2 = np.array([22000002.5, 22500003.0, 23000002.0])

        tec = dual_frequency_tec(p_l1, p_l2)

        assert len(tec) == 3
        assert np.all(tec > 0)


# =============================================================================
# Tests for ionospheric_delay_from_tec
# =============================================================================


class TestIonosphericDelayFromTec:
    """Tests for ionospheric delay computation from TEC."""

    def test_delay_from_tec_basic(self):
        """Test basic delay computation from TEC."""
        tec = 20.0  # 20 TECU (typical mid-latitude value)

        delay = ionospheric_delay_from_tec(tec)

        assert delay > 0
        assert delay < 15  # Typical L1 delay for 20 TECU

    def test_delay_proportional_to_tec(self):
        """Test that delay is proportional to TEC."""
        delay_10 = ionospheric_delay_from_tec(10.0)
        delay_20 = ionospheric_delay_from_tec(20.0)

        assert delay_20 == pytest.approx(2 * delay_10)

    def test_delay_l2_larger_than_l1(self):
        """Test that L2 delay is larger than L1 delay."""
        tec = 20.0

        delay_l1 = ionospheric_delay_from_tec(tec, frequency=F_L1)
        delay_l2 = ionospheric_delay_from_tec(tec, frequency=F_L2)

        assert delay_l2 > delay_l1  # Lower frequency has more delay

    def test_delay_frequency_dependence(self):
        """Test 1/f² frequency dependence."""
        tec = 20.0

        delay_l1 = ionospheric_delay_from_tec(tec, frequency=F_L1)
        delay_l2 = ionospheric_delay_from_tec(tec, frequency=F_L2)

        # Ratio should be (f1/f2)²
        expected_ratio = (F_L1 / F_L2) ** 2
        actual_ratio = delay_l2 / delay_l1

        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-10)

    def test_delay_array_input(self):
        """Test delay computation with array input."""
        tec = np.array([10.0, 20.0, 30.0])

        delay = ionospheric_delay_from_tec(tec)

        assert len(delay) == 3
        assert np.all(delay > 0)


# =============================================================================
# Tests for simple_iri
# =============================================================================


class TestSimpleIri:
    """Tests for simplified IRI model."""

    def test_simple_iri_basic(self):
        """Test basic IRI state computation."""
        state = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
        )

        assert isinstance(state, IonosphereState)
        assert state.tec > 0
        assert state.delay_l1 > 0
        assert state.delay_l2 > 0
        assert state.f_peak > 0
        assert state.h_peak > 0

    def test_simple_iri_noon_vs_night(self):
        """Test that TEC is higher at noon than at night."""
        state_noon = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=14,  # Afternoon
        )

        state_night = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=2,  # Night
        )

        assert state_noon.tec > state_night.tec

    def test_simple_iri_solar_activity(self):
        """Test effect of solar activity on TEC."""
        state_low = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
            solar_flux=80,  # Low activity
        )

        state_high = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
            solar_flux=200,  # High activity
        )

        assert state_high.tec > state_low.tec

    def test_simple_iri_seasonal_variation(self):
        """Test seasonal variation (summer vs winter)."""
        state_summer = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
            month=6,  # June (summer in NH)
        )

        state_winter = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
            month=12,  # December (winter in NH)
        )

        # Both should be valid
        assert state_summer.tec > 0
        assert state_winter.tec > 0

    def test_simple_iri_array_input(self):
        """Test IRI with array inputs."""
        latitudes = np.radians([30, 40, 50])
        longitudes = np.radians([-105, -105, -105])
        altitudes = np.array([300e3, 300e3, 300e3])
        hours = np.array([12, 12, 12])

        state = simple_iri(latitudes, longitudes, altitudes, hours)

        assert len(state.tec) == 3
        assert np.all(state.tec > 0)

    def test_simple_iri_delay_consistency(self):
        """Test that delays are consistent with TEC."""
        state = simple_iri(
            latitude=np.radians(40),
            longitude=np.radians(-105),
            altitude=300e3,
            hour=12,
        )

        # Compute expected delay from TEC
        expected_l1 = ionospheric_delay_from_tec(state.tec, F_L1)
        expected_l2 = ionospheric_delay_from_tec(state.tec, F_L2)

        assert state.delay_l1 == pytest.approx(expected_l1)
        assert state.delay_l2 == pytest.approx(expected_l2)


# =============================================================================
# Tests for magnetic_latitude
# =============================================================================


class TestMagneticLatitude:
    """Tests for magnetic latitude computation."""

    def test_magnetic_latitude_new_york(self):
        """Test magnetic latitude for New York City."""
        # NYC: 40.7°N, 74°W
        mag_lat = magnetic_latitude(np.radians(40.7), np.radians(-74))

        # Should be around 51° magnetic latitude
        assert 45 < np.degrees(mag_lat) < 55

    def test_magnetic_latitude_equator(self):
        """Test magnetic latitude at geographic equator."""
        mag_lat = magnetic_latitude(0.0, 0.0)

        # Geographic equator at 0° longitude is near magnetic equator
        assert np.abs(np.degrees(mag_lat)) < 15

    def test_magnetic_latitude_pole(self):
        """Test magnetic latitude near magnetic pole."""
        # Near the magnetic pole location (80.5°N, 72.8°W)
        mag_lat = magnetic_latitude(np.radians(80.5), np.radians(-72.8))

        # Should be very high magnetic latitude
        assert np.degrees(mag_lat) > 80

    def test_magnetic_latitude_array_input(self):
        """Test magnetic latitude with array inputs."""
        latitudes = np.radians([0, 30, 60])
        longitudes = np.radians([0, 0, 0])

        mag_lats = magnetic_latitude(latitudes, longitudes)

        assert len(mag_lats) == 3

    def test_magnetic_latitude_symmetry(self):
        """Test that magnetic latitude has expected asymmetry."""
        # Due to offset of magnetic pole from geographic pole,
        # positive and negative geographic latitudes give different results
        mag_lat_pos = magnetic_latitude(np.radians(45), np.radians(0))
        mag_lat_neg = magnetic_latitude(np.radians(-45), np.radians(0))

        # They should not be exactly symmetric
        assert not np.isclose(np.abs(mag_lat_pos), np.abs(mag_lat_neg))


# =============================================================================
# Tests for scintillation_index
# =============================================================================


class TestScintillationIndex:
    """Tests for scintillation index estimation."""

    def test_scintillation_equatorial_night(self):
        """Test high scintillation at equatorial regions at night."""
        s4 = scintillation_index(
            magnetic_latitude=np.radians(10),  # Near magnetic equator
            hour=21,  # Post-sunset
            kp_index=5.0,
        )

        assert s4 > 0.3  # Moderate to strong scintillation expected

    def test_scintillation_midlat_daytime(self):
        """Test low scintillation at mid-latitudes during daytime."""
        s4 = scintillation_index(
            magnetic_latitude=np.radians(45),
            hour=12,  # Noon
            kp_index=1.0,  # Quiet conditions
        )

        assert s4 < 0.2  # Low scintillation expected

    def test_scintillation_auroral_zone(self):
        """Test scintillation in auroral zone during storm."""
        s4 = scintillation_index(
            magnetic_latitude=np.radians(70),  # Auroral zone
            hour=21,
            kp_index=7.0,  # Storm conditions
        )

        assert s4 > 0.1  # Some scintillation expected

    def test_scintillation_kp_dependence(self):
        """Test that scintillation increases with Kp index."""
        s4_quiet = scintillation_index(
            magnetic_latitude=np.radians(15),
            hour=20,
            kp_index=1.0,
        )

        s4_storm = scintillation_index(
            magnetic_latitude=np.radians(15),
            hour=20,
            kp_index=8.0,
        )

        assert s4_storm > s4_quiet

    def test_scintillation_bounded(self):
        """Test that S4 is bounded between 0 and 1."""
        # Test extreme conditions
        s4_max = scintillation_index(
            magnetic_latitude=np.radians(15),
            hour=20,
            kp_index=9.0,
        )

        s4_min = scintillation_index(
            magnetic_latitude=np.radians(50),
            hour=6,
            kp_index=0.0,
        )

        assert 0 <= s4_max <= 1
        assert 0 <= s4_min <= 1

    def test_scintillation_array_input(self):
        """Test scintillation with array inputs."""
        mag_lats = np.radians([10, 45, 70])
        hours = np.array([20, 12, 20])

        s4 = scintillation_index(mag_lats, hours, kp_index=3.0)

        assert len(s4) == 3
        assert np.all(s4 >= 0)
        assert np.all(s4 <= 1)


# =============================================================================
# Tests for IonosphereState NamedTuple
# =============================================================================


class TestIonosphereState:
    """Tests for IonosphereState type."""

    def test_ionosphere_state_creation(self):
        """Test creating IonosphereState."""
        state = IonosphereState(
            tec=20.0,
            delay_l1=3.5,
            delay_l2=5.8,
            f_peak=8.0,
            h_peak=300.0,
        )

        assert state.tec == 20.0
        assert state.delay_l1 == 3.5
        assert state.delay_l2 == 5.8
        assert state.f_peak == 8.0
        assert state.h_peak == 300.0

    def test_ionosphere_state_with_arrays(self):
        """Test IonosphereState with array values."""
        tec = np.array([10.0, 20.0, 30.0])
        delay_l1 = np.array([1.75, 3.5, 5.25])
        delay_l2 = np.array([2.9, 5.8, 8.7])
        f_peak = np.array([6.0, 8.0, 10.0])
        h_peak = np.array([280.0, 300.0, 320.0])

        state = IonosphereState(
            tec=tec,
            delay_l1=delay_l1,
            delay_l2=delay_l2,
            f_peak=f_peak,
            h_peak=h_peak,
        )

        assert len(state.tec) == 3
        np.testing.assert_array_equal(state.tec, tec)


# =============================================================================
# Integration tests
# =============================================================================


class TestIonosphereIntegration:
    """Integration tests for ionosphere module."""

    def test_tec_delay_roundtrip(self):
        """Test TEC -> delay -> approximate TEC roundtrip."""
        original_tec = 25.0

        # Compute delays
        delay_l1 = ionospheric_delay_from_tec(original_tec, F_L1)
        delay_l2 = ionospheric_delay_from_tec(original_tec, F_L2)

        # Create fake pseudoranges (arbitrary base range)
        base_range = 22000000.0
        p_l1 = base_range + delay_l1
        p_l2 = base_range + delay_l2

        # Recover TEC from pseudorange difference
        recovered_tec = dual_frequency_tec(p_l1, p_l2)

        # Should match original (within floating point tolerance)
        assert recovered_tec == pytest.approx(original_tec, rel=1e-6)

    def test_iri_klobuchar_comparison(self):
        """Test that IRI and Klobuchar give comparable results."""
        # Same location and time
        lat = np.radians(40)
        lon = np.radians(-105)
        hour = 12

        # Get IRI prediction
        iri_state = simple_iri(lat, lon, 300e3, hour)

        # Get Klobuchar prediction (need elevation/azimuth for zenith)
        klobuchar_delay_val = klobuchar_delay(
            lat, lon, np.radians(90), 0, hour * 3600  # Zenith
        )

        # Both should be same order of magnitude
        assert 0.1 < klobuchar_delay_val / iri_state.delay_l1 < 10
