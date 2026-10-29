"""
Tests for ionospheric models.

Klobuchar values are checked against a step-by-step evaluation of the
IS-GPS-200 algorithm (section 20.3.3.5.2.5, "Ionospheric Model") written out in
``_is_gps_200_klobuchar`` below, and against closed-form anchors.

Tests cover:
- Klobuchar ionospheric delay model
- Dual-frequency TEC computation
- Ionospheric delay from TEC
- Simplified IRI model
- Magnetic latitude calculation
- Scintillation index estimation
"""

import math

import numpy as np
import pytest

from pytcl.atmosphere.ionosphere import (
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


def _is_gps_200_klobuchar(lat_deg, lon_deg, el_deg, az_deg, gps_time, alpha, beta):
    """Klobuchar L1 delay in metres, evaluated step by step from IS-GPS-200.

    IS-GPS-200 section 20.3.3.5.2.5: all angles in semi-circles (1 semi-circle
    = 180 degrees); the trigonometric arguments are converted back to radians.
    """
    phi_u = lat_deg / 180.0  # user geodetic latitude, semi-circles
    lam_u = lon_deg / 180.0  # user geodetic longitude, semi-circles
    elev = el_deg / 180.0  # satellite elevation E, semi-circles
    azim = math.radians(az_deg)  # satellite azimuth A, radians

    # Earth's central angle between user and ionospheric pierce point
    psi = 0.0137 / (elev + 0.11) - 0.022
    # subionospheric latitude, limited to +/-0.416 semi-circles
    phi_i = min(max(phi_u + psi * math.cos(azim), -0.416), 0.416)
    # subionospheric longitude
    lam_i = lam_u + psi * math.sin(azim) / math.cos(phi_i * math.pi)
    # geomagnetic latitude of the pierce point
    phi_m = phi_i + 0.064 * math.cos((lam_i - 1.617) * math.pi)
    # local time at the pierce point, seconds, in [0, 86400)
    t_local = (4.32e4 * lam_i + gps_time) % 86400.0
    # obliquity (slant) factor
    slant = 1.0 + 16.0 * (0.53 - elev) ** 3
    # amplitude and period of the cosine, from cubic polynomials in phi_m
    amp = max(sum(alpha[n] * phi_m**n for n in range(4)), 0.0)
    per = max(sum(beta[n] * phi_m**n for n in range(4)), 72000.0)
    # phase, radians
    x = 2.0 * math.pi * (t_local - 50400.0) / per
    if abs(x) < 1.57:
        t_iono = slant * (5.0e-9 + amp * (1.0 - x**2 / 2.0 + x**4 / 24.0))
    else:
        t_iono = slant * 5.0e-9
    return t_iono * 299792458.0


# pytcl's own default coefficients, not broadcast ICD values: the tests check the
# algorithm, not any particular satellite message.
_DEFAULT_ALPHA = [3.82e-8, 1.49e-8, -5.96e-8, -5.96e-8]
_DEFAULT_BETA = [1.43e5, 0.0, -3.28e5, 1.13e5]


# =============================================================================
# Tests for constants
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_speed_of_light(self):
        """Test speed of light constant."""
        assert SPEED_OF_LIGHT == 299792458.0

    def test_gps_frequencies(self):
        """Test GPS frequency constants."""
        assert F_L1 == 1575.42e6
        assert F_L2 == 1227.60e6


# =============================================================================
# Tests for Klobuchar coefficients
# =============================================================================


class TestKlobucharCoefficients:
    """Tests for Klobuchar coefficient handling."""

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
    """Klobuchar delay against the IS-GPS-200 step-by-step evaluation.

    No external numeric worked example exists for this model, so the oracle is
    a transcription of IS-GPS-200 section 20.3.3.5.2.5 (``_is_gps_200_klobuchar``).
    The phase x = 2 pi (t - 50400) / PER selects the branch: |x| < 1.57 is the
    cosine (daytime) branch, otherwise the delay is the constant night floor
    F * 5 ns. Each case below states its x.
    """

    def test_evaluator_pins(self):
        """Evaluator values computed independently in mpmath at 30 digits."""
        args = (_DEFAULT_ALPHA, _DEFAULT_BETA)
        night = _is_gps_200_klobuchar(40, -105, 45, 180, 43200, *args)  # x = -1.653
        peak = _is_gps_200_klobuchar(0, 0, 90, 0, 50400, *args)  # x = 0
        day = _is_gps_200_klobuchar(40, 0, 45, 180, 46800, *args)  # x = -0.178
        day2 = _is_gps_200_klobuchar(40, -105, 45, 180, 64800, *args)  # x = -0.551
        assert night == pytest.approx(2.02544581304128, rel=1e-13, abs=0)
        assert peak == pytest.approx(13.051388706360187, rel=1e-13, abs=0)
        assert day == pytest.approx(17.083060349781694, rel=1e-13, abs=0)
        assert day2 == pytest.approx(14.814267584787064, rel=1e-13, abs=0)

    @pytest.mark.parametrize(
        ("lat", "lon", "el", "az", "t"),
        [
            # night branch, |x| >= 1.57: delay is F * 5 ns
            (40, -105, 45, 180, 43200),  # x = -1.653
            (40, -105, 80, 180, 43200),  # x = -1.681
            (40, -105, 15, 180, 43200),  # x = -1.595, low elevation, large F
            (40, 0, 45, 180, 18000),  # x = -1.601
            (40, 0, 45, 180, 18550),  # x = -1.574, just outside 1.57
            # cosine branch, |x| < 1.57
            (40, 0, 45, 180, 18700),  # x = -1.567, just inside 1.57
            (40, 0, 30, 90, 18000),  # x = -1.540
            (40, 0, 45, 180, 46800),  # x = -0.178
            (40, -105, 45, 180, 64800),  # x = -0.551
            (40, 0, 45, 180, 50400),  # x = 0, daytime peak
            (0, 0, 90, 0, 50400),  # x = 0, zenith
            (-35, 150, 60, 270, 3600),  # x = -0.583, southern hemisphere
            (40, -150, 45, 180, 20000),  # local time wraps below 0 (mod 86400)
            # pierce point clipped to +/-0.416 semi-circles
            (80, 0, 45, 0, 43200),  # +0.416, x = -0.506
            (-80, 20, 20, 180, 30000),  # -0.416, x = -1.249
        ],
    )
    def test_default_coefficients(self, lat, lon, el, az, t):
        got = klobuchar_delay(
            np.radians(lat), np.radians(lon), np.radians(el), np.radians(az), t
        )
        want = _is_gps_200_klobuchar(lat, lon, el, az, t, _DEFAULT_ALPHA, _DEFAULT_BETA)
        assert float(got) == pytest.approx(want, rel=1e-12, abs=0)

    def test_branch_switch_at_x_1_57(self):
        """Just outside |x| = 1.57 the delay is the night floor; just inside it is not."""
        floor = _is_gps_200_klobuchar(40, 0, 45, 180, 18550, [0.0] * 4, _DEFAULT_BETA)
        outside = float(
            klobuchar_delay(np.radians(40), 0.0, np.radians(45), np.radians(180), 18550)
        )
        inside = float(
            klobuchar_delay(np.radians(40), 0.0, np.radians(45), np.radians(180), 18700)
        )
        assert outside == pytest.approx(floor, rel=1e-13, abs=0)
        assert inside > outside * 1.1

    def test_custom_coefficients(self):
        alpha = [5e-8, 1e-8, -5e-8, 0.0]
        beta = [1.5e5, 0.0, -3e5, 1e5]
        coef = KlobucharCoefficients(alpha=np.array(alpha), beta=np.array(beta))
        got = klobuchar_delay(
            np.radians(40),
            np.radians(-105),
            np.radians(45),
            np.radians(180),
            43200,
            coefficients=coef,
        )
        want = _is_gps_200_klobuchar(40, -105, 45, 180, 43200, alpha, beta)
        assert float(got) == pytest.approx(want, rel=1e-12, abs=0)

    def test_negative_amplitude_is_floored_at_zero(self):
        """AMP < 0 is set to 0: the delay is then the 5 ns night-time floor."""
        alpha = [-1e-8, 0.0, 0.0, 0.0]
        coef = KlobucharCoefficients(
            alpha=np.array(alpha), beta=np.array(_DEFAULT_BETA)
        )
        got = klobuchar_delay(0.0, 0.0, np.radians(90), 0.0, 50400, coefficients=coef)
        # E = 0.5 semi-circles: F = 1 + 16 * 0.03^3
        slant = 1.0 + 16.0 * 0.03**3
        assert float(got) == pytest.approx(
            slant * 5.0e-9 * 299792458.0, rel=1e-13, abs=0
        )

    def test_short_period_is_floored_at_72000_s(self):
        """PER < 72000 is set to 72000: a 50000 s period gives the same delay."""
        alpha = _DEFAULT_ALPHA
        short = KlobucharCoefficients(
            alpha=np.array(alpha), beta=np.array([50000.0, 0.0, 0.0, 0.0])
        )
        floor = KlobucharCoefficients(
            alpha=np.array(alpha), beta=np.array([72000.0, 0.0, 0.0, 0.0])
        )
        args = (np.radians(40), 0.0, np.radians(45), np.radians(180), 40000)
        assert float(klobuchar_delay(*args, coefficients=short)) == float(
            klobuchar_delay(*args, coefficients=floor)
        )

    def test_zenith_at_local_peak_is_slant_times_five_ns_plus_amplitude(self):
        """lam_i = 0, t = 50400 s => x = 0, so T = F (5 ns + AMP).

        E = 0.5 semi-circles gives psi = 0.0137/0.61 - 0.022 and F = 1.000432;
        with A = 0 the pierce point is at phi_i = psi, lam_i = 0.
        """
        psi = 0.0137 / 0.61 - 0.022
        phi_m = psi + 0.064 * math.cos(-1.617 * math.pi)
        amp = sum(a * phi_m**n for n, a in enumerate(_DEFAULT_ALPHA))
        slant = 1.0 + 16.0 * 0.03**3
        expected = slant * (5.0e-9 + amp) * 299792458.0
        got = klobuchar_delay(0.0, 0.0, np.radians(90), 0.0, 50400)
        assert float(got) == pytest.approx(expected, rel=1e-12, abs=0)

    def test_array_input_matches_scalar_evaluation(self):
        lats = np.array([30.0, 40.0, 50.0])
        got = klobuchar_delay(
            np.radians(lats),
            np.radians([-105.0, -105.0, -105.0]),
            np.radians([45.0, 45.0, 45.0]),
            np.radians([180.0, 180.0, 180.0]),
            np.array([43200.0, 43200.0, 43200.0]),
        )
        want = [
            _is_gps_200_klobuchar(
                lat, -105, 45, 180, 43200, _DEFAULT_ALPHA, _DEFAULT_BETA
            )
            for lat in lats
        ]
        assert got.shape == (3,)
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


# =============================================================================
# Tests for dual_frequency_tec
# =============================================================================


class TestDualFrequencyTec:
    """Dual-frequency TEC: delay_i = 40.3 TEC / f_i^2 (TEC in electrons/m^2)."""

    def test_one_tecu_l2_minus_l1_is_about_0_105_m(self):
        """Textbook: 1 TECU delays L1 by 0.1624 m and L2 by 0.2674 m."""
        tec = dual_frequency_tec(0.0, 0.2674 - 0.1624)
        assert tec == pytest.approx(1.0, rel=2e-3, abs=0)

    def test_matches_inverse_of_dispersion_formula(self):
        """P2 - P1 = 40.3 * TEC * (1/f2^2 - 1/f1^2), solved for TEC."""
        p_l1 = 22000000.0
        p_l2 = 22000002.5
        expected = 2.5 / (40.3 * (1 / 1227.60e6**2 - 1 / 1575.42e6**2)) / 1e16
        assert dual_frequency_tec(p_l1, p_l2) == pytest.approx(
            expected, rel=1e-12, abs=0
        )

    def test_zero_difference(self):
        assert dual_frequency_tec(22000000.0, 22000000.0) == 0.0

    def test_negative_difference_gives_negative_tec(self):
        assert dual_frequency_tec(22000002.5, 22000000.0) == pytest.approx(
            -dual_frequency_tec(22000000.0, 22000002.5), rel=1e-12, abs=0
        )

    def test_array_input(self):
        p_l1 = np.array([22000000.0, 22500000.0, 23000000.0])
        p_l2 = np.array([22000002.5, 22500003.0, 23000002.0])
        k = 40.3 * (1 / 1227.60e6**2 - 1 / 1575.42e6**2) * 1e16
        np.testing.assert_allclose(
            dual_frequency_tec(p_l1, p_l2),
            np.array([2.5, 3.0, 2.0]) / k,
            rtol=1e-9,
            atol=0,
        )


# =============================================================================
# Tests for ionospheric_delay_from_tec
# =============================================================================


class TestIonosphericDelayFromTec:
    """delay = 40.3 * TEC * 1e16 / f^2."""

    def test_20_tecu_on_l1(self):
        expected = 40.3 * 20.0e16 / 1575.42e6**2
        assert ionospheric_delay_from_tec(20.0) == pytest.approx(
            expected, rel=1e-13, abs=0
        )

    def test_one_tecu_on_l1_is_0_1624_m(self):
        """Textbook value 0.162 m / TECU at L1."""
        assert ionospheric_delay_from_tec(1.0) == pytest.approx(0.1624, rel=1e-3, abs=0)

    def test_one_tecu_on_l2_is_0_2674_m(self):
        assert ionospheric_delay_from_tec(1.0, frequency=F_L2) == pytest.approx(
            0.2674, rel=1e-3, abs=0
        )

    def test_delay_proportional_to_tec(self):
        assert ionospheric_delay_from_tec(20.0) == pytest.approx(
            2 * ionospheric_delay_from_tec(10.0), rel=1e-14, abs=0
        )

    def test_frequency_dependence_is_inverse_square(self):
        ratio = ionospheric_delay_from_tec(20.0, F_L2) / ionospheric_delay_from_tec(
            20.0, F_L1
        )
        assert ratio == pytest.approx((1575.42 / 1227.60) ** 2, rel=1e-13, abs=0)

    def test_array_input(self):
        np.testing.assert_allclose(
            ionospheric_delay_from_tec(np.array([10.0, 20.0, 30.0])),
            np.array([10.0, 20.0, 30.0]) * 40.3e16 / 1575.42e6**2,
            rtol=1e-13,
            atol=0,
        )


# =============================================================================
# Tests for simple_iri
# =============================================================================


class TestSimpleIri:
    """Tests for simplified IRI model."""

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

    def test_simple_iri_warns_that_altitude_and_longitude_are_ignored(self):
        """simple_iri accepts altitude and longitude but its result does
        not depend on either -- see test_simple_iri_tec_is_altitude_and_
        longitude_independent_as_documented below. Callers get a warning
        naming both parameters rather than a silently wrong answer."""
        with pytest.warns(UserWarning, match="altitude and longitude are ignored"):
            simple_iri(
                latitude=np.radians(40),
                longitude=np.radians(-105),
                altitude=300e3,
                hour=12,
            )

    def test_simple_iri_tec_is_altitude_and_longitude_independent_as_documented(
        self,
    ):
        """Pins a known limitation (task 2.9): simple_iri is a
        latitude/local-time/solar-flux/season fit with no altitude or
        longitude dependence at all. A future fix that makes it actually
        altitude- or longitude-sensitive must update this test."""
        with pytest.warns(UserWarning):
            low = simple_iri(
                latitude=np.radians(40),
                longitude=np.radians(-105),
                altitude=100e3,
                hour=12,
            )
        with pytest.warns(UserWarning):
            high = simple_iri(
                latitude=np.radians(40),
                longitude=np.radians(-105),
                altitude=1_000e3,
                hour=12,
            )
        with pytest.warns(UserWarning):
            other_lon = simple_iri(
                latitude=np.radians(40),
                longitude=np.radians(170),
                altitude=300e3,
                hour=12,
            )

        assert low.tec == pytest.approx(high.tec)
        assert low.tec == pytest.approx(other_lon.tec)


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
