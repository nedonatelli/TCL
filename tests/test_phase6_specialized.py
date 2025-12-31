"""Tests for Phase 6: Specialized Domains (astronomical, navigation, atmosphere)."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.astronomical import (
    cal_to_jd,
    get_leap_seconds,
    gmst,
    gps_week_seconds,
    jd_to_cal,
    jd_to_mjd,
    jd_to_unix,
    mjd_to_jd,
    tai_to_tt,
    unix_to_jd,
    utc_to_tai,
)
from pytcl.atmosphere import (
    P0,
    T0,
    isa_atmosphere,
    mach_number,
    us_standard_atmosphere_1976,
)
from pytcl.navigation import (
    WGS84,
    direct_geodetic,
    ecef_to_enu,
    ecef_to_geodetic,
    enu_to_ecef,
    geodetic_to_ecef,
    haversine_distance,
    inverse_geodetic,
)


class TestCalendarJulianDate:
    """Tests for calendar to Julian Date conversions."""

    def test_j2000_epoch(self):
        """J2000.0 is 2000-01-01 12:00:00 TT = JD 2451545.0."""
        jd = cal_to_jd(2000, 1, 1, 12, 0, 0)
        assert_allclose(jd, 2451545.0, rtol=1e-10)

    def test_unix_epoch(self):
        """Unix epoch is 1970-01-01 00:00:00 UTC = JD 2440587.5."""
        jd = cal_to_jd(1970, 1, 1, 0, 0, 0)
        assert_allclose(jd, 2440587.5, rtol=1e-10)

    def test_roundtrip(self):
        """Test calendar -> JD -> calendar roundtrip."""
        year, month, day = 2024, 6, 15
        hour, minute, second = 14, 30, 45.5

        jd = cal_to_jd(year, month, day, hour, minute, second)
        y2, m2, d2, h2, min2, sec2 = jd_to_cal(jd)

        assert y2 == year
        assert m2 == month
        assert d2 == day
        assert h2 == hour
        assert min2 == minute
        assert_allclose(sec2, second, atol=1e-6)


class TestMJD:
    """Tests for Modified Julian Date conversions."""

    def test_mjd_offset(self):
        """MJD = JD - 2400000.5."""
        jd = 2451545.0
        mjd = jd_to_mjd(jd)
        assert_allclose(mjd, 51544.5, rtol=1e-10)

        jd_back = mjd_to_jd(mjd)
        assert_allclose(jd_back, jd, rtol=1e-10)


class TestUnixTime:
    """Tests for Unix time conversions."""

    def test_unix_epoch(self):
        """Unix time 0 should be JD 2440587.5."""
        jd = unix_to_jd(0.0)
        assert_allclose(jd, 2440587.5, rtol=1e-10)

    def test_roundtrip(self):
        """Test Unix time roundtrip."""
        unix_time = 1700000000.0  # Some time in 2023
        jd = unix_to_jd(unix_time)
        unix_back = jd_to_unix(jd)
        assert_allclose(unix_back, unix_time, rtol=1e-10)


class TestTimeScales:
    """Tests for time scale conversions."""

    def test_leap_seconds(self):
        """Test leap second lookup."""
        # Before leap seconds
        assert get_leap_seconds(1970, 1, 1) == 0

        # After several leap seconds
        assert get_leap_seconds(1980, 1, 6) == 19  # GPS epoch
        assert get_leap_seconds(2020, 1, 1) == 37

    def test_utc_to_tai(self):
        """Test UTC to TAI conversion."""
        # 2020-01-01 00:00:00 UTC should have 37 leap seconds
        jd_tai = utc_to_tai(2020, 1, 1, 0, 0, 0)
        jd_utc = cal_to_jd(2020, 1, 1, 0, 0, 0)

        # TAI should be 37 seconds ahead
        diff_seconds = (jd_tai - jd_utc) * 86400
        assert_allclose(diff_seconds, 37.0, atol=0.1)

    def test_tai_to_tt(self):
        """TT = TAI + 32.184 seconds."""
        jd_tai = 2451545.0
        jd_tt = tai_to_tt(jd_tai)

        diff_seconds = (jd_tt - jd_tai) * 86400
        assert_allclose(diff_seconds, 32.184, atol=1e-3)


class TestGPSTime:
    """Tests for GPS time conversions."""

    def test_gps_week_at_epoch(self):
        """GPS week 0, second 0 should be at GPS epoch."""
        from pytcl.astronomical.time_systems import JD_GPS_EPOCH

        week, seconds = gps_week_seconds(JD_GPS_EPOCH)
        assert week == 0
        assert_allclose(seconds, 0.0, atol=1e-6)

    def test_gps_week_calculation(self):
        """Test GPS week calculation."""
        from pytcl.astronomical.time_systems import JD_GPS_EPOCH

        # One week after GPS epoch
        jd_gps = JD_GPS_EPOCH + 7
        week, seconds = gps_week_seconds(jd_gps)
        assert week == 1
        assert_allclose(seconds, 0.0, atol=1e-6)


class TestSiderealTime:
    """Tests for sidereal time calculations."""

    def test_gmst_reasonable(self):
        """GMST should be in [0, 2*pi]."""
        jd = cal_to_jd(2024, 1, 1, 0, 0, 0)
        theta = gmst(jd)
        assert 0 <= theta < 2 * np.pi


class TestGeodeticECEF:
    """Tests for geodetic to ECEF conversions."""

    def test_equator_prime_meridian(self):
        """Point at equator and prime meridian."""
        lat, lon, alt = 0.0, 0.0, 0.0
        x, y, z = geodetic_to_ecef(lat, lon, alt)

        # Should be at (a, 0, 0) approximately
        assert_allclose(x, WGS84.a, rtol=1e-10)
        assert_allclose(y, 0.0, atol=1e-6)
        assert_allclose(z, 0.0, atol=1e-6)

    def test_north_pole(self):
        """Point at north pole."""
        lat, lon, alt = np.pi / 2, 0.0, 0.0
        x, y, z = geodetic_to_ecef(lat, lon, alt)

        # Should be at (0, 0, b) approximately
        assert_allclose(x, 0.0, atol=1e-6)
        assert_allclose(y, 0.0, atol=1e-6)
        assert_allclose(z, WGS84.b, rtol=1e-10)

    def test_roundtrip(self):
        """Test geodetic -> ECEF -> geodetic roundtrip."""
        lat = np.radians(40.0)
        lon = np.radians(-75.0)
        alt = 1000.0

        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)

        assert_allclose(lat2, lat, rtol=1e-10)
        assert_allclose(lon2, lon, rtol=1e-10)
        assert_allclose(alt2, alt, rtol=1e-6)

    def test_array_input(self):
        """Test with array inputs."""
        lat = np.radians([0, 30, 60, 90])
        lon = np.radians([0, 45, 90, 0])
        alt = np.array([0, 1000, 5000, 10000])

        x, y, z = geodetic_to_ecef(lat, lon, alt)
        assert x.shape == (4,)
        assert y.shape == (4,)
        assert z.shape == (4,)


class TestENU:
    """Tests for ENU coordinate conversions."""

    def test_enu_at_origin(self):
        """Reference point should map to ENU (0, 0, 0)."""
        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-75.0)
        alt_ref = 100.0

        x, y, z = geodetic_to_ecef(lat_ref, lon_ref, alt_ref)
        e, n, u = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)

        assert_allclose(e, 0.0, atol=1e-6)
        assert_allclose(n, 0.0, atol=1e-6)
        assert_allclose(u, 0.0, atol=1e-6)

    def test_enu_roundtrip(self):
        """Test ENU -> ECEF -> ENU roundtrip."""
        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-75.0)
        alt_ref = 0.0

        e, n, u = 1000.0, 2000.0, 500.0

        x, y, z = enu_to_ecef(e, n, u, lat_ref, lon_ref, alt_ref)
        e2, n2, u2 = ecef_to_enu(x, y, z, lat_ref, lon_ref, alt_ref)

        assert_allclose(e2, e, rtol=1e-10)
        assert_allclose(n2, n, rtol=1e-10)
        assert_allclose(u2, u, rtol=1e-10)


class TestGeodeticProblems:
    """Tests for direct and inverse geodetic problems."""

    def test_direct_geodetic(self):
        """Test direct geodetic problem."""
        lat1 = np.radians(40.0)
        lon1 = np.radians(-75.0)
        azimuth = np.radians(45.0)  # Northeast
        distance = 10000.0  # 10 km

        lat2, lon2, az2 = direct_geodetic(lat1, lon1, azimuth, distance)

        # Result should be northeast of start
        assert lat2 > lat1
        assert lon2 > lon1

    def test_inverse_geodetic(self):
        """Test inverse geodetic problem."""
        lat1 = np.radians(40.0)
        lon1 = np.radians(-75.0)
        lat2 = np.radians(41.0)
        lon2 = np.radians(-74.0)

        distance, az1, az2 = inverse_geodetic(lat1, lon1, lat2, lon2)

        # Distance should be positive
        assert distance > 0

        # Azimuth should be in [0, 2*pi] or [-pi, pi]
        assert -np.pi <= az1 <= 2 * np.pi

    def test_direct_inverse_roundtrip(self):
        """Direct then inverse should recover distance and azimuth."""
        lat1 = np.radians(45.0)
        lon1 = np.radians(0.0)
        azimuth = np.radians(30.0)
        distance = 50000.0

        lat2, lon2, _ = direct_geodetic(lat1, lon1, azimuth, distance)
        dist_back, az_back, _ = inverse_geodetic(lat1, lon1, lat2, lon2)

        assert_allclose(dist_back, distance, rtol=1e-6)
        assert_allclose(az_back, azimuth, rtol=1e-6)

    def test_haversine(self):
        """Test haversine distance."""
        lat1 = np.radians(0.0)
        lon1 = np.radians(0.0)
        lat2 = np.radians(0.0)
        lon2 = np.radians(1.0)

        # At equator, 1 degree longitude is about 111 km
        dist = haversine_distance(lat1, lon1, lat2, lon2)
        assert_allclose(dist, 111195, rtol=0.01)


class TestAtmosphereModels:
    """Tests for atmospheric models."""

    def test_sea_level_conditions(self):
        """Test sea level conditions match standards."""
        state = us_standard_atmosphere_1976(0)

        assert_allclose(state.temperature, T0, rtol=1e-10)
        assert_allclose(state.pressure, P0, rtol=1e-10)
        assert_allclose(state.density, 1.225, rtol=0.001)

    def test_temperature_decreases_troposphere(self):
        """Temperature should decrease in troposphere."""
        state_0 = us_standard_atmosphere_1976(0)
        state_5000 = us_standard_atmosphere_1976(5000)
        state_10000 = us_standard_atmosphere_1976(10000)

        assert state_5000.temperature < state_0.temperature
        assert state_10000.temperature < state_5000.temperature

    def test_tropopause(self):
        """Temperature should be constant in tropopause."""
        state_11km = us_standard_atmosphere_1976(11000)
        state_15km = us_standard_atmosphere_1976(15000)

        # Both should be around 216.65 K
        assert_allclose(state_11km.temperature, 216.65, rtol=0.01)
        assert_allclose(state_15km.temperature, 216.65, rtol=0.01)

    def test_pressure_decreases(self):
        """Pressure should always decrease with altitude."""
        altitudes = [0, 5000, 10000, 20000, 30000]
        states = [us_standard_atmosphere_1976(h) for h in altitudes]

        for i in range(len(states) - 1):
            assert states[i + 1].pressure < states[i].pressure

    def test_speed_of_sound(self):
        """Test speed of sound calculation."""
        state = us_standard_atmosphere_1976(0)
        # Sea level speed of sound is about 340 m/s
        assert_allclose(state.speed_of_sound, 340.3, rtol=0.01)

    def test_array_input(self):
        """Test with array altitude input."""
        altitudes = np.array([0, 5000, 10000, 15000])
        state = us_standard_atmosphere_1976(altitudes)

        assert state.temperature.shape == (4,)
        assert state.pressure.shape == (4,)


class TestISAAtmosphere:
    """Tests for ISA atmosphere model."""

    def test_matches_us76_at_standard(self):
        """ISA should match US76 with no temperature offset."""
        for h in [0, 5000, 10000]:
            isa = isa_atmosphere(h, temperature_offset=0)
            us76 = us_standard_atmosphere_1976(h)

            assert_allclose(isa.temperature, us76.temperature, rtol=0.01)
            assert_allclose(isa.pressure, us76.pressure, rtol=0.01)

    def test_hot_day_warmer(self):
        """Hot day should have higher temperature."""
        standard = isa_atmosphere(5000, temperature_offset=0)
        hot_day = isa_atmosphere(5000, temperature_offset=15)

        assert hot_day.temperature > standard.temperature


class TestMachNumber:
    """Tests for Mach number calculations."""

    def test_mach_at_sea_level(self):
        """Test Mach number at sea level."""
        velocity = 340.3  # Speed of sound at sea level
        mach = mach_number(velocity, 0)
        assert_allclose(mach, 1.0, rtol=0.01)

    def test_mach_subsonic(self):
        """Subsonic velocity should give Mach < 1."""
        velocity = 200.0  # m/s
        mach = mach_number(velocity, 0)
        assert mach < 1.0

    def test_mach_supersonic(self):
        """Supersonic velocity should give Mach > 1."""
        velocity = 500.0  # m/s
        mach = mach_number(velocity, 0)
        assert mach > 1.0


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_aircraft_tracking_scenario(self):
        """Test a typical aircraft tracking scenario."""
        # Aircraft position
        lat = np.radians(40.0)
        lon = np.radians(-75.0)
        alt = 10000.0  # 10 km altitude

        # Convert to ECEF
        x, y, z = geodetic_to_ecef(lat, lon, alt)

        # Get atmosphere at altitude
        atm = us_standard_atmosphere_1976(alt)

        # Aircraft flying at Mach 0.8
        true_airspeed = 0.8 * atm.speed_of_sound

        # Verify reasonable values
        assert true_airspeed > 200  # m/s
        assert true_airspeed < 300  # m/s
        assert atm.temperature < T0  # Colder at altitude
        assert atm.pressure < P0  # Lower pressure at altitude
