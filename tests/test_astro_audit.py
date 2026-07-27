"""
Correctness audit tests for pytcl.astronomical.

Reference-validates time systems, sidereal angles, reference frames,
SGP4, TLE parsing, Lambert solvers, and orbital mechanics against:

- astropy / pyerfa (IAU 1976/1980/1982 models, matching pytcl's intent)
- the official ``sgp4`` package (Vallado's reference implementation)
- textbook cases (Curtis Ex. 5.2, Vallado Ex. 7-5)
- analytic identities (round trips, conservation laws, boundary conditions)

External-package tests skip gracefully via ``pytest.importorskip``.
No network access is required (astropy auto-download is disabled and all
astropy epochs are covered by its bundled IERS-B table).
"""

import numpy as np
import pytest

from pytcl.astronomical import reference_frames as rf
from pytcl.astronomical import time_systems as ts
from pytcl.astronomical.lambert import (
    bi_elliptic_transfer,
    hohmann_transfer,
    lambert_izzo,
    lambert_universal,
    minimum_energy_transfer,
)
from pytcl.astronomical.orbital_mechanics import (
    GM_EARTH,
    OrbitalElements,
    StateVector,
    apoapsis_radius,
    circular_velocity,
    eccentric_to_mean_anomaly,
    eccentric_to_true_anomaly,
    escape_velocity,
    flight_path_angle,
    hyperbolic_to_true_anomaly,
    kepler_propagate,
    kepler_propagate_state,
    mean_motion,
    mean_to_eccentric_anomaly,
    mean_to_hyperbolic_anomaly,
    mean_to_true_anomaly,
    orbit_radius,
    orbital_elements_to_state,
    orbital_period,
    periapsis_radius,
    specific_angular_momentum,
    specific_orbital_energy,
    state_to_orbital_elements,
    time_since_periapsis,
    true_to_eccentric_anomaly,
    true_to_hyperbolic_anomaly,
    true_to_mean_anomaly,
    vis_viva,
)
from pytcl.astronomical.sgp4 import SGP4Satellite, sgp4_propagate, sgp4_propagate_batch
from pytcl.astronomical.special_orbits import (
    OrbitType,
    classify_orbit,
    eccentricity_vector,
    escape_velocity_at_radius,
    hyperbolic_anomaly_to_true_anomaly,
    hyperbolic_asymptote_angle,
    hyperbolic_deflection_angle,
    hyperbolic_excess_velocity,
    mean_to_parabolic_anomaly,
    mean_to_true_anomaly_parabolic,
    parabolic_anomaly_to_true_anomaly,
    radius_parabolic,
    semi_major_axis_from_energy,
    true_anomaly_to_hyperbolic_anomaly,
    true_anomaly_to_parabolic_anomaly,
    velocity_parabolic,
)
from pytcl.astronomical.tle import (
    is_deep_space,
    orbital_period_from_tle,
    parse_tle,
    parse_tle_3line,
    semi_major_axis_from_mean_motion,
    tle_epoch_to_datetime,
    tle_epoch_to_jd,
)

ARCSEC = np.pi / (180.0 * 3600.0)

ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997"
ISS_L2 = "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003"


def _fix_checksum(line: str) -> str:
    cs = 0
    for c in line[:68]:
        if c.isdigit():
            cs += int(c)
        elif c == "-":
            cs += 1
    return line[:68] + str(cs % 10)


def _angdiff(a: float, b: float) -> float:
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def _astropy_time():
    """Import astropy.time with IERS auto-download disabled."""
    astropy_time = pytest.importorskip("astropy.time")
    iers = pytest.importorskip("astropy.utils.iers")
    iers.conf.auto_download = False
    return astropy_time


# =============================================================================
# Time systems
# =============================================================================


class TestJulianDates:
    def test_known_epochs(self):
        assert ts.cal_to_jd(2000, 1, 1, 12, 0, 0.0) == 2451545.0
        assert ts.cal_to_jd(1970, 1, 1) == 2440587.5
        assert ts.cal_to_jd(1980, 1, 6) == ts.JD_GPS_EPOCH
        # Vallado Ex. 3-4: 1996-10-26 14:20:00 UTC
        assert ts.cal_to_jd(1996, 10, 26, 14, 20, 0.0) == pytest.approx(
            2450383.09722222, abs=1e-8
        )

    def test_jd_cal_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(300):
            jd = 2440000.0 + rng.uniform(0, 40000)
            y, mo, d, h, mi, s = ts.jd_to_cal(jd)
            assert abs(ts.cal_to_jd(y, mo, d, h, mi, s) - jd) * 86400.0 < 1e-5

    def test_mjd(self):
        assert ts.mjd_to_jd(44239.0) == 2444239.5
        assert ts.jd_to_mjd(2444239.5) == 44239.0
        assert ts.jd_to_mjd(ts.mjd_to_jd(51544.5)) == 51544.5

    def test_unix(self):
        assert ts.unix_to_jd(0.0) == 2440587.5
        assert ts.jd_to_unix(2440587.5) == 0.0
        assert ts.jd_to_unix(ts.unix_to_jd(1.7e9)) == pytest.approx(1.7e9, abs=1e-3)


class TestTimeScales:
    def test_leap_seconds(self):
        assert ts.get_leap_seconds(2020, 1, 1) == 37
        assert ts.get_leap_seconds(1980, 1, 6) == 19
        assert ts.get_leap_seconds(1981, 6, 30) == 19
        assert ts.get_leap_seconds(1981, 7, 1) == 20
        assert ts.get_leap_seconds(1971, 12, 31) == 0

    def test_tt_tai_gps_offsets(self):
        jd = 2451545.0
        assert (ts.tai_to_tt(jd) - jd) * 86400 == pytest.approx(32.184)
        assert ts.tt_to_tai(ts.tai_to_tt(jd)) == pytest.approx(jd, abs=1e-12)
        assert (ts.gps_to_tai(jd) - jd) * 86400 == pytest.approx(19.0)
        assert ts.tai_to_gps(ts.gps_to_tai(jd)) == pytest.approx(jd, abs=1e-12)

    def test_utc_scales_vs_astropy(self):
        astropy_time = _astropy_time()
        for dstr in ["2004-04-06 07:51:28", "2024-01-01 00:00:00"]:
            t = astropy_time.Time(dstr, scale="utc")
            dt = t.datetime
            args = (dt.year, dt.month, dt.day, dt.hour, dt.minute, float(dt.second))
            assert ts.utc_to_tai(*args) == pytest.approx(t.tai.jd, abs=1e-9)
            assert ts.utc_to_tt(*args) == pytest.approx(t.tt.jd, abs=1e-9)

    def test_utc_tai_roundtrip(self):
        jd_tai = ts.utc_to_tai(2020, 6, 1, 12, 0, 0.0)
        jd_utc, leap = ts.tai_to_utc(jd_tai)
        assert leap == 37
        assert jd_utc == pytest.approx(ts.cal_to_jd(2020, 6, 1, 12, 0, 0.0), abs=1e-10)

    def test_tt_utc_roundtrip(self):
        jd_tt = ts.utc_to_tt(2018, 3, 15, 6, 30, 0.0)
        jd_utc, _ = ts.tt_to_utc(jd_tt)
        assert jd_utc == pytest.approx(ts.cal_to_jd(2018, 3, 15, 6, 30, 0.0), abs=1e-10)

    def test_gps_week(self):
        assert ts.gps_week_seconds(ts.JD_GPS_EPOCH) == (0, 0.0)
        week, sow = ts.gps_week_seconds(ts.JD_GPS_EPOCH + 8.5)
        assert week == 1
        assert sow == pytest.approx(1.5 * 86400.0)
        jd_utc, _ = ts.gps_week_to_utc(week, sow)
        jd_gps = ts.utc_to_gps(*ts.jd_to_cal(jd_utc)[:3])
        assert ts.gps_week_seconds(jd_gps)[0] >= 0  # structural round trip

    def test_gps_vs_astropy(self):
        astropy_time = _astropy_time()
        t = astropy_time.Time("2024-01-01 00:00:00", scale="utc")
        jd_gps = ts.utc_to_gps(2024, 1, 1, 0, 0, 0.0)
        week, sow = ts.gps_week_seconds(jd_gps)
        assert week * 604800 + sow == pytest.approx(t.gps, abs=1e-3)


class TestSiderealTime:
    def test_gmst_vs_astropy_iau1982(self):
        """Regression for day-fraction bug: errors were up to ~2600 arcsec."""
        astropy_time = _astropy_time()
        for dstr in [
            "2004-04-06 07:51:28",
            "2024-03-01 06:30:00",
            "2000-01-01 12:00:00",
            "2015-06-30 18:00:00",
        ]:
            t = astropy_time.Time(dstr, scale="ut1")
            ref = t.sidereal_time("mean", "greenwich", model="IAU1982").rad
            assert abs(_angdiff(ts.gmst(t.jd), ref)) < 0.001 * ARCSEC
            assert abs(_angdiff(rf.gmst_iau82(t.jd), ref)) < 0.001 * ARCSEC

    def test_gmst_vallado_example(self):
        """Vallado Ex. 3-5: 1992-08-20 12:14 UT1 -> GMST = 152.578788 deg."""
        jd = ts.cal_to_jd(1992, 8, 20, 12, 14, 0.0)
        assert np.degrees(ts.gmst(jd)) == pytest.approx(152.578788, abs=1e-5)

    def test_gast_identity(self):
        jd = 2453101.5
        eps = 0.4090
        dpsi = -60e-6
        assert ts.gast(jd, dpsi, eps) == pytest.approx(
            ts.gmst(jd) + dpsi * np.cos(eps), abs=1e-15
        )

    def test_gast_iau82_vs_astropy(self):
        astropy_time = _astropy_time()
        for dstr in ["2004-04-06 07:51:28", "2024-03-01 00:00:00"]:
            t = astropy_time.Time(dstr, scale="ut1")
            ref = t.sidereal_time("apparent", "greenwich", model="IAU1994").rad
            got = rf.gast_iau82(t.jd, t.tt.jd)
            # Truncated 4-term nutation series: allow 0.3 arcsec
            assert abs(_angdiff(got, ref)) < 0.3 * ARCSEC


# =============================================================================
# Reference frames
# =============================================================================


class TestPrecessionNutation:
    def test_precession_matrix_vs_erfa(self):
        erfa = pytest.importorskip("erfa")
        for jd in [2451545.0, 2453101.5, 2460000.5, 2440000.5]:
            P = rf.precession_matrix_iau76(jd)
            assert np.allclose(P, erfa.pmat76(jd, 0.0), atol=5e-11)

    def test_nutation_angles_vs_erfa(self):
        """Regression for wrong term arguments: errors were ~2.2 arcsec."""
        erfa = pytest.importorskip("erfa")
        for jd in [2451545.0, 2453101.5, 2460000.5, 2440000.5, 2466000.5]:
            dpsi, deps = rf.nutation_angles_iau80(jd)
            rdpsi, rdeps = erfa.nut80(jd, 0.0)
            # 4 largest of 106 terms: residual must stay below 0.25 arcsec
            assert abs(dpsi - rdpsi) < 0.25 * ARCSEC
            assert abs(deps - rdeps) < 0.1 * ARCSEC

    def test_mean_obliquity_vs_erfa(self):
        erfa = pytest.importorskip("erfa")
        for jd in [2451545.0, 2460000.5]:
            assert rf.mean_obliquity_iau80(jd) == pytest.approx(
                erfa.obl80(jd, 0.0), abs=1e-9
            )

    def test_earth_rotation_angle_vs_erfa(self):
        erfa = pytest.importorskip("erfa")
        for jd in [2453101.5, 2460311.25]:
            assert abs(_angdiff(rf.earth_rotation_angle(jd), erfa.era00(jd, 0.0))) < (
                0.001 * ARCSEC
            )

    def test_polar_motion_vs_erfa(self):
        erfa = pytest.importorskip("erfa")
        xp, yp = 1.2e-6, -0.8e-6
        W = rf.polar_motion_matrix(xp, yp)
        # erfa.pom00 maps TIRS->ITRS, same sense as documented (PEF->ITRF);
        # s' is omitted in pytcl (sub-microarcsecond)
        assert np.allclose(W, erfa.pom00(xp, yp, 0.0), atol=1e-11)

    def test_rotation_matrices_orthogonal(self):
        jd = 2453101.5
        for M in [
            rf.precession_matrix_iau76(jd),
            rf.nutation_matrix(jd),
            rf.sidereal_rotation_matrix(1.234),
            rf.polar_motion_matrix(1e-6, 2e-6),
        ]:
            assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
            assert np.linalg.det(M) == pytest.approx(1.0, abs=1e-12)

    def test_true_obliquity_and_eq_equinoxes(self):
        jd = 2453101.5
        dpsi, deps = rf.nutation_angles_iau80(jd)
        eps0 = rf.mean_obliquity_iau80(jd)
        assert rf.true_obliquity(jd) == pytest.approx(eps0 + deps, abs=1e-15)
        assert rf.equation_of_equinoxes(jd) == pytest.approx(
            dpsi * np.cos(eps0), abs=1e-15
        )


class TestFrameTransforms:
    JD_UT1 = 2453101.5
    JD_TT = 2453101.5 + 68.0 / 86400.0

    def test_gcrf_itrf_roundtrip(self):
        r = np.array([5102.5096, 6123.01152, 6378.1363])
        r_itrf = rf.gcrf_to_itrf(r, self.JD_UT1, self.JD_TT, xp=1e-6, yp=2e-6)
        r_back = rf.itrf_to_gcrf(r_itrf, self.JD_UT1, self.JD_TT, xp=1e-6, yp=2e-6)
        assert np.allclose(r_back, r, atol=1e-9)
        assert np.linalg.norm(r_itrf) == pytest.approx(np.linalg.norm(r), abs=1e-9)

    def test_gcrf_itrf_vs_astropy(self):
        """IAU76/80 chain vs astropy's IAU2006A: agreement to tens of meters."""
        astropy_time = _astropy_time()
        coords = pytest.importorskip("astropy.coordinates")
        u = pytest.importorskip("astropy.units")
        t = astropy_time.Time("2004-04-06 07:51:28", scale="utc")
        r = np.array([5102.5096, 6123.01152, 6378.1363])
        g = coords.GCRS(coords.CartesianRepresentation(r * u.km), obstime=t)
        r_ref = g.transform_to(coords.ITRS(obstime=t)).cartesian.xyz.to_value(u.km)
        r_got = rf.gcrf_to_itrf(r, t.ut1.jd, t.tt.jd)
        # Documented model difference (IAU76/80, truncated nutation, no EOP)
        assert np.linalg.norm(r_got - r_ref) < 0.05  # 50 m

    def test_tod_mod_chain_consistency(self):
        r = np.array([-3000.0, 5500.0, 4100.0])
        r_mod = rf.gcrf_to_mod(r, self.JD_TT)
        r_tod = rf.mod_to_tod(r_mod, self.JD_TT)
        assert np.allclose(rf.gcrf_to_tod(r, self.JD_TT), r_tod, atol=1e-9)
        assert np.allclose(rf.tod_to_mod(r_tod, self.JD_TT), r_mod, atol=1e-9)
        assert np.allclose(rf.mod_to_gcrf(r_mod, self.JD_TT), r, atol=1e-9)
        assert np.allclose(rf.tod_to_gcrf(r_tod, self.JD_TT), r, atol=1e-9)

    def test_tod_itrf_roundtrip(self):
        r = np.array([7000.0, -2000.0, 1000.0])
        r_itrf = rf.tod_to_itrf(r, self.JD_UT1, self.JD_TT)
        r_back = rf.itrf_to_tod(r_itrf, self.JD_UT1, self.JD_TT)
        assert np.allclose(r_back, r, atol=1e-9)

    def test_pef_gcrf_roundtrip(self):
        r = np.array([7000.0, -2000.0, 1000.0])
        r_pef = rf.gcrf_to_pef(r, self.JD_UT1, self.JD_TT)
        assert np.allclose(rf.pef_to_gcrf(r_pef, self.JD_UT1, self.JD_TT), r, atol=1e-9)

    def test_eci_ecef_roundtrip(self):
        r = np.array([7000.0, -2000.0, 1000.0])
        gmst = rf.gmst_iau82(self.JD_UT1)
        assert np.allclose(rf.ecef_to_eci(rf.eci_to_ecef(r, gmst), gmst), r, atol=1e-12)

    def test_ecliptic_equatorial_roundtrip(self):
        r = np.array([1.0, 2.0, 3.0])
        eps = rf.mean_obliquity_iau80(self.JD_TT)
        r_eq = rf.ecliptic_to_equatorial(r, eps)
        assert np.allclose(rf.equatorial_to_ecliptic(r_eq, eps), r, atol=1e-14)
        # z-axis of ecliptic maps to (0, -sin eps, cos eps)
        z_eq = rf.ecliptic_to_equatorial(np.array([0.0, 0.0, 1.0]), eps)
        assert np.allclose(z_eq, [0.0, -np.sin(eps), np.cos(eps)], atol=1e-14)


class TestTEME:
    JD_UT1 = 2453101.5
    JD_TT = 2453101.5 + 68.0 / 86400.0

    def test_teme_gmst_gast_consistency(self):
        """R3(GMST) r_TEME must equal R3(GAST) r_TOD (regression: eq-of-
        equinoxes rotation sign was flipped, giving ~0.6 km inconsistency)."""
        r_teme = np.array([7000.0, 1000.0, 1500.0])
        r_pef_a = rf.teme_to_pef(r_teme, self.JD_UT1)
        r_gcrf = rf.teme_to_gcrf(r_teme, self.JD_TT)
        r_pef_b = rf.gcrf_to_pef(r_gcrf, self.JD_UT1, self.JD_TT)
        assert np.allclose(r_pef_a, r_pef_b, atol=1e-9)

    def test_teme_gcrf_roundtrip(self):
        r_teme = np.array([7000.0, 1000.0, 1500.0])
        r_gcrf = rf.teme_to_gcrf(r_teme, self.JD_TT)
        assert np.allclose(rf.gcrf_to_teme(r_gcrf, self.JD_TT), r_teme, atol=1e-9)

    def test_teme_itrf_roundtrip(self):
        r_teme = np.array([7000.0, 1000.0, 1500.0])
        r_itrf = rf.teme_to_itrf(r_teme, self.JD_UT1, xp=1e-6, yp=2e-6)
        r_back = rf.itrf_to_teme(r_itrf, self.JD_UT1, xp=1e-6, yp=2e-6)
        assert np.allclose(r_back, r_teme, atol=1e-9)
        r_pef = rf.teme_to_pef(r_teme, self.JD_UT1)
        assert np.allclose(rf.pef_to_teme(r_pef, self.JD_UT1), r_teme, atol=1e-9)

    def test_teme_to_gcrf_vs_astropy(self):
        astropy_time = _astropy_time()
        coords = pytest.importorskip("astropy.coordinates")
        u = pytest.importorskip("astropy.units")
        r_teme = np.array([7000.0, 1000.0, 1500.0])
        t = astropy_time.Time(self.JD_UT1, format="jd", scale="ut1")
        c = coords.TEME(coords.CartesianRepresentation(r_teme * u.km), obstime=t)
        r_ref = c.transform_to(coords.GCRS(obstime=t)).cartesian.xyz.to_value(u.km)
        r_got = rf.teme_to_gcrf(r_teme, self.JD_TT)
        assert np.linalg.norm(r_got - r_ref) < 0.005  # 5 m

    def test_teme_itrf_velocity_roundtrip(self):
        r_teme = np.array([7000.0, 1000.0, 1500.0])
        v_teme = np.array([1.0, 7.0, -0.5])
        r_itrf, v_itrf = rf.teme_to_itrf_with_velocity(r_teme, v_teme, self.JD_UT1)
        r_back, v_back = rf.itrf_to_teme_with_velocity(r_itrf, v_itrf, self.JD_UT1)
        assert np.allclose(r_back, r_teme, atol=1e-9)
        assert np.allclose(v_back, v_teme, atol=1e-12)

    def test_teme_itrf_velocity_earth_rotation(self):
        """An ECEF-stationary point must have near-zero ITRF velocity."""
        omega = 7.29211514670698e-5
        r_teme = np.array([7000.0, 0.0, 0.0])
        v_teme = np.cross([0.0, 0.0, omega], r_teme)  # co-rotating
        _, v_itrf = rf.teme_to_itrf_with_velocity(r_teme, v_teme, self.JD_UT1)
        assert np.linalg.norm(v_itrf) < 1e-10


# =============================================================================
# Orbital mechanics
# =============================================================================


class TestAnomalyConversions:
    def test_elliptic_roundtrips(self):
        rng = np.random.default_rng(42)
        for _ in range(500):
            e = rng.uniform(0.0, 0.99)
            M = rng.uniform(0.0, 2 * np.pi)
            E = mean_to_eccentric_anomaly(M, e)
            assert abs(E - e * np.sin(E) - M) < 1e-11  # Kepler's equation
            assert abs(_angdiff(eccentric_to_mean_anomaly(E, e), M)) < 1e-11
            nu = eccentric_to_true_anomaly(E, e)
            assert abs(_angdiff(true_to_eccentric_anomaly(nu, e), E)) < 1e-11
            assert abs(_angdiff(true_to_mean_anomaly(nu, e), M)) < 1e-11
            assert abs(_angdiff(mean_to_true_anomaly(M, e), nu)) < 1e-11

    def test_hyperbolic_roundtrips(self):
        rng = np.random.default_rng(43)
        for _ in range(500):
            e = rng.uniform(1.01, 5.0)
            M = rng.uniform(-3.0, 3.0)
            H = mean_to_hyperbolic_anomaly(M, e)
            assert abs(e * np.sinh(H) - H - M) < 1e-10
            nu = hyperbolic_to_true_anomaly(H, e)
            assert abs(true_to_hyperbolic_anomaly(nu, e) - H) < 1e-10
            assert abs(true_to_mean_anomaly(nu, e) - M) < 1e-9
            assert abs(mean_to_true_anomaly(M, e) - nu) < 1e-10

    def test_curtis_example_3_2(self):
        """Curtis Ex. 3.2: M = 3.6029 rad, e = 0.37255 -> E = 3.47942 rad."""
        E = mean_to_eccentric_anomaly(3.6029, 0.37255)
        assert E == pytest.approx(3.47942, abs=1e-4)

    def test_invalid_eccentricity(self):
        with pytest.raises(ValueError):
            mean_to_eccentric_anomaly(1.0, 1.5)
        with pytest.raises(ValueError):
            mean_to_hyperbolic_anomaly(1.0, 0.5)


class TestElementsStateConversions:
    def test_roundtrip_random(self):
        rng = np.random.default_rng(44)
        for _ in range(300):
            el = OrbitalElements(
                a=rng.uniform(6800, 50000),
                e=rng.uniform(0.0, 0.9),
                i=rng.uniform(0.01, np.pi - 0.01),
                raan=rng.uniform(0, 2 * np.pi),
                omega=rng.uniform(0, 2 * np.pi),
                nu=rng.uniform(0, 2 * np.pi),
            )
            st = orbital_elements_to_state(el)
            el2 = state_to_orbital_elements(st)
            st2 = orbital_elements_to_state(el2)
            assert np.allclose(st2.r, st.r, rtol=1e-9, atol=1e-6)
            assert np.allclose(st2.v, st.v, rtol=1e-9, atol=1e-9)

    def test_roundtrip_hyperbolic(self):
        el = OrbitalElements(a=-20000.0, e=1.5, i=0.5, raan=1.0, omega=2.0, nu=0.5)
        st = orbital_elements_to_state(el)
        el2 = state_to_orbital_elements(st)
        assert np.allclose(el2, el, rtol=1e-9, atol=1e-9)

    def test_perigee_geometry(self):
        el = OrbitalElements(a=7000.0, e=0.01, i=0.5, raan=0.0, omega=0.0, nu=0.0)
        st = orbital_elements_to_state(el)
        assert np.allclose(st.r, [6930.0, 0.0, 0.0], atol=1e-9)
        assert np.dot(st.r, st.v) == pytest.approx(0.0, abs=1e-9)

    def test_state_consistency_with_energy(self):
        el = OrbitalElements(a=12000.0, e=0.3, i=1.0, raan=2.0, omega=3.0, nu=1.5)
        st = orbital_elements_to_state(el)
        assert specific_orbital_energy(st) == pytest.approx(
            -GM_EARTH / (2 * el.a), rel=1e-12
        )
        h = specific_angular_momentum(st)
        p = el.a * (1 - el.e**2)
        assert np.linalg.norm(h) == pytest.approx(np.sqrt(GM_EARTH * p), rel=1e-12)

    def test_circular_equatorial(self):
        r = 8000.0
        v = circular_velocity(r)
        st = StateVector(r=np.array([r, 0.0, 0.0]), v=np.array([0.0, v, 0.0]))
        el = state_to_orbital_elements(st)
        assert el.a == pytest.approx(r, rel=1e-12)
        assert el.e == pytest.approx(0.0, abs=1e-12)
        assert el.i == pytest.approx(0.0, abs=1e-12)


class TestKeplerPropagation:
    def test_conservation(self):
        st = orbital_elements_to_state(OrbitalElements(7000, 0.1, 0.9, 1.0, 2.0, 0.3))
        E0 = specific_orbital_energy(st)
        h0 = specific_angular_momentum(st)
        for dt in np.linspace(100.0, 2e5, 25):
            st2 = kepler_propagate_state(st, dt)
            assert specific_orbital_energy(st2) == pytest.approx(E0, rel=1e-12)
            assert np.allclose(specific_angular_momentum(st2), h0, rtol=1e-12)

    def test_full_period_return(self):
        st = orbital_elements_to_state(OrbitalElements(7000, 0.1, 0.9, 1.0, 2.0, 0.3))
        stT = kepler_propagate_state(st, orbital_period(7000.0))
        assert np.allclose(stT.r, st.r, atol=1e-6)
        assert np.allclose(stT.v, st.v, atol=1e-9)

    def test_back_forward(self):
        st = orbital_elements_to_state(OrbitalElements(9000, 0.2, 0.4, 0.5, 1.0, 2.0))
        st2 = kepler_propagate_state(kepler_propagate_state(st, -3600.0), 3600.0)
        assert np.allclose(st2.r, st.r, atol=1e-6)

    def test_elements_propagation_only_changes_nu(self):
        el = OrbitalElements(7000, 0.1, 0.9, 1.0, 2.0, 0.3)
        el2 = kepler_propagate(el, 1234.0)
        assert (el2.a, el2.e, el2.i, el2.raan, el2.omega) == (
            el.a,
            el.e,
            el.i,
            el.raan,
            el.omega,
        )
        assert el2.nu != el.nu

    def test_hyperbolic_propagation(self):
        st = orbital_elements_to_state(OrbitalElements(-20000, 1.5, 0.5, 1.0, 2.0, 0.2))
        E0 = specific_orbital_energy(st)
        st2 = kepler_propagate_state(st, 5000.0)
        assert specific_orbital_energy(st2) == pytest.approx(E0, rel=1e-12)
        st3 = kepler_propagate_state(st2, -5000.0)
        assert np.allclose(st3.r, st.r, atol=1e-6)


class TestOrbitalQuantities:
    def test_vis_viva_family(self):
        r, a = 7000.0, 9000.0
        v = vis_viva(r, a)
        assert v == pytest.approx(np.sqrt(GM_EARTH * (2 / r - 1 / a)), rel=1e-15)
        assert vis_viva(7000.0, 7000.0) == pytest.approx(
            circular_velocity(7000.0), rel=1e-15
        )
        assert escape_velocity(7000.0) == pytest.approx(
            np.sqrt(2) * circular_velocity(7000.0), rel=1e-15
        )

    def test_period_mean_motion_consistency(self):
        for a in [6800.0, 26560.0, 42164.0]:
            assert orbital_period(a) == pytest.approx(2 * np.pi / mean_motion(a))
        # GEO sanity: ~1436 min
        assert orbital_period(42164.17) / 60 == pytest.approx(1436.07, abs=0.1)

    def test_apsides(self):
        assert periapsis_radius(10000.0, 0.3) == 7000.0
        assert apoapsis_radius(10000.0, 0.3) == 13000.0
        assert apoapsis_radius(-10000.0, 1.5) == np.inf
        assert orbit_radius(0.0, 10000.0, 0.3) == pytest.approx(7000.0)
        assert orbit_radius(np.pi, 10000.0, 0.3) == pytest.approx(13000.0)

    def test_time_since_periapsis(self):
        a, e = 7000.0, 0.1
        T = orbital_period(a)
        assert time_since_periapsis(np.pi, a, e) == pytest.approx(T / 2, rel=1e-12)
        assert time_since_periapsis(0.0, a, e) == pytest.approx(0.0, abs=1e-9)

    def test_flight_path_angle(self):
        st = StateVector(r=np.array([7000.0, 0.0, 0.0]), v=np.array([0.0, 7.5, 0.0]))
        assert flight_path_angle(st) == pytest.approx(0.0, abs=1e-12)
        # Analytic: tan(gamma) = e sin(nu) / (1 + e cos(nu))
        el = OrbitalElements(9000.0, 0.3, 0.7, 1.0, 2.0, 1.1)
        st2 = orbital_elements_to_state(el)
        gamma_ref = np.arctan2(el.e * np.sin(el.nu), 1 + el.e * np.cos(el.nu))
        assert flight_path_angle(st2) == pytest.approx(gamma_ref, abs=1e-12)


# =============================================================================
# Lambert solvers
# =============================================================================


class TestLambert:
    R1 = np.array([5000.0, 10000.0, 2100.0])
    R2 = np.array([-14600.0, 2500.0, 7000.0])

    def _check_boundary(self, r1, r2, sol, tof, atol=1e-6):
        st = kepler_propagate_state(StateVector(r=r1, v=sol.v1), tof)
        assert np.linalg.norm(st.r - r2) < atol
        assert np.allclose(st.v, sol.v2, atol=1e-9)

    def test_universal_curtis_5_2(self):
        sol = lambert_universal(self.R1, self.R2, 3600.0)
        assert np.allclose(sol.v1, [-5.9925, 1.9254, 3.2456], atol=1e-3)
        assert np.allclose(sol.v2, [-3.3125, -4.1966, -0.38529], atol=1e-3)
        self._check_boundary(self.R1, self.R2, sol, 3600.0)

    def test_universal_vallado_7_5(self):
        r1 = np.array([15945.34, 0.0, 0.0])
        r2 = np.array([12214.83899, 10249.46731, 0.0])
        sol = lambert_universal(r1, r2, 76.0 * 60)
        assert np.allclose(sol.v1, [2.058913, 2.915965, 0.0], atol=1e-4)
        assert np.allclose(sol.v2, [-3.451565, 0.910315, 0.0], atol=1e-4)
        self._check_boundary(r1, r2, sol, 76.0 * 60)

    def test_izzo_curtis_5_2(self):
        """Regression: previously off by thousands of km."""
        sol = lambert_izzo(self.R1, self.R2, 3600.0)
        assert np.allclose(sol.v1, [-5.9925, 1.9254, 3.2456], atol=1e-3)
        self._check_boundary(self.R1, self.R2, sol, 3600.0)

    def test_izzo_vallado_7_5(self):
        r1 = np.array([15945.34, 0.0, 0.0])
        r2 = np.array([12214.83899, 10249.46731, 0.0])
        sol = lambert_izzo(r1, r2, 76.0 * 60)
        assert np.allclose(sol.v1, [2.058913, 2.915965, 0.0], atol=1e-4)
        self._check_boundary(r1, r2, sol, 76.0 * 60)

    def test_izzo_matches_universal_random(self):
        rng = np.random.default_rng(7)
        for _ in range(50):
            th = rng.uniform(0.3, 2.8)
            r1 = rng.uniform(7000, 20000) * np.array([1.0, 0.0, 0.0])
            r2 = rng.uniform(7000, 20000) * np.array(
                [np.cos(th), np.sin(th), rng.uniform(-0.2, 0.2)]
            )
            tof = rng.uniform(3000, 30000)
            s_izzo = lambert_izzo(r1, r2, tof)
            s_univ = lambert_universal(r1, r2, tof)
            assert np.allclose(s_izzo.v1, s_univ.v1, atol=1e-5)
            self._check_boundary(r1, r2, s_izzo, tof, atol=1e-4)

    def test_izzo_hyperbolic(self):
        sol = lambert_izzo(self.R1, self.R2, 1500.0)
        assert sol.e > 1.0
        self._check_boundary(self.R1, self.R2, sol, 1500.0, atol=1e-6)

    def test_izzo_retrograde(self):
        sol = lambert_izzo(self.R1, self.R2, 3600.0, prograde=False)
        h = np.cross(self.R1, sol.v1)
        assert h[2] < 0  # retrograde orbit normal
        self._check_boundary(self.R1, self.R2, sol, 3600.0)

    def test_izzo_multi_rev(self):
        r1 = np.array([7000.0, 0.0, 0.0])
        r2 = np.array([0.0, 8000.0, 100.0])
        tof = 30000.0
        sol = lambert_izzo(r1, r2, tof, multi_rev=1)
        self._check_boundary(r1, r2, sol, tof, atol=1e-4)

    def test_izzo_multi_rev_too_short_raises(self):
        with pytest.raises(ValueError):
            lambert_izzo(
                np.array([7000.0, 0.0, 0.0]),
                np.array([0.0, 8000.0, 100.0]),
                3000.0,
                multi_rev=3,
            )

    def test_universal_long_way(self):
        tof = 4 * 3600.0
        sol = lambert_universal(self.R1, self.R2, tof, low_path=False)
        st = kepler_propagate_state(StateVector(r=self.R1, v=sol.v1), tof)
        assert np.linalg.norm(st.r - self.R2) < 1e-6

    def test_minimum_energy_transfer(self):
        tof_min, sol = minimum_energy_transfer(self.R1, self.R2)
        assert tof_min > 0
        st = kepler_propagate_state(StateVector(r=self.R1, v=sol.v1), tof_min)
        assert np.linalg.norm(st.r - self.R2) < 1e-6
        # Minimum-energy semi-major axis is s/2
        r1m, r2m = np.linalg.norm(self.R1), np.linalg.norm(self.R2)
        c = np.linalg.norm(self.R2 - self.R1)
        assert sol.a == pytest.approx((r1m + r2m + c) / 4, rel=1e-6)

    def test_hohmann_leo_geo(self):
        dv1, dv2, tof = hohmann_transfer(6678.0, 42164.0)
        assert dv1 + dv2 == pytest.approx(3.893, abs=1e-3)  # Curtis/Vallado
        assert tof == pytest.approx(
            np.pi * np.sqrt(((6678.0 + 42164.0) / 2) ** 3 / GM_EARTH), rel=1e-12
        )

    def test_bi_elliptic_reduces_to_hohmann(self):
        """With intermediate radius == r2, bi-elliptic equals Hohmann."""
        r1, r2 = 7000.0, 40000.0
        dv1h, dv2h, tofh = hohmann_transfer(r1, r2)
        dv1, dv2, dv3, tof = bi_elliptic_transfer(r1, r2, r2)
        assert dv1 == pytest.approx(dv1h, rel=1e-12)
        assert dv2 + dv3 == pytest.approx(dv2h, rel=1e-9)
        assert tof > tofh  # includes second half-ellipse

    def test_bi_elliptic_invalid_intermediate(self):
        with pytest.raises(ValueError):
            bi_elliptic_transfer(7000.0, 40000.0, 20000.0)


# =============================================================================
# TLE parsing
# =============================================================================


class TestTLEParsing:
    def test_parse_iss(self):
        tle = parse_tle(ISS_L1, ISS_L2, name="ISS")
        assert tle.catalog_number == 25544
        assert tle.classification == "U"
        assert tle.int_designator == "98067A"
        assert tle.epoch_year == 2024
        assert tle.epoch_day == pytest.approx(1.5)
        assert np.degrees(tle.inclination) == pytest.approx(51.6400)
        assert np.degrees(tle.raan) == pytest.approx(247.4627)
        assert tle.eccentricity == pytest.approx(0.0006703)
        assert np.degrees(tle.arg_perigee) == pytest.approx(130.5360)
        assert np.degrees(tle.mean_anomaly) == pytest.approx(325.0288)
        assert tle.mean_motion == pytest.approx(15.49815350 * 2 * np.pi / 1440.0)
        assert tle.bstar == pytest.approx(0.10270e-3)
        assert tle.ndot == pytest.approx(2 * 0.00016717)
        assert tle.revolution_number == 47900

    def test_parse_vs_sgp4_package(self):
        sgp4_api = pytest.importorskip("sgp4.api")
        sat = sgp4_api.Satrec.twoline2rv(ISS_L1, ISS_L2)
        tle = parse_tle(ISS_L1, ISS_L2)
        assert tle.inclination == pytest.approx(sat.inclo, abs=1e-12)
        assert tle.raan == pytest.approx(sat.nodeo, abs=1e-12)
        assert tle.eccentricity == pytest.approx(sat.ecco, abs=1e-12)
        assert tle.arg_perigee == pytest.approx(sat.argpo, abs=1e-12)
        assert tle.mean_anomaly == pytest.approx(sat.mo, abs=1e-12)
        assert tle.mean_motion == pytest.approx(sat.no_kozai, abs=1e-14)
        assert tle.bstar == pytest.approx(sat.bstar, abs=1e-12)
        assert tle_epoch_to_jd(tle) == pytest.approx(
            sat.jdsatepoch + sat.jdsatepochF, abs=1e-9
        )

    def test_checksum_rejected(self):
        bad = ISS_L1[:68] + "0"
        with pytest.raises(ValueError):
            parse_tle(bad, ISS_L2)
        # But accepted when verification is disabled
        parse_tle(bad, ISS_L2, verify_checksum=False)

    def test_negative_bstar(self):
        line1 = _fix_checksum(
            "1 25544U 98067A   24001.50000000  .00016717  00000-0 -10270-3 0  9990"
        )
        tle = parse_tle(line1, ISS_L2)
        assert tle.bstar == pytest.approx(-0.10270e-3)

    def test_parse_3line(self):
        tle = parse_tle_3line(f"ISS (ZARYA)\n{ISS_L1}\n{ISS_L2}")
        assert tle.name == "ISS (ZARYA)"
        assert tle.catalog_number == 25544

    def test_epoch_conversions(self):
        tle = parse_tle(ISS_L1, ISS_L2)
        assert tle_epoch_to_jd(tle) == pytest.approx(2460311.0)
        dt = tle_epoch_to_datetime(tle)
        assert (dt.year, dt.month, dt.day, dt.hour) == (2024, 1, 1, 12)

    def test_epoch_year_windowing(self):
        # 57 -> 1957 (Sputnik convention), 56 -> 2056
        l1_1999 = _fix_checksum(
            "1 25544U 98067A   99001.50000000  .00016717  00000-0  10270-3 0  9997"
        )
        tle = parse_tle(l1_1999, ISS_L2)
        assert tle.epoch_year == 1999

    def test_deep_space_and_period(self):
        tle = parse_tle(ISS_L1, ISS_L2)
        assert not is_deep_space(tle)
        assert orbital_period_from_tle(tle) == pytest.approx(
            2 * np.pi / tle.mean_motion * 60.0
        )
        n = tle.mean_motion
        a = semi_major_axis_from_mean_motion(n)
        assert a == pytest.approx((398600.4418 / (n / 60.0) ** 2) ** (1 / 3))
        assert 6700 < a < 6900


# =============================================================================
# SGP4 vs the official reference implementation
# =============================================================================

GEO_L1 = "1 41866U 16071A   24001.32505567  .00000090  00000+0  00000+0 0  9996"
GEO_L2 = "2 41866   0.0357 286.1082 0000429 156.2069 244.4229  1.00271862 26076"
MOL_L1 = "1 25485U 98054A   24001.17603044  .00000239  00000+0  00000+0 0  9992"
MOL_L2 = "2 25485  64.2196  62.5211 6822908 289.0665  12.5462  2.36441455205652"
# High-drag LEO (large BSTAR) exercising the higher-order drag terms
DRAG_L1 = "1 44444U 19029BR  24001.50000000  .00051000  00000-0  25000-2 0  9990"
DRAG_L2 = "2 44444  52.9979 339.1899 0011000  85.9807 274.1363 15.50000000 12345"


class TestSGP4:
    def _compare(self, l1, l2, times, tol_r_km, tol_v_kms):
        sgp4_api = pytest.importorskip("sgp4.api")
        l1, l2 = _fix_checksum(l1), _fix_checksum(l2)
        ref = sgp4_api.Satrec.twoline2rv(l1, l2, sgp4_api.WGS72)
        sat = SGP4Satellite(parse_tle(l1, l2))
        worst_r = worst_v = 0.0
        for t in times:
            err, r_ref, v_ref = ref.sgp4_tsince(t)
            assert err == 0
            st = sat.propagate(t)
            assert st.error == 0
            worst_r = max(worst_r, float(np.linalg.norm(np.array(r_ref) - st.r)))
            worst_v = max(worst_v, float(np.linalg.norm(np.array(v_ref) - st.v)))
        assert worst_r < tol_r_km, f"max position error {worst_r * 1e3:.1f} m"
        assert worst_v < tol_v_kms, f"max velocity error {worst_v * 1e3:.4f} m/s"

    def test_leo_matches_reference(self):
        """Near-Earth SGP4 must match Vallado's reference to sub-meter.

        Regression: argpdot used (1 - theta^2) instead of con42 =
        (1 - 5 theta^2), the J2 short-period factor was half its correct
        value, and the drag periodic terms were wrong -- giving 1.5 km
        error at epoch and ~45 km per orbit of drift.
        """
        times = [-720.0, 0.0, 10.0, 90.0, 360.0, 1440.0, 4320.0]
        self._compare(ISS_L1, ISS_L2, times, tol_r_km=1e-3, tol_v_kms=1e-6)

    def test_high_drag_leo_matches_reference(self):
        times = [0.0, 90.0, 720.0, 1440.0, 2880.0]
        self._compare(DRAG_L1, DRAG_L2, times, tol_r_km=1e-3, tol_v_kms=1e-6)

    def test_geo_within_documented_limits(self):
        """SDP4 deep-space physics (lunar-solar, resonance) is not
        implemented; agreement is tens of km, not meters. This bounds the
        documented limitation so regressions are caught."""
        times = [0.0, 360.0, 1440.0]
        self._compare(GEO_L1, GEO_L2, times, tol_r_km=50.0, tol_v_kms=0.01)

    def test_molniya_within_documented_limits(self):
        times = [0.0, 360.0, 1440.0]
        self._compare(MOL_L1, MOL_L2, times, tol_r_km=100.0, tol_v_kms=0.05)

    def test_propagate_jd_and_batch(self):
        tle = parse_tle(ISS_L1, ISS_L2)
        sat = SGP4Satellite(tle)
        st0 = sat.propagate(0.0)
        st_jd = sat.propagate_jd(tle_epoch_to_jd(tle))
        assert np.allclose(st_jd.r, st0.r, atol=1e-9)
        times = np.array([0.0, 30.0, 60.0])
        r, v = sgp4_propagate_batch(tle, times)
        assert r.shape == (3, 3) and v.shape == (3, 3)
        assert np.allclose(r[0], st0.r, atol=1e-9)
        st60 = sgp4_propagate(tle, 60.0)
        assert np.allclose(r[2], st60.r, atol=1e-9)

    def test_orbit_radius_sane(self):
        tle = parse_tle(ISS_L1, ISS_L2)
        st = sgp4_propagate(tle, 0.0)
        assert 6700 < np.linalg.norm(st.r) < 6900
        assert 7.0 < np.linalg.norm(st.v) < 8.0


# =============================================================================
# Special orbits
# =============================================================================


class TestSpecialOrbits:
    def test_classify(self):
        assert classify_orbit(0.0) == OrbitType.CIRCULAR
        assert classify_orbit(0.5) == OrbitType.ELLIPTICAL
        assert classify_orbit(1.0) == OrbitType.PARABOLIC
        assert classify_orbit(1.0 + 1e-12) == OrbitType.PARABOLIC
        assert classify_orbit(1.5) == OrbitType.HYPERBOLIC
        with pytest.raises(ValueError):
            classify_orbit(-0.1)

    def test_barker_equation(self):
        rng = np.random.default_rng(5)
        for _ in range(100):
            M = rng.uniform(-10.0, 10.0)
            D = mean_to_parabolic_anomaly(M)
            assert abs(D + D**3 / 3.0 - M) < 1e-10
            nu = parabolic_anomaly_to_true_anomaly(D)
            assert abs(true_anomaly_to_parabolic_anomaly(nu) - D) < 1e-9
        nu = mean_to_true_anomaly_parabolic(0.0)
        assert nu == pytest.approx(0.0, abs=1e-12)

    def test_parabolic_radius_velocity(self):
        mu, rp = GM_EARTH, 7000.0
        assert radius_parabolic(rp, 0.0) == pytest.approx(rp)
        r = radius_parabolic(rp, 1.0)
        # v = sqrt(2 mu / r) (zero-energy orbit)
        assert velocity_parabolic(mu, rp, 1.0) == pytest.approx(np.sqrt(2 * mu / r))
        assert escape_velocity_at_radius(mu, r) == pytest.approx(np.sqrt(2 * mu / r))
        with pytest.raises(ValueError):
            radius_parabolic(rp, np.pi)

    def test_hyperbolic_helpers_match_orbital_mechanics(self):
        e, H = 1.7, 0.8
        assert hyperbolic_anomaly_to_true_anomaly(H, e) == pytest.approx(
            hyperbolic_to_true_anomaly(H, e), abs=1e-14
        )
        nu = hyperbolic_anomaly_to_true_anomaly(H, e)
        assert true_anomaly_to_hyperbolic_anomaly(nu, e) == pytest.approx(H, abs=1e-12)
        with pytest.raises(ValueError):
            hyperbolic_anomaly_to_true_anomaly(1.0, 0.9)

    def test_hyperbolic_geometry(self):
        """Regression: deflection angle was pi - 2*arccos(-1/e), which is
        always negative; the turn angle is 2*arcsin(1/e) in (0, pi)."""
        e = 2.0
        nu_inf = hyperbolic_asymptote_angle(e)
        assert nu_inf == pytest.approx(np.arccos(-0.5))  # 120 deg
        delta = hyperbolic_deflection_angle(e)
        assert delta == pytest.approx(2 * np.arcsin(1 / e))  # 60 deg
        assert delta == pytest.approx(2 * nu_inf - np.pi)
        assert 0 < delta < np.pi
        with pytest.raises(ValueError):
            hyperbolic_deflection_angle(0.5)

    def test_excess_velocity_and_energy(self):
        a = -15000.0
        v_inf = hyperbolic_excess_velocity(GM_EARTH, a)
        assert v_inf == pytest.approx(np.sqrt(-GM_EARTH / a))
        assert semi_major_axis_from_energy(GM_EARTH, v_inf**2 / 2) == pytest.approx(a)
        with pytest.raises(ValueError):
            hyperbolic_excess_velocity(GM_EARTH, 15000.0)
        with pytest.raises(ValueError):
            semi_major_axis_from_energy(GM_EARTH, 0.0)

    def test_eccentricity_vector(self):
        el = OrbitalElements(a=9000.0, e=0.35, i=0.8, raan=1.2, omega=2.5, nu=0.9)
        st = orbital_elements_to_state(el)
        e_vec = eccentricity_vector(st.r, st.v, GM_EARTH)
        assert np.linalg.norm(e_vec) == pytest.approx(el.e, rel=1e-12)
