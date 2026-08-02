"""Correctness audit tests for pytcl.navigation.

Reference validation against geographiclib (Vincenty geodesics), independent
vector/spherical trigonometry (great circle), ODE integration of the loxodrome
equations (rhumb lines), analytic WGS84 values (gravity, radii), and
hand-computed geometry (DOP, pseudorange Jacobians).
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pytcl.coordinate_systems.rotations import quat2rotmat
from pytcl.navigation.geodesy import (
    WGS84,
    direct_geodetic,
    ecef_to_enu,
    ecef_to_geodetic,
    ecef_to_ned,
    enu_to_ecef,
    geodetic_to_ecef,
    haversine_distance,
    inverse_geodetic,
    ned_to_ecef,
)
from pytcl.navigation.great_circle import (
    EARTH_RADIUS,
    angular_distance,
    cross_track_distance,
    destination_point,
    great_circle_azimuth,
    great_circle_direct,
    great_circle_distance,
    great_circle_intersect,
    great_circle_inverse,
    great_circle_path_intersect,
    great_circle_tdoa_loc,
    great_circle_waypoint,
    great_circle_waypoints,
)
from pytcl.navigation.ins import (
    OMEGA_EARTH,
    IMUData,
    coarse_alignment,
    compensate_imu_data,
    coning_correction,
    earth_rate_ned,
    gyrocompass_alignment,
    initialize_ins_state,
    mechanize_ins_ned,
    normal_gravity,
    radii_of_curvature,
    sculling_correction,
    skew_symmetric,
    transport_rate_ned,
    update_attitude_ned,
    update_quaternion,
)
from pytcl.navigation.ins_gnss import (
    GNSSMeasurement,
    SatelliteInfo,
    compute_dop,
    compute_line_of_sight,
    gnss_outage_detection,
    initialize_ins_gnss,
    loose_coupled_predict,
    loose_coupled_update,
    loose_coupled_update_position,
    loose_coupled_update_velocity,
    position_measurement_matrix,
    position_std_to_error_state_units,
    position_velocity_measurement_matrix,
    pseudorange_measurement_matrix,
    satellite_elevation_azimuth,
    tight_coupled_measurement_matrix,
    tight_coupled_pseudorange_innovation,
    tight_coupled_update,
    velocity_measurement_matrix,
)
from pytcl.navigation.rhumb import (
    compare_great_circle_rhumb,
    direct_rhumb,
    direct_rhumb_spherical,
    indirect_rhumb,
    indirect_rhumb_spherical,
    rhumb_bearing,
    rhumb_distance_ellipsoidal,
    rhumb_distance_spherical,
    rhumb_intersect,
    rhumb_midpoint,
    rhumb_waypoints,
)

geographiclib = pytest.importorskip("geographiclib.geodesic")
GEO = geographiclib.Geodesic.WGS84

d = np.radians

CITY_PAIRS = [
    ("NYC-London", 40.7128, -74.0060, 51.5074, -0.1278),
    ("Sydney-Santiago", -33.8688, 151.2093, -33.4489, -70.6693),
    ("Tokyo-SaoPaulo", 35.6762, 139.6503, -23.5505, -46.6333),
    ("polar", 85.0, 10.0, 80.0, -170.0),
    ("near-antipodal", 10.0, 20.0, -9.5, -159.0),
    ("short", 40.0, -75.0, 40.001, -75.001),
    ("equatorial", 0.0, 0.0, 0.0, 90.0),
    ("same-meridian", 10.0, 30.0, 60.0, 30.0),
]


def sph2vec(lat, lon):
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


class TestGeodesyReference:
    """Vincenty direct/inverse vs geographiclib.Geodesic.WGS84."""

    @pytest.mark.parametrize("name,la1,lo1,la2,lo2", CITY_PAIRS)
    def test_inverse_vs_geographiclib(self, name, la1, lo1, la2, lo2):
        ref = GEO.Inverse(la1, lo1, la2, lo2)
        dist, az1, az2 = inverse_geodetic(d(la1), d(lo1), d(la2), d(lo2))
        assert dist == pytest.approx(ref["s12"], abs=1e-3)
        assert np.degrees(az1) % 360 == pytest.approx(ref["azi1"] % 360, abs=1e-6)
        assert np.degrees(az2) % 360 == pytest.approx(ref["azi2"] % 360, abs=1e-6)

    @pytest.mark.parametrize(
        "la1,lo1,azdeg,dist",
        [
            (40.7, -74.0, 45.0, 1_000_000),
            (85.0, 10.0, 10.0, 800_000),
            (0.0, 0.0, 90.0, 5_000_000),
            (-30.0, 100.0, 250.0, 15_000_000),
            (-60.0, -70.0, 180.0, 2_000_000),
        ],
    )
    def test_direct_vs_geographiclib(self, la1, lo1, azdeg, dist):
        ref = GEO.Direct(la1, lo1, azdeg, dist)
        lat2, lon2, az2 = direct_geodetic(d(la1), d(lo1), d(azdeg), dist)
        assert np.degrees(lat2) == pytest.approx(ref["lat2"], abs=1e-7)
        dlon = (np.degrees(lon2) - ref["lon2"] + 180) % 360 - 180
        assert dlon == pytest.approx(0.0, abs=1e-7)
        assert np.degrees(az2) % 360 == pytest.approx(ref["azi2"] % 360, abs=1e-6)

    @pytest.mark.parametrize("name,la1,lo1,la2,lo2", CITY_PAIRS)
    def test_direct_inverse_roundtrip(self, name, la1, lo1, la2, lo2):
        dist, az1, _ = inverse_geodetic(d(la1), d(lo1), d(la2), d(lo2))
        lat2, lon2, _ = direct_geodetic(d(la1), d(lo1), az1, dist)
        closure = GEO.Inverse(np.degrees(lat2), np.degrees(lon2), la2, lo2)["s12"]
        assert closure < 5e-3  # sub-cm closure (1mm distance quantization)

    def test_ecef_roundtrip_random(self):
        rng = np.random.default_rng(42)
        for _ in range(100):
            la = rng.uniform(-89.9, 89.9)
            lo = rng.uniform(-180, 180)
            al = rng.uniform(-5000, 50000)
            x, y, z = geodetic_to_ecef(d(la), d(lo), al)
            la2, lo2, al2 = ecef_to_geodetic(x, y, z)
            assert np.degrees(la2) == pytest.approx(la, abs=1e-9)
            assert np.degrees(lo2) == pytest.approx(lo, abs=1e-9)
            assert float(al2) == pytest.approx(al, abs=1e-4)

    def test_ecef_pole(self):
        x, y, z = geodetic_to_ecef(np.pi / 2, 0.0, 100.0)
        la2, _, al2 = ecef_to_geodetic(x, y, z)
        assert float(la2) == pytest.approx(np.pi / 2, abs=1e-9)
        assert float(al2) == pytest.approx(100.0, abs=1e-4)

    def test_enu_ned_roundtrips(self):
        ref = (d(40.0), d(-75.0), 30.0)
        x, y, z = enu_to_ecef(1000.0, 500.0, 100.0, *ref)
        e, n, u = ecef_to_enu(x, y, z, *ref)
        np.testing.assert_allclose([e, n, u], [1000.0, 500.0, 100.0], atol=1e-6)
        x, y, z = ned_to_ecef(100.0, 50.0, 10.0, *ref)
        n, e, dn = ecef_to_ned(x, y, z, *ref)
        np.testing.assert_allclose([n, e, dn], [100.0, 50.0, 10.0], atol=1e-6)

    def test_ned_is_enu_swapped(self):
        ref = (d(40.0), d(-75.0), 0.0)
        x, y, z = geodetic_to_ecef(d(40.01), d(-74.99), 200.0)
        e, n, u = ecef_to_enu(x, y, z, *ref)
        nn, ee, dd = ecef_to_ned(x, y, z, *ref)
        assert nn == pytest.approx(float(n))
        assert ee == pytest.approx(float(e))
        assert dd == pytest.approx(float(-u))

    def test_haversine_analytic(self):
        # Equator to 45N along meridian is exactly R*pi/4
        got = haversine_distance(0.0, 0.0, d(45), 0.0)
        assert got == pytest.approx(6371000.0 * np.pi / 4, abs=1e-6)
        assert haversine_distance(0.0, 0.0, 0.0, 0.0) == 0.0


class TestGreatCircleReference:
    """Great circle formulas vs independent vector math."""

    def test_distance_vs_vector_angle(self):
        rng = np.random.default_rng(7)
        for _ in range(100):
            la1, lo1 = rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-np.pi, np.pi)
            la2, lo2 = rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-np.pi, np.pi)
            ref = (
                np.arccos(np.clip(sph2vec(la1, lo1) @ sph2vec(la2, lo2), -1, 1))
                * EARTH_RADIUS
            )
            assert great_circle_distance(la1, lo1, la2, lo2) == pytest.approx(
                ref, abs=1e-2
            )

    def test_azimuth_vs_tangent_plane(self):
        rng = np.random.default_rng(8)
        north = np.array([0.0, 0.0, 1.0])
        for _ in range(100):
            la1, lo1 = rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)
            la2, lo2 = rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)
            p1, p2 = sph2vec(la1, lo1), sph2vec(la2, lo2)
            e = np.cross(north, p1)
            e /= np.linalg.norm(e)
            n = np.cross(p1, e)
            t = p2 - (p2 @ p1) * p1
            if np.linalg.norm(t) < 1e-9:
                continue
            ref = np.arctan2(t @ e, t @ n) % (2 * np.pi)
            got = great_circle_azimuth(la1, lo1, la2, lo2)
            err = min(abs(got - ref), 2 * np.pi - abs(got - ref))
            assert err < 1e-8

    def test_direct_inverse_roundtrip(self):
        rng = np.random.default_rng(9)
        for _ in range(100):
            la1, lo1 = rng.uniform(-1.4, 1.4), rng.uniform(-np.pi, np.pi)
            az = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(1e3, 1.5e7)
            wp = great_circle_direct(la1, lo1, az, dist)
            inv = great_circle_inverse(la1, lo1, wp.lat, wp.lon)
            assert inv.distance == pytest.approx(dist, abs=1e-2)
            err = min(abs(inv.azimuth1 - az), 2 * np.pi - abs(inv.azimuth1 - az))
            assert err < 1e-7

    def test_destination_point_matches_direct(self):
        wp1 = great_circle_direct(d(30), d(40), d(60), 2e6)
        wp2 = destination_point(d(30), d(40), d(60), 2e6 / EARTH_RADIUS)
        assert wp1.lat == pytest.approx(wp2.lat, abs=1e-12)
        assert wp1.lon == pytest.approx(wp2.lon, abs=1e-12)

    def test_waypoint_endpoints_and_fractions(self):
        la1, lo1, la2, lo2 = d(40.7), d(-74.0), d(51.5), d(-0.1)
        w0 = great_circle_waypoint(la1, lo1, la2, lo2, 0.0)
        w1 = great_circle_waypoint(la1, lo1, la2, lo2, 1.0)
        assert great_circle_distance(w0.lat, w0.lon, la1, lo1) < 1e-6
        assert great_circle_distance(w1.lat, w1.lon, la2, lo2) < 1e-6
        total = great_circle_distance(la1, lo1, la2, lo2)
        for f in (0.25, 0.5, 0.75):
            w = great_circle_waypoint(la1, lo1, la2, lo2, f)
            # Waypoint lies on the path at the right fraction
            assert great_circle_distance(la1, lo1, w.lat, w.lon) == pytest.approx(
                f * total, abs=1e-2
            )
            ct = cross_track_distance(w.lat, w.lon, la1, lo1, la2, lo2)
            assert abs(ct.cross_track) < 1e-2

    def test_waypoints_array(self):
        la1, lo1, la2, lo2 = 0.0, 0.0, d(45), d(45)
        lats, lons = great_circle_waypoints(la1, lo1, la2, lo2, 7)
        assert len(lats) == 7
        assert lats[0] == pytest.approx(la1, abs=1e-12)
        assert lats[-1] == pytest.approx(la2, abs=1e-12)
        assert lons[-1] == pytest.approx(lo2, abs=1e-12)

    def test_cross_track_sign_convention(self):
        # Northward path along lon 0; point to the east is right (+)
        ct = cross_track_distance(d(5), d(1), 0.0, 0.0, d(10), 0.0)
        assert ct.cross_track > 0
        ct = cross_track_distance(d(5), d(-1), 0.0, 0.0, d(10), 0.0)
        assert ct.cross_track < 0
        # Eastward path along equator; 1 deg north is left, magnitude R*1deg
        ct = cross_track_distance(d(1), d(5), 0.0, 0.0, 0.0, d(10))
        assert ct.cross_track == pytest.approx(-d(1) * EARTH_RADIUS, rel=1e-6)

    def test_along_track_for_on_path_point(self):
        la1, lo1, la2, lo2 = d(40.7), d(-74.0), d(51.5), d(-0.1)
        total = great_circle_distance(la1, lo1, la2, lo2)
        w = great_circle_waypoint(la1, lo1, la2, lo2, 0.3)
        ct = cross_track_distance(w.lat, w.lon, la1, lo1, la2, lo2)
        assert ct.along_track == pytest.approx(0.3 * total, abs=1e-2)

    def test_intersect_bearings_consistent(self):
        res = great_circle_intersect(0.0, 0.0, d(45), 0.0, d(10), d(315))
        assert res.valid
        b1 = great_circle_azimuth(0.0, 0.0, res.lat1, res.lon1)
        b2 = great_circle_azimuth(0.0, d(10), res.lat1, res.lon1)
        assert np.degrees(b1) == pytest.approx(45.0, abs=1e-6)
        assert np.degrees(b2) == pytest.approx(315.0, abs=1e-6)
        # Second point is antipodal to the first
        assert res.lat2 == pytest.approx(-res.lat1)

    def test_intersect_identical_circles_invalid(self):
        # Same great circle (equator) from two points heading east
        res = great_circle_intersect(0.0, 0.0, d(90), 0.0, d(10), d(90))
        assert not res.valid

    def test_path_intersect(self):
        res = great_circle_path_intersect(
            0.0, 0.0, d(10), d(10), 0.0, d(10), d(10), 0.0
        )
        assert res.valid
        assert np.degrees(res.lon1) == pytest.approx(5.0, abs=1e-6)
        assert 4.9 < np.degrees(res.lat1) < 5.2  # slightly above 5 (gc bulge)

    def test_angular_distance(self):
        assert angular_distance(0.0, 0.0, d(45), 0.0) == pytest.approx(d(45), abs=1e-10)

    def test_tdoa_recovers_emitter(self):
        c = 299792458.0
        em = (d(20), d(30))
        rec = [(d(10), d(10)), (d(40), d(20)), (d(15), d(50))]
        dists = [great_circle_distance(em[0], em[1], la, lo) for la, lo in rec]
        loc1, loc2 = great_circle_tdoa_loc(
            *rec[0],
            *rec[1],
            *rec[2],
            (dists[0] - dists[1]) / c,
            (dists[0] - dists[2]) / c,
        )
        assert loc1 is not None
        candidates = [loc1] + ([loc2] if loc2 is not None else [])
        best = min(
            great_circle_distance(loc.lat, loc.lon, em[0], em[1]) for loc in candidates
        )
        assert best < 100.0  # within 100 m on Earth-sized sphere


class TestRhumbReference:
    """Rhumb line vs analytic loxodrome and ODE integration."""

    @staticmethod
    def _lox_ode(bearing, dist, lat0, lon0, ellipsoidal):
        a, e2 = WGS84.a, WGS84.e2

        def f(_s, y):
            lat = y[0]
            if ellipsoidal:
                s2 = np.sin(lat) ** 2
                M = a * (1 - e2) / (1 - e2 * s2) ** 1.5
                N = a / np.sqrt(1 - e2 * s2)
            else:
                M = N = EARTH_RADIUS
            return [np.cos(bearing) / M, np.sin(bearing) / (N * np.cos(lat))]

        sol = solve_ivp(f, [0, dist], [lat0, lon0], rtol=1e-12, atol=1e-14)
        return sol.y[0][-1], sol.y[1][-1]

    @pytest.mark.parametrize(
        "lat1,lon1,brg,dist",
        [
            (d(40), d(-74), d(78), 5.8e6),
            (d(-30), d(100), d(200), 4e6),
            (d(10), 0.0, 0.0, 3e6),
            (d(10), 0.0, d(90), 3e6),
            (d(60), d(20), d(170), 2e6),
        ],
    )
    def test_direct_spherical_vs_ode(self, lat1, lon1, brg, dist):
        got = direct_rhumb_spherical(lat1, lon1, brg, dist)
        ref = self._lox_ode(brg, dist, lat1, lon1, ellipsoidal=False)
        assert got.lat == pytest.approx(ref[0], abs=1e-10)
        assert got.lon == pytest.approx(ref[1], abs=1e-10)

    def test_spherical_roundtrip(self):
        rng = np.random.default_rng(3)
        for _ in range(100):
            la1, lo1 = rng.uniform(-1.2, 1.2), rng.uniform(-np.pi, np.pi)
            la2, lo2 = rng.uniform(-1.2, 1.2), rng.uniform(-np.pi, np.pi)
            res = indirect_rhumb_spherical(la1, lo1, la2, lo2)
            dst = direct_rhumb_spherical(la1, lo1, res.bearing, res.distance)
            assert dst.lat == pytest.approx(la2, abs=1e-9)
            dlon = (dst.lon - lo2 + np.pi) % (2 * np.pi) - np.pi
            assert abs(dlon) * np.cos(la2) < 1e-9

    def test_distance_spherical_analytic(self):
        R = EARTH_RADIUS
        # Meridian
        assert rhumb_distance_spherical(d(10), d(5), d(50), d(5)) == pytest.approx(
            R * d(40), abs=1e-6
        )
        # Equator
        assert rhumb_distance_spherical(0.0, 0.0, 0.0, d(30)) == pytest.approx(
            R * d(30), abs=1e-6
        )
        # Along the 60N parallel
        assert rhumb_distance_spherical(d(60), 0.0, d(60), d(40)) == pytest.approx(
            R * np.cos(d(60)) * d(40), abs=1e-6
        )

    def test_bearing_analytic(self):
        def psi(la):
            return np.log(np.tan(np.pi / 4 + la / 2))

        ref = np.arctan2(d(10), psi(d(10)))
        assert rhumb_bearing(0.0, 0.0, d(10), d(10)) == pytest.approx(ref, abs=1e-12)
        # Due south
        assert rhumb_bearing(d(30), d(5), d(10), d(5)) == pytest.approx(np.pi)

    def test_bearing_dateline_wraparound(self):
        # From 170E to 170W at the same latitude: due east, not west
        b = rhumb_bearing(d(10), d(170), d(10), d(-170))
        assert b == pytest.approx(np.pi / 2, abs=1e-9)

    def test_direct_ellipsoidal_vs_ode(self):
        # Ellipsoidal direct uses a midpoint meridional-radius approximation;
        # verify it stays within a few hundred meters over multi-1000 km legs.
        for lat1, lon1, brg, dist in [
            (d(40), d(-74), d(78), 5.8e6),
            (d(-30), d(100), d(200), 4e6),
            (d(10), 0.0, 0.0, 3e6),
            (d(10), 0.0, d(90), 3e6),
        ]:
            got = direct_rhumb(lat1, lon1, brg, dist)
            ref = self._lox_ode(brg, dist, lat1, lon1, ellipsoidal=True)
            lat_err_m = abs(got.lat - ref[0]) * 6.4e6
            dlon = (got.lon - ref[1] + np.pi) % (2 * np.pi) - np.pi
            lon_err_m = abs(dlon) * 6.4e6 * np.cos(ref[0])
            assert lat_err_m < 500.0
            assert lon_err_m < 500.0

    def test_ellipsoidal_roundtrip(self):
        for la1, lo1, la2, lo2 in [
            (d(40), d(-74), d(51), 0.0),
            (d(-35), d(20), d(10), d(100)),
            (d(30), 0.0, d(30.00001), d(40)),
        ]:
            res = indirect_rhumb(la1, lo1, la2, lo2)
            dst = direct_rhumb(la1, lo1, res.bearing, res.distance)
            assert dst.lat == pytest.approx(la2, abs=1e-8)
            dlon = (dst.lon - lo2 + np.pi) % (2 * np.pi) - np.pi
            assert abs(dlon) < 1e-8

    def test_distance_ellipsoidal_moderate_leg(self):
        # NYC->London leg: reference distance = meridian arc / cos(bearing)
        from scipy.integrate import quad

        a, e2 = WGS84.a, WGS84.e2
        la1, lo1, la2, lo2 = d(40), d(-74), d(51), 0.0
        res = indirect_rhumb(la1, lo1, la2, lo2)
        arc, _ = quad(
            lambda la: a * (1 - e2) / (1 - e2 * np.sin(la) ** 2) ** 1.5, la1, la2
        )
        ref = abs(arc / np.cos(res.bearing))
        assert rhumb_distance_ellipsoidal(la1, lo1, la2, lo2) == pytest.approx(
            ref, abs=10.0
        )

    def test_rhumb_intersect_lies_on_both_lines(self):
        la1, lo1, b1 = d(10), 0.0, d(45)
        la2, lo2, b2 = d(10), d(20), d(315)
        res = rhumb_intersect(la1, lo1, b1, la2, lo2, b2)
        assert res.valid
        assert np.degrees(rhumb_bearing(la1, lo1, res.lat, res.lon)) == pytest.approx(
            45.0, abs=1e-6
        )
        assert np.degrees(rhumb_bearing(la2, lo2, res.lat, res.lon)) == pytest.approx(
            315.0, abs=1e-6
        )

    def test_rhumb_intersect_meridional(self):
        res = rhumb_intersect(d(10), d(5), 0.0, d(20), 0.0, d(90))
        assert res.valid
        assert np.degrees(res.lon) == pytest.approx(5.0, abs=1e-9)
        assert np.degrees(res.lat) == pytest.approx(20.0, abs=1e-9)

    def test_rhumb_intersect_parallel_invalid(self):
        res = rhumb_intersect(d(10), 0.0, d(45), d(20), d(10), d(45))
        assert not res.valid

    def test_midpoint_and_waypoints(self):
        mid = rhumb_midpoint(0.0, 0.0, d(10), d(10))
        assert np.degrees(mid.lat) == pytest.approx(5.0, abs=0.01)
        lats, lons = rhumb_waypoints(d(40), d(-74), d(51), 0.0, 5)
        assert len(lats) == 5
        assert lats[0] == pytest.approx(d(40), abs=1e-12)
        assert lats[-1] == pytest.approx(d(51), abs=1e-9)
        assert lons[-1] == pytest.approx(0.0, abs=1e-9)
        # Interior waypoints share the constant bearing
        b = rhumb_bearing(d(40), d(-74), d(51), 0.0)
        for i in range(1, 5):
            assert rhumb_bearing(lats[0], lons[0], lats[i], lons[i]) == pytest.approx(
                b, abs=1e-6
            )

    def test_compare_great_circle_rhumb(self):
        gc, rmb, diff = compare_great_circle_rhumb(d(40), d(-74), d(51), 0.0)
        assert rmb >= gc
        assert diff == pytest.approx((rmb - gc) / gc * 100)


class TestINSReference:
    """INS gravity/rates vs WGS84 analytic values; strapdown properties."""

    def test_normal_gravity_wgs84(self):
        assert normal_gravity(0.0) == pytest.approx(9.7803253359, abs=1e-9)
        assert normal_gravity(np.pi / 2) == pytest.approx(9.8321849378, abs=1e-9)
        # 45 deg: known WGS84 value ~9.806199
        assert normal_gravity(d(45)) == pytest.approx(9.8061992, abs=1e-5)
        # Free-air gradient ~ -3.086e-6 /m (relative tolerance: first-order model)
        dg = normal_gravity(d(45), 1000.0) - normal_gravity(d(45))
        assert dg == pytest.approx(-3.086e-3, rel=0.01)

    def test_radii_of_curvature_analytic(self):
        a, e2 = WGS84.a, WGS84.e2
        RN, RE = radii_of_curvature(0.0)
        assert RN == pytest.approx(a * (1 - e2), rel=1e-12)
        assert RE == pytest.approx(a, rel=1e-12)
        RN90, RE90 = radii_of_curvature(np.pi / 2)
        assert RN90 == pytest.approx(a / np.sqrt(1 - e2), rel=1e-12)
        assert RE90 == pytest.approx(a / np.sqrt(1 - e2), rel=1e-12)

    def test_earth_rate_and_transport_rate(self):
        lat = d(45)
        np.testing.assert_allclose(
            earth_rate_ned(lat),
            [OMEGA_EARTH * np.cos(lat), 0.0, -OMEGA_EARTH * np.sin(lat)],
            rtol=1e-12,
        )
        RN, RE = radii_of_curvature(lat)
        np.testing.assert_allclose(
            transport_rate_ned(lat, 0.0, 100.0, 50.0),
            [50.0 / RE, -100.0 / RN, -50.0 * np.tan(lat) / RE],
            rtol=1e-12,
        )

    def test_skew_symmetric(self):
        v = np.array([1.0, 2.0, 3.0])
        S = skew_symmetric(v)
        np.testing.assert_allclose(S.T, -S)
        w = np.array([-0.4, 0.5, 2.0])
        np.testing.assert_allclose(S @ w, np.cross(v, w), atol=1e-15)

    def test_coning_zero_for_coplanar_rotation(self):
        # Constant rotation axis (coplanar angular rate): no coning
        g1 = np.array([0.01, 0.02, 0.0])
        g2 = 2.0 * g1
        np.testing.assert_allclose(coning_correction(g1, g2), 0.0, atol=1e-18)

    def test_sculling_zero_for_parallel_motion(self):
        g1 = np.array([0.01, 0.02, 0.0])
        np.testing.assert_allclose(
            sculling_correction(g1, 2 * g1, g1, 2 * g1), 0.0, atol=1e-18
        )

    def test_compensate_imu_constant_inputs(self):
        accel = np.array([0.1, 0.0, -9.8])
        gyro = np.array([0.0, 0.0, 0.01])
        dtheta, dv = compensate_imu_data(accel, accel, gyro, gyro, 0.01)
        np.testing.assert_allclose(dtheta, gyro * 0.01, atol=1e-15)
        # Rotation compensation term 0.5*dtheta x dv_raw is the only extra
        expected = accel * 0.01 + 0.5 * np.cross(gyro * 0.01, accel * 0.01)
        np.testing.assert_allclose(dv, expected, atol=1e-15)

    def test_update_quaternion_constant_rate_analytic(self):
        q = np.array([1.0, 0.0, 0.0, 0.0])
        w = np.array([0.0, 0.0, 0.1])
        dt = 0.01
        for _ in range(1000):
            q = update_quaternion(q, w * dt)
        R = quat2rotmat(q)
        c, s = np.cos(1.0), np.sin(1.0)
        ref = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(R, ref, atol=1e-12)

    def test_update_attitude_ned_tracks_nav_frame(self):
        # Body rotating exactly with the nav frame: attitude unchanged
        lat = d(45)
        st = initialize_ins_state(lat, 0.0, 0.0, yaw=d(30))
        omega_in_n = earth_rate_ned(lat)
        Rbn = quat2rotmat(st.quaternion)
        omega_ib_b = Rbn.T @ omega_in_n
        q_new = update_attitude_ned(st.quaternion, omega_ib_b, omega_in_n, 1.0)
        np.testing.assert_allclose(q_new, st.quaternion, atol=1e-12)

    def test_stationary_mechanization_zero_drift(self):
        # Perfect IMU on stationary, level platform: no drift at all
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        state = initialize_ins_state(lat0, lon0, alt0)
        g = normal_gravity(lat0, alt0)
        imu = IMUData(
            accel=np.array([0.0, 0.0, -g]),
            gyro=earth_rate_ned(lat0),
            dt=0.01,
        )
        for _ in range(1000):  # 10 s
            state = mechanize_ins_ned(state, imu)
        assert abs(state.position[0] - lat0) * 6.4e6 < 1e-3
        assert abs(state.position[1] - lon0) * 6.4e6 < 1e-3
        assert abs(state.position[2] - alt0) < 1e-3
        np.testing.assert_allclose(state.velocity, 0.0, atol=1e-6)
        assert state.time == pytest.approx(10.0)

    def test_stationary_mechanization_with_coning_path(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        state = initialize_ins_state(lat0, lon0, alt0)
        g = normal_gravity(lat0, alt0)
        accel = np.array([0.0, 0.0, -g])
        gyro = earth_rate_ned(lat0)
        imu = IMUData(accel=accel, gyro=gyro, dt=0.01)
        for _ in range(500):
            state = mechanize_ins_ned(state, imu, accel_prev=accel, gyro_prev=gyro)
        assert abs(state.position[2] - alt0) < 1e-3
        # The rotation-compensation term overlaps slightly with the average-DCM
        # transform, leaving a bounded ~0.25 microgravity artifact (dt-scaled).
        np.testing.assert_allclose(state.velocity, 0.0, atol=1e-4)

    def test_mechanization_constant_climb(self):
        # Constant upward velocity, no horizontal motion
        lat0, lon0, alt0 = d(10), 0.0, 0.0
        state = initialize_ins_state(lat0, lon0, alt0, vD=-10.0)
        for _ in range(100):
            g = normal_gravity(state.position[0], state.position[2])
            imu = IMUData(
                accel=np.array([0.0, 0.0, -g]),
                gyro=earth_rate_ned(state.position[0]),
                dt=0.01,
            )
            state = mechanize_ins_ned(state, imu)
        # After 1 s at 10 m/s climb
        assert state.position[2] == pytest.approx(10.0, abs=0.05)

    def test_coarse_alignment_recovers_tilt(self):
        roll_t, pitch_t = d(5), d(10)
        st = initialize_ins_state(0.0, 0.0, 0.0, roll=roll_t, pitch=pitch_t, yaw=0.7)
        f_b = quat2rotmat(st.quaternion).T @ np.array([0.0, 0.0, -9.81])
        r, p = coarse_alignment(f_b, 0.0)
        assert r == pytest.approx(roll_t, abs=1e-10)
        assert p == pytest.approx(pitch_t, abs=1e-10)

    @pytest.mark.parametrize(
        "yaw_true", [0.0, 30.0, 45.0, 90.0, 135.0, 180.0, 270.0, -30.0]
    )
    def test_gyrocompass_recovers_heading(self, yaw_true):
        lat = d(45)
        st = initialize_ins_state(lat, 0.0, 0.0, yaw=d(yaw_true))
        gyro_b = quat2rotmat(st.quaternion).T @ earth_rate_ned(lat)
        yaw_est = gyrocompass_alignment(gyro_b, 0.0, 0.0, lat)
        err = (np.degrees(yaw_est) - yaw_true + 180.0) % 360.0 - 180.0
        assert abs(err) < 1e-6

    def test_gyrocompass_with_tilt(self):
        lat = d(45)
        for yaw_true in (45.0, 120.0, -60.0):
            st = initialize_ins_state(
                lat, 0.0, 0.0, roll=d(3), pitch=d(-7), yaw=d(yaw_true)
            )
            gyro_b = quat2rotmat(st.quaternion).T @ earth_rate_ned(lat)
            yaw_est = gyrocompass_alignment(gyro_b, d(3), d(-7), lat)
            err = (np.degrees(yaw_est) - yaw_true + 180.0) % 360.0 - 180.0
            assert abs(err) < 1e-6


class TestGNSSReference:
    """DOP vs hand-computed geometry; INS/GNSS integration properties."""

    def test_compute_dop_hand_computed(self):
        # Symmetric constellation in ENU: 4 sats at 45 deg elevation N/E/S/W
        # plus one at zenith. Q = inv(H^T H) computed by hand via numpy.
        el = d(45)
        us = [
            [np.sin(d(az)) * np.cos(el), np.cos(d(az)) * np.cos(el), np.sin(el)]
            for az in (0, 90, 180, 270)
        ]
        us.append([0.0, 0.0, 1.0])
        H = np.hstack([-np.array(us), np.ones((5, 1))])
        gdop, pdop, hdop, vdop = compute_dop(H)
        Q = np.linalg.inv(H.T @ H)
        assert gdop == pytest.approx(np.sqrt(np.trace(Q)), rel=1e-12)
        assert pdop == pytest.approx(np.sqrt(Q[0, 0] + Q[1, 1] + Q[2, 2]), rel=1e-12)
        assert hdop == pytest.approx(np.sqrt(Q[0, 0] + Q[1, 1]), rel=1e-12)
        assert vdop == pytest.approx(np.sqrt(Q[2, 2]), rel=1e-12)
        # By symmetry the 4-corner constellation gives HDOP = sqrt(2)
        assert hdop == pytest.approx(np.sqrt(2), rel=1e-9)
        assert gdop**2 == pytest.approx(pdop**2 + Q[3, 3], rel=1e-9)

    def test_compute_dop_singular(self):
        # Coplanar constellation: singular geometry -> inf
        H = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]])
        gdop, pdop, hdop, vdop = compute_dop(H)
        assert not np.isfinite(gdop) or gdop > 1e6

    def test_line_of_sight(self):
        los, rng = compute_line_of_sight([0.0, 0.0, 0.0], [3.0, 4.0, 0.0])
        np.testing.assert_allclose(los, [0.6, 0.8, 0.0])
        assert rng == 5.0

    def test_satellite_elevation_azimuth(self):
        lat0, lon0 = d(45), d(-75)
        # Directly overhead
        xs, ys, zs = geodetic_to_ecef(lat0, lon0, 20200e3)
        el, az = satellite_elevation_azimuth([lat0, lon0, 0.0], [xs, ys, zs])
        assert np.degrees(el) == pytest.approx(90.0, abs=1e-6)
        # Due north, 45 deg elevation
        xs, ys, zs = enu_to_ecef(0.0, 1000e3, 1000e3, lat0, lon0, 0.0)
        el, az = satellite_elevation_azimuth(
            [lat0, lon0, 0.0], [float(xs), float(ys), float(zs)]
        )
        assert np.degrees(el) == pytest.approx(45.0, abs=1e-9)
        assert np.degrees(az) == pytest.approx(0.0, abs=1e-9)
        # Due east on horizon
        xs, ys, zs = enu_to_ecef(1000e3, 0.0, 0.0, lat0, lon0, 0.0)
        el, az = satellite_elevation_azimuth(
            [lat0, lon0, 0.0], [float(xs), float(ys), float(zs)]
        )
        assert np.degrees(el) == pytest.approx(0.0, abs=1e-9)
        assert np.degrees(az) == pytest.approx(90.0, abs=1e-9)

    def test_measurement_matrices_structure(self):
        Hp = position_measurement_matrix()
        Hv = velocity_measurement_matrix()
        Hpv = position_velocity_measurement_matrix()
        np.testing.assert_array_equal(Hp[:, :3], np.eye(3))
        assert not Hp[:, 3:].any()
        np.testing.assert_array_equal(Hv[:, 3:6], np.eye(3))
        assert not Hv[:, :3].any() and not Hv[:, 6:].any()
        np.testing.assert_array_equal(Hpv[:, :6], np.eye(6))
        assert not Hpv[:, 6:].any()

    @staticmethod
    def _make_sats(user_ecef, lat0, lon0, true_ecef=None, clock=0.0, n=6, seed=11):
        rng = np.random.default_rng(seed)
        ref = true_ecef if true_ecef is not None else user_ecef
        sats = []
        for i in range(n):
            e = rng.uniform(-2e7, 2e7)
            no = rng.uniform(-2e7, 2e7)
            u = rng.uniform(1e7, 2.5e7)
            sx, sy, sz = enu_to_ecef(e, no, u, lat0, lon0, 0.0)
            spos = np.array([float(sx), float(sy), float(sz)])
            _, r = compute_line_of_sight(ref, spos)
            sats.append(
                SatelliteInfo(
                    prn=i,
                    position=spos,
                    velocity=np.zeros(3),
                    pseudorange=r + clock,
                )
            )
        return sats

    def test_pseudorange_matrix_matches_numerical_jacobian(self):
        lat0, lon0 = d(45), d(-75)
        x, y, z = geodetic_to_ecef(lat0, lon0, 0.0)
        user = np.array([float(x), float(y), float(z)])
        sats = self._make_sats(user, lat0, lon0)
        H = pseudorange_measurement_matrix(user, sats)
        assert H.shape == (len(sats), 4)
        np.testing.assert_allclose(H[:, 3], 1.0)
        eps = 1.0
        for i, sat in enumerate(sats):
            _, r0 = compute_line_of_sight(user, sat.position)
            for k in range(3):
                up = user.copy()
                up[k] += eps
                _, rp = compute_line_of_sight(up, sat.position)
                assert H[i, k] == pytest.approx((rp - r0) / eps, abs=1e-6)

    def test_tight_coupled_H_matches_numerical_jacobian(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0)
        x, y, z = geodetic_to_ecef(lat0, lon0, alt0)
        user = np.array([float(x), float(y), float(z)])
        sats = self._make_sats(user, lat0, lon0, n=4)
        H = tight_coupled_measurement_matrix(st, sats)
        assert H.shape == (4, 17)

        # Numerical d(range)/d(lat, lon, alt)
        def ranges(lat, lon, alt):
            xx, yy, zz = geodetic_to_ecef(lat, lon, alt)
            u = np.array([float(xx), float(yy), float(zz)])
            return np.array([compute_line_of_sight(u, s.position)[1] for s in sats])

        base = ranges(lat0, lon0, alt0)
        eps_ang, eps_alt = 1e-8, 1e-2
        num_lat = (ranges(lat0 + eps_ang, lon0, alt0) - base) / eps_ang
        num_lon = (ranges(lat0, lon0 + eps_ang, alt0) - base) / eps_ang
        num_alt = (ranges(lat0, lon0, alt0 + eps_alt) - base) / eps_alt
        # The implementation uses a simplified Jacobian (N, ignoring alt and
        # dN/dlat ~ 21 km/rad at 45 deg): agree to ~0.4% of the LOS scale.
        scale = 6.4e6
        np.testing.assert_allclose(H[:, 0], num_lat, atol=0.004 * scale)
        np.testing.assert_allclose(H[:, 1], num_lon, atol=0.004 * scale)
        np.testing.assert_allclose(H[:, 2], num_alt, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(H[:, 15], 1.0)

    def test_loose_coupled_position_update_pulls_to_gnss(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0, position_std=10.0)
        dlat = 50.0 / 6.4e6
        # A GNSS fix good to a millimeter, expressed in the error-state units
        # the filter actually uses: [rad, rad, m]. This used to read
        # `np.eye(3) * 1e-12`, which was negligible against a position
        # covariance wrongly carrying meters-squared on the radian diagonal.
        # Once the units agree (gh-19), 1e-12 rad^2 is a 1.6 m uncertainty --
        # comparable to the filter's own 10 m -- and the fix stops dominating.
        millimeter = position_std_to_error_state_units(1e-3, lat0, alt0)
        gnss = GNSSMeasurement(
            position=np.array([lat0 + dlat, lon0, alt0]),
            velocity=None,
            position_cov=np.diag(millimeter**2),
            velocity_cov=None,
            time=0.0,
        )
        res = loose_coupled_update_position(st, gnss)
        moved = (res.state.ins_state.position[0] - lat0) / dlat
        assert moved == pytest.approx(1.0, abs=1e-6)
        # Closed-loop reset
        np.testing.assert_allclose(res.state.error_state, 0.0)

    def test_loose_coupled_velocity_update(self):
        ins0 = initialize_ins_state(d(45), d(-75), 100.0)
        st = initialize_ins_gnss(ins0)
        gnss = GNSSMeasurement(
            position=None,
            velocity=np.array([1.0, 0.0, 0.0]),
            position_cov=None,
            velocity_cov=np.eye(3) * 1e-6,
            time=0.0,
        )
        res = loose_coupled_update_velocity(st, gnss)
        assert res.state.ins_state.velocity[0] == pytest.approx(1.0, abs=1e-4)

    def test_loose_coupled_combined_update(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0)
        dlat = 20.0 / 6.4e6
        # Millimeter-accurate fix in error-state units; see the note in
        # test_loose_coupled_position_update_pulls_to_gnss (gh-19).
        millimeter = position_std_to_error_state_units(1e-3, lat0, alt0)
        gnss = GNSSMeasurement(
            position=np.array([lat0 + dlat, lon0, alt0 + 5.0]),
            velocity=np.array([0.5, -0.5, 0.0]),
            position_cov=np.diag(millimeter**2),
            velocity_cov=np.eye(3) * 1e-8,
            time=0.0,
        )
        res = loose_coupled_update(st, gnss)
        assert res.state.ins_state.position[0] == pytest.approx(
            lat0 + dlat, abs=dlat * 1e-3
        )
        assert res.state.ins_state.position[2] == pytest.approx(alt0 + 5.0, abs=0.01)
        np.testing.assert_allclose(
            res.state.ins_state.velocity, [0.5, -0.5, 0.0], atol=1e-3
        )

    def test_loose_coupled_invalid_measurement_noop(self):
        ins0 = initialize_ins_state(d(45), d(-75), 100.0)
        st = initialize_ins_gnss(ins0)
        gnss = GNSSMeasurement(
            position=None,
            velocity=None,
            position_cov=None,
            velocity_cov=None,
            time=0.0,
            valid=False,
        )
        res = loose_coupled_update(st, gnss)
        assert res.state.ins_state is ins0

    def test_loose_coupled_predict_stationary(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0)
        g = normal_gravity(lat0, alt0)
        imu = IMUData(accel=np.array([0.0, 0.0, -g]), gyro=earth_rate_ned(lat0), dt=0.1)
        stp = loose_coupled_predict(st, imu)
        # Covariance grows during prediction; INS solution stays put
        assert np.trace(stp.error_cov) > np.trace(st.error_cov)
        np.testing.assert_allclose(stp.ins_state.position, ins0.position, atol=1e-12)

    def test_tight_coupled_update_recovers_position_and_clock(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0, position_std=10.0)
        true_lat = lat0 + 30.0 / 6.4e6
        tx, ty, tz = geodetic_to_ecef(true_lat, lon0, alt0)
        true_ecef = np.array([float(tx), float(ty), float(tz)])
        sats = self._make_sats(
            true_ecef, lat0, lon0, true_ecef=true_ecef, clock=1000.0, n=8, seed=5
        )
        res = tight_coupled_update(st, sats, pseudorange_std=0.1)
        err_north = abs(true_lat - res.state.ins_state.position[0]) * 6.4e6
        assert err_north < 0.1  # from 30 m initial error to < 10 cm
        assert res.state.clock_bias == pytest.approx(1000.0, abs=0.1)
        assert all(np.isfinite(res.dop))

    def test_tight_coupled_too_few_sats(self):
        ins0 = initialize_ins_state(d(45), d(-75), 100.0)
        st = initialize_ins_gnss(ins0)
        x, y, z = geodetic_to_ecef(d(45), d(-75), 0.0)
        user = np.array([float(x), float(y), float(z)])
        sats = self._make_sats(user, d(45), d(-75), n=3)
        res = tight_coupled_update(st, sats)
        assert res.state.ins_state is ins0
        assert not np.isfinite(res.dop[0])

    def test_tight_coupled_innovation_zero_at_truth(self):
        lat0, lon0, alt0 = d(45), d(-75), 100.0
        ins0 = initialize_ins_state(lat0, lon0, alt0)
        st = initialize_ins_gnss(ins0)
        x, y, z = geodetic_to_ecef(lat0, lon0, alt0)
        user = np.array([float(x), float(y), float(z)])
        sats = self._make_sats(user, lat0, lon0, n=5)
        innov, pred = tight_coupled_pseudorange_innovation(st, sats)
        np.testing.assert_allclose(innov, 0.0, atol=1e-6)
        assert np.all(pred > 1e7)  # plausible satellite ranges

    def test_gnss_outage_detection(self):
        # NIS below/above chi2 threshold
        assert not gnss_outage_detection(np.array([0.1, 0.1]), np.eye(2))
        assert gnss_outage_detection(np.array([10.0, 10.0]), np.eye(2))
        # Singular covariance flags a fault
        assert gnss_outage_detection(np.array([1.0, 1.0]), np.zeros((2, 2)))
