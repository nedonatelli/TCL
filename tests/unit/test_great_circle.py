"""Tests for great circle navigation functions.

Agreement is asserted only for the listed city pairs, start points and
bearings (mid-latitude and southern-hemisphere routes up to ~20,000 km),
at 1e-9 relative for distances and 1e-9 rad for angles; nothing is claimed
for points within a few metres of a pole.

The oracle is geographiclib's geodesic solver on a sphere (flattening 0,
radius equal to the module's mean Earth radius), whose direct and inverse
solutions are accurate to nanometres. Cross-track distances use vector
algebra instead, since geographiclib has no such function.
"""

import numpy as np
import pytest
from geographiclib.geodesic import Geodesic

from pytcl.core.constants import EARTH_MEAN_RADIUS
from pytcl.navigation.great_circle import (
    GreatCircleResult,
    angular_distance,
    cross_track_distance,
    destination_point,
    great_circle_azimuth,
    great_circle_direct,
    great_circle_distance,
    great_circle_intersect,
    great_circle_inverse,
    great_circle_path_intersect,
    great_circle_waypoint,
    great_circle_waypoints,
)

R = EARTH_MEAN_RADIUS
SPHERE = Geodesic(R, 0.0)

NYC = (40.7128, -74.0060)
LONDON = (51.5074, -0.1278)
SYDNEY = (-33.8688, 151.2093)
SANTIAGO = (-33.4489, -70.6693)


def _rad(point):
    return np.radians(point[0]), np.radians(point[1])


def _inverse(p1, p2, radius=R):
    """geographiclib sphere inverse: distance (m), azimuths (rad, [0, 2pi))."""
    out = Geodesic(radius, 0.0).Inverse(*p1, *p2)
    return (
        out["s12"],
        np.radians(out["azi1"]) % (2 * np.pi),
        np.radians(out["azi2"]) % (2 * np.pi),
    )


def _angle_error(a, b):
    """Smallest absolute difference between two angles."""
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _direct(p1, azimuth_rad, distance, radius=R):
    out = Geodesic(radius, 0.0).Direct(*p1, np.degrees(azimuth_rad), distance)
    return np.radians(out["lat2"]), np.radians(out["lon2"])


PAIRS = [
    pytest.param(NYC, LONDON, id="nyc-london"),
    pytest.param(LONDON, SYDNEY, id="london-sydney"),
    pytest.param(SYDNEY, SANTIAGO, id="sydney-santiago"),
    pytest.param((10.0, 20.0), (-35.0, -160.0), id="south-pacific"),
]


class TestGreatCircleDistance:
    @pytest.mark.parametrize("p1,p2", PAIRS)
    def test_matches_geographiclib(self, p1, p2):
        dist = great_circle_distance(*_rad(p1), *_rad(p2))

        assert dist == pytest.approx(_inverse(p1, p2)[0], rel=1e-9, abs=0)

    def test_zero_distance(self):
        lat, lon = np.radians(40), np.radians(-74)
        assert great_circle_distance(lat, lon, lat, lon) < 1e-6

    def test_antipodal_points_are_half_circumference(self):
        dist = great_circle_distance(0.0, 0.0, 0.0, np.pi)
        assert dist == pytest.approx(np.pi * R, rel=1e-12, abs=0)

    def test_poles_are_half_circumference(self):
        dist = great_circle_distance(np.pi / 2, 0.0, -np.pi / 2, 0.0)
        assert dist == pytest.approx(np.pi * R, rel=1e-12, abs=0)

    def test_custom_radius(self):
        dist = great_circle_distance(0.0, 0.0, 0.0, np.pi / 2, radius=1000.0)
        assert dist == pytest.approx(1000.0 * np.pi / 2, rel=1e-12, abs=0)


class TestGreatCircleAzimuth:
    @pytest.mark.parametrize("p1,p2", PAIRS)
    def test_matches_geographiclib(self, p1, p2):
        az = great_circle_azimuth(*_rad(p1), *_rad(p2))

        assert _angle_error(az, _inverse(p1, p2)[1]) < 1e-9

    @pytest.mark.parametrize(
        "p1,p2,expected",
        [
            ((40.0, -74.0), (50.0, -74.0), 0.0),
            ((0.0, 0.0), (0.0, 10.0), np.pi / 2),
            ((40.0, -74.0), (30.0, -74.0), np.pi),
            ((0.0, 0.0), (0.0, -10.0), 3 * np.pi / 2),
        ],
        ids=["north", "east", "south", "west"],
    )
    def test_cardinal_directions(self, p1, p2, expected):
        assert _angle_error(great_circle_azimuth(*_rad(p1), *_rad(p2)), expected) < 1e-9

    def test_normalized_range(self):
        az = great_circle_azimuth(*_rad(NYC), *_rad(LONDON))
        assert 0 <= az < 2 * np.pi


class TestGreatCircleInverse:
    @pytest.mark.parametrize("p1,p2", PAIRS)
    def test_matches_geographiclib(self, p1, p2):
        result = great_circle_inverse(*_rad(p1), *_rad(p2))
        s12, azi1, azi2 = _inverse(p1, p2)

        assert isinstance(result, GreatCircleResult)
        assert result.distance == pytest.approx(s12, rel=1e-9, abs=0)
        assert _angle_error(result.azimuth1, azi1) < 1e-9
        assert _angle_error(result.azimuth2, azi2) < 1e-9


class TestGreatCircleWaypoint:
    @pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_matches_geographiclib_direct(self, fraction):
        wp = great_circle_waypoint(*_rad(NYC), *_rad(LONDON), fraction)
        s12, azi1, _ = _inverse(NYC, LONDON)
        lat, lon = _direct(NYC, azi1, fraction * s12)

        assert abs(wp.lat - lat) < 1e-9
        assert _angle_error(wp.lon, lon) < 1e-9


class TestGreatCircleWaypoints:
    def test_waypoints_match_geographiclib_direct(self):
        lats, lons = great_circle_waypoints(*_rad(NYC), *_rad(LONDON), 6)
        s12, azi1, _ = _inverse(NYC, LONDON)

        assert lats.shape == (6,)
        assert lons.shape == (6,)
        for i, fraction in enumerate(np.linspace(0.0, 1.0, 6)):
            lat, lon = _direct(NYC, azi1, fraction * s12)
            assert abs(lats[i] - lat) < 1e-9
            assert _angle_error(lons[i], lon) < 1e-9


class TestGreatCircleDirect:
    @pytest.mark.parametrize(
        "start,azimuth_deg,distance",
        [
            ((40.0, -74.0), 45.0, 1_000_000.0),
            ((40.0, -74.0), 0.0, 1_000_000.0),
            ((0.0, 0.0), 90.0, 1_000_000.0),
            ((-33.0, 151.0), 250.0, 8_000_000.0),
            ((40.0, -74.0), 45.0, 0.0),
        ],
    )
    def test_matches_geographiclib(self, start, azimuth_deg, distance):
        dest = great_circle_direct(*_rad(start), np.radians(azimuth_deg), distance)
        lat, lon = _direct(start, np.radians(azimuth_deg), distance)

        assert abs(dest.lat - lat) < 1e-9
        assert _angle_error(dest.lon, lon) < 1e-9

    def test_custom_radius(self):
        dest = great_circle_direct(0.0, 0.0, np.pi / 2, 500.0, radius=1000.0)

        assert abs(dest.lat) < 1e-12
        assert dest.lon == pytest.approx(0.5, rel=1e-12, abs=0)


def _cross_track_oracle(point, start, end):
    """Signed (right-positive) cross-track and along-track distances in metres,
    from unit-vector algebra: the path's pole n = p1 x p2, point p."""

    def unit(lat_lon):
        lat, lon = np.radians(lat_lon[0]), np.radians(lat_lon[1])
        return np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )

    p1, p2, p = unit(start), unit(end), unit(point)
    pole = np.cross(p1, p2)
    pole /= np.linalg.norm(pole)
    heading = np.cross(pole, p1)
    cross = -np.arcsin(np.dot(p, pole)) * R
    along = np.arctan2(np.dot(p, heading), np.dot(p, p1)) * R
    return cross, along


class TestCrossTrackDistance:
    @pytest.mark.parametrize(
        "point",
        [
            (1.0, 5.0),
            (-1.0, 5.0),
            (0.0, 5.0),
            (2.0, 12.0),
            (0.5, -3.0),
        ],
    )
    def test_matches_vector_algebra_on_equatorial_path(self, point):
        start, end = (0.0, 0.0), (0.0, 10.0)

        result = cross_track_distance(*_rad(point), *_rad(start), *_rad(end))
        cross, along = _cross_track_oracle(point, start, end)

        assert abs(result.cross_track - cross) < 1e-3
        assert abs(result.along_track - along) < 1e-3

    def test_oblique_path(self):
        point = (45.0, -40.0)

        result = cross_track_distance(*_rad(point), *_rad(NYC), *_rad(LONDON))
        cross, along = _cross_track_oracle(point, NYC, LONDON)

        assert abs(result.cross_track - cross) < 1e-3
        assert abs(result.along_track - along) < 1e-3

    def test_one_degree_north_of_equator_is_left_and_111_km(self):
        """Eastward along the equator, a point at 1 deg N is left of track:
        cross-track -R * 1 deg in radians."""
        result = cross_track_distance(
            *_rad((1.0, 5.0)), *_rad((0.0, 0.0)), *_rad((0.0, 10.0))
        )

        assert result.cross_track == pytest.approx(
            -R * np.radians(1.0), rel=1e-9, abs=0
        )

    def test_foot_of_perpendicular_is_cross_track_away(self):
        """Walk the along-track distance with geographiclib, then the point
        lies exactly |cross-track| from the foot."""
        point = (45.0, -40.0)
        result = cross_track_distance(*_rad(point), *_rad(NYC), *_rad(LONDON))
        _, azi1, _ = _inverse(NYC, LONDON)
        lat, lon = _direct(NYC, azi1, result.along_track)

        foot_to_point = SPHERE.Inverse(np.degrees(lat), np.degrees(lon), *point)["s12"]

        assert foot_to_point == pytest.approx(abs(result.cross_track), rel=1e-9, abs=0)


class TestGreatCircleIntersect:
    def test_meridian_crosses_equator(self):
        """Eastward along the equator from 20W meets southward from 40N, 0E."""
        result = great_circle_intersect(
            *_rad((0.0, -20.0)), np.pi / 2, *_rad((40.0, 0.0)), np.pi
        )

        assert result.valid
        assert abs(result.lat1) < 1e-12
        assert abs(result.lon1) < 1e-12
        assert abs(result.lat2) < 1e-12
        assert _angle_error(result.lon2, np.pi) < 1e-12

    def test_generic_intersection_lies_on_both_geodesics(self):
        p1, az1 = (10.0, -20.0), np.radians(60.0)
        p2, az2 = (40.0, 30.0), np.radians(200.0)

        result = great_circle_intersect(*_rad(p1), az1, *_rad(p2), az2)
        hit = (np.degrees(result.lat1), np.degrees(result.lon1))

        assert result.valid
        for start, azimuth in ((p1, az1), (p2, az2)):
            s12, bearing, _ = _inverse(start, hit)
            assert _angle_error(bearing, azimuth) < 1e-9
            lat, lon = _direct(start, azimuth, s12)
            assert abs(lat - result.lat1) < 1e-9
            assert _angle_error(lon, result.lon1) < 1e-9

    def test_second_point_is_antipode_of_first(self):
        result = great_circle_intersect(
            *_rad((10.0, -20.0)),
            np.radians(60.0),
            *_rad((40.0, 30.0)),
            np.radians(200.0),
        )

        assert result.lat2 == pytest.approx(-result.lat1, rel=1e-12, abs=0)
        assert _angle_error(result.lon2, result.lon1 + np.pi) < 1e-12

    def test_meridians_meet_at_the_pole(self):
        result = great_circle_intersect(*_rad((0.0, 0.0)), 0.0, *_rad((0.0, 10.0)), 0.0)

        assert result.valid
        assert result.lat1 == pytest.approx(np.pi / 2, rel=1e-6, abs=0)
        assert result.lat2 == pytest.approx(-np.pi / 2, rel=1e-6, abs=0)

    def test_identical_great_circles_are_invalid(self):
        result = great_circle_intersect(
            *_rad((0.0, 0.0)), np.pi / 2, *_rad((0.0, 30.0)), np.pi / 2
        )

        assert not result.valid


class TestGreatCirclePathIntersect:
    def test_crossing_paths(self):
        result = great_circle_path_intersect(
            *_rad((0.0, -10.0)),
            *_rad((0.0, 10.0)),
            *_rad((-10.0, 0.0)),
            *_rad((10.0, 0.0)),
        )

        assert result.valid
        assert abs(result.lat1) < 1e-12
        assert abs(result.lon1) < 1e-12

    def test_oblique_paths_lie_on_both_geodesics(self):
        a1, a2 = NYC, LONDON
        b1, b2 = (30.0, -50.0), (60.0, -20.0)

        result = great_circle_path_intersect(*_rad(a1), *_rad(a2), *_rad(b1), *_rad(b2))
        hit = (np.degrees(result.lat1), np.degrees(result.lon1))

        assert result.valid
        for start, end in ((a1, a2), (b1, b2)):
            _, azimuth, _ = _inverse(start, end)
            s12, bearing, _ = _inverse(start, hit)
            assert _angle_error(bearing, azimuth) < 1e-9
            lat, lon = _direct(start, azimuth, s12)
            assert abs(lat - result.lat1) < 1e-9
            assert _angle_error(lon, result.lon1) < 1e-9


class TestAngularDistance:
    def test_quarter_sphere(self):
        assert angular_distance(0.0, 0.0, 0.0, np.pi / 2) == pytest.approx(
            np.pi / 2, rel=1e-12, abs=0
        )

    @pytest.mark.parametrize("p1,p2", PAIRS)
    def test_matches_geographiclib(self, p1, p2):
        assert angular_distance(*_rad(p1), *_rad(p2)) == pytest.approx(
            _inverse(p1, p2)[0] / R, rel=1e-9, abs=0
        )


class TestDestinationPoint:
    @pytest.mark.parametrize("bearing_deg", [0.0, 45.0, 135.0, 270.0])
    def test_matches_geographiclib(self, bearing_deg):
        start = (40.0, -74.0)
        angular = 1_000_000.0 / R

        dest = destination_point(*_rad(start), np.radians(bearing_deg), angular)
        lat, lon = _direct(start, np.radians(bearing_deg), 1_000_000.0)

        assert abs(dest.lat - lat) < 1e-9
        assert _angle_error(dest.lon, lon) < 1e-9

    def test_zero_distance_returns_start(self):
        lat, lon = np.radians(40), np.radians(-74)

        dest = destination_point(lat, lon, 0.0, 0.0)

        assert dest.lat == pytest.approx(lat, rel=1e-12, abs=0)
        assert dest.lon == pytest.approx(lon, rel=1e-12, abs=0)
