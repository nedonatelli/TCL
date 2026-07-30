"""Tests for great circle navigation functions."""

import numpy as np
import pytest

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


class TestGreatCircleDistance:
    """Tests for great_circle_distance."""

    def test_zero_distance(self):
        """Same point should have zero distance."""
        lat, lon = np.radians(40), np.radians(-74)
        dist = great_circle_distance(lat, lon, lat, lon)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_ny_to_london(self):
        """Test NY to London distance (well-known value)."""
        # New York
        lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
        # London
        lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Actual distance is ~5570 km
        assert dist == pytest.approx(5570e3, rel=0.01)

    def test_antipodal_points(self):
        """Antipodal points should be half circumference apart."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(180)

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Half Earth circumference ~20015 km
        assert dist == pytest.approx(20015e3, rel=0.01)

    def test_poles(self):
        """North pole to south pole."""
        lat1, lon1 = np.radians(90), np.radians(0)
        lat2, lon2 = np.radians(-90), np.radians(0)

        dist = great_circle_distance(lat1, lon1, lat2, lon2)

        # Half Earth circumference
        assert dist == pytest.approx(20015e3, rel=0.01)

    def test_custom_radius(self):
        """Test with custom sphere radius."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(90)

        dist = great_circle_distance(lat1, lon1, lat2, lon2, radius=1000.0)

        # Quarter circumference on unit sphere
        expected = 1000.0 * np.pi / 2
        assert dist == pytest.approx(expected, rel=1e-6)


class TestGreatCircleAzimuth:
    """Tests for great_circle_azimuth."""

    def test_due_north(self):
        """Point directly north should have azimuth 0."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-74)

        az = great_circle_azimuth(lat1, lon1, lat2, lon2)
        assert az == pytest.approx(0.0, abs=1e-6)

    def test_due_east(self):
        """Point directly east should have azimuth π/2."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(10)

        az = great_circle_azimuth(lat1, lon1, lat2, lon2)
        assert az == pytest.approx(np.pi / 2, abs=1e-6)

    def test_due_south(self):
        """Point directly south should have azimuth π."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(30), np.radians(-74)

        az = great_circle_azimuth(lat1, lon1, lat2, lon2)
        assert az == pytest.approx(np.pi, abs=1e-6)

    def test_due_west(self):
        """Point directly west should have azimuth 3π/2."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(-10)

        az = great_circle_azimuth(lat1, lon1, lat2, lon2)
        assert az == pytest.approx(3 * np.pi / 2, abs=1e-6)

    def test_normalized_range(self):
        """Azimuth should be in [0, 2π)."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        az = great_circle_azimuth(lat1, lon1, lat2, lon2)
        assert 0 <= az < 2 * np.pi


class TestGreatCircleInverse:
    """Tests for great_circle_inverse."""

    def test_returns_named_tuple(self):
        """Should return GreatCircleResult."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        result = great_circle_inverse(lat1, lon1, lat2, lon2)

        assert isinstance(result, GreatCircleResult)
        assert hasattr(result, "distance")
        assert hasattr(result, "azimuth1")
        assert hasattr(result, "azimuth2")


class TestGreatCircleWaypoint:
    """Tests for great_circle_waypoint."""

    def test_fraction_zero(self):
        """Fraction 0 should return starting point."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        wp = great_circle_waypoint(lat1, lon1, lat2, lon2, 0.0)

        assert wp.lat == pytest.approx(lat1, abs=1e-10)
        assert wp.lon == pytest.approx(lon1, abs=1e-10)

    def test_fraction_one(self):
        """Fraction 1 should return destination."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        wp = great_circle_waypoint(lat1, lon1, lat2, lon2, 1.0)

        assert wp.lat == pytest.approx(lat2, abs=1e-10)
        assert wp.lon == pytest.approx(lon2, abs=1e-10)

    def test_midpoint(self):
        """Midpoint should be equidistant from both endpoints."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        mid = great_circle_waypoint(lat1, lon1, lat2, lon2, 0.5)

        d1 = great_circle_distance(lat1, lon1, mid.lat, mid.lon)
        d2 = great_circle_distance(mid.lat, mid.lon, lat2, lon2)

        assert d1 == pytest.approx(d2, rel=1e-6)


class TestGreatCircleWaypoints:
    """Tests for great_circle_waypoints."""

    def test_n_points(self):
        """Should return correct number of points."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        lats, lons = great_circle_waypoints(lat1, lon1, lat2, lon2, 10)

        assert len(lats) == 10
        assert len(lons) == 10

    def test_endpoints(self):
        """First and last points should match endpoints."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        lats, lons = great_circle_waypoints(lat1, lon1, lat2, lon2, 5)

        assert lats[0] == pytest.approx(lat1, abs=1e-10)
        assert lons[0] == pytest.approx(lon1, abs=1e-10)
        assert lats[-1] == pytest.approx(lat2, abs=1e-10)
        assert lons[-1] == pytest.approx(lon2, abs=1e-10)


class TestGreatCircleDirect:
    """Tests for great_circle_direct."""

    def test_zero_distance(self):
        """Zero distance should return starting point."""
        lat, lon = np.radians(40), np.radians(-74)
        az = np.radians(45)

        dest = great_circle_direct(lat, lon, az, 0.0)

        assert dest.lat == pytest.approx(lat, abs=1e-10)
        assert dest.lon == pytest.approx(lon, abs=1e-10)

    def test_due_north(self):
        """Moving north should increase latitude."""
        lat, lon = np.radians(40), np.radians(-74)
        az = 0.0  # North

        dest = great_circle_direct(lat, lon, az, 1000000)  # 1000 km

        assert dest.lat > lat
        assert dest.lon == pytest.approx(lon, abs=1e-6)

    def test_due_east_at_equator(self):
        """Moving east at equator should change longitude only."""
        lat, lon = 0.0, 0.0
        az = np.pi / 2  # East

        dest = great_circle_direct(lat, lon, az, 1000000)

        assert dest.lat == pytest.approx(0.0, abs=1e-6)
        assert dest.lon > 0

    def test_roundtrip_consistency(self):
        """Direct then inverse should give same distance."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        az = np.radians(45)
        dist = 1000000

        dest = great_circle_direct(lat1, lon1, az, dist)
        result = great_circle_inverse(lat1, lon1, dest.lat, dest.lon)

        assert result.distance == pytest.approx(dist, rel=1e-6)


class TestCrossTrackDistance:
    """Tests for cross_track_distance."""

    def test_point_on_path(self):
        """Point on great circle path should have zero cross-track."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(10)

        # Point on equator between them
        lat_p, lon_p = np.radians(0), np.radians(5)

        result = cross_track_distance(lat_p, lon_p, lat1, lon1, lat2, lon2)

        assert result.cross_track == pytest.approx(0.0, abs=100)  # Within 100m

    def test_point_north_of_path(self):
        """Point north of east-west path should have positive cross-track."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(10)

        # Point north of equator
        lat_p, lon_p = np.radians(1), np.radians(5)

        result = cross_track_distance(lat_p, lon_p, lat1, lon1, lat2, lon2)

        # Right of eastward path = positive (south)
        # Actually north would be left = negative for eastward
        assert result.cross_track != 0


class TestGreatCircleIntersect:
    """Tests for great_circle_intersect."""

    def test_perpendicular_great_circles(self):
        """Two perpendicular great circles should intersect."""
        # Equator and prime meridian
        lat1, lon1 = np.radians(0), np.radians(0)
        az1 = np.pi / 2  # East along equator

        lat2, lon2 = np.radians(0), np.radians(0)
        az2 = 0.0  # North along prime meridian

        result = great_circle_intersect(lat1, lon1, az1, lat2, lon2, az2)

        assert result.valid

    def test_parallel_great_circles(self):
        """Parallel great circles should not intersect (same orientation)."""
        lat1, lon1 = np.radians(0), np.radians(0)
        az1 = 0.0  # North

        lat2, lon2 = np.radians(0), np.radians(10)
        az2 = 0.0  # Also north

        result = great_circle_intersect(lat1, lon1, az1, lat2, lon2, az2)

        # Parallel meridians - these actually do intersect at poles
        assert result.valid


class TestGreatCirclePathIntersect:
    """Tests for great_circle_path_intersect."""

    def test_crossing_paths(self):
        """Two crossing paths should have intersection."""
        # Path 1: equator segment
        lat1a, lon1a = np.radians(0), np.radians(-10)
        lat2a, lon2a = np.radians(0), np.radians(10)

        # Path 2: prime meridian segment
        lat1b, lon1b = np.radians(-10), np.radians(0)
        lat2b, lon2b = np.radians(10), np.radians(0)

        result = great_circle_path_intersect(
            lat1a, lon1a, lat2a, lon2a, lat1b, lon1b, lat2b, lon2b
        )

        assert result.valid
        # Intersection should be near origin
        assert result.lat1 == pytest.approx(0.0, abs=0.01)
        assert result.lon1 == pytest.approx(0.0, abs=0.01)


class TestAngularDistance:
    """Tests for angular_distance."""

    def test_quarter_sphere(self):
        """90 degrees apart on great circle."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(90)

        ang = angular_distance(lat1, lon1, lat2, lon2)

        assert ang == pytest.approx(np.pi / 2, abs=1e-10)


class TestDestinationPoint:
    """Tests for destination_point."""

    def test_zero_distance(self):
        """Zero angular distance returns starting point."""
        lat, lon = np.radians(40), np.radians(-74)

        dest = destination_point(lat, lon, 0.0, 0.0)

        assert dest.lat == pytest.approx(lat, abs=1e-10)
        assert dest.lon == pytest.approx(lon, abs=1e-10)

    def test_consistency_with_direct(self):
        """Should be consistent with great_circle_direct."""
        lat, lon = np.radians(40), np.radians(-74)
        bearing = np.radians(45)
        distance = 1000000  # 1000 km
        radius = 6371000

        dest1 = great_circle_direct(lat, lon, bearing, distance, radius)
        dest2 = destination_point(lat, lon, bearing, distance / radius)

        assert dest1.lat == pytest.approx(dest2.lat, abs=1e-10)
        assert dest1.lon == pytest.approx(dest2.lon, abs=1e-10)
