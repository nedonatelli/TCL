"""Tests for rhumb line navigation functions."""

import numpy as np
import pytest

from pytcl.navigation.rhumb import (
    RhumbDirectResult,
    RhumbResult,
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


class TestRhumbDistanceSpherical:
    """Tests for rhumb_distance_spherical."""

    def test_zero_distance(self):
        """Same point should have zero distance."""
        lat, lon = np.radians(40), np.radians(-74)
        dist = rhumb_distance_spherical(lat, lon, lat, lon)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_due_north(self):
        """Due north rhumb should equal great circle distance."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-74)

        rhumb_dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)

        # For same longitude, rhumb = great circle
        from pytcl.navigation.great_circle import great_circle_distance

        gc_dist = great_circle_distance(lat1, lon1, lat2, lon2)

        assert rhumb_dist == pytest.approx(gc_dist, rel=0.001)

    def test_due_east_at_equator(self):
        """Due east at equator rhumb equals great circle."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(10)

        rhumb_dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)

        from pytcl.navigation.great_circle import great_circle_distance

        gc_dist = great_circle_distance(lat1, lon1, lat2, lon2)

        assert rhumb_dist == pytest.approx(gc_dist, rel=0.001)

    def test_rhumb_longer_than_great_circle(self):
        """Rhumb line should be longer than great circle for diagonal paths."""
        # New York to London
        lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
        lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)

        rhumb_dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)

        from pytcl.navigation.great_circle import great_circle_distance

        gc_dist = great_circle_distance(lat1, lon1, lat2, lon2)

        assert rhumb_dist > gc_dist

    def test_wraparound_longitude(self):
        """Should handle longitude wraparound."""
        lat1, lon1 = np.radians(0), np.radians(170)
        lat2, lon2 = np.radians(0), np.radians(-170)

        dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)

        # Should cross date line (20 degrees), not go around (340 degrees)
        expected_approx = 6371000 * np.radians(20)  # ~2200 km
        assert dist == pytest.approx(expected_approx, rel=0.01)


class TestRhumbBearing:
    """Tests for rhumb_bearing."""

    def test_due_north(self):
        """Due north should have bearing 0."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-74)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
        assert bearing == pytest.approx(0.0, abs=1e-6)

    def test_due_east(self):
        """Due east should have bearing π/2."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(40), np.radians(-64)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
        assert bearing == pytest.approx(np.pi / 2, abs=1e-6)

    def test_due_south(self):
        """Due south should have bearing π."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(30), np.radians(-74)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
        assert bearing == pytest.approx(np.pi, abs=1e-6)

    def test_due_west(self):
        """Due west should have bearing 3π/2."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(40), np.radians(-84)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
        assert bearing == pytest.approx(3 * np.pi / 2, abs=1e-6)

    def test_normalized_range(self):
        """Bearing should be in [0, 2π)."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)
        assert 0 <= bearing < 2 * np.pi


class TestIndirectRhumbSpherical:
    """Tests for indirect_rhumb_spherical."""

    def test_returns_named_tuple(self):
        """Should return RhumbResult."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        result = indirect_rhumb_spherical(lat1, lon1, lat2, lon2)

        assert isinstance(result, RhumbResult)
        assert hasattr(result, "distance")
        assert hasattr(result, "bearing")


class TestDirectRhumbSpherical:
    """Tests for direct_rhumb_spherical."""

    def test_zero_distance(self):
        """Zero distance should return starting point."""
        lat, lon = np.radians(40), np.radians(-74)
        bearing = np.radians(45)

        dest = direct_rhumb_spherical(lat, lon, bearing, 0.0)

        assert dest.lat == pytest.approx(lat, abs=1e-10)
        assert dest.lon == pytest.approx(lon, abs=1e-10)

    def test_due_north(self):
        """Moving north should increase latitude only."""
        lat, lon = np.radians(40), np.radians(-74)
        bearing = 0.0

        dest = direct_rhumb_spherical(lat, lon, bearing, 1000000)

        assert dest.lat > lat
        assert dest.lon == pytest.approx(lon, abs=1e-6)

    def test_due_east(self):
        """Moving east should change longitude only."""
        lat, lon = np.radians(40), np.radians(-74)
        bearing = np.pi / 2

        dest = direct_rhumb_spherical(lat, lon, bearing, 1000000)

        assert dest.lat == pytest.approx(lat, abs=1e-6)
        assert dest.lon > lon

    def test_roundtrip_consistency(self):
        """Direct then indirect should give same values."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        bearing = np.radians(45)
        dist = 1000000

        dest = direct_rhumb_spherical(lat1, lon1, bearing, dist)
        result = indirect_rhumb_spherical(lat1, lon1, dest.lat, dest.lon)

        assert result.distance == pytest.approx(dist, rel=1e-4)
        assert result.bearing == pytest.approx(bearing, abs=1e-4)


class TestRhumbDistanceEllipsoidal:
    """Tests for rhumb_distance_ellipsoidal."""

    def test_zero_distance(self):
        """Same point should have zero distance."""
        lat, lon = np.radians(40), np.radians(-74)
        dist = rhumb_distance_ellipsoidal(lat, lon, lat, lon)
        assert dist == pytest.approx(0.0, abs=1)

    def test_similar_to_spherical(self):
        """Ellipsoidal should be close to spherical for short distances."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(41), np.radians(-73)

        sph_dist = rhumb_distance_spherical(lat1, lon1, lat2, lon2)
        ell_dist = rhumb_distance_ellipsoidal(lat1, lon1, lat2, lon2)

        # Should be within a few percent
        assert ell_dist == pytest.approx(sph_dist, rel=0.02)


class TestIndirectRhumb:
    """Tests for indirect_rhumb (ellipsoidal)."""

    def test_returns_named_tuple(self):
        """Should return RhumbResult."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        result = indirect_rhumb(lat1, lon1, lat2, lon2)

        assert isinstance(result, RhumbResult)


class TestDirectRhumb:
    """Tests for direct_rhumb (ellipsoidal)."""

    def test_returns_named_tuple(self):
        """Should return RhumbDirectResult."""
        lat, lon = np.radians(40), np.radians(-74)
        bearing = np.radians(45)

        dest = direct_rhumb(lat, lon, bearing, 1000000)

        assert isinstance(dest, RhumbDirectResult)

    def test_roundtrip_consistency(self):
        """Direct then indirect should give same values."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        bearing = np.radians(45)
        dist = 1000000

        dest = direct_rhumb(lat1, lon1, bearing, dist)
        result = indirect_rhumb(lat1, lon1, dest.lat, dest.lon)

        assert result.distance == pytest.approx(dist, rel=0.01)
        assert result.bearing == pytest.approx(bearing, abs=0.01)


class TestRhumbIntersect:
    """Tests for rhumb_intersect."""

    def test_perpendicular_rhumbs(self):
        """Perpendicular rhumb lines should intersect."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        bearing1 = np.radians(45)  # Northeast

        lat2, lon2 = np.radians(40), np.radians(-70)
        bearing2 = np.radians(315)  # Northwest

        result = rhumb_intersect(lat1, lon1, bearing1, lat2, lon2, bearing2)

        # Should intersect somewhere
        assert result.valid

    def test_crossing_rhumbs(self):
        """Two crossing rhumb lines."""
        lat1, lon1 = np.radians(0), np.radians(0)
        bearing1 = np.radians(45)  # Northeast

        lat2, lon2 = np.radians(10), np.radians(0)
        bearing2 = np.radians(135)  # Southeast

        result = rhumb_intersect(lat1, lon1, bearing1, lat2, lon2, bearing2)

        assert result.valid

    def test_parallel_rhumbs_no_intersect(self):
        """Parallel rhumb lines should not intersect."""
        lat1, lon1 = np.radians(0), np.radians(0)
        bearing1 = np.radians(45)

        lat2, lon2 = np.radians(0), np.radians(10)
        bearing2 = np.radians(45)  # Same bearing

        result = rhumb_intersect(lat1, lon1, bearing1, lat2, lon2, bearing2)

        assert not result.valid


class TestRhumbMidpoint:
    """Tests for rhumb_midpoint."""

    def test_midpoint(self):
        """Midpoint should be at half distance."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-64)

        mid = rhumb_midpoint(lat1, lon1, lat2, lon2)

        # Distance from start to mid should equal mid to end
        d1 = rhumb_distance_spherical(lat1, lon1, mid.lat, mid.lon)
        d2 = rhumb_distance_spherical(mid.lat, mid.lon, lat2, lon2)

        assert d1 == pytest.approx(d2, rel=0.01)


class TestRhumbWaypoints:
    """Tests for rhumb_waypoints."""

    def test_n_points(self):
        """Should return correct number of points."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-64)

        lats, lons = rhumb_waypoints(lat1, lon1, lat2, lon2, 10)

        assert len(lats) == 10
        assert len(lons) == 10

    def test_endpoints(self):
        """First and last points should match endpoints."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-64)

        lats, lons = rhumb_waypoints(lat1, lon1, lat2, lon2, 5)

        assert lats[0] == pytest.approx(lat1, abs=1e-6)
        assert lons[0] == pytest.approx(lon1, abs=1e-6)
        assert lats[-1] == pytest.approx(lat2, abs=1e-4)
        assert lons[-1] == pytest.approx(lon2, abs=1e-4)


class TestCompareGreatCircleRhumb:
    """Tests for compare_great_circle_rhumb."""

    def test_returns_three_values(self):
        """Should return gc_distance, rhumb_distance, difference."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        gc, rhumb, diff = compare_great_circle_rhumb(lat1, lon1, lat2, lon2)

        assert gc > 0
        assert rhumb > 0
        assert diff >= 0  # Rhumb should be >= great circle

    def test_meridional_path_equal(self):
        """Along meridian, both should be equal."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(50), np.radians(-74)

        gc, rhumb, diff = compare_great_circle_rhumb(lat1, lon1, lat2, lon2)

        assert diff == pytest.approx(0.0, abs=0.1)

    def test_equatorial_path_equal(self):
        """Along equator, both should be equal."""
        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(30)

        gc, rhumb, diff = compare_great_circle_rhumb(lat1, lon1, lat2, lon2)

        assert diff == pytest.approx(0.0, abs=0.1)

    def test_diagonal_path_rhumb_longer(self):
        """Diagonal path should have rhumb > great circle."""
        lat1, lon1 = np.radians(40), np.radians(-74)
        lat2, lon2 = np.radians(51), np.radians(-0.1)

        gc, rhumb, diff = compare_great_circle_rhumb(lat1, lon1, lat2, lon2)

        assert rhumb > gc
        assert diff > 0
