"""Tests for map projections."""

import numpy as np
from numpy.testing import assert_allclose

from pytcl.coordinate_systems.projections import (
    ProjectionResult,
    UTMResult,
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    geodetic2utm,
    geodetic2utm_batch,
    lambert_conformal_conic,
    lambert_conformal_conic_inverse,
    mercator,
    mercator_inverse,
    polar_stereographic,
    stereographic,
    stereographic_inverse,
    transverse_mercator,
    transverse_mercator_inverse,
    utm2geodetic,
    utm_central_meridian,
    utm_zone,
)


class TestMercator:
    """Tests for Mercator projection."""

    def test_mercator_equator(self):
        """Mercator at equator should have unit scale."""
        result = mercator(0, 0)
        assert_allclose(result.x, 0, atol=1e-6)
        assert_allclose(result.y, 0, atol=1e-6)
        assert_allclose(result.scale, 1.0, atol=0.01)

    def test_mercator_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(45)
        lon = np.radians(-75)

        result = mercator(lat, lon)
        lat_inv, lon_inv = mercator_inverse(result.x, result.y)

        assert_allclose(lat_inv, lat, atol=1e-10)
        assert_allclose(lon_inv, lon, atol=1e-10)

    def test_mercator_scale_increases_with_latitude(self):
        """Scale should increase with latitude."""
        result_30 = mercator(np.radians(30), 0)
        result_60 = mercator(np.radians(60), 0)

        assert result_60.scale > result_30.scale

    def test_mercator_convergence_zero(self):
        """Mercator convergence should be zero."""
        result = mercator(np.radians(45), np.radians(30))
        assert result.convergence == 0.0

    def test_mercator_symmetric(self):
        """Mercator should be symmetric about equator."""
        result_n = mercator(np.radians(45), 0)
        result_s = mercator(np.radians(-45), 0)

        assert_allclose(result_n.x, result_s.x)
        assert_allclose(result_n.y, -result_s.y)

    def test_mercator_with_central_meridian(self):
        """Test Mercator with non-zero central meridian."""
        lon0 = np.radians(-75)
        result = mercator(np.radians(40), np.radians(-75), lon0=lon0)
        assert_allclose(result.x, 0, atol=1e-10)


class TestTransverseMercator:
    """Tests for Transverse Mercator projection."""

    def test_tm_on_central_meridian(self):
        """On central meridian, x should be zero."""
        lon0 = np.radians(-75)
        result = transverse_mercator(np.radians(45), lon0, lon0=lon0)
        assert_allclose(result.x, 0, atol=1e-6)

    def test_tm_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(45)
        lon = np.radians(-74)  # 1 degree from central meridian
        lon0 = np.radians(-75)

        result = transverse_mercator(lat, lon, lon0=lon0)
        lat_inv, lon_inv = transverse_mercator_inverse(result.x, result.y, lon0=lon0)

        assert_allclose(lat_inv, lat, atol=1e-8)
        assert_allclose(lon_inv, lon, atol=1e-8)

    def test_tm_scale_at_central_meridian(self):
        """Scale at central meridian should equal k0."""
        k0 = 0.9996
        lon0 = np.radians(-75)
        result = transverse_mercator(np.radians(45), lon0, lon0=lon0, k0=k0)
        assert_allclose(result.scale, k0, rtol=1e-4)

    def test_tm_scale_increases_away_from_meridian(self):
        """Scale should increase away from central meridian."""
        lon0 = np.radians(-75)
        k0 = 0.9996

        result_0 = transverse_mercator(np.radians(45), lon0, lon0=lon0, k0=k0)
        result_2 = transverse_mercator(
            np.radians(45), lon0 + np.radians(2), lon0=lon0, k0=k0
        )

        assert result_2.scale > result_0.scale


class TestUTM:
    """Tests for UTM projection."""

    def test_utm_zone_calculation(self):
        """Test UTM zone calculation."""
        # Zone 18 covers -78 to -72 degrees
        assert utm_zone(np.radians(-75)) == 18
        assert utm_zone(np.radians(-73)) == 18

        # Zone 1 starts at -180
        assert utm_zone(np.radians(-177)) == 1

        # Zone 60 ends at 180
        assert utm_zone(np.radians(177)) == 60

    def test_utm_central_meridian(self):
        """Test UTM central meridian calculation."""
        # Zone 18: central meridian at -75 degrees
        lon0 = utm_central_meridian(18)
        assert_allclose(np.degrees(lon0), -75, atol=0.1)

        # Zone 31: central meridian at 3 degrees
        lon0 = utm_central_meridian(31)
        assert_allclose(np.degrees(lon0), 3, atol=0.1)

    def test_utm_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(45)
        lon = np.radians(-75.5)

        result = geodetic2utm(lat, lon)
        lat_inv, lon_inv = utm2geodetic(
            result.easting, result.northing, result.zone, result.hemisphere
        )

        assert_allclose(lat_inv, lat, atol=1e-8)
        assert_allclose(lon_inv, lon, atol=1e-8)

    def test_utm_false_easting(self):
        """Central meridian should have 500000 easting."""
        lat = np.radians(45)
        lon = np.radians(-75)  # Central meridian of zone 18

        result = geodetic2utm(lat, lon, zone=18)
        assert_allclose(result.easting, 500000, atol=1)

    def test_utm_northern_hemisphere(self):
        """Northern hemisphere should have hemisphere='N'."""
        result = geodetic2utm(np.radians(45), np.radians(-75))
        assert result.hemisphere == "N"

    def test_utm_southern_hemisphere(self):
        """Southern hemisphere should have hemisphere='S'."""
        result = geodetic2utm(np.radians(-45), np.radians(-75))
        assert result.hemisphere == "S"

    def test_utm_false_northing_south(self):
        """Southern hemisphere should have 10,000,000 added to northing."""
        result = geodetic2utm(np.radians(-1), np.radians(-75))
        assert result.northing > 9000000

    def test_utm_scale_factor(self):
        """UTM scale factor at central meridian should be ~0.9996."""
        lat = np.radians(45)
        lon = np.radians(-75)  # Central meridian

        result = geodetic2utm(lat, lon, zone=18)
        assert_allclose(result.scale, 0.9996, rtol=1e-3)

    def test_utm_result_type(self):
        """Result should be UTMResult namedtuple."""
        result = geodetic2utm(np.radians(45), np.radians(-75))
        assert isinstance(result, UTMResult)
        assert hasattr(result, "easting")
        assert hasattr(result, "northing")
        assert hasattr(result, "zone")
        assert hasattr(result, "hemisphere")


class TestStereographic:
    """Tests for Stereographic projection."""

    def test_stereographic_center(self):
        """At center point, x and y should be zero."""
        lat0 = np.radians(45)
        lon0 = np.radians(-75)

        result = stereographic(lat0, lon0, lat0, lon0)
        assert_allclose(result.x, 0, atol=1e-6)
        assert_allclose(result.y, 0, atol=1e-6)

    def test_stereographic_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(46)
        lon = np.radians(-74)
        lat0 = np.radians(45)
        lon0 = np.radians(-75)

        result = stereographic(lat, lon, lat0, lon0)
        lat_inv, lon_inv = stereographic_inverse(result.x, result.y, lat0, lon0)

        assert_allclose(lat_inv, lat, atol=1e-8)
        assert_allclose(lon_inv, lon, atol=1e-8)

    def test_stereographic_scale_at_center(self):
        """Scale at center should equal k0."""
        lat0 = np.radians(45)
        lon0 = np.radians(-75)
        k0 = 1.0

        result = stereographic(lat0, lon0, lat0, lon0, k0=k0)
        # At the center point, the scale factor is k0 (not 2*k0)
        assert_allclose(result.scale, k0, rtol=0.1)

    def test_polar_stereographic_north(self):
        """Test polar stereographic at north pole."""
        result = polar_stereographic(np.radians(85), np.radians(45), north=True)
        assert isinstance(result, ProjectionResult)
        # Distance from pole should be positive
        assert np.sqrt(result.x**2 + result.y**2) > 0

    def test_polar_stereographic_south(self):
        """Test polar stereographic at south pole."""
        result = polar_stereographic(np.radians(-85), np.radians(45), north=False)
        assert isinstance(result, ProjectionResult)

    def test_polar_stereographic_at_pole(self):
        """At pole, x and y should be zero."""
        result = polar_stereographic(np.radians(90), 0, north=True)
        assert_allclose(result.x, 0, atol=1e-6)
        assert_allclose(result.y, 0, atol=1e-6)


class TestLambertConformalConic:
    """Tests for Lambert Conformal Conic projection."""

    def test_lcc_on_central_meridian(self):
        """On central meridian, x should be zero."""
        lat0 = np.radians(39)
        lon0 = np.radians(-96)
        lat1 = np.radians(33)
        lat2 = np.radians(45)

        result = lambert_conformal_conic(np.radians(40), lon0, lat0, lon0, lat1, lat2)
        assert_allclose(result.x, 0, atol=1)

    def test_lcc_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(40)
        lon = np.radians(-100)
        lat0 = np.radians(39)
        lon0 = np.radians(-96)
        lat1 = np.radians(33)
        lat2 = np.radians(45)

        result = lambert_conformal_conic(lat, lon, lat0, lon0, lat1, lat2)
        lat_inv, lon_inv = lambert_conformal_conic_inverse(
            result.x, result.y, lat0, lon0, lat1, lat2
        )

        assert_allclose(lat_inv, lat, atol=1e-8)
        assert_allclose(lon_inv, lon, atol=1e-8)

    def test_lcc_scale_at_standard_parallels(self):
        """Scale should be close to 1 at standard parallels."""
        lat1 = np.radians(33)
        lat2 = np.radians(45)
        lat0 = np.radians(39)
        lon0 = np.radians(-96)

        result1 = lambert_conformal_conic(lat1, lon0, lat0, lon0, lat1, lat2)
        result2 = lambert_conformal_conic(lat2, lon0, lat0, lon0, lat1, lat2)

        assert_allclose(result1.scale, 1.0, rtol=0.01)
        assert_allclose(result2.scale, 1.0, rtol=0.01)


class TestAzimuthalEquidistant:
    """Tests for Azimuthal Equidistant projection."""

    def test_azeq_center(self):
        """At center point, x and y should be zero."""
        lat0 = np.radians(38.9)
        lon0 = np.radians(-77)

        result = azimuthal_equidistant(lat0, lon0, lat0, lon0)
        assert_allclose(result.x, 0, atol=1e-6)
        assert_allclose(result.y, 0, atol=1e-6)

    def test_azeq_roundtrip(self):
        """Forward and inverse should be consistent."""
        lat = np.radians(40)
        lon = np.radians(-75)
        lat0 = np.radians(38.9)
        lon0 = np.radians(-77)

        result = azimuthal_equidistant(lat, lon, lat0, lon0)
        lat_inv, lon_inv = azimuthal_equidistant_inverse(result.x, result.y, lat0, lon0)

        assert_allclose(lat_inv, lat, atol=1e-6)
        assert_allclose(lon_inv, lon, atol=1e-6)

    def test_azeq_distance_preservation(self):
        """Distance from center should be preserved."""
        lat0 = np.radians(38.9)
        lon0 = np.radians(-77)

        # Point at known distance
        lat = np.radians(40)
        lon = np.radians(-77)  # Due north

        result = azimuthal_equidistant(lat, lon, lat0, lon0)

        # The projected distance should approximately equal the geodesic distance
        # The projected point should be north of the origin (positive y)
        assert result.y > 0
        assert_allclose(result.x, 0, atol=1000)  # Due north

        # Distance should be reasonable (about 120-140 km for 1.1 degrees)
        dist = np.sqrt(result.x**2 + result.y**2)
        assert 100000 < dist < 200000  # 100-200 km range


class TestBatchOperations:
    """Tests for batch projection operations."""

    def test_geodetic2utm_batch(self):
        """Test batch UTM conversion."""
        lats = np.radians(np.array([40, 41, 42, 43]))
        lons = np.radians(np.array([-75, -75, -75, -75]))

        eastings, northings, zones, hemispheres = geodetic2utm_batch(lats, lons)

        assert len(eastings) == 4
        assert len(northings) == 4
        assert np.all(zones == 18)
        assert np.all(hemispheres == "N")

    def test_batch_matches_single(self):
        """Batch results should match individual conversions."""
        lats = np.radians(np.array([40, 41, -42]))
        lons = np.radians(np.array([-75, -74, -73]))

        eastings, northings, zones, hemispheres = geodetic2utm_batch(lats, lons)

        for i in range(len(lats)):
            single = geodetic2utm(lats[i], lons[i])
            assert_allclose(eastings[i], single.easting, atol=1e-6)
            assert_allclose(northings[i], single.northing, atol=1e-6)


class TestProjectionResult:
    """Tests for ProjectionResult namedtuple."""

    def test_projection_result_fields(self):
        """ProjectionResult should have expected fields."""
        result = ProjectionResult(1000, 2000, 1.0, 0.01)
        assert result.x == 1000
        assert result.y == 2000
        assert result.scale == 1.0
        assert result.convergence == 0.01


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_mercator_high_latitude(self):
        """Mercator should handle high latitudes."""
        result = mercator(np.radians(85), 0)
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)
        assert result.scale > 10  # Very large scale

    def test_utm_equator(self):
        """UTM should handle equator correctly."""
        result = geodetic2utm(0, np.radians(-75))
        assert_allclose(result.northing, 0, atol=1)
        assert result.hemisphere == "N"

    def test_stereographic_antipode(self):
        """Test stereographic far from center."""
        lat0 = np.radians(45)
        lon0 = np.radians(0)

        # Point far from center
        result = stereographic(np.radians(-30), np.radians(90), lat0, lon0)
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)

    def test_lcc_at_pole(self):
        """LCC should handle points approaching poles."""
        lat0 = np.radians(39)
        lon0 = np.radians(-96)
        lat1 = np.radians(33)
        lat2 = np.radians(45)

        result = lambert_conformal_conic(np.radians(80), lon0, lat0, lon0, lat1, lat2)
        assert np.isfinite(result.x)
        assert np.isfinite(result.y)


class TestNorwayException:
    """Tests for UTM Norway/Svalbard zone exceptions."""

    def test_norway_zone_32(self):
        """Norway exception: zone 32 extended."""
        # Bergen, Norway (60°N, 5°E) should be zone 32, not 31
        lat = np.radians(60)
        lon = np.radians(5)
        assert utm_zone(lon, lat) == 32

    def test_svalbard_zones(self):
        """Svalbard uses special zones."""
        lat = np.radians(78)

        # 0-9°E should be zone 31
        assert utm_zone(np.radians(5), lat) == 31

        # 9-21°E should be zone 33
        assert utm_zone(np.radians(15), lat) == 33

        # 21-33°E should be zone 35
        assert utm_zone(np.radians(27), lat) == 35

        # 33-42°E should be zone 37
        assert utm_zone(np.radians(38), lat) == 37
