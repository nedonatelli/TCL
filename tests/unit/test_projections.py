"""Tests for map projections.

Coordinates, scale factors and grid convergence are checked against PROJ
through pyproj, an independent implementation. Where a pytcl projection is a
documented approximation (the spherical azimuthal equidistant, the Gaussian-
sphere ``stereographic``), the oracle is the PROJ projection that does the
same computation (``+proj=aeqd +R=<authalic radius>``, ``+proj=stere +R=a``).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.coordinate_systems.projections import (
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    geodetic2utm,
    geodetic2utm_batch,
    lambert_conformal_conic,
    lambert_conformal_conic_inverse,
    mercator,
    mercator_inverse,
    oblique_stereographic,
    polar_stereographic,
    stereographic,
    stereographic_inverse,
    transverse_mercator,
    transverse_mercator_inverse,
    utm2geodetic,
    utm_central_meridian,
    utm_zone,
)

WGS84 = "+ellps=WGS84 +units=m +no_defs"
A = 6378137.0
F = 1.0 / 298.257223563
E2 = F * (2.0 - F)


@pytest.fixture(scope="module")
def pyproj():
    return pytest.importorskip("pyproj")


def _factors(proj, lat_deg, lon_deg):
    return proj.get_factors(lon_deg, lat_deg)


def _authalic_radius():
    """Radius of the sphere with the WGS84 surface area (closed form)."""
    e = np.sqrt(E2)
    b = A * (1.0 - F)
    return float(np.sqrt(A**2 / 2.0 + b**2 / 2.0 * np.arctanh(e) / e))


class TestMercator:
    """Mercator against ``+proj=merc`` at four points (lat -30 to 85, lon -100 to 20)."""

    POINTS = [(45.0, -70.0), (-30.0, 20.0), (85.0, -75.0), (10.0, -100.0)]

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_forward_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"+proj=merc +lon_0=-75 {WGS84}")
        result = mercator(np.radians(lat), np.radians(lon), lon0=np.radians(-75))
        x, y = proj(lon, lat)

        assert abs(result.x - x) < 1e-6
        assert abs(result.y - y) < 1e-6
        assert result.scale == pytest.approx(
            _factors(proj, lat, lon).parallel_scale, rel=1e-9, abs=0
        )
        assert result.convergence == 0.0

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_inverse_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"+proj=merc +lon_0=-75 {WGS84}")
        x, y = proj(lon, lat)

        lat_inv, lon_inv = mercator_inverse(x, y, lon0=np.radians(-75))

        assert abs(np.degrees(lat_inv) - lat) < 1e-9
        assert abs(np.degrees(lon_inv) - lon) < 1e-9

    def test_scale_is_secant_of_latitude_on_the_sphere(self):
        """Closed form: with e = 0, k = 1/cos(lat) and y = R ln tan(pi/4 + lat/2)."""
        lat = np.radians(60.0)
        result = mercator(lat, 0.0, e=0.0)

        assert result.scale == pytest.approx(2.0, rel=1e-12, abs=0)
        assert result.y == pytest.approx(
            A * np.log(np.tan(np.pi / 4 + lat / 2)), rel=1e-12, abs=0
        )

    def test_equator_origin(self):
        result = mercator(0.0, 0.0)
        assert result.x == 0.0
        assert abs(result.y) < 1e-6
        assert result.scale == 1.0

    def test_symmetric_about_equator(self):
        north = mercator(np.radians(45), np.radians(10))
        south = mercator(np.radians(-45), np.radians(10))

        assert north.x == south.x
        assert north.y == -south.y

    def test_central_meridian_has_zero_easting(self):
        lon0 = np.radians(-75)
        assert mercator(np.radians(40), lon0, lon0=lon0).x == 0.0


class TestTransverseMercator:
    """Transverse Mercator against ``+proj=tmerc`` (PROJ uses the exact
    Poder/Engsager series; pytcl's Helmert series agrees to a few millimetres
    within the 5 degrees of the central meridian covered by the test points,
    on WGS84, k0 = 0.9996 and, for the custom-ellipsoid case, f = 1/150)."""

    POINTS = [(45.0, -74.0), (10.0, -73.0), (60.0, -76.0), (-30.0, -70.0)]

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_forward_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"+proj=tmerc +lon_0=-75 +k_0=0.9996 {WGS84}")
        result = transverse_mercator(
            np.radians(lat), np.radians(lon), lon0=np.radians(-75), k0=0.9996
        )
        x, y = proj(lon, lat)
        factors = _factors(proj, lat, lon)

        assert abs(result.x - x) < 5e-3
        assert abs(result.y - y) < 5e-3
        assert result.scale == pytest.approx(factors.parallel_scale, rel=1e-8, abs=0)
        assert abs(result.convergence - np.radians(factors.meridian_convergence)) < 1e-8

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_inverse_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"+proj=tmerc +lon_0=-75 +k_0=0.9996 {WGS84}")
        x, y = proj(lon, lat)

        lat_inv, lon_inv = transverse_mercator_inverse(
            x, y, lon0=np.radians(-75), k0=0.9996
        )

        assert abs(np.degrees(lat_inv) - lat) < 1e-7
        assert abs(np.degrees(lon_inv) - lon) < 1e-7

    @pytest.mark.parametrize("lat,lon", [(10.0, -73.0), (45.0, -74.0), (60.0, -76.0)])
    def test_custom_ellipsoid_matches_proj(self, pyproj, lat, lon):
        """A non-WGS84 ellipsoid (f = 1/150), forward and inverse.

        Regression for the inverse once taking the WGS84 semi-minor axis
        regardless of the caller's ``a`` and ``e2`` (gh-25 class).
        """
        f = 1.0 / 150.0
        e2 = 2 * f - f**2
        proj = pyproj.Proj(f"+proj=tmerc +lon_0=-75 +a={A} +rf=150 +units=m +no_defs")
        x, y = proj(lon, lat)

        result = transverse_mercator(
            np.radians(lat), np.radians(lon), lon0=np.radians(-75), a=A, e2=e2
        )
        lat_inv, lon_inv = transverse_mercator_inverse(
            x, y, lon0=np.radians(-75), a=A, e2=e2
        )

        assert abs(result.x - x) < 5e-3
        assert abs(result.y - y) < 5e-3
        assert abs(np.degrees(lat_inv) - lat) < 1e-7
        assert abs(np.degrees(lon_inv) - lon) < 1e-7

    def test_central_meridian_has_zero_easting_and_scale_k0(self, pyproj):
        lon0 = np.radians(-75)
        result = transverse_mercator(np.radians(45), lon0, lon0=lon0, k0=0.9996)

        assert abs(result.x) < 1e-6
        assert result.scale == pytest.approx(0.9996, rel=1e-12, abs=0)

    def test_scale_grows_away_from_the_central_meridian(self, pyproj):
        """On the sphere-like closed form k = k0 (1 + (dlon cos(lat))^2 / 2)."""
        proj = pyproj.Proj(f"+proj=tmerc +lon_0=-75 +k_0=0.9996 {WGS84}")
        result = transverse_mercator(
            np.radians(45), np.radians(-73), lon0=np.radians(-75), k0=0.9996
        )

        assert result.scale == pytest.approx(
            _factors(proj, 45.0, -73.0).parallel_scale, rel=1e-8, abs=0
        )
        assert result.scale > 0.9996


class TestUTM:
    """UTM against EPSG:326xx / 327xx at the six listed points (zones 18, 30, 32, 56)."""

    CASES = [
        (45.0, -75.5, 32618, 18, "N"),
        (-33.9, 151.2, 32756, 56, "S"),
        (0.0, -75.0, 32618, 18, "N"),
        (-1.0, -75.0, 32718, 18, "S"),
        (60.0, 5.0, 32632, 32, "N"),
        (51.5, -0.1, 32630, 30, "N"),
    ]

    @pytest.mark.parametrize("lat,lon,epsg,zone,hemisphere", CASES)
    def test_forward_matches_epsg(self, pyproj, lat, lon, epsg, zone, hemisphere):
        transformer = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
        easting, northing = transformer.transform(lon, lat)

        result = geodetic2utm(np.radians(lat), np.radians(lon))

        assert result.zone == zone
        assert result.hemisphere == hemisphere
        assert abs(result.easting - easting) < 1e-3
        assert abs(result.northing - northing) < 1e-3

    @pytest.mark.parametrize("lat,lon,epsg,zone,hemisphere", CASES)
    def test_inverse_matches_epsg(self, pyproj, lat, lon, epsg, zone, hemisphere):
        transformer = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
        easting, northing = transformer.transform(lon, lat)

        lat_inv, lon_inv = utm2geodetic(easting, northing, zone, hemisphere)

        assert abs(np.degrees(lat_inv) - lat) < 1e-7
        assert abs(np.degrees(lon_inv) - lon) < 1e-7

    def test_central_meridian_false_easting_and_scale(self, pyproj):
        result = geodetic2utm(np.radians(45), np.radians(-75), zone=18)

        assert result.easting == pytest.approx(500000.0, rel=1e-12, abs=0)
        assert result.scale == pytest.approx(0.9996, rel=1e-12, abs=0)

    def test_equator_northing_is_zero(self):
        result = geodetic2utm(0.0, np.radians(-75))

        assert abs(result.northing) < 1e-6
        assert result.hemisphere == "N"

    def test_zone_calculation(self):
        assert utm_zone(np.radians(-75)) == 18
        assert utm_zone(np.radians(-73)) == 18
        assert utm_zone(np.radians(-177)) == 1
        assert utm_zone(np.radians(177)) == 60

    @pytest.mark.parametrize(
        "zone,central_deg", [(18, -75.0), (31, 3.0), (1, -177.0), (60, 177.0)]
    )
    def test_central_meridian(self, zone, central_deg):
        """Zone n is centred on 6n - 183 degrees."""
        assert np.degrees(utm_central_meridian(zone)) == pytest.approx(
            central_deg, rel=1e-12, abs=0
        )


class TestStereographic:
    """Stereographic projections against PROJ ``stere`` / ``sterea``."""

    POINTS = [(46.0, -74.0), (30.0, -60.0), (60.0, -100.0)]

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_oblique_sphere_matches_proj(self, pyproj, lat, lon):
        """With e2 = 0 ``stereographic`` is the spherical oblique stereographic."""
        proj = pyproj.Proj(
            f"+proj=stere +R={A} +lat_0=45 +lon_0=-75 +k_0=1 +units=m +no_defs"
        )
        result = stereographic(
            np.radians(lat), np.radians(lon), np.radians(45), np.radians(-75), e2=0.0
        )
        x, y = proj(lon, lat)
        factors = _factors(proj, lat, lon)

        assert abs(result.x - x) < 1e-6
        assert abs(result.y - y) < 1e-6
        assert result.scale == pytest.approx(factors.parallel_scale, rel=1e-9, abs=0)
        assert abs(result.convergence - np.radians(factors.meridian_convergence)) < 1e-9

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_oblique_sphere_inverse_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(
            f"+proj=stere +R={A} +lat_0=45 +lon_0=-75 +k_0=1 +units=m +no_defs"
        )
        x, y = proj(lon, lat)

        lat_inv, lon_inv = stereographic_inverse(
            x, y, np.radians(45), np.radians(-75), e2=0.0
        )

        assert abs(np.degrees(lat_inv) - lat) < 1e-9
        assert abs(np.degrees(lon_inv) - lon) < 1e-9

    @pytest.mark.parametrize("lat,lon", POINTS)
    def test_ellipsoidal_oblique_matches_sterea(self, pyproj, lat, lon):
        """EPSG method 9809 against ``+proj=sterea``."""
        proj = pyproj.Proj(f"+proj=sterea +lat_0=45 +lon_0=-75 +k_0=0.9999 {WGS84}")
        result = oblique_stereographic(
            np.radians(lat), np.radians(lon), np.radians(45), np.radians(-75), k0=0.9999
        )
        x, y = proj(lon, lat)

        assert abs(result.x - x) < 1e-6
        assert abs(result.y - y) < 1e-6
        assert result.scale == pytest.approx(
            _factors(proj, lat, lon).parallel_scale, rel=1e-9, abs=0
        )

    @pytest.mark.xfail(
        strict=True,
        reason="stereographic's ellipsoidal oblique form (Gaussian-radius sphere) "
        "deviates from PROJ's sterea by hundreds of metres at 1 degree; documented "
        "in its docstring, use oblique_stereographic instead",
    )
    def test_ellipsoidal_oblique_matches_proj(self, pyproj):
        proj = pyproj.Proj(f"+proj=sterea +lat_0=45 +lon_0=-75 +k_0=1 {WGS84}")
        result = stereographic(
            np.radians(46.0), np.radians(-74.0), np.radians(45), np.radians(-75)
        )
        x, y = proj(-74.0, 46.0)

        assert abs(result.x - x) < 1e-3
        assert abs(result.y - y) < 1e-3

    def test_scale_at_center_is_k0(self):
        lat0, lon0 = np.radians(45), np.radians(-75)
        result = stereographic(lat0, lon0, lat0, lon0, k0=0.9996)

        assert result.x == 0.0
        assert result.y == 0.0
        assert result.scale == pytest.approx(0.9996, rel=1e-12, abs=0)

    @pytest.mark.parametrize(
        "lat,lon,north,proj_def",
        [
            (85.0, 45.0, True, "+proj=stere +lat_0=90 +lon_0=0 +k_0=0.994"),
            (70.0, -120.0, True, "+proj=stere +lat_0=90 +lon_0=0 +k_0=0.994"),
            (-85.0, 45.0, False, "+proj=stere +lat_0=-90 +lon_0=0 +k_0=0.994"),
            (-60.0, 170.0, False, "+proj=stere +lat_0=-90 +lon_0=0 +k_0=0.994"),
        ],
    )
    def test_polar_matches_proj(self, pyproj, lat, lon, north, proj_def):
        proj = pyproj.Proj(f"{proj_def} {WGS84}")
        result = polar_stereographic(np.radians(lat), np.radians(lon), north=north)
        x, y = proj(lon, lat)
        factors = _factors(proj, lat, lon)

        assert abs(result.x - x) < 1e-6
        assert abs(result.y - y) < 1e-6
        assert result.scale == pytest.approx(factors.parallel_scale, rel=1e-9, abs=0)
        assert abs(result.convergence - np.radians(factors.meridian_convergence)) < 1e-9

    def test_polar_origin_at_pole(self):
        result = polar_stereographic(np.radians(90), 0.0, north=True)

        assert abs(result.x) < 1e-6
        assert abs(result.y) < 1e-6
        assert result.scale == pytest.approx(0.994, rel=1e-12, abs=0)


class TestLambertConformalConic:
    """Lambert Conformal Conic (2SP, parallels 33/45 N, origin 39 N 96 W, WGS84)
    against ``+proj=lcc`` at the listed points only."""

    PARAMS = "+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96"
    ARGS = (np.radians(39), np.radians(-96), np.radians(33), np.radians(45))

    @pytest.mark.parametrize(
        "lat,lon", [(40.0, -100.0), (80.0, -96.0), (30.0, -90.0), (33.0, -96.0)]
    )
    def test_forward_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"{self.PARAMS} {WGS84}")
        result = lambert_conformal_conic(np.radians(lat), np.radians(lon), *self.ARGS)
        x, y = proj(lon, lat)
        factors = _factors(proj, lat, lon)

        assert abs(result.x - x) < 1e-6
        assert abs(result.y - y) < 1e-6
        assert result.scale == pytest.approx(factors.parallel_scale, rel=1e-9, abs=0)
        assert abs(result.convergence - np.radians(factors.meridian_convergence)) < 1e-9

    @pytest.mark.parametrize("lat,lon", [(40.0, -100.0), (30.0, -90.0)])
    def test_inverse_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(f"{self.PARAMS} {WGS84}")
        x, y = proj(lon, lat)

        lat_inv, lon_inv = lambert_conformal_conic_inverse(x, y, *self.ARGS)

        assert abs(np.degrees(lat_inv) - lat) < 1e-9
        assert abs(np.degrees(lon_inv) - lon) < 1e-9

    def test_scale_is_exactly_one_at_the_standard_parallels(self):
        for lat_deg in (33.0, 45.0):
            result = lambert_conformal_conic(
                np.radians(lat_deg), np.radians(-96), *self.ARGS
            )
            assert result.scale == pytest.approx(1.0, rel=1e-9, abs=0)


class TestAzimuthalEquidistant:
    """Spherical azimuthal equidistant (authalic radius) against
    ``+proj=aeqd +R=<authalic radius>``."""

    LAT0, LON0 = 38.9, -77.0

    @pytest.mark.parametrize(
        "lat,lon", [(40.0, -77.0), (40.0, -75.0), (20.0, 30.0), (38.9, -77.0)]
    )
    def test_forward_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(
            f"+proj=aeqd +R={_authalic_radius()} +lat_0={self.LAT0} +lon_0={self.LON0} +units=m +no_defs"
        )
        result = azimuthal_equidistant(
            np.radians(lat),
            np.radians(lon),
            np.radians(self.LAT0),
            np.radians(self.LON0),
        )
        x, y = proj(lon, lat)

        assert abs(result.x - x) < 1e-5
        assert abs(result.y - y) < 1e-5

    @pytest.mark.parametrize("lat,lon", [(40.0, -77.0), (40.0, -75.0), (20.0, 30.0)])
    def test_inverse_matches_proj(self, pyproj, lat, lon):
        proj = pyproj.Proj(
            f"+proj=aeqd +R={_authalic_radius()} +lat_0={self.LAT0} +lon_0={self.LON0} +units=m +no_defs"
        )
        x, y = proj(lon, lat)

        lat_inv, lon_inv = azimuthal_equidistant_inverse(
            x, y, np.radians(self.LAT0), np.radians(self.LON0)
        )

        assert abs(np.degrees(lat_inv) - lat) < 1e-9
        assert abs(np.degrees(lon_inv) - lon) < 1e-9

    def test_due_north_distance_is_radius_times_latitude_difference(self):
        """Along the central meridian the radial distance is R * dlat exactly."""
        result = azimuthal_equidistant(
            np.radians(40.0),
            np.radians(self.LON0),
            np.radians(self.LAT0),
            np.radians(self.LON0),
        )

        assert abs(result.x) < 1e-6
        assert result.y == pytest.approx(
            _authalic_radius() * np.radians(40.0 - self.LAT0), rel=1e-12, abs=0
        )


class TestBatchOperations:
    """Batch UTM against EPSG:32618 / 32718."""

    def test_geodetic2utm_batch_matches_epsg(self, pyproj):
        lat_deg = np.array([40.0, 41.0, 42.0, 43.0])
        lon_deg = np.array([-75.0, -75.0, -74.0, -73.5])
        transformer = pyproj.Transformer.from_crs(4326, 32618, always_xy=True)
        east, north = transformer.transform(lon_deg, lat_deg)

        eastings, northings, zones, hemispheres = geodetic2utm_batch(
            np.radians(lat_deg), np.radians(lon_deg)
        )

        assert_allclose(eastings, east, rtol=0, atol=1e-3)
        assert_allclose(northings, north, rtol=0, atol=1e-3)
        assert np.all(zones == 18)
        assert np.all(hemispheres == "N")

    def test_batch_mixed_hemispheres_matches_epsg(self, pyproj):
        lat_deg = np.array([40.0, -42.0])
        lon_deg = np.array([-75.0, -73.0])

        eastings, northings, zones, hemispheres = geodetic2utm_batch(
            np.radians(lat_deg), np.radians(lon_deg)
        )

        for i, epsg in enumerate((32618, 32718)):
            transformer = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
            east, north = transformer.transform(lon_deg[i], lat_deg[i])
            assert abs(eastings[i] - east) < 1e-3
            assert abs(northings[i] - north) < 1e-3
        assert list(hemispheres) == ["N", "S"]


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


class TestGridConvergenceMatchesTheProjection:
    """Every producer's ``convergence`` must be the real grid convergence.

    ``ProjectionResult.convergence`` is documented as "angle from true north
    to grid north". It is checkable directly: project a point, project it
    again a hair further north, and the bearing of that step in grid axes is
    the negated convergence. Nothing did that, and two of the six producers
    were filling the field with a back-azimuth instead -- ``stereographic``
    and ``azimuthal_equidistant`` both reported 180 degrees for a point
    sitting on the projection's own central meridian, where the true answer
    is zero.
    """

    @staticmethod
    def _measure(fn, args, h=1e-7):
        """Convergence from finite differences along the meridian."""
        moved = list(args)
        moved[0] += h
        north = fn(*moved)
        moved[0] -= 2 * h
        south = fn(*moved)
        return -np.arctan2(north.x - south.x, north.y - south.y)

    CASES = [
        ("transverse_mercator", transverse_mercator, (np.radians(45), np.radians(1))),
        ("polar_stereographic", polar_stereographic, (np.radians(80), np.radians(45))),
        (
            "lambert_conformal_conic",
            lambert_conformal_conic,
            (
                np.radians(40),
                np.radians(-100),
                np.radians(39),
                np.radians(-96),
                np.radians(33),
                np.radians(45),
            ),
        ),
        (
            "stereographic_on_central_meridian",
            stereographic,
            (np.radians(1), 0.0, 0.0, 0.0),
        ),
        (
            "stereographic_off_meridian",
            stereographic,
            (np.radians(45), np.radians(1), np.radians(45), 0.0),
        ),
        (
            "azimuthal_equidistant_due_north",
            azimuthal_equidistant,
            (np.radians(1), 0.0, 0.0, 0.0),
        ),
        (
            "azimuthal_equidistant_due_east",
            azimuthal_equidistant,
            (0.0, np.radians(1), 0.0, 0.0),
        ),
        (
            "azimuthal_equidistant_far_field",
            azimuthal_equidistant,
            (np.radians(20), np.radians(30), np.radians(-5), np.radians(-10)),
        ),
    ]

    @pytest.mark.parametrize("name,fn,args", CASES, ids=[c[0] for c in CASES])
    def test_reported_convergence_equals_measured(self, name, fn, args):
        reported = fn(*args).convergence
        measured = self._measure(fn, args)
        assert_allclose(reported, measured, atol=1e-6)

    def test_a_point_on_the_central_meridian_has_zero_convergence(self):
        """The case the old formulas got most visibly wrong.

        Directly north of the centre, the meridian through the point *is*
        the map's y-axis, so grid north and true north coincide. Both
        projections used to report pi here.
        """
        for fn in (stereographic, azimuthal_equidistant):
            result = fn(np.radians(1), 0.0, 0.0, 0.0)
            assert abs(result.convergence) < 1e-9, fn.__name__

    def test_azimuthal_equidistant_is_not_merely_the_conformal_form(self):
        """AEQD is not conformal, so it needs the c/sin(c) tangential term.

        Far from the centre the two forms diverge; this pins that the
        non-conformal one is what ships.
        """
        args = (np.radians(35), np.radians(60), np.radians(-10), np.radians(-20))
        reported = azimuthal_equidistant(*args).convergence
        measured = self._measure(azimuthal_equidistant, args)
        assert_allclose(reported, measured, atol=1e-6)

        lat, lon, lat0, lon0 = args
        dlon = lon - lon0
        azimuth = np.arctan2(
            np.cos(lat) * np.sin(dlon),
            np.cos(lat0) * np.sin(lat) - np.sin(lat0) * np.cos(lat) * np.cos(dlon),
        )
        back = np.arctan2(
            -np.cos(lat0) * np.sin(dlon),
            np.cos(lat) * np.sin(lat0) - np.sin(lat) * np.cos(lat0) * np.cos(dlon),
        )
        conformal_form = np.arctan2(
            np.sin(back - azimuth - np.pi), np.cos(back - azimuth - np.pi)
        )
        assert abs(reported - conformal_form) > 1e-3


class TestArrayInputIsNotMutated:
    """`transverse_mercator` (and `geodetic2utm`, which calls it) used to
    alias its `lat`/`lat0` array arguments (`sigma = lat` followed by an
    in-place `sigma -= ...`) and corrupt them in place across the Helmert
    series loop, so a second call on the same array silently returned a
    different, wrong answer (gh-25 follow-up).
    """

    def test_geodetic2utm_repeated_calls_agree_and_leave_lat_unchanged(self):
        lat = np.array(np.radians(45.0))
        lon = np.array(np.radians(-75.0))
        lat_before = lat.copy()

        first = geodetic2utm(lat, lon, zone=18)
        assert_allclose(lat, lat_before)

        second = geodetic2utm(lat, lon, zone=18)
        third = geodetic2utm(lat, lon, zone=18)

        assert_allclose(lat, lat_before)
        assert_allclose(second.northing, first.northing)
        assert_allclose(third.northing, first.northing)
        assert_allclose(second.easting, first.easting)

    def test_transverse_mercator_leaves_lat_and_lat0_unchanged(self):
        lat = np.array(np.radians(45.0))
        lat0 = np.array(np.radians(40.0))
        lon = np.array(np.radians(-75.0))
        lon0 = np.radians(-75.0)
        lat_before, lat0_before = lat.copy(), lat0.copy()

        first = transverse_mercator(lat, lon, lat0, lon0)
        second = transverse_mercator(lat, lon, lat0, lon0)

        assert_allclose(lat, lat_before)
        assert_allclose(lat0, lat0_before)
        assert_allclose(second.y, first.y)
        assert_allclose(second.x, first.x)

    def test_transverse_mercator_inverse_leaves_lat0_unchanged(self):
        x = np.array(500000.0)
        y = np.array(4000000.0)
        lat0 = np.array(np.radians(40.0))
        lat0_before = lat0.copy()

        first = transverse_mercator_inverse(x, y, lat0, np.radians(-75), k0=0.9996)
        transverse_mercator_inverse(x, y, lat0, np.radians(-75), k0=0.9996)

        assert_allclose(lat0, lat0_before)
        assert_allclose(
            transverse_mercator_inverse(x, y, lat0, np.radians(-75), k0=0.9996),
            first,
        )


class TestWrapLongitudeDifferenceArrayPassthrough:
    """The gh-25 wrap helper used to cast its result through `float()`,
    which raises for array input; before gh-25 it silently accepted arrays
    (per-element scalar calls agreed), so array support must come back
    without reintroducing the pre-gh-25 unwrapped-difference bug -- hence
    the same case is exercised against `mercator`, `lambert_conformal_conic`,
    and `transverse_mercator_inverse` below.
    """

    def test_mercator_accepts_size_one_array(self):
        # Only size-1 (or 0-d) array input is claimed here: `mercator`'s
        # scale factor branches on `cos_lat > 1e-10` with a Python `if`,
        # which is ambiguous for a multi-element array and is a pre-existing
        # limitation unrelated to the wrap helper -- out of scope for this
        # patch (not part of gh-25 or its follow-up).
        lat = np.array([np.radians(45.0)])
        lon = np.array([np.radians(-75.0)])
        result = mercator(lat, lon, lon0=0.0)
        assert result.x.shape == (1,)
        scalar = mercator(lat[0], lon[0], lon0=0.0)
        assert_allclose(result.x[0], scalar.x)

    def test_lambert_conformal_conic_accepts_array_input(self):
        lat = np.radians(np.array([30.0, 45.0, 60.0]))
        lon = np.radians(np.array([-100.0, -95.0, -90.0]))
        lat0, lon0 = np.radians(40.0), np.radians(-96.0)
        lat1, lat2 = np.radians(33.0), np.radians(45.0)

        result = lambert_conformal_conic(lat, lon, lat0, lon0, lat1, lat2)
        for i in range(lat.size):
            scalar = lambert_conformal_conic(lat[i], lon[i], lat0, lon0, lat1, lat2)
            assert_allclose(result.x[i], scalar.x)
            assert_allclose(result.y[i], scalar.y)

    def test_transverse_mercator_inverse_accepts_array_input(self):
        x = np.array([500000.0, 500100.0])
        y = np.array([4000000.0, 4000100.0])
        lat, lon = transverse_mercator_inverse(x, y, lon0=np.radians(-75), k0=0.9996)
        for i in range(x.size):
            slat, slon = transverse_mercator_inverse(
                x[i], y[i], lon0=np.radians(-75), k0=0.9996
            )
            # Bit-exact, not merely close: any in-place accumulation left in
            # the footpoint-latitude series would show up here as ~1e-10 rad.
            assert lat[i] == slat
            assert lon[i] == slon
