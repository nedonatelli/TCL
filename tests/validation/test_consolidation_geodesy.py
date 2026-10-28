"""Geodesy on the sphere and ellipsoid consolidated (v2.12.0 Task 6.5).

Row 15, haversine. ``great_circle_distance`` evaluated the haversine on
inputs rounded to 10 decimals (up to 0.64 mm on the ground); it now calls
``geodesy.haversine_distance`` on the exact coordinates, caching on them.
The terrain inline copies call it too.

Row 16, meridional radius. The inline ``a (1 - e2) / (1 - e2 sin^2)^1.5``
copies call one body, ``geodetic._meridional_radius_e2``; STABLE
``meridional_radius(lat, a, f)`` is a wrapper over it.

Row 18, the 6371000 m mean Earth radius is ``EARTH_MEAN_RADIUS``.
"""

import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from pytcl.coordinate_systems.conversions import meridional_radius
from pytcl.coordinate_systems.projections import projections
from pytcl.core import constants
from pytcl.core.constants import EARTH_MEAN_RADIUS
from pytcl.navigation import geodesy, great_circle, ins, rhumb
from pytcl.terrain import dem as dem_mod
from pytcl.terrain import visibility
from tests.validation._v2111 import shipped_module

ELLIPSOIDS = {
    "wgs84": (geodesy.WGS84.a, geodesy.WGS84.f),
    "grs80": (geodesy.GRS80.a, geodesy.GRS80.f),
    "sphere": (EARTH_MEAN_RADIUS, 0.0),
    "hayford": (6378388.0, 1 / 297.0),
}


def _ellipsoids(mod):
    return {k: mod.Ellipsoid(a=a, f=f) for k, (a, f) in ELLIPSOIDS.items()}


@pytest.fixture(scope="module")
def old_geodesy():
    return shipped_module("pytcl/navigation/geodesy.py")


@pytest.fixture(scope="module")
def old_gc():
    return shipped_module("pytcl/navigation/great_circle.py")


@pytest.fixture(scope="module")
def old_ins():
    return shipped_module("pytcl/navigation/ins.py")


@pytest.fixture(scope="module")
def old_rhumb():
    return shipped_module("pytcl/navigation/rhumb.py")


@pytest.fixture(scope="module")
def old_proj():
    return shipped_module("pytcl/coordinate_systems/projections/projections.py")


@pytest.fixture(scope="module")
def old_conv():
    return shipped_module("pytcl/coordinate_systems/conversions/geodetic.py")


@pytest.fixture(scope="module")
def old_vis():
    return shipped_module("pytcl/terrain/visibility.py")


@pytest.fixture(scope="module")
def old_dem():
    return shipped_module("pytcl/terrain/dem.py")


def _same(new, old):
    """Bitwise equal, same type, same shape, same dtype."""
    assert type(new) is type(old)
    new_a, old_a = np.asarray(new), np.asarray(old)
    assert new_a.shape == old_a.shape
    assert new_a.dtype == old_a.dtype
    np.testing.assert_array_equal(new_a, old_a)


def _pairs(n, seed):
    rng = np.random.default_rng(seed)
    lat = rng.uniform(-np.pi / 2, np.pi / 2, (n, 2))
    lon = rng.uniform(-np.pi, np.pi, (n, 2))
    return lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1]


LAT_INPUTS = {
    "float": 0.7,
    "np_float64": np.float64(0.7),
    "np_float32": np.float32(0.7),
    "int": 1,
    "array": np.linspace(-1.5, 1.5, 33),
    "array_f32": np.linspace(-1.5, 1.5, 33, dtype=np.float32),
    "array_2d": np.linspace(-1.5, 1.5, 12).reshape(3, 4),
    "list": [0.1, 0.2, 0.3],
}


class TestMeridionalRadius:
    @pytest.mark.parametrize("name", list(LAT_INPUTS))
    @pytest.mark.parametrize("ell", list(ELLIPSOIDS))
    def test_stable_wrapper_matches_v2111(self, old_conv, name, ell):
        a, f = ELLIPSOIDS[ell]
        _same(
            meridional_radius(LAT_INPUTS[name], a, f),
            old_conv.meridional_radius(LAT_INPUTS[name], a, f),
        )

    def test_stable_wrapper_defaults_match_v2111(self, old_conv):
        lat = np.linspace(-1.5, 1.5, 200)
        _same(meridional_radius(lat), old_conv.meridional_radius(lat))

    @pytest.mark.parametrize("name", list(LAT_INPUTS))
    @pytest.mark.parametrize("ell", list(ELLIPSOIDS))
    def test_radii_of_curvature_bitwise(self, old_ins, name, ell):
        new_e, old_e = _ellipsoids(geodesy)[ell], _ellipsoids(old_ins)[ell]
        new = ins.radii_of_curvature(LAT_INPUTS[name], new_e)
        old = old_ins.radii_of_curvature(LAT_INPUTS[name], old_e)
        for n, o in zip(new, old, strict=True):
            _same(n, o)

    @pytest.mark.parametrize("ell", list(ELLIPSOIDS))
    def test_rhumb_ellipsoidal_bitwise(self, old_rhumb, ell):
        new_e, old_e = _ellipsoids(geodesy)[ell], _ellipsoids(old_rhumb)[ell]
        lat1, lon1, lat2, lon2 = _pairs(150, 3)
        # includes the east-west branch, which reads the sin(lat_mid) too
        lat2 = np.concatenate([lat2, lat1[:10]])
        lat1 = np.concatenate([lat1, lat1[:10]])
        lon1 = np.concatenate([lon1, lon1[:10]])
        lon2 = np.concatenate([lon2, lon2[:10] + 0.3])
        for args in zip(lat1, lon1, lat2, lon2, strict=True):
            _same(
                rhumb.rhumb_distance_ellipsoidal(*args, new_e),
                old_rhumb.rhumb_distance_ellipsoidal(*args, old_e),
            )

    @pytest.mark.parametrize("ell", list(ELLIPSOIDS))
    def test_direct_rhumb_bitwise(self, old_rhumb, ell):
        new_e, old_e = _ellipsoids(geodesy)[ell], _ellipsoids(old_rhumb)[ell]
        rng = np.random.default_rng(4)
        for _ in range(150):
            lat = rng.uniform(-1.4, 1.4)
            lon = rng.uniform(-3, 3)
            brg = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(1e3, 5e6)
            new = rhumb.direct_rhumb(lat, lon, brg, dist, new_e)
            old = old_rhumb.direct_rhumb(lat, lon, brg, dist, old_e)
            assert tuple(new) == tuple(old)

    @pytest.mark.parametrize("e2", [0.0, 0.00669437999014, 0.0067, 0.05])
    def test_projections_bitwise(self, old_proj, e2):
        a = 6378137.0
        rng = np.random.default_rng(5)
        for _ in range(100):
            lat0 = rng.uniform(-1.2, 1.2)
            lon0 = rng.uniform(-3, 3)
            lat = lat0 + rng.uniform(-0.05, 0.05)
            lon = lon0 + rng.uniform(-0.05, 0.05)
            x, y = rng.uniform(-2e5, 2e5, 2)
            fwd = projections.oblique_stereographic(lat, lon, lat0, lon0, 0.9999, a, e2)
            fwd_o = old_proj.oblique_stereographic(lat, lon, lat0, lon0, 0.9999, a, e2)
            assert tuple(fwd) == tuple(fwd_o)
            inv = projections.oblique_stereographic_inverse(
                x, y, lat0, lon0, 0.9999, a, e2
            )
            inv_o = old_proj.oblique_stereographic_inverse(
                x, y, lat0, lon0, 0.9999, a, e2
            )
            assert tuple(inv) == tuple(inv_o)
            tm = projections.transverse_mercator_inverse(
                x, y, lat0, lon0, 0.9996, a, e2
            )
            tm_o = old_proj.transverse_mercator_inverse(x, y, lat0, lon0, 0.9996, a, e2)
            assert tuple(tm) == tuple(tm_o)


class TestHaversine:
    def test_haversine_distance_matches_v2111(self, old_geodesy):
        lat1, lon1, lat2, lon2 = _pairs(500, 6)
        _same(
            geodesy.haversine_distance(lat1, lon1, lat2, lon2),
            old_geodesy.haversine_distance(lat1, lon1, lat2, lon2),
        )
        _same(
            geodesy.haversine_distance(0.3, 0.2, 0.5, -0.4, 1234.5),
            old_geodesy.haversine_distance(0.3, 0.2, 0.5, -0.4, 1234.5),
        )

    def test_great_circle_distance_within_quantization_bound(self, old_gc):
        lat1, lon1, lat2, lon2 = _pairs(2000, 7)
        worst = 0.0
        for args in zip(lat1, lon1, lat2, lon2, strict=True):
            new = great_circle.great_circle_distance(*args)
            old = old_gc.great_circle_distance(*args)
            assert type(new) is type(old)
            worst = max(worst, abs(new - old))
        # Four coordinates, each moved by at most half of 1e-10 rad, and a
        # unit change in any one moves the distance by at most R rad.
        assert 1e-4 < worst <= 4 * 0.5e-10 * EARTH_MEAN_RADIUS

    def test_great_circle_distance_radius_and_types(self, old_gc):
        for args in [
            (0, 0, 1, 1),
            (np.float32(0.1), np.float32(0.2), np.float32(0.3), np.float32(0.4)),
            (np.float64(0.1), 0.2, 0.3, np.float64(0.4)),
        ]:
            for radius in (None, 1.0, 1000.0):
                kw = {} if radius is None else {"radius": radius}
                new = great_circle.great_circle_distance(*args, **kw)
                old = old_gc.great_circle_distance(*args, **kw)
                assert type(new) is type(old)
                assert new == pytest.approx(old, abs=7e-4 * (radius or 1e7) / 6.4e6)

    def test_great_circle_distance_array_input_still_unhashable(self, old_gc):
        a = np.array([0.1, 0.2])
        with pytest.raises(TypeError):
            old_gc.great_circle_distance(a, 0.0, 0.3, 0.4)
        with pytest.raises(TypeError):
            great_circle.great_circle_distance(a, 0.0, 0.3, 0.4)

    def test_great_circle_distance_matches_geographiclib_sphere(self):
        geographiclib = pytest.importorskip("geographiclib.geodesic")
        sphere = geographiclib.Geodesic(6371000.0, 0)
        lat1, lon1, lat2, lon2 = _pairs(2000, 8)
        # close pairs: the failure of rounding is relative, so it shows there
        rng = np.random.default_rng(9)
        lat1 = np.concatenate([lat1, lat1[:500]])
        lon1 = np.concatenate([lon1, lon1[:500]])
        lat2 = np.concatenate([lat2, lat1[:500] + rng.uniform(-1e-5, 1e-5, 500)])
        lon2 = np.concatenate([lon2, lon1[:500] + rng.uniform(-1e-5, 1e-5, 500)])
        worst = 0.0
        for args in zip(lat1, lon1, lat2, lon2, strict=True):
            ref = sphere.Inverse(*np.degrees(args))["s12"]
            got = great_circle.great_circle_distance(*args)
            worst = max(worst, abs(got - ref) / ref)
        assert worst < 1e-9

    def test_cache_is_keyed_on_exact_coordinates(self):
        great_circle.clear_great_circle_cache()
        lat1, lon1, lat2, lon2 = 0.7, 0.1, 0.9, 0.5
        d0 = great_circle.great_circle_distance(lat1, lon1, lat2, lon2)
        d1 = great_circle.great_circle_distance(lat1 + 3e-11, lon1, lat2, lon2)
        assert d0 != d1
        assert great_circle.get_cache_info()["distance"]["currsize"] == 2
        assert great_circle.great_circle_distance(lat1, lon1, lat2, lon2) == d0
        assert great_circle.get_cache_info()["distance"]["hits"] == 1


@pytest.fixture(scope="module")
def dem():
    rng = np.random.default_rng(10)
    return dem_mod.DEMGrid(
        rng.uniform(0, 800, (40, 50)),
        np.radians(35.0),
        np.radians(36.0),
        np.radians(-120.0),
        np.radians(-119.0),
    )


class TestTerrainSites:
    def test_line_of_sight_bitwise(self, old_vis, dem):
        rng = np.random.default_rng(11)
        for _ in range(30):
            la = np.radians(rng.uniform(35.05, 35.95, 2))
            lo = np.radians(rng.uniform(-119.95, -119.05, 2))
            args = (dem, la[0], lo[0], 20.0, la[1], lo[1], 15.0, 64)
            for kw in ({}, {"earth_radius": 6378137.0, "refraction_coeff": 0.13}):
                assert tuple(visibility.line_of_sight(*args, **kw)) == tuple(
                    old_vis.line_of_sight(*args, **kw)
                )

    def test_radar_coverage_map_bitwise(self, old_vis, dem):
        args = (dem, np.radians(35.5), np.radians(-119.5), 30.0)
        kw = dict(
            min_elevation=np.radians(0.05),
            max_range=20000.0,
            n_radials=12,
            samples_per_radial=15,
        )
        new = visibility.radar_coverage_map(*args, **kw)
        old = old_vis.radar_coverage_map(*args, **kw)
        for n, o in zip(new, old, strict=True):
            np.testing.assert_array_equal(n, o)

    def test_radar_coverage_cell_distances_are_the_inline_haversine(
        self, dem, monkeypatch
    ):
        seen = []
        real = visibility.haversine_distance

        def spy(lat1, lon1, lat2, lon2, radius):
            out = real(lat1, lon1, lat2, lon2, radius)
            seen.append((lat1, lon1, lat2, lon2, radius, out))
            return out

        monkeypatch.setattr(visibility, "haversine_distance", spy)
        visibility.radar_coverage_map(
            dem,
            np.radians(35.5),
            np.radians(-119.5),
            30.0,
            min_elevation=np.radians(0.05),
            max_range=20000.0,
            n_radials=12,
            samples_per_radial=15,
        )
        cells = [c for c in seen if np.ndim(c[2]) == 0]
        assert cells
        for lat1, lon1, lat2, lon2, radius, out in cells:
            assert radius == EARTH_MEAN_RADIUS
            dlat, dlon = lat2 - lat1, lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            assert out == radius * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

    def test_elevation_profile_bitwise(self, old_dem, dem):
        args = (dem, np.radians(35.1), np.radians(-119.9), np.radians(35.9))
        args += (np.radians(-119.1), 57)
        new = dem_mod.get_elevation_profile(*args)
        old = old_dem.get_elevation_profile(*args)
        for n, o in zip(new, old, strict=True):
            _same(n, o)


class TestEarthRadius:
    def test_constants_are_the_one_radius(self, old_geodesy, old_gc, old_rhumb):
        assert geodesy.SPHERE.a == EARTH_MEAN_RADIUS == old_geodesy.SPHERE.a
        assert great_circle.EARTH_RADIUS == EARTH_MEAN_RADIUS
        assert great_circle.EARTH_RADIUS == old_gc.EARTH_RADIUS
        assert rhumb.EARTH_RADIUS == EARTH_MEAN_RADIUS == old_rhumb.EARTH_RADIUS

    @staticmethod
    def _radius_defaults(mod):
        out = {}
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if fn.__module__ != mod.__name__:
                continue
            for pname in ("radius", "earth_radius"):
                p = inspect.signature(fn).parameters.get(pname)
                if p is not None:
                    out[f"{name}.{pname}"] = p.default
        return out

    @pytest.mark.parametrize("mod", [geodesy, great_circle, rhumb, visibility])
    def test_every_default_is_the_constant_itself(self, mod):
        defaults = self._radius_defaults(mod)
        assert defaults
        assert all(v is EARTH_MEAN_RADIUS for v in defaults.values()), defaults

    def test_dem_gradient_default(self):
        p = inspect.signature(dem_mod.DEMGrid.get_gradient).parameters
        assert p["earth_radius"].default is EARTH_MEAN_RADIUS

    def test_sphere_earth_constant(self):
        assert constants.SPHERE_EARTH.a is EARTH_MEAN_RADIUS

    # Spellings of the mean radius in metres: 6371000, 6_371_000, 6371000.0,
    # 6.371e6, 6371e3, with any exponent sign or case; and in kilometres:
    # 6371.0, 6371e0. The 6371.2 km geomagnetic reference radius is a different
    # constant and does not match.
    _LITERAL = re.compile(
        r"6_?371_?000(?![\d_])|6\.371[eE]\+?0?6\b|6371[eE]\+?0?3\b"
        r"|(?<![\d.])6371\.0+(?!\d)|(?<![\d.])6371[eE]\+?0+(?!\d)"
    )
    # (path, line text) pairs that spell it and are not a second definition.
    _ALLOWED = (
        ("core/constants.py", "EARTH_MEAN_RADIUS: Final[float] = 6_371_000.0"),
        # The prose in these docstrings names the constant being defaulted.
        ("terrain/dem.py", "``EARTH_MEAN_RADIUS`` (6371000.0)"),
        ("terrain/visibility.py", "``EARTH_MEAN_RADIUS`` (6371000.0)"),
        # Doctest examples of an illustrative surface radius for the
        # time-dilation formula; not a default and not the mean radius's job.
        ("astronomical/relativity.py", "r_earth = 6.371e6"),
        ("astronomical/relativity.py", "r_ground = 6.371e6"),
    )

    def test_mean_radius_is_spelled_once_in_the_package(self):
        root = Path(inspect.getfile(constants)).parents[1]
        hits = []
        for path in root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if self._LITERAL.search(line) and not any(
                    rel == r and text in line for r, text in self._ALLOWED
                ):
                    hits.append(f"{rel}:{n}: {line.strip()}")
        assert not hits, hits

    def test_literal_scan_sees_every_spelling(self):
        for text in (
            "6371000",
            "6_371_000",
            "6371000.0",
            "6.371e6",
            "6371e3",
            "6371.0",
            "6371.00",
            "6371e0",
        ):
            assert self._LITERAL.search(f"a = {text}"), text
        assert not self._LITERAL.search("r = 6371.2")
        assert not self._LITERAL.search("r = 16371.0")


class TestAzimuth:
    def _short_pairs(self, sep, seed):
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(1000):
            lat, lon = rng.uniform(-1.4, 1.4), rng.uniform(-3, 3)
            th = rng.uniform(0, 2 * np.pi)
            s = sep / EARTH_MEAN_RADIUS
            out.append(
                (lat, lon, lat + s * np.cos(th), lon + s * np.sin(th) / np.cos(lat))
            )
        return out

    @staticmethod
    def _err(fn, args, sphere):
        ref = np.radians(sphere.Inverse(*np.degrees(args))["azi1"]) % (2 * np.pi)
        d = abs(fn(*args) - ref)
        return min(d, 2 * np.pi - d)

    # Measured (1000-2000 pairs, seed-dependent maxima): global 2.2e-15 rad;
    # separations 64 m / 1 m / 0.01 m: 4.3e-11 / 2.3e-9 / 2.9e-7 rad
    # (v2.11.1: 1.3e-9 / 1.0e-5 / 7.3e-4 / 6.3e-2). The reference takes
    # degrees, so part of the short-pair figure is the oracle's.
    @pytest.mark.parametrize(
        ("sep", "bound"), [(None, 1e-12), (64.0, 1e-9), (1.0, 1e-7), (0.01, 1e-5)]
    )
    def test_matches_geographiclib_sphere(self, sep, bound):
        geographiclib = pytest.importorskip("geographiclib.geodesic")
        sphere = geographiclib.Geodesic(6371000.0, 0)
        if sep is None:
            rng = np.random.default_rng(1)
            pairs = list(
                zip(
                    rng.uniform(-1.4, 1.4, 1000),
                    rng.uniform(-3, 3, 1000),
                    rng.uniform(-1.4, 1.4, 1000),
                    rng.uniform(-3, 3, 1000),
                    strict=True,
                )
            )
        else:
            pairs = self._short_pairs(sep, 2)
        worst = max(
            self._err(great_circle.great_circle_azimuth, p, sphere) for p in pairs
        )
        assert worst < bound

    def test_inverse_agrees_with_the_individual_functions_bitwise(self):
        near = (0.7, 0.1, 0.7 + 3e-11, 0.1 + 2e-11)
        for args in [*zip(*_pairs(300, 12), strict=True), near]:
            r = great_circle.great_circle_inverse(*args)
            lat1, lon1, lat2, lon2 = args
            assert r.distance == great_circle.great_circle_distance(*args)
            assert r.azimuth1 == great_circle.great_circle_azimuth(*args)
            back = great_circle.great_circle_azimuth(lat2, lon2, lat1, lon1)
            assert r.azimuth2 == (back + np.pi) % (2 * np.pi)

    def test_sub_millimetre_pair_has_a_real_azimuth(self, old_gc):
        args = (0.7, 0.1, 0.7 + 3e-11, 0.1 + 2e-11)
        assert old_gc.great_circle_azimuth(*args) == 0.0
        assert great_circle.great_circle_azimuth(*args) == pytest.approx(0.46, abs=0.02)

    def test_global_pairs_move_only_by_the_old_rounding(self, old_gc):
        # measured 1.3e-9 rad over 2000 pairs
        for args in zip(*_pairs(500, 13), strict=True):
            d = abs(
                great_circle.great_circle_azimuth(*args)
                - old_gc.great_circle_azimuth(*args)
            )
            assert min(d, 2 * np.pi - d) < 1e-7

    def test_cache_is_keyed_on_exact_coordinates(self):
        great_circle.clear_great_circle_cache()
        a = great_circle.great_circle_azimuth(0.7, 0.1, 0.9, 0.5)
        b = great_circle.great_circle_azimuth(0.7 + 3e-11, 0.1, 0.9, 0.5)
        assert a != b
        info = great_circle.get_cache_info()["azimuth"]
        assert info["currsize"] == 2 and info["hits"] == 0
        assert not hasattr(great_circle, "_quantize_coord")


class TestOsculatingSphere:
    @pytest.mark.parametrize("ell", ["wgs84", "grs80", "hayford"])
    def test_matches_v2111_to_one_ulp(self, old_geodesy, ell):
        new_e, old_e = _ellipsoids(geodesy)[ell], _ellipsoids(old_geodesy)[ell]
        worst = 0.0
        for lat in np.linspace(-1.55, 1.55, 400):
            n = geodesy.osculating_sphere(lat, 0.3, new_e)
            o = old_geodesy.osculating_sphere(lat, 0.3, old_e)
            assert type(n.radius) is type(o.radius)
            worst = max(worst, abs(n.radius - o.radius) / o.radius)
            np.testing.assert_allclose(n.center, o.center, rtol=1e-12, atol=1e-8)
        # measured 2.9e-16 (1 ulp) in about 28% of cases
        assert worst <= 4.5e-16

    def test_uses_the_shared_meridional_body(self, monkeypatch):
        calls = []
        real = geodesy._meridional_radius_e2

        def spy(lat, a, e2):
            calls.append((lat, a, e2))
            return real(lat, a, e2)

        monkeypatch.setattr(geodesy, "_meridional_radius_e2", spy)
        geodesy.osculating_sphere(0.7, 0.1)
        assert calls == [(0.7, geodesy.WGS84.a, geodesy.WGS84.e2)]

    def test_shared_body_is_the_formula(self):
        from pytcl.coordinate_systems.conversions import geodetic

        lat, a, e2 = np.linspace(-1.5, 1.5, 7), 6378137.0, 0.00669437999014
        want = a * (1 - e2) / (1 - e2 * np.sin(lat) ** 2) ** 1.5
        np.testing.assert_array_equal(geodetic._meridional_radius_e2(lat, a, e2), want)
