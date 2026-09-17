"""Antimeridian-crossing regression tests for the projection forward paths.

Oracle: PROJ via pyproj. `geodetic2utm`, `mercator`, and `transverse_mercator`
computed `x = a * (lon - lon0)` (or its series equivalent) without wrapping
the longitude difference into [-pi, pi], so a forced UTM zone or a central
meridian near +-180 degrees produced eastings in the billions of metres for
any point on the far side of the seam (gh-25 follow-up). Skips without
pyproj.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.projections import (
    geodetic2utm,
    geodetic2utm_batch,
    mercator,
    transverse_mercator,
    utm2geodetic,
    utm_central_meridian,
)

pyproj = pytest.importorskip("pyproj")
from pyproj import Transformer  # noqa: E402

ANTIMERIDIAN_CASES = [
    (10.0, 179.9, 1),
    (-20.0, 178.0, 1),
    (40.0, -179.5, 60),
    (0.0, -180.0, 1),
    (65.0, 179.999, 60),
]


def _utm_proj(zone: int, lat: float) -> Transformer:
    south = " +south" if lat < 0 else ""
    return Transformer.from_crs(
        "EPSG:4326", f"+proj=utm +zone={zone}{south} +datum=WGS84", always_xy=True
    )


@pytest.mark.parametrize("lat,lon,zone", ANTIMERIDIAN_CASES)
def test_forced_zone_utm_matches_proj_across_the_antimeridian(lat, lon, zone):
    want_e, want_n = _utm_proj(zone, lat).transform(lon, lat)
    got = geodetic2utm(np.radians(lat), np.radians(lon), zone=zone)
    assert got.easting == pytest.approx(want_e, abs=1e-3)
    assert got.northing == pytest.approx(want_n, abs=1e-3)
    assert got.scale == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("lat,lon,zone", ANTIMERIDIAN_CASES)
def test_geodetic2utm_batch_matches_forced_zone_across_the_antimeridian(lat, lon, zone):
    want_e, want_n = _utm_proj(zone, lat).transform(lon, lat)
    eastings, northings, zones, hemispheres = geodetic2utm_batch(
        np.array([np.radians(lat)]), np.array([np.radians(lon)]), zone=zone
    )
    assert eastings[0] == pytest.approx(want_e, abs=1e-3)
    assert northings[0] == pytest.approx(want_n, abs=1e-3)


def test_mercator_matches_proj_across_the_antimeridian():
    lon0, lat, lon = np.radians(170.0), np.radians(0.0), np.radians(-170.0)
    proj = Transformer.from_crs(
        "EPSG:4326",
        "+proj=merc +lon_0=170 +datum=WGS84",
        always_xy=True,
    )
    want_x, want_y = proj.transform(-170.0, 0.0)
    r = mercator(lat, lon, lon0=lon0)
    assert r.x == pytest.approx(want_x, abs=1e-3)
    assert r.y == pytest.approx(want_y, abs=1e-3)


@pytest.mark.parametrize("lat,lon,zone", ANTIMERIDIAN_CASES)
def test_transverse_mercator_matches_proj_utm_across_the_antimeridian(lat, lon, zone):
    lon0 = utm_central_meridian(zone)
    want_e, want_n = _utm_proj(zone, lat).transform(lon, lat)
    r = transverse_mercator(
        np.radians(lat), np.radians(lon), lat0=0.0, lon0=lon0, k0=0.9996
    )
    got_e = r.x + 500000.0
    got_n = r.y if lat >= 0 else r.y + 10000000.0
    assert got_e == pytest.approx(want_e, abs=1e-3)
    assert got_n == pytest.approx(want_n, abs=1e-3)
    assert r.scale == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("lat,lon,zone", ANTIMERIDIAN_CASES)
def test_utm_roundtrip_closes_across_the_antimeridian(lat, lon, zone):
    fwd = geodetic2utm(np.radians(lat), np.radians(lon), zone=zone)
    got_lat, got_lon = utm2geodetic(fwd.easting, fwd.northing, fwd.zone, fwd.hemisphere)
    assert got_lat == pytest.approx(np.radians(lat), abs=1e-9)
    lon_error = np.arctan2(
        np.sin(got_lon - np.radians(lon)), np.cos(got_lon - np.radians(lon))
    )
    assert lon_error == pytest.approx(0.0, abs=1e-9)
