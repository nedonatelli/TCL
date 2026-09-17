"""Antimeridian-crossing regression tests for the projections module.

Oracle: PROJ via pyproj. `geodetic2utm`, `mercator`, and `transverse_mercator`
computed `x = a * (lon - lon0)` (or its series equivalent) without wrapping
the longitude difference into [-pi, pi], so a forced UTM zone or a central
meridian near +-180 degrees produced eastings in the billions of metres for
any point on the far side of the seam (gh-25 follow-up). Skips without
pyproj.

The paired inverse functions (`utm2geodetic` / `transverse_mercator_inverse`,
`mercator_inverse`, `stereographic_inverse`, `azimuthal_equidistant_inverse`)
recovered the correct point but as a longitude outside [-pi, pi] whenever
`lon0 + offset` crossed the seam -- mathematically exact, but a value a naive
caller would not expect and could not directly compare against a canonical
reference. They now canonicalize their output through the same
`_wrap_longitude_difference` helper used by the forward fix above.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.projections import (
    azimuthal_equidistant,
    azimuthal_equidistant_inverse,
    geodetic2utm,
    geodetic2utm_batch,
    mercator,
    mercator_inverse,
    stereographic,
    stereographic_inverse,
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
    # The range check is the part that would have caught the un-canonicalized
    # inverse: -180.1, -182.0 and 180.5 degrees all "closed" under the
    # wrap-aware comparison above while sitting outside [-pi, pi].
    assert -np.pi <= got_lon <= np.pi


MERIDIAN_SEAM_CASES = [
    (11.0, -179.5, 179.0),
    (11.0, 178.5, -179.0),
    (5.0, 179.9, -179.9),
]


@pytest.mark.parametrize("lat,lon,lon0", MERIDIAN_SEAM_CASES)
def test_mercator_roundtrip_closes_and_is_canonical_across_the_antimeridian(
    lat, lon, lon0
):
    lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(lon0)
    fwd = mercator(lat_r, lon_r, lon0=lon0_r)
    _, got_lon = mercator_inverse(fwd.x, fwd.y, lon0=lon0_r)
    lon_error = np.arctan2(np.sin(got_lon - lon_r), np.cos(got_lon - lon_r))
    assert lon_error == pytest.approx(0.0, abs=1e-9)
    assert -np.pi <= got_lon <= np.pi


@pytest.mark.parametrize("lat,lon,lon0", MERIDIAN_SEAM_CASES)
def test_stereographic_roundtrip_closes_and_is_canonical_across_the_antimeridian(
    lat, lon, lon0
):
    lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(lon0)
    lat0_r = np.radians(10.0)
    fwd = stereographic(lat_r, lon_r, lat0_r, lon0_r)
    got_lat, got_lon = stereographic_inverse(fwd.x, fwd.y, lat0_r, lon0_r)
    assert got_lat == pytest.approx(lat_r, abs=1e-9)
    lon_error = np.arctan2(np.sin(got_lon - lon_r), np.cos(got_lon - lon_r))
    assert lon_error == pytest.approx(0.0, abs=1e-9)
    assert -np.pi <= got_lon <= np.pi


@pytest.mark.parametrize("lat,lon,lon0", MERIDIAN_SEAM_CASES)
def test_azimuthal_equidistant_roundtrip_is_canonical_across_the_antimeridian(
    lat, lon, lon0
):
    lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(lon0)
    lat0_r = np.radians(10.0)
    fwd = azimuthal_equidistant(lat_r, lon_r, lat0_r, lon0_r)
    got_lat, got_lon = azimuthal_equidistant_inverse(fwd.x, fwd.y, lat0_r, lon0_r)
    assert got_lat == pytest.approx(lat_r, abs=1e-9)
    lon_error = np.arctan2(np.sin(got_lon - lon_r), np.cos(got_lon - lon_r))
    assert lon_error == pytest.approx(0.0, abs=1e-9)
    assert -np.pi <= got_lon <= np.pi
