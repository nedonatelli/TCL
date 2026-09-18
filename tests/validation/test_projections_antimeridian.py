"""Antimeridian-crossing regression tests for the projections module.

Oracle: PROJ via pyproj. `geodetic2utm`, `mercator`, and `transverse_mercator`
computed `x = a * (lon - lon0)` (or its series equivalent) without wrapping
the longitude difference into [-pi, pi), so a forced UTM zone or a central
meridian near +-180 degrees produced eastings in the billions of metres for
any point on the far side of the seam (gh-25 follow-up). Skips without
pyproj.

The paired inverse functions (`utm2geodetic` / `transverse_mercator_inverse`,
`mercator_inverse`, `stereographic_inverse`, `azimuthal_equidistant_inverse`)
recovered the correct point but as a longitude outside [-pi, pi) whenever
`lon0 + offset` crossed the seam -- mathematically exact, but a value a naive
caller would not expect and could not directly compare against a canonical
reference. They now canonicalize their output through the same
`_wrap_longitude_difference` helper used by the forward fix above.

`oblique_stereographic` and `lambert_conformal_conic` scale the wrapped
difference by a non-unity factor (`n`) before taking sin/cos, so unlike
`mercator`/`transverse_mercator`, an unwrapped difference does not merely
alias back to the right answer through 2*pi-periodicity: it silently
computes the wrong map coordinate (124.6 km for `oblique_stereographic`,
tens of millions of metres for `lambert_conformal_conic`) while still
round-tripping through its own equally-wrong inverse -- the self-consistency
check alone would not have caught this class, hence the PROJ-oracle
comparisons below rather than round-trip checks alone.

Exactly +-180 degrees is a genuine ambiguity, not a defect: both signs name
the same meridian, `wrap_to_pi`'s half-open [-pi, pi) convention picks one
canonical answer (+pi wraps to -pi), and PROJ does not enforce the same
convention (it returns the literal, unwrapped input's projection, so a
literal +180 and a literal -180 disagree in PROJ by a sign flip). The tests
below pin pytcl's own self-consistency at that point rather than comparing
to PROJ, which has no single canonical answer to offer there.
"""

import numpy as np
import pytest

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
    oblique_stereographic_inverse,
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
    # wrap-aware comparison above while sitting outside [-pi, pi).
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


# lat, lon, lon0 -- a non-seam case (small |lon - lon0|) followed by two
# seam-crossing cases, so a regression to the ordinary (non-antimeridian)
# path would also fail these parametrizations.
STEREO_LCC_CASES = [
    (10.0, -176.0, -177.0),
    (10.0, 179.9, -177.0),
    (-20.0, 178.0, -177.0),
]


@pytest.mark.parametrize("lat,lon,lon0", STEREO_LCC_CASES)
def test_oblique_stereographic_matches_proj_sterea_across_the_antimeridian(
    lat, lon, lon0
):
    lat0 = 10.0
    sterea = pyproj.Proj(proj="sterea", lat_0=lat0, lon_0=lon0, ellps="WGS84")
    want_x, want_y = sterea(lon, lat)
    r = oblique_stereographic(
        np.radians(lat), np.radians(lon), np.radians(lat0), np.radians(lon0)
    )
    assert r.x == pytest.approx(want_x, abs=1e-6)
    assert r.y == pytest.approx(want_y, abs=1e-6)


@pytest.mark.parametrize("lat,lon,lon0", STEREO_LCC_CASES)
def test_oblique_stereographic_roundtrip_is_canonical_across_the_antimeridian(
    lat, lon, lon0
):
    lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(lon0)
    lat0_r = np.radians(10.0)
    fwd = oblique_stereographic(lat_r, lon_r, lat0_r, lon0_r)
    got_lat, got_lon = oblique_stereographic_inverse(fwd.x, fwd.y, lat0_r, lon0_r)
    assert got_lat == pytest.approx(lat_r, abs=1e-9)
    lon_error = np.arctan2(np.sin(got_lon - lon_r), np.cos(got_lon - lon_r))
    assert lon_error == pytest.approx(0.0, abs=1e-9)
    assert -np.pi <= got_lon <= np.pi


@pytest.mark.parametrize("lat,lon,lon0", STEREO_LCC_CASES)
def test_lambert_conformal_conic_matches_proj_across_the_antimeridian(lat, lon, lon0):
    lat0, lat1, lat2 = 5.0, 1.0, 9.0
    lcc = pyproj.Proj(proj="lcc", lat_0=lat0, lon_0=lon0, lat_1=lat1, lat_2=lat2)
    want_x, want_y = lcc(lon, lat)
    r = lambert_conformal_conic(
        np.radians(lat),
        np.radians(lon),
        np.radians(lat0),
        np.radians(lon0),
        np.radians(lat1),
        np.radians(lat2),
    )
    assert r.x == pytest.approx(want_x, abs=1e-3)
    assert r.y == pytest.approx(want_y, abs=1e-3)


@pytest.mark.parametrize("lat,lon,lon0", STEREO_LCC_CASES)
def test_lambert_conformal_conic_roundtrip_is_canonical_across_the_antimeridian(
    lat, lon, lon0
):
    lat_r, lon_r, lon0_r = np.radians(lat), np.radians(lon), np.radians(lon0)
    lat0_r, lat1_r, lat2_r = np.radians(5.0), np.radians(1.0), np.radians(9.0)
    fwd = lambert_conformal_conic(lat_r, lon_r, lat0_r, lon0_r, lat1_r, lat2_r)
    got_lat, got_lon = lambert_conformal_conic_inverse(
        fwd.x, fwd.y, lat0_r, lon0_r, lat1_r, lat2_r
    )
    assert got_lat == pytest.approx(lat_r, abs=1e-6)
    lon_error = np.arctan2(np.sin(got_lon - lon_r), np.cos(got_lon - lon_r))
    assert lon_error == pytest.approx(0.0, abs=1e-9)
    assert -np.pi <= got_lon <= np.pi


def test_wrap_is_identical_for_both_signs_of_the_exact_antimeridian():
    """+180 and -180 degrees name the same meridian; pytcl must treat them
    identically rather than giving a representation-dependent answer.

    Not compared against PROJ: PROJ returns the literal, unwrapped input's
    projection here (a literal +180 deg input and a literal -180 deg input
    disagree in PROJ by a sign flip, of equal magnitude), so there is no
    single external "correct" value at this exact boundary -- only pytcl's
    own canonicalization invariant to check. `transverse_mercator`/UTM is
    deliberately excluded: at exactly +-180 degrees from the central
    meridian its Redfearn series argument is not small (the series is only
    ever documented/valid within a handful of degrees of the meridian), so
    it diverges there for reasons unrelated to wrapping.
    """
    lat, lon0 = np.radians(10.0), np.radians(0.0)
    lon_pos, lon_neg = np.radians(180.0), np.radians(-180.0)

    r_pos = mercator(lat, lon_pos, lon0=lon0)
    r_neg = mercator(lat, lon_neg, lon0=lon0)
    assert r_pos.x == pytest.approx(r_neg.x, abs=1e-9)
    assert r_pos.y == pytest.approx(r_neg.y, abs=1e-9)

    for lon_input in (lon_pos, lon_neg):
        fwd = mercator(lat, lon_input, lon0=lon0)
        got_lat, got_lon = mercator_inverse(fwd.x, fwd.y, lon0=lon0)
        assert got_lat == pytest.approx(lat, abs=1e-9)
        # Compare against pi itself, wrap-aware: the recovered longitude may
        # land on either edge of the half-open interval depending on
        # floating-point rounding in x / a, and +pi and -pi are the same
        # antimeridian either way.
        lon_error = np.arctan2(np.sin(got_lon - np.pi), np.cos(got_lon - np.pi))
        assert lon_error == pytest.approx(0.0, abs=1e-6)
        assert -np.pi <= got_lon <= np.pi
