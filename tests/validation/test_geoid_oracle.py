"""Geoid-height oracle tests for pytcl.gravity.egm and pytcl.gravity.clenshaw.

Task 2.1 (v2.11.1 correctness patch): ``egm.geoid_height`` fed geodetic
latitude into a spherical-harmonic synthesis that requires geocentric
latitude, and evaluated it at the mean reference radius ``R`` instead of
the true ellipsoid radius under the point. Task 2.2: ``clenshaw_geoid``
never removed the reference field -- it zeroed the ``n=0,1`` terms and
called that "the reference field", leaving ``C20`` (the dominant even
zonal harmonic) in -- and had the same latitude/radius bug. See
``.superpowers/sdd/2026-09-14-v2.11.1-tier1-patch/task-2.{1,2}-brief.md``.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems import geodetic2ecef
from pytcl.core.exceptions import DependencyError
from pytcl.gravity.clenshaw import clenshaw_geoid, clenshaw_potential
from pytcl.gravity.egm import (
    _subtract_reference_field,
    create_test_coefficients,
    geoid_height,
    load_egm_coefficients,
)
from pytcl.gravity.models import WGS84, ellips_grav_coeffs, normal_gravity_somigliana

_data_skip = (FileNotFoundError, DependencyError)


def _geocentric(lat, lon):
    """Geocentric latitude and radius at the ellipsoid surface below (lat, lon).

    Derived from :func:`geodetic2ecef` rather than the closed-form
    ``arctan((1-f)**2 * tan(lat))``, per CLAUDE.md's "use the existing
    conversion" convention.
    """
    ecef = geodetic2ecef(lat, lon, 0.0)
    r = float(np.linalg.norm(ecef))
    lat_gc = float(np.arctan2(ecef[2], np.hypot(ecef[0], ecef[1])))
    return lat_gc, r


def _geoid_via_explicit_geocentric_path(lat, lon, coef):
    """Independent reimplementation of the corrected ``geoid_height`` body.

    Uses the same production helpers (``_subtract_reference_field``,
    ``clenshaw_potential``) but recomputes the geocentric latitude and
    radius itself, so this pins the fix against reintroduction of the
    geodetic-latitude / ``r = R`` bug rather than testing a tautology.
    """
    C_dist = coef.C.copy()
    S_dist = coef.S.copy()
    C_dist[0, 0] = 0.0
    if coef.n_max >= 1:
        C_dist[1, :] = 0.0
        S_dist[1, :] = 0.0
    _subtract_reference_field(C_dist, coef.n_max)

    lat_gc, r = _geocentric(lat, lon)
    gamma = normal_gravity_somigliana(lat, WGS84)
    T = clenshaw_potential(lat_gc, lon, r, C_dist, S_dist, coef.R, coef.GM, coef.n_max)
    return T / gamma


POINTS_DEG = [
    (45.0, 10.0),
    (60.0, -150.0),
    (-70.0, 45.0),
    (-23.6174, 133.8747),
]


@pytest.mark.parametrize("lat_deg,lon_deg", POINTS_DEG)
def test_geoid_height_uses_geocentric_latitude_at_the_ellipsoid_radius(
    lat_deg, lon_deg
):
    """Direct check of the two terms the audit decomposed (task 2.1)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    coef = create_test_coefficients(n_max=10)
    got = geoid_height(lat, lon, coefficients=coef)
    manual = _geoid_via_explicit_geocentric_path(lat, lon, coef)
    assert got == pytest.approx(manual, abs=1e-9)


@pytest.mark.parametrize(
    "lat_deg,lon_deg",
    [(0.0, 0.0), (45.0, 10.0), (-70.0, 45.0), (89.9, 0.0), (-89.9, 90.0)],
)
def test_clenshaw_geoid_of_pure_reference_field_is_zero(lat_deg, lon_deg):
    """The disturbing potential of the reference field against itself is 0.

    Direct test of the task 2.2 defect: ``clenshaw_geoid`` zeroed only the
    ``n=0,1`` terms and called that "the reference field", leaving C20 --
    the dominant even zonal harmonic -- in, so a pure reference field
    produced a large false "geoid" instead of the zero it is by
    construction.
    """
    n_max = 10
    C, S, R, GM = ellips_grav_coeffs(max_order=n_max, is_normalized=True)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    gamma = normal_gravity_somigliana(lat, WGS84)
    N = clenshaw_geoid(lat, lon, C, S, R, GM, gamma, n_max=n_max)
    assert N == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("lat_deg,lon_deg", [(0.0, 0.0), (45.0, 10.0), (-70.0, 45.0)])
def test_clenshaw_geoid_agrees_with_egm_geoid_height(lat_deg, lon_deg):
    """clenshaw_geoid matches geoid_height and stays within the physical range.

    Real EGM96 (degree 360) puts the pre-fix ``clenshaw_geoid`` at
    +3482.31 m, -1689.60 m and -5643.72 m at these three points -- 30x
    outside the true geoid's ±110 m range.
    """
    try:
        coef = load_egm_coefficients("EGM96")
    except _data_skip as exc:
        pytest.skip(f"EGM96 coefficients unavailable: {exc}")
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    gamma = normal_gravity_somigliana(lat, WGS84)
    got = clenshaw_geoid(lat, lon, coef.C, coef.S, coef.R, coef.GM, gamma)
    expected = geoid_height(lat, lon, coefficients=coef)
    assert abs(got) < 110.0, "geoid undulation exceeds its physical range"
    assert got == pytest.approx(expected, abs=0.05)
