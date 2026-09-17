"""Geoid/gravity-disturbance oracle tests for pytcl.gravity.egm and
pytcl.gravity.clenshaw.

Task 2.1 (v2.11.1 correctness patch): ``egm.geoid_height`` fed geodetic
latitude into a spherical-harmonic synthesis that requires geocentric
latitude, and evaluated it at the mean reference radius ``R`` instead of
the true ellipsoid radius under the point. Task 2.2: ``clenshaw_geoid``
never removed the reference field -- it zeroed the ``n=0,1`` terms and
called that "the reference field", leaving ``C20`` (the dominant even
zonal harmonic) in -- and had the same latitude/radius bug. Also fixed,
as a controller-owned scope addition found by code review:
``egm.gravity_disturbance`` carried the identical ``r = R + h`` /
geodetic-as-geocentric defect. See
``.superpowers/sdd/2026-09-14-v2.11.1-tier1-patch/task-2.{1,2}-brief.md``
and ``task-2.1-2.3-report.md`` (the "fix report" appendix covers the
review round).
"""

from math import factorial

import numpy as np
import pytest
from scipy.special import lpmv

from pytcl.coordinate_systems import geodetic2ecef
from pytcl.core.exceptions import DependencyError
from pytcl.gravity.clenshaw import clenshaw_geoid
from pytcl.gravity.egm import (
    EGMCoefficients,
    _subtract_reference_field,
    create_test_coefficients,
    geoid_height,
    gravity_disturbance,
    load_egm_coefficients,
)
from pytcl.gravity.models import WGS84, ellips_grav_coeffs, normal_gravity_somigliana

_data_skip = (FileNotFoundError, DependencyError)

# EGM96 constants, matched to pytcl.gravity.egm.EGM_PARAMETERS["EGM96"].
_GM = 3.986004415e14
_R = 6378136.3


def full_norm_legendre(n: int, m: int, x: float) -> float:
    """Geodesy fully normalized Pbar_nm via scipy (no Condon-Shortley phase).

    Independent of pytcl's own Legendre/Clenshaw code -- the same
    from-scratch pattern as ``tests/validation/test_gravity_audit.py``.
    """
    p = lpmv(m, n, x) * (-1.0) ** m
    norm = np.sqrt((2 - (m == 0)) * (2 * n + 1) * factorial(n - m) / factorial(n + m))
    return norm * p


def _geocentric(lat, lon, h=0.0):
    """Geocentric latitude and radius at height h above (lat, lon).

    Derived from :func:`geodetic2ecef` rather than the closed-form
    ``arctan((1-f)**2 * tan(lat))``, per CLAUDE.md's "use the existing
    conversion" convention.
    """
    ecef = geodetic2ecef(lat, lon, h)
    r = float(np.linalg.norm(ecef))
    lat_gc = float(np.arctan2(ecef[2], np.hypot(ecef[0], ecef[1])))
    return lat_gc, r


def _hand_coded_reference_zonals(n_max):
    """Fully normalized C_2k,0 of the WGS84 level ellipsoid's normal
    field (k=1..5, degrees 2/4/6/8/10), typed directly from Heiskanen &
    Moritz, "Physical Geodesy", eq. 2-92 -- no pytcl gravity code is
    called. See the fix report for a verification of these numbers
    against ``models.ellips_grav_coeffs``.
    """
    if n_max > 10:
        raise ValueError(
            f"_hand_coded_reference_zonals only types out k=1..5 (degrees "
            f"up to 10); n_max={n_max} would silently under-subtract the "
            "untyped higher-degree zonals instead of raising here."
        )
    f = WGS84.f
    e2 = f * (2.0 - f)
    J2 = WGS84.J2
    C = np.zeros(n_max + 1)
    for k in range(1, 6):
        n = 2 * k
        if n > n_max:
            break
        J2k = (
            (-1) ** (k + 1)
            * (3.0 * e2**k)
            / ((2 * k + 1) * (2 * k + 3))
            * (1.0 - k + 5.0 * k * J2 / e2)
        )
        C[n] = -J2k / np.sqrt(2 * n + 1)
    return C


def _independent_geoid_via_legendre_sum(lat, lon, coef):
    """From-scratch spherical-harmonic sum (scipy ``lpmv``), independent
    of every piece of pytcl gravity code -- not ``clenshaw_potential``/
    ``clenshaw_sum_order`` (a different, scipy-only summation), and not
    ``_subtract_reference_field`` either (a hand-coded reference field,
    see :func:`_hand_coded_reference_zonals`). This is a REFERENCE-class
    oracle: an independent implementation of the actual synthesis, not a
    second call into any production code.
    """
    C_dist = coef.C.copy()
    S_dist = coef.S.copy()
    C_dist[0, 0] = 0.0
    if coef.n_max >= 1:
        C_dist[1, :] = 0.0
        S_dist[1, :] = 0.0
    C_dist[:, 0] -= _hand_coded_reference_zonals(coef.n_max)

    lat_gc, r = _geocentric(lat, lon)
    x = np.sin(lat_gc)
    T = 0.0
    for n in range(2, coef.n_max + 1):
        for m in range(n + 1):
            T += (
                (coef.R / r) ** n
                * full_norm_legendre(n, m, x)
                * (C_dist[n, m] * np.cos(m * lon) + S_dist[n, m] * np.sin(m * lon))
            )
    T *= coef.GM / r
    gamma = normal_gravity_somigliana(lat, WGS84)
    return T / gamma


def _c20_plus_delta_for_egm(delta, n_max=2, model_name="TEST"):
    """Coefficients whose disturbing field, after ``egm.geoid_height`` /
    ``egm.gravity_disturbance``'s internal ``_subtract_reference_field``
    step, is *exactly* a single C20 term of magnitude ``delta`` --
    everything else (including the reference ellipsoid's own C20, which
    would otherwise dominate) cancels.

    Same construction as ``test_gravity_audit.py::test_c20_only_analytic``.
    Matched to the *egm.py* subtraction routine specifically: it and
    ``models.ellips_grav_coeffs`` (used by ``clenshaw_geoid``, see
    ``_c20_plus_delta_for_clenshaw`` below) differ at the ~1.6e-7
    relative level (task 2.2's noted, accepted discrepancy between the
    two reference-field routines), which is negligible against a
    physical C20 but not against a ``delta`` chosen this small -- using
    the wrong one of the two here left an 0.8% residual that failed a
    tight tolerance during review.
    """
    ref = np.zeros((n_max + 1, n_max + 1))
    _subtract_reference_field(ref, n_max)
    c20_ref = -ref[2, 0]
    C = np.zeros((n_max + 1, n_max + 1))
    S = np.zeros((n_max + 1, n_max + 1))
    C[0, 0] = 1.0
    C[2, 0] = c20_ref + delta
    return EGMCoefficients(C=C, S=S, GM=_GM, R=_R, n_max=n_max, model_name=model_name)


def _c20_plus_delta_for_clenshaw(delta, n_max=2):
    """Like :func:`_c20_plus_delta_for_egm`, but matched to
    ``clenshaw_geoid``'s internal subtraction routine,
    ``models.ellips_grav_coeffs`` (task 2.2), not ``egm.py``'s."""
    C_ref, S_ref, R, GM = ellips_grav_coeffs(max_order=n_max, is_normalized=True)
    C = C_ref.copy()
    C[2, 0] += delta
    return EGMCoefficients(
        C=C, S=S_ref.copy(), GM=GM, R=R, n_max=n_max, model_name="TEST"
    )


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
    """geoid_height matches a from-scratch Legendre-sum oracle (task 2.1)."""
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    coef = create_test_coefficients(n_max=10)
    got = geoid_height(lat, lon, coefficients=coef)
    expected = _independent_geoid_via_legendre_sum(lat, lon, coef)
    assert got == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize(
    "lat_deg,lon_deg",
    [(0.0, 0.0), (45.0, 10.0), (-70.0, 45.0), (89.9, 0.0), (-89.9, 90.0)],
)
def test_clenshaw_geoid_of_pure_reference_field_is_zero(lat_deg, lon_deg):
    """The disturbing potential of the reference field against itself is 0.

    A basic identity check, but on its own it is *not* the discriminating
    oracle for this task: ``clenshaw_geoid``'s synthesis is linear in
    ``C, S``, so once the reference field is fully subtracted the input
    to the summation is the zero matrix and the result is 0.0 at *any*
    latitude or radius -- correct or reverted. See
    ``test_clenshaw_geoid_matches_bruns_formula_at_the_true_point`` below
    for the test that actually exercises the geocentric latitude / true
    radius fix (only that one was reverted against, per the fix report).
    """
    n_max = 10
    C, S, R, GM = ellips_grav_coeffs(max_order=n_max, is_normalized=True)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    gamma = normal_gravity_somigliana(lat, WGS84)
    N = clenshaw_geoid(lat, lon, C, S, R, GM, gamma, n_max=n_max)
    assert N == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("lat_deg,lon_deg", [(0.0, 0.0), (45.0, 10.0), (-70.0, 45.0)])
def test_clenshaw_geoid_matches_bruns_formula_at_the_true_point(lat_deg, lon_deg):
    """A C20 delta over the reference field follows Bruns' formula at the
    true geocentric latitude and radius -- not at (R, geodetic lat).

    Unlike the zero-field test above, the disturbing field here is
    genuinely non-trivial (a single non-zero C20 delta), so this test
    actually depends on both halves of the task 2.2 fix; see the fix
    report's revert-check for confirmation that it fails if either half
    (geocentric latitude, or true radius) is reverted alone.
    """
    n_max = 2
    delta = 1.0e-8
    coef = _c20_plus_delta_for_clenshaw(delta, n_max=n_max)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    gamma = normal_gravity_somigliana(lat, WGS84)
    N = clenshaw_geoid(lat, lon, coef.C, coef.S, coef.R, coef.GM, gamma, n_max=n_max)

    lat_gc, r = _geocentric(lat, lon)
    expected = (
        coef.GM
        / (r * gamma)
        * (coef.R / r) ** 2
        * delta
        * full_norm_legendre(2, 0, np.sin(lat_gc))
    )
    assert N == pytest.approx(expected, rel=1e-9)


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


def _independent_zonal_disturbing_potential(r, lat_gc, R, GM, delta, n=2):
    """From-scratch (scipy ``lpmv``) potential of a single C_n0 = delta
    disturbing term, independent of any pytcl spherical-harmonic code."""
    x = np.sin(lat_gc)
    return GM / r * (R / r) ** n * full_norm_legendre(n, 0, x) * delta


@pytest.mark.parametrize(
    "lat_deg,lon_deg,h",
    [(45.0, 10.0, 1000.0), (-70.0, 45.0, 500.0), (89.9, 0.0, 2000.0)],
)
def test_gravity_disturbance_uses_geocentric_latitude_at_true_radius(
    lat_deg, lon_deg, h
):
    """gravity_disturbance matches a from-scratch numerical-gradient
    oracle at the true geocentric latitude and radius -- not at
    (R + h, geodetic lat).

    ``h != 0`` and a non-equatorial, non-polar latitude are both needed
    to discriminate the defect: at h=0 the radius half is invisible
    (R + 0 == R), and at the pole/equator geodetic and geocentric
    latitude coincide.
    """
    n_max = 2
    delta = 1.0e-8
    coef = _c20_plus_delta_for_egm(delta, n_max=n_max)
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)

    dist = gravity_disturbance(lat, lon, h=h, coefficients=coef)

    lat_gc, r = _geocentric(lat, lon, h)
    eps_r, eps_lat = 1.0, 1e-6

    def T(rr, ll):
        return _independent_zonal_disturbing_potential(rr, ll, coef.R, coef.GM, delta)

    g_r_expected = (T(r + eps_r, lat_gc) - T(r - eps_r, lat_gc)) / (2 * eps_r)
    g_lat_expected = (
        (T(r, lat_gc + eps_lat) - T(r, lat_gc - eps_lat)) / (2 * eps_lat) / r
    )

    assert dist.delta_g_r == pytest.approx(g_r_expected, rel=1e-6)
    assert dist.delta_g_lat == pytest.approx(g_lat_expected, rel=1e-6)
    assert dist.delta_g_lon == pytest.approx(0.0, abs=1e-15)
