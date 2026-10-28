"""Geophysical fields consolidated to one body each (v2.12.0 Task 6.4).

Row 13, EGM reference field. ``egm`` now subtracts MATLAB
``getEGMWGS84TCoeffs``'s reference field: ``ellips_grav_coeffs`` for the
WGS84 ellipsoid, rescaled to the model's ``(GM, R)``. v2.11.1 typed the
Heiskanen-Moritz 2-92 series with the rounded GRS80 ``J2 = 1.08263e-3``
and no rescaling. Against the closed-form normal potential (H-M 2-124
in ellipsoidal coordinates, evaluated in mpmath) v2.11.1's reference
series was off by up to 3.8e-3 m^2/s^2 (0.38 mm of geoid); the new one
is off by at most 3.5e-6 m^2/s^2. Geoid heights move by -0.191 mm at
the equator to +0.384 mm at the poles.

Row 14, Schmidt dP/dtheta. EMM's inline recursion divided by
``sin(theta)``: zero derivatives at the poles (B_theta off by 2.9% of
the field with the degree-36 test set) and cancellation near them. The
shared ``_schmidt_legendre_with_derivative`` now uses the same-degree
identity, which has no division. WMM and IGRF share it.

Row 17, decay constant. ``_std_decay_const`` returns
``atmos_exp_decay_const(ns).ce``.
"""

import functools
import importlib
from math import comb, factorial
from unittest import mock

import mpmath as mp
import numpy as np
import pytest

from pytcl.atmosphere import refraction
from pytcl.core.exceptions import DependencyError
from pytcl.gravity import egm
from pytcl.gravity.models import WGS84, ellips_grav_coeffs
from pytcl.magnetism import _schmidt
from tests.validation._v2111 import shipped_module

emm = importlib.import_module("pytcl.magnetism.emm")
wmm = importlib.import_module("pytcl.magnetism.wmm")

_data_skip = (FileNotFoundError, DependencyError)

EGM_GM = 3.986004415e14
EGM_R = 6378136.3


@pytest.fixture(scope="module")
def old_egm():
    return shipped_module("pytcl/gravity/egm.py")


@pytest.fixture(scope="module")
def old_schmidt():
    return shipped_module("pytcl/magnetism/_schmidt.py")


@pytest.fixture(scope="module")
def old_emm(old_schmidt):
    """v2.11.1 emm.py; call it inside ``_shipped_schmidt`` so its
    function-level import of ``_schmidt`` sees the v2.11.1 helpers."""
    return shipped_module("pytcl/magnetism/emm.py")


@pytest.fixture(scope="module")
def old_wmm(old_schmidt):
    module = shipped_module("pytcl/magnetism/wmm.py")
    module._schmidt_legendre_with_derivative = (
        old_schmidt._schmidt_legendre_with_derivative
    )
    return module


def _shipped_schmidt(old_schmidt):
    return mock.patch.multiple(
        _schmidt,
        _schmidt_legendre=old_schmidt._schmidt_legendre,
        _schmidt_legendre_with_derivative=old_schmidt._schmidt_legendre_with_derivative,
    )


@pytest.fixture(scope="module")
def old_refraction():
    return shipped_module("pytcl/atmosphere/refraction.py")


# ---------------------------------------------------------------------------
# Row 13: EGM reference field
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_max", [2, 3, 10, 36, 360])
@pytest.mark.parametrize("gm,r", [(EGM_GM, EGM_R), (WGS84.GM, WGS84.a)])
def test_reference_zonals_are_ellips_grav_coeffs_exactly(n_max, gm, r):
    C = np.zeros((n_max + 1, n_max + 1))
    egm._subtract_reference_field(C, n_max, gm, r)

    C_ell = ellips_grav_coeffs(
        max_order=n_max,
        is_normalized=True,
        omega=WGS84.omega,
        a=WGS84.a,
        f=WGS84.f,
        gm=WGS84.GM,
    )[0]
    degrees = np.arange(n_max + 1)
    expected = -(WGS84.GM / gm) * (WGS84.a / r) ** degrees * C_ell[:, 0]
    expected[:2] = 0.0
    np.testing.assert_array_equal(C[:, 0], expected)
    np.testing.assert_array_equal(C[:, 1:], 0.0)
    if (gm, r) == (WGS84.GM, WGS84.a):
        np.testing.assert_array_equal(C[2:, 0], -C_ell[2:, 0])


mp.mp.dps = 40
_A = mp.mpf(WGS84.a)
_F = mp.mpf(WGS84.f)
_GM_E = mp.mpf(WGS84.GM)
_OMEGA = mp.mpf(WGS84.omega)
_B = _A * (1 - _F)
_E = mp.sqrt(_A**2 - _B**2)
_E2 = _F * (2 - _F)


def _q(u):
    return ((1 + 3 * u**2 / _E**2) * mp.atan(_E / u) - 3 * u / _E) / 2


def _closed_form_gravitational_potential(p, z):
    """WGS84 normal gravitational potential (no centrifugal term) at
    cylindrical ``(p, z)``: Heiskanen & Moritz eq. 2-124 in ellipsoidal
    coordinates ``(u, beta)`` with the centrifugal potential
    ``omega^2 (u^2 + E^2) cos^2(beta) / 2`` removed."""
    r2 = p**2 + z**2
    u = mp.sqrt(
        (r2 - _E**2) / 2 * (1 + mp.sqrt(1 + 4 * _E**2 * z**2 / (r2 - _E**2) ** 2))
    )
    sin_beta = z / u
    return _GM_E / _E * mp.atan(_E / u) + _OMEGA**2 * _A**2 / 2 * _q(u) / _q(_B) * (
        sin_beta**2 - mp.mpf(1) / 3
    )


@pytest.mark.parametrize("lat_deg", [-40.0, 0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
@pytest.mark.parametrize("h", [0.0, 1.0e4, 1.0e6])
def test_reference_field_matches_closed_form_normal_potential(lat_deg, h):
    """The zonals egm subtracts, summed in the model's ``GM/r (R/r)^n``
    series, reproduce the WGS84 normal potential beyond degree 0.

    Measured residual at most 3.5e-6 m^2/s^2 (0.35 um of geoid), from
    the cancellation in ``ellipsGravCoeffs``'s ``q0`` (its C20 is 5.1e-11
    relative off the mpmath value). v2.11.1's series missed by up to
    3.8e-3 m^2/s^2; unscaled ``ellips_grav_coeffs`` by up to 1.5e-2.
    """
    n_max = 20
    C = np.zeros((n_max + 1, n_max + 1))
    egm._subtract_reference_field(C, n_max, EGM_GM, EGM_R)
    zonals = -C[:, 0]

    phi = mp.radians(lat_deg)
    n_radius = _A / mp.sqrt(1 - _E2 * mp.sin(phi) ** 2)
    p = (n_radius + h) * mp.cos(phi)
    z = (n_radius * (1 - _E2) + h) * mp.sin(phi)
    r = mp.sqrt(p**2 + z**2)
    truth = _closed_form_gravitational_potential(p, z) - _GM_E / r
    series = (
        EGM_GM
        / r
        * mp.fsum(
            (mp.mpf(EGM_R) / r) ** n
            * mp.mpf(zonals[n])
            * mp.sqrt(2 * n + 1)
            * mp.legendre(n, z / r)
            for n in range(2, n_max + 1)
        )
    )
    assert abs(float(series - truth)) < 1.0e-5


def test_geoid_change_from_v2111_is_the_reference_field_change(old_egm):
    """Measured: -0.191 mm at the equator to +0.384 mm at the poles, the
    same for EGM96 and EGM2008 (shared GM and R), scalar and array."""
    coef = egm.create_test_coefficients(n_max=36)
    lats = np.radians(np.linspace(-90.0, 90.0, 181))
    lons = np.full_like(lats, 0.3)
    delta = egm.geoid_heights(lats, lons, coefficients=coef) - old_egm.geoid_heights(
        lats, lons, coefficients=coef
    )
    assert delta.min() == pytest.approx(-0.191e-3, rel=0, abs=2e-6)
    assert delta.max() == pytest.approx(0.384e-3, rel=0, abs=2e-6)
    assert delta[90] == pytest.approx(delta.min(), rel=0, abs=1e-12)
    assert delta[0] == pytest.approx(delta.max(), rel=0, abs=1e-12)
    for i in (0, 45, 90, 150):
        scalar = egm.geoid_height(lats[i], lons[i], coefficients=coef) - (
            old_egm.geoid_height(lats[i], lons[i], coefficients=coef)
        )
        assert scalar == delta[i]


def test_gravity_outputs_move_by_the_measured_amounts(old_egm):
    """gravity_disturbance / gravity_anomaly move by at most 1.8e-4 mGal
    and deflection_of_vertical by at most 1.9e-5 arcsec (latitudes -60
    to 60 deg for the deflection, -89.5 to 89.5 deg for the rest; heights 0
    and 10 km)."""
    coef = egm.create_test_coefficients(n_max=36)
    for lat_deg in (-89.5, -60.0, -30.0, 0.0, 30.0, 45.0, 60.0, 89.5):
        lat = np.radians(lat_deg)
        for h in (0.0, 1.0e4):
            new = egm.gravity_disturbance(lat, 0.3, h, coefficients=coef)
            old = old_egm.gravity_disturbance(lat, 0.3, h, coefficients=coef)
            for name in ("delta_g_r", "delta_g_lat", "magnitude"):
                assert abs(getattr(new, name) - getattr(old, name)) < 2.0e-9
            assert new.delta_g_lon == old.delta_g_lon
            anomaly_change = egm.gravity_anomaly(
                lat, 0.3, h, coefficients=coef
            ) - old_egm.gravity_anomaly(lat, 0.3, h, coefficients=coef)
            assert abs(anomaly_change) < 2.0e-9
    for lat_deg in (-60.0, -30.0, 0.0, 30.0, 45.0, 60.0):
        lat = np.radians(lat_deg)
        new = egm.deflection_of_vertical(lat, 0.3, coefficients=coef)
        old = old_egm.deflection_of_vertical(lat, 0.3, coefficients=coef)
        assert np.max(np.abs(np.subtract(new, old))) < 2.0e-5


@pytest.mark.parametrize("model", ["EGM96", "EGM2008"])
def test_real_model_geoid_moves_sub_millimetre(model, old_egm):
    try:
        coef = egm.load_egm_coefficients(model, n_max=360)
    except _data_skip as exc:
        pytest.skip(f"{model} coefficients unavailable: {exc}")
    for lat_deg, lon_deg in [(0.0, 0.0), (45.0, 10.0), (-70.0, 45.0), (89.5, 0.0)]:
        lat, lon = np.radians(lat_deg), np.radians(lon_deg)
        new = egm.geoid_height(lat, lon, coefficients=coef)
        old = old_egm.geoid_height(lat, lon, coefficients=coef)
        assert abs(new - old) < 0.4e-3


# ---------------------------------------------------------------------------
# Row 14: Schmidt dP/dtheta
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _legendre_derivative_poly(n, k):
    """``d^k P_n / dx^k`` as exact integer coefficients over a common
    denominator, from Rodrigues' formula."""
    coeffs = {}
    for i in range(n + 1):
        j = 2 * i
        if j >= n + k:
            coeffs[j - n - k] = (
                comb(n, i) * (-1) ** (n - i) * factorial(j) // factorial(j - n - k)
            )
    return coeffs, 2**n * factorial(n)


def _poly_at(n, k, x):
    coeffs, denominator = _legendre_derivative_poly(n, k)
    return mp.fsum(mp.mpf(c) * x**j for j, c in coeffs.items()) / denominator


def _exact_schmidt(n, m, theta):
    """Schmidt ``P[n,m](theta) = N sin^m(theta) d^m P_n/dx^m (cos theta)``
    and its exact theta derivative, in mpmath (signed through the poles,
    unlike anything evaluated from ``cos(theta)`` alone)."""
    s, c = mp.sin(theta), mp.cos(theta)
    norm = mp.sqrt(mp.mpf(2 if m else 1) * mp.factorial(n - m) / mp.factorial(n + m))
    q = _poly_at(n, m, c)
    d = -(s ** (m + 1)) * _poly_at(n, m + 1, c)
    if m:
        d += m * s ** (m - 1) * c * q
    return norm * s**m * q, norm * d


def _exact_field(theta, lon, r, coeffs, year):
    """B_r, B_theta of a Schmidt Gauss expansion at the exact colatitude."""
    dt = year - coeffs.epoch
    a = mp.mpf(_schmidt.GEOMAGNETIC_REFERENCE_RADIUS_KM)
    B_r = B_theta = mp.mpf(0)
    for n in range(1, coeffs.n_max + 1):
        rp = (a / mp.mpf(r)) ** (n + 2)
        secular = n <= getattr(coeffs, "n_max_sv", coeffs.n_max)
        for m in range(n + 1):
            g, h = coeffs.g[n, m], coeffs.h[n, m]
            if secular:
                g += dt * coeffs.g_dot[n, m]
                h += dt * coeffs.h_dot[n, m]
            gc = g * mp.cos(m * lon) + h * mp.sin(m * lon)
            p, d = _exact_schmidt(n, m, theta)
            B_r += (n + 1) * rp * p * gc
            B_theta -= rp * d * gc
    return float(B_r), float(B_theta)


_COLATITUDES = [0.0, 1e-9, 1e-6, 1e-3, 0.3, 1.5, np.pi - 1e-6, np.pi]


@pytest.mark.parametrize("theta", _COLATITUDES)
def test_schmidt_table_and_derivative_are_exact(theta):
    """P and dP/dtheta against the exact mpmath values, relative to each
    degree's largest entry. Measured at most 7.8e-15 (P) and 7.8e-15
    (dP), except 4.3e-14 for both at 1e-6 from each pole."""
    n_max = 36
    P, dP = _schmidt._schmidt_legendre_with_derivative(
        n_max, np.cos(theta), np.sin(theta)
    )
    assert np.all(np.isfinite(dP))
    for n in range(n_max + 1):
        exact = [_exact_schmidt(n, m, mp.mpf(theta)) for m in range(n + 1)]
        for k, got in ((0, P), (1, dP)):
            scale = max(float(abs(v[k])) for v in exact) or 1.0
            err = max(abs(float(exact[m][k]) - got[n, m]) for m in range(n + 1))
            assert err <= 1e-13 * scale, f"n={n}, {'dP' if k else 'P'}"
    upper = np.triu_indices(n_max + 1, 1)
    assert np.all(P[upper] == 0.0)
    assert np.all(dP[upper] == 0.0)


@pytest.mark.parametrize(
    "theta", [0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.06, 0.2, 0.6, 1.2]
)
@pytest.mark.parametrize("north", [True, False])
def test_emm_field_matches_exact_synthesis(theta, north):
    """EMM's (B_r, B_theta) against an exact mpmath synthesis at the true
    colatitude, relative to the field magnitude. Measured at most 5.6e-16.

    v2.11.1 was off by 2.9% (north) and 8.9% (south) at colatitude 0,
    1e-9 and 1e-6 -- zero derivatives at the pole, and a Legendre table
    whose 12-decimal ``cos(theta)`` cache key tabulates the pole itself
    there -- and by 4.0e-11, 1.2e-9, 1.1e-10, 2.7e-12 and 5.1e-13 (north)
    at 1e-4, 1e-3, 1e-2, 0.06 and 0.2 rad. That table rounding is also
    why EMM moves by up to 1.8e-12 at mid-latitudes (see below).
    """
    coef = emm.create_test_coefficients(n_max=36)
    lat = np.pi / 2 - theta if north else -(np.pi / 2 - theta)
    lon, r, year = 0.7, 6380.0, 2020.0
    got = np.array(emm._high_res_field_spherical(lat, lon, r, year, coef))
    assert np.all(np.isfinite(got))
    exact = np.array(_exact_field(mp.mpf(np.pi / 2) - mp.mpf(lat), lon, r, coef, year))
    err = np.max(np.abs(got[:2] - exact)) / np.linalg.norm(exact)
    assert err <= 1e-14


def _emm_grid():
    lats = np.radians(np.linspace(-70.0, 70.0, 15))
    lons = np.radians(np.array([-150.0, -40.0, 0.0, 40.0, 165.0]))
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    return lat_grid, lon_grid


_AWAY_RTOL = 2e-12


def _assert_results_close(new, old, rtol):
    F = np.asarray(old.F)
    for name in ("X", "Y", "Z", "H", "F"):
        diff = np.abs(np.asarray(getattr(new, name)) - np.asarray(getattr(old, name)))
        assert np.all(diff <= rtol * F), name


@pytest.mark.parametrize("h", [0.0, 400.0])
def test_emm_away_from_poles_matches_v2111(h, old_emm, old_schmidt):
    """|lat| <= 70 deg geodetic: within 2e-12 of |F| (measured 9.4e-13
    on this grid; 1.8e-12 for WMMHR2025 on a 1-deg by 6-longitude
    grid), array and scalar paths. Not 1e-12: v2.11.1's Legendre table
    rounded ``cos(theta)`` to 12 decimals, and this is that error."""
    coef = emm.create_test_coefficients(n_max=36)
    lat, lon = _emm_grid()
    with pytest.warns(UserWarning, match="validity window"):
        new = emm.emm(lat, lon, h, 2020.0, coefficients=coef)
    with _shipped_schmidt(old_schmidt), pytest.warns(UserWarning, match="validity"):
        old = old_emm.emm(lat, lon, h, 2020.0, coefficients=coef)
    assert np.shape(new.X) == lat.shape
    _assert_results_close(new, old, _AWAY_RTOL)
    with pytest.warns(UserWarning, match="validity window"):
        scalar = emm.emm(
            float(lat[3, 2]), float(lon[3, 2]), h, 2020.0, coefficients=coef
        )
    assert isinstance(scalar.X, float)
    assert scalar.X == new.X[3, 2]


def test_wmmhr_away_from_poles_matches_v2111(old_emm, old_schmidt):
    try:
        coef = emm.load_emm_coefficients("WMMHR2025")
    except _data_skip as exc:
        pytest.skip(f"WMMHR2025 coefficients unavailable: {exc}")
    lat, lon = _emm_grid()
    new = emm.wmmhr(lat, lon, 0.0, 2026.0, coefficients=coef)
    with _shipped_schmidt(old_schmidt):
        old = old_emm.wmmhr(lat, lon, 0.0, 2026.0, coefficients=coef)
    _assert_results_close(new, old, _AWAY_RTOL)


@pytest.mark.parametrize("north", [True, False])
def test_wmm_at_the_pole_matches_exact_synthesis(north):
    """WMM shares the helper. v2.11.1's B_theta at the pole was zero,
    12% of the field off for WMM2020. Measured 1.4e-16."""
    coef = wmm.create_wmm2020_coefficients()
    lat = np.pi / 2 if north else -np.pi / 2
    lon, r, year = 0.7, 6380.0, 2022.0
    got = np.array(wmm._compute_magnetic_field_spherical_impl(lat, lon, r, year, coef))
    exact = np.array(_exact_field(mp.mpf(np.pi / 2) - mp.mpf(lat), lon, r, coef, year))
    assert np.max(np.abs(got[:2] - exact)) / np.linalg.norm(exact) <= 1e-14


def test_wmm_away_from_poles_matches_v2111(old_wmm):
    """|lat| <= 70 deg: within 2e-12 of |B| (measured 9.9e-13)."""
    coef = wmm.create_wmm2020_coefficients()
    lat, lon = _emm_grid()
    for la, lo in zip(lat.ravel(), lon.ravel(), strict=True):
        for r in (6371.2, 7000.0):
            new = np.array(
                wmm._compute_magnetic_field_spherical_impl(la, lo, r, 2022.0, coef)
            )
            old = np.array(
                old_wmm._compute_magnetic_field_spherical_impl(la, lo, r, 2022.0, coef)
            )
            assert np.max(np.abs(new - old)) <= _AWAY_RTOL * np.linalg.norm(old)


def test_magnetic_coordinates_move_at_roundoff_level(old_schmidt):
    """itrs2magnetic_apex and itrs2qd read the field and the potential
    through the shared tables. Measured movement against the v2.11.1
    helpers: at most 3.1e-12 of each output's magnitude."""
    coordinates = importlib.import_module("pytcl.magnetism.coordinates")
    points = [
        np.array([6.4e6, 1e5, 2e6]),
        np.array([6.9e6, 2e5, 2.4e6]),
        np.array([6.5e6, -3e6, -1.8e6]),
        np.array([1e6, 5e5, 6.3e6]),
    ]

    def evaluate():
        out = []
        for x in points:
            for fn in (coordinates.itrs2magnetic_apex, coordinates.itrs2qd):
                z, apex = fn(x)
                out.append((np.asarray(z), np.asarray(apex)))
        return out

    new = evaluate()
    with (
        mock.patch.object(
            _schmidt,
            "_schmidt_legendre",
            lambda n, c, s: old_schmidt._schmidt_legendre(n, c),
        ),
        mock.patch.object(
            wmm,
            "_schmidt_legendre_with_derivative",
            old_schmidt._schmidt_legendre_with_derivative,
        ),
    ):
        old = evaluate()
    for (z_new, apex_new), (z_old, apex_old) in zip(new, old, strict=True):
        np.testing.assert_allclose(z_new, z_old, rtol=1e-11, atol=0)
        np.testing.assert_allclose(apex_new, apex_old, rtol=1e-11, atol=0)


# ---------------------------------------------------------------------------
# Row 17: refraction decay constant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ns",
    [313.0, 313, 250.5, np.float64(400.0), np.int64(300), np.array(313.0)],
)
def test_std_decay_const_equals_v2111(ns, old_refraction):
    new = refraction._std_decay_const(ns)
    assert type(new) is float
    assert new == old_refraction._std_decay_const(ns)
    assert new == refraction.atmos_exp_decay_const(ns).ce


def test_std_decay_const_float32_is_now_computed_in_float64(old_refraction):
    """v2.11.1 evaluated a float32 ``ns`` in float32 (6.5e-7 relative
    off); it now gets the float64 value."""
    ns = np.float32(313.0)
    new = refraction._std_decay_const(ns)
    assert new == refraction._std_decay_const(313.0)
    old = old_refraction._std_decay_const(ns)
    assert abs(old - new) / new == pytest.approx(6.5e-7, rel=0.05, abs=0)


def test_std_decay_const_shape_error_unchanged(old_refraction):
    with pytest.raises(TypeError):
        old_refraction._std_decay_const(np.array([313.0]))
    with pytest.raises(TypeError):
        refraction._std_decay_const(np.array([313.0]))


def test_refraction_defaults_equal_v2111(old_refraction):
    for algorithm in (0, 1):
        new = refraction.std_refrac_bias_approx(100e3, 0.1, 100.0, algorithm=algorithm)
        old = old_refraction.std_refrac_bias_approx(
            100e3, 0.1, 100.0, algorithm=algorithm
        )
        assert tuple(new) == tuple(old)
    z_rx = np.array([6378137.0, 0.0, 0.0])
    z_tar = np.array([[6428137.0, 6400137.0], [100e3, -50e3], [0.0, 20e3]])
    new = refraction.cart2ruv_std_refrac(z_tar, True, z_rx, z_rx)
    old = old_refraction.cart2ruv_std_refrac(z_tar, True, z_rx, z_rx)
    np.testing.assert_array_equal(new.z, old.z)
