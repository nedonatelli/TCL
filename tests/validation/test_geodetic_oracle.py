"""Oracle tests for ``ecef2geodetic`` (v2.12.0 Task 3.5).

The oracle solves ``p = (N + h) cos(lat)``, ``z = (N (1 - e2) + h) sin(lat)``
by 2-D Newton in mpmath at 40 digits. Measured on 20000 seeded points per
band. This file samples 2000 points per band to stay fast, but the
tolerances are 1.5x the 20000-point maxima below.

Maximum error, ``direct`` -- latitude (rad) / altitude (m):

    surface   3.0e-16 / 3.2e-09
    leo       3.0e-16 / 2.7e-09
    geo       2.9e-16 / 1.7e-08
    interior  2.9e-16 / 3.6e-09

Maximum error, ``iterative`` -- latitude (rad) / altitude (m):

    surface   5.0e-14 / 2.8e-07
    leo       2.4e-14 / 1.5e-07
    geo       2.6e-15 / 8.8e-08
    interior  2.8e-04 / 6.1

Near the Earth's centre, forward residual (distance between the input and
the point rebuilt from the result, mpmath at 40 digits) over 309 directions
(300 seeded random, 9 axis and near-axis). Radii are in units of ``e2 a``
(42697.7 m for WGS84). The edges were bisected to 2e-5 relative; the 1 mm
criterion is scaled by ``a / 6378137`` for the ``a = 1`` case.

    ellipsoid             e2 a       direct NaN/wrong   iterative > 1 mm
    WGS84                 42697.7 m  r < 1.0034 e2 a    r < 7.352 e2 a
    WGS84 scaled to a=1   0.0066944  r < 1.0034 e2 a    r < 7.352 e2 a
    a = WGS84.a, f=1/50   252574 m   r < 1.0204 e2 a    r < 10.501 e2 a

Inside its edge ``direct`` gives NaN on 67487 of 74160 WGS84 rows (240
radii, 1e-300 m .. 1 e2 a) and finite values up to 3.5e159 m wrong (WGS84;
8.3e159 m at f = 1/50); outside it, at most 3.4e-9 m (WGS84) and 4.2e-9 m
(f = 1/50) at the 70 log-spaced radii up to 60 e2 a. ``iterative`` is
wrong by up to 0.995 e2 a (42501 m, at 16 m from the centre) on WGS84 and
1.012 e2 a at f = 1/50; at those radii beyond its edge it is within
4.4e-4 m (WGS84), limited by its 1e-12 rad stopping tolerance. Warning radii: ``direct`` 10 e2 a (about
10x its edge), ``iterative`` 30 e2 a (4.1x its WGS84 edge, 2.9x at
f = 1/50); 4.27e5 m and 1.28e6 m for WGS84.

On a sphere (f = 0, e2 a = 0) neither method warns. Over 306 directions at
radii 1e-300 m .. 1.5 a: ``iterative`` within 1.7e-8 m everywhere;
``direct`` within 2.9e-9 m from 7.8e-47 m up, NaN at 1.3e-47 m and below
(squared coordinates underflow).

Before the fix, ``method='Direct'`` (any non-``'iterative'`` string) silently
ran the closed form, and the closed form returned NaN with two
RuntimeWarnings at the ECEF origin.
"""

import warnings

import mpmath as mp
import numpy as np
import pytest

from pytcl.coordinate_systems import ecef2geodetic, geodetic2ecef
from pytcl.core.constants import WGS84
from pytcl.core.exceptions import MethodError
from pytcl.navigation.geodesy import SPHERE
from tests.validation._v2111 import shipped_module

mp.mp.dps = 40
_A = mp.mpf(float(WGS84.a))
_F = mp.mpf(float(WGS84.f))
_E2 = 2 * _F - _F * _F

BANDS = {
    "surface": (-10e3, 100e3),
    "leo": (100e3, 2000e3),
    "geo": (35_000e3, 36_000e3),
    "interior": (-6.3e6, -1e6),
}
N_POINTS = 2000
LAT_TOL = 4.5e-16
H_TOL = {"surface": 4.8e-9, "leo": 4.1e-9, "geo": 2.6e-8, "interior": 5.4e-9}


def _geodetic_mp(x, y, z, lat0, h0):
    p = mp.sqrt(mp.mpf(x) ** 2 + mp.mpf(y) ** 2)
    z = mp.mpf(z)
    lat, h = mp.mpf(lat0), mp.mpf(h0)
    for _ in range(30):
        s, c = mp.sin(lat), mp.cos(lat)
        n = _A / mp.sqrt(1 - _E2 * s * s)
        dn = n**3 * _E2 * s * c / _A**2
        f1 = (n + h) * c - p
        f2 = (n * (1 - _E2) + h) * s - z
        j11, j12 = dn * c - (n + h) * s, c
        j21, j22 = dn * (1 - _E2) * s + (n * (1 - _E2) + h) * c, s
        det = j11 * j22 - j12 * j21
        dl = (f1 * j22 - j12 * f2) / det
        dh = (j11 * f2 - j21 * f1) / det
        lat -= dl
        h -= dh
        if abs(dl) < mp.mpf(10) ** -35 and abs(dh) < mp.mpf(10) ** -30:
            break
    return lat, h


def _max_errors(method, band):
    rng = np.random.default_rng(0)
    lo, hi = BANDS[band]
    lat = np.arcsin(rng.uniform(-1, 1, N_POINTS))
    lon = rng.uniform(-np.pi, np.pi, N_POINTS)
    h = rng.uniform(lo, hi, N_POINTS)
    ecef = np.array([geodetic2ecef(a, b, c) for a, b, c in zip(lat, lon, h)]).T
    lat_out, _, h_out = ecef2geodetic(ecef, method=method)
    lat_err = h_err = 0.0
    for i in range(N_POINTS):
        ml, mh = _geodetic_mp(ecef[0, i], ecef[1, i], ecef[2, i], lat[i], h[i])
        lat_err = max(lat_err, float(abs(mp.mpf(float(lat_out[i])) - ml)))
        h_err = max(h_err, float(abs(mp.mpf(float(h_out[i])) - mh)))
    return lat_err, h_err


@pytest.mark.filterwarnings("ignore:ecef2geodetic:RuntimeWarning")
@pytest.mark.parametrize("band", BANDS)
def test_direct_matches_the_oracle(band):
    lat_err, h_err = _max_errors("direct", band)
    assert lat_err <= LAT_TOL
    assert h_err <= H_TOL[band]


@pytest.mark.filterwarnings("ignore:ecef2geodetic:RuntimeWarning")
def test_iterative_is_off_deep_inside():
    _, h_err = _max_errors("iterative", "interior")
    assert h_err > 1.0


def test_direct_handles_origin_inside_an_array(recwarn):
    ecef = np.array([[0.0, 0.0, 0.0], [6378137.0, 0.0, 0.0]]).T
    lat, lon, h = ecef2geodetic(ecef, method="direct")
    assert lat[0] == pytest.approx(np.pi / 2, rel=1e-15, abs=0)
    assert lon[0] == 0.0
    assert h[0] == pytest.approx(-WGS84.b, rel=1e-15, abs=0)
    assert lat[1] == 0.0 and h[1] == pytest.approx(0.0, abs=1e-9)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_origin_matches_between_methods():
    origin = np.zeros(3)
    assert ecef2geodetic(origin, method="direct") == ecef2geodetic(origin)


@pytest.mark.parametrize("method", ["Direct", "ITERATIVE", "closed", "iterativ", ""])
def test_method_is_case_sensitive(method):
    with pytest.raises(MethodError, match="'iterative' or 'direct'"):
        ecef2geodetic(np.array([7e6, 0.0, 0.0]), method=method)


_E2A = (2 * WGS84.f - WGS84.f**2) * WGS84.a
R = {"direct": 10.0 * _E2A, "iterative": 30.0 * _E2A}
METHODS = ["iterative", "direct"]


def _shipped():
    return shipped_module("pytcl/coordinate_systems/conversions/geodetic.py")


def _no_warning(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return ecef2geodetic(*args, **kwargs)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", [0.99, 0.5, 1e-3])
@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, -0.48, 0.64)])
def test_rows_inside_the_radius_warn_once(method, scale, axis):
    point = scale * R[method] * np.array(axis)
    with pytest.warns(RuntimeWarning, match=rf"\(method='{method}'\): 1 point") as rec:
        ecef2geodetic(point, method=method)
    assert len(rec) == 1


def test_warning_text_names_each_methods_failure():
    with pytest.warns(RuntimeWarning, match="'direct' returns NaN") as rec:
        ecef2geodetic(np.array([1e3, 0.0, 0.0]), method="direct")
    assert "4.27e+05 m" in str(rec[0].message)
    with pytest.warns(RuntimeWarning, match="'iterative' is off by up to") as rec:
        ecef2geodetic(np.array([1e3, 0.0, 0.0]), method="iterative")
    assert "1.28e+06 m" in str(rec[0].message)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("axis", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.6, -0.48, 0.64)])
def test_rows_just_outside_the_radius_do_not_warn_and_match_v2_11_1(method, axis):
    point = 1.01 * R[method] * np.array(axis)
    got = _no_warning(point, method=method)
    want = _shipped().ecef2geodetic(point, method=method)
    assert tuple(float(v).hex() for v in got) == tuple(float(v).hex() for v in want)


@pytest.mark.parametrize("method", METHODS)
def test_every_row_outside_the_radius_is_bit_identical_to_v2_11_1(method):
    rng = np.random.default_rng(7)
    lat = np.arcsin(rng.uniform(-1, 1, 500))
    lon = rng.uniform(-np.pi, np.pi, 500)
    r = np.exp(rng.uniform(np.log(R[method]), np.log(4.2e7), 500))
    ecef = np.array(
        [r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)]
    )
    shipped = _shipped().ecef2geodetic
    for layout in (ecef, ecef.T, ecef[:, 0].copy()):
        got = _no_warning(layout, method=method)
        want = shipped(layout, method=method)
        for g, w in zip(got, want):
            assert np.array_equal(g, w)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "point", [(3e5, 2e5, 4e5), (1.0, 2.0, 3.0), (SPHERE.a, 0.0, 0.0)]
)
def test_sphere_never_warns_and_is_accurate(method, point):
    lat, lon, h = _no_warning(np.array(point), SPHERE.a, SPHERE.f, method=method)
    x, y, z = point
    r = np.sqrt(x**2 + y**2 + z**2)
    assert lat == pytest.approx(np.arcsin(z / r), rel=0, abs=1e-15)
    assert lon == pytest.approx(np.arctan2(y, x), rel=0, abs=1e-15)
    assert h == pytest.approx(r - SPHERE.a, rel=0, abs=2e-8)


@pytest.mark.parametrize("method", METHODS)
def test_radius_scales_with_the_ellipsoid(method):
    a, f = 1.0, WGS84.f
    _no_warning(np.array([1.0, 0.0, 0.0]), a, f, method=method)
    with pytest.warns(RuntimeWarning, match="1 point"):
        ecef2geodetic(np.array([0.05, 0.0, 0.0]), a, f, method=method)


@pytest.mark.parametrize("method", METHODS)
def test_exact_origin_does_not_warn(method):
    lat, lon, h = _no_warning(np.zeros(3), method=method)
    assert (lat, lon) == (np.pi / 2, 0.0)
    assert h == pytest.approx(-WGS84.b, rel=1e-15, abs=0)


@pytest.mark.parametrize("method", METHODS)
def test_mixed_array_warns_once_and_surface_rows_are_unchanged(method):
    surface = np.array(
        [geodetic2ecef(0.3, 0.4, 100.0), geodetic2ecef(-0.9, 2.0, 5e5)]
    ).T
    alone = ecef2geodetic(surface, method=method)
    mixed = np.column_stack(
        [surface[:, 0], [1e5, 2e5, 1e5], np.zeros(3), surface[:, 1]]
    )
    with pytest.warns(RuntimeWarning, match="1 point") as rec:
        out = ecef2geodetic(mixed, method=method)
    assert len(rec) == 1
    # 'iterative' stops when every row has converged, so its rows are coupled
    # (as in v2.11.1): the altitude moves by up to 5e-6 m, not bitwise.
    tol = 0.0 if method == "direct" else 1e-5
    for got, want in zip(out, alone):
        assert abs(got[0] - want[0]) <= tol and abs(got[3] - want[1]) <= tol
