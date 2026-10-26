"""Oracle tests for ``ecef2geodetic`` (v2.12.0 Task 3.5).

The oracle solves ``p = (N + h) cos(lat)``, ``z = (N (1 - e2) + h) sin(lat)``
by 2-D Newton in mpmath at 40 digits. Measured on 20000 seeded points per
band (this file samples fewer so the suite stays fast; tolerances are 1.5x
the 20000-point maximum).

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

Before the fix, ``method='Direct'`` (any non-``'iterative'`` string) silently
ran the closed form, and the closed form returned NaN with two
RuntimeWarnings at the ECEF origin.
"""

import mpmath as mp
import numpy as np
import pytest

from pytcl.coordinate_systems import ecef2geodetic, geodetic2ecef
from pytcl.core.constants import WGS84
from pytcl.core.exceptions import MethodError

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


@pytest.mark.parametrize("band", BANDS)
def test_direct_matches_the_oracle(band):
    lat_err, h_err = _max_errors("direct", band)
    assert lat_err <= LAT_TOL
    assert h_err <= H_TOL[band]


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
