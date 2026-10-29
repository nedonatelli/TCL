"""
Tests for the EGM (Earth Gravitational Model) geoid-height evaluation.

UNVERIFIED LOCALLY. EGM2008 coefficients are not shipped with pytcl and were
absent on the machine these tests were written on, so every test here skips
there and has never been run. Whoever has ``EGM2008.cof`` in the data
directory should run this file first: a failure may be a wrong tolerance or
reference rather than a library defect.

The gravity disturbance, gravity anomaly and deflection-of-the-vertical
evaluations have no tests here for want of reference values; see the v2.13.0
entry in ROADMAP.md ("EGM gravity disturbance/anomaly/deflection need
reference values").
"""

import numpy as np
import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.gravity.egm import geoid_height, geoid_heights

_data_skip = (FileNotFoundError, DependencyError)

# Geoid height N (m) above WGS84 at (geodetic latitude deg, east longitude
# deg, class), N from the EGM2008 "egm2008-1-height" column of GeographicLib's
# GeoidHeights.dat (https://sourceforge.net/projects/geographiclib/files/testdata/).
# GeographicLib states its EGM2008 grids come from NGA's binary gridded files
# (https://geographiclib.sourceforge.io/C++/doc/geoid.html); the first rows of
# the data file are transcribed from the copy in PyGeodesy 26.10.26
# test/testGeoids.py. Class from the Earth2014 SUR surface at the point:
# "ocean" is 0 m (open water), "land" is 870-1170 m, and the Antarctic point
# (3735 m) is excluded as high topography, see below.
_OCEAN = [
    (79.695484, 88.805571, 0.749472),
    (-15.244804, 168.747961, 65.108852),
    (43.377784, 229.448418, -29.55705),
    (69.016564, 344.078885, 60.69041),
]
_LAND = [
    (-19.379357, 15.85511, 24.888926),
    (-16.074554, 21.79547, 9.741516),
    (-11.256389, 286.249745, 30.058864),
]
# (-76.981466, 34.17016, 12.63629) is left out: 3735 m of Antarctic ice, where
# the height-anomaly-to-geoid correction is metres, not decimetres.

# Degree-0 term N0 that pytcl omits (egm.py geoid_height, gh-25). The pytcl
# docstring says "roughly -0.5 m"; NGA's EGM2008 README gives N0 = -0.41 m for
# WGS84 (as relayed in the task ruling; the README itself was not reachable
# from the machine these tests were written on). NGA's value is used. N_pytcl
# should equal N_reference - N0.
_N0_M = -0.41

# Ocean: bound = GeographicLib's quoted egm2008-1 bilinear interpolation error
# (max 25 mm, 0.8 mm rms; same page, "Interpolation and quantization errors")
# plus a share for unmodelled terms.
_OCEAN_TOLERANCE_M = 0.05
# Land, 0.9-1.2 km: the published N is the geoid height while pytcl's T/gamma
# is the height anomaly zeta; their difference grows with elevation (a few
# dm per km). This bound is a stated allowance for that, not a measured value.
_LAND_TOLERANCE_M = 0.5

# EGM2008 degree-2 zonal in the tide-free system (the published C-bar(2,0) of
# EGM2008_to2190_TideFree). The reference N is tide-free per NGA's EGM2008
# README (relayed, not re-read); a zero-tide file differs by a latitude-
# dependent 0.03-0.2 m, so the tests skip rather than guess when the installed
# coefficients are not tide-free.
_C20_TIDE_FREE = -0.484165143790815e-3


@pytest.fixture(scope="module")
def egm2008():
    from pytcl.gravity.egm import load_egm_coefficients

    try:
        coefficients = load_egm_coefficients("EGM2008")
    except _data_skip as exc:
        pytest.skip(f"EGM2008 coefficients unavailable: {exc}")
    if abs(coefficients.C[2, 0] - _C20_TIDE_FREE) > 1e-11:
        pytest.skip(
            "installed EGM2008 coefficients are not the tide-free set "
            f"(C20 = {coefficients.C[2, 0]!r}); reference heights are tide-free"
        )
    return coefficients


CASES = [(*p, _OCEAN_TOLERANCE_M) for p in _OCEAN] + [
    (*p, _LAND_TOLERANCE_M) for p in _LAND
]


class TestGeoidHeightAgainstNGA:
    """geoid_height against NGA's published EGM2008 geoid heights."""

    @pytest.mark.parametrize("lat_deg,lon_deg,n_ref,tolerance", CASES)
    def test_geoid_height_matches_nga(
        self, egm2008, lat_deg, lon_deg, n_ref, tolerance
    ):
        h = geoid_height(np.radians(lat_deg), np.radians(lon_deg), coefficients=egm2008)
        assert abs(h - (n_ref - _N0_M)) < tolerance

    def test_geoid_heights_array_matches_nga(self, egm2008):
        lats = np.radians([p[0] for p in CASES])
        lons = np.radians([p[1] for p in CASES])
        expected = np.array([p[2] - _N0_M for p in CASES])
        tolerances = np.array([p[3] for p in CASES])

        heights = geoid_heights(lats, lons, coefficients=egm2008)

        assert np.all(np.abs(heights - expected) < tolerances)

    def test_longitude_periodicity(self, egm2008):
        lat = np.radians(45.0)
        h1 = geoid_height(lat, 0.0, coefficients=egm2008)
        h2 = geoid_height(lat, 2.0 * np.pi, coefficients=egm2008)

        assert h1 == pytest.approx(h2, rel=1e-6, abs=0)

    def test_north_pole_independent_of_longitude(self, egm2008):
        h0 = geoid_height(np.pi / 2, 0.0, coefficients=egm2008)
        h1 = geoid_height(np.pi / 2, np.radians(137.0), coefficients=egm2008)

        assert h0 == pytest.approx(h1, rel=1e-6, abs=0)
