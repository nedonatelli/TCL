"""
Tests for the EGM (Earth Gravitational Model) geoid-height evaluation.

The geoid heights are checked against NGA's EGM2008 values. EGM2008
coefficients are not shipped with pytcl, so these tests run only where
``EGM2008.cof`` is installed in the data directory and skip elsewhere.
"""

import numpy as np
import pytest

from pytcl.core.exceptions import DependencyError
from pytcl.gravity.egm import geoid_height, geoid_heights

_data_skip = (FileNotFoundError, DependencyError)

# Geoid height N (m) above WGS84 from the EGM2008 spherical-harmonic model
# at (geodetic latitude deg, east longitude deg). Source: the "egm2008-1-height"
# column of GeographicLib's GeoidHeights.dat test data
# (https://sourceforge.net/projects/geographiclib/files/testdata/), whose
# EGM2008 values derive from NGA's EGM2008 release
# (https://earth-info.nga.mil). First rows of that file,
# transcribed from the copy embedded in PyGeodesy 26.10.26 test/testGeoids.py.
_NGA_EGM2008 = [
    (-76.981466, 34.17016, 12.63629),
    (79.695484, 88.805571, 0.749472),
    (-15.244804, 168.747961, 65.108852),
    (-19.379357, 15.85511, 24.888926),
    (43.377784, 229.448418, -29.55705),
    (-16.074554, 21.79547, 9.741516),
    (-11.256389, 286.249745, 30.058864),
    (69.016564, 344.078885, 60.69041),
]

# pytcl's synthesis omits the degree-0 term (about -0.5 m, gh-25) and
# evaluates on the ellipsoid rather than the geoid, so agreement with the
# full-model value is bounded at the metre level, not millimetres. The bound
# is unverified on this machine (no EGM2008 file), see the task report.
_MODEL_DIFFERENCE_M = 1.0


@pytest.fixture(scope="module")
def egm2008():
    from pytcl.gravity.egm import load_egm_coefficients

    try:
        return load_egm_coefficients("EGM2008")
    except _data_skip as exc:
        pytest.skip(f"EGM2008 coefficients unavailable: {exc}")


class TestGeoidHeightAgainstNGA:
    """geoid_height against NGA's published EGM2008 geoid heights."""

    @pytest.mark.parametrize("lat_deg,lon_deg,n_ref", _NGA_EGM2008)
    def test_geoid_height_matches_nga(self, egm2008, lat_deg, lon_deg, n_ref):
        h = geoid_height(np.radians(lat_deg), np.radians(lon_deg), coefficients=egm2008)
        assert abs(h - n_ref) < _MODEL_DIFFERENCE_M

    def test_geoid_heights_array_matches_nga(self, egm2008):
        lats = np.radians([p[0] for p in _NGA_EGM2008])
        lons = np.radians([p[1] for p in _NGA_EGM2008])
        refs = np.array([p[2] for p in _NGA_EGM2008])

        heights = geoid_heights(lats, lons, coefficients=egm2008)

        assert np.max(np.abs(heights - refs)) < _MODEL_DIFFERENCE_M

    def test_longitude_periodicity(self, egm2008):
        lat = np.radians(45.0)
        h1 = geoid_height(lat, 0.0, coefficients=egm2008)
        h2 = geoid_height(lat, 2.0 * np.pi, coefficients=egm2008)

        assert h1 == pytest.approx(h2, rel=1e-6, abs=0)

    def test_north_pole_independent_of_longitude(self, egm2008):
        h0 = geoid_height(np.pi / 2, 0.0, coefficients=egm2008)
        h1 = geoid_height(np.pi / 2, np.radians(137.0), coefficients=egm2008)

        assert h0 == pytest.approx(h1, rel=1e-6, abs=0)
