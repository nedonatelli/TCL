"""Validation of the exact projections added for the gh-25 close-out.

Oracles: PROJ via pyproj (``+proj=sterea`` for EPSG method 9809,
``+proj=aeqd`` for the geodesic azimuthal equidistant), plus internal
consistency against the spherical implementations in their limits.
Both oracles live in optional extras, so tests skip without them.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.projections import (
    azimuthal_equidistant,
    azimuthal_equidistant_exact,
    azimuthal_equidistant_exact_inverse,
    oblique_stereographic,
    oblique_stereographic_inverse,
)

pyproj = pytest.importorskip("pyproj")
pytest.importorskip("geographiclib")

RD_LAT0 = np.radians(52.156160556)
RD_LON0 = np.radians(5.387638889)
RD_K0 = 0.9999079

TEST_POINTS = [
    (53.0, 6.0),
    (50.5, 3.5),
    (54.9, 9.9),
    (45.0, -5.0),
    (60.0, 20.0),
    (52.156160556, 5.387638889),  # the origin itself
]


class TestObliqueStereographic:
    @pytest.fixture(scope="class")
    def sterea(self):
        return pyproj.Proj(
            proj="sterea",
            lat_0=np.degrees(RD_LAT0),
            lon_0=np.degrees(RD_LON0),
            k_0=RD_K0,
            ellps="WGS84",
        )

    @pytest.mark.parametrize("latd,lond", TEST_POINTS)
    def test_coordinates_match_proj_sterea(self, sterea, latd, lond):
        xe, ye = sterea(lond, latd)
        r = oblique_stereographic(
            np.radians(latd), np.radians(lond), RD_LAT0, RD_LON0, k0=RD_K0
        )
        # gh-25 documented km-scale divergence for the old `stereographic`;
        # the EPSG-9809 implementation must match PROJ to sub-micrometer.
        assert abs(r.x - xe) < 1e-6 and abs(r.y - ye) < 1e-6

    @pytest.mark.parametrize("latd,lond", TEST_POINTS)
    def test_roundtrip(self, latd, lond):
        r = oblique_stereographic(
            np.radians(latd), np.radians(lond), RD_LAT0, RD_LON0, k0=RD_K0
        )
        lat, lon = oblique_stereographic_inverse(r.x, r.y, RD_LAT0, RD_LON0, k0=RD_K0)
        np.testing.assert_allclose(np.degrees(lat), latd, atol=1e-9)
        np.testing.assert_allclose(np.degrees(lon), lond, atol=1e-9)

    @pytest.mark.parametrize("latd,lond", [(53.0, 6.0), (45.0, -5.0), (60.0, 20.0)])
    def test_scale_and_convergence_match_proj_factors(self, sterea, latd, lond):
        f = sterea.get_factors(lond, latd)
        r = oblique_stereographic(
            np.radians(latd), np.radians(lond), RD_LAT0, RD_LON0, k0=RD_K0
        )
        np.testing.assert_allclose(r.scale, f.meridional_scale, rtol=1e-8)
        np.testing.assert_allclose(
            np.degrees(r.convergence), f.meridian_convergence, atol=1e-8
        )

    def test_origin_scale_is_k0(self):
        r = oblique_stereographic(RD_LAT0, RD_LON0, RD_LAT0, RD_LON0, k0=RD_K0)
        np.testing.assert_allclose(r.scale, RD_K0, rtol=1e-12)
        assert r.x == 0.0 and r.y == 0.0


class TestAzimuthalEquidistantExact:
    LAT0, LON0 = np.radians(38.9), np.radians(-77.0)

    @pytest.fixture(scope="class")
    def aeqd(self):
        return pyproj.Proj(proj="aeqd", lat_0=38.9, lon_0=-77.0, ellps="WGS84")

    @pytest.mark.parametrize(
        "latd,lond",
        [
            (40.0, -75.0),
            (0.0, 10.0),
            (-38.0, 102.5),  # near the antipode of the centre
            (52.0, 170.0),
            (-90.0, 0.0),
        ],
    )
    def test_coordinates_match_proj_aeqd(self, aeqd, latd, lond):
        xe, ye = aeqd(lond, latd)
        r = azimuthal_equidistant_exact(
            np.radians(latd), np.radians(lond), self.LAT0, self.LON0
        )
        assert abs(r.x - xe) < 1e-6 and abs(r.y - ye) < 1e-6
        assert r.scale == 1.0

    @pytest.mark.parametrize("latd,lond", [(40.0, -75.0), (0.0, 10.0), (52.0, 170.0)])
    def test_roundtrip(self, latd, lond):
        r = azimuthal_equidistant_exact(
            np.radians(latd), np.radians(lond), self.LAT0, self.LON0
        )
        lat, lon = azimuthal_equidistant_exact_inverse(r.x, r.y, self.LAT0, self.LON0)
        np.testing.assert_allclose(np.degrees(lat), latd, atol=1e-9)
        np.testing.assert_allclose(np.degrees(lon), lond, atol=1e-9)

    def test_reduces_to_spherical_on_the_sphere(self):
        # With e2 = 0 the geodesics are great circles, so the exact
        # projection must agree with the spherical implementation
        # (authalic radius = a on a sphere) -- convergence included,
        # which pins the s12/m12 generalization of c/sin(c).
        lat0, lon0 = np.radians(38.9), np.radians(-77.0)
        for latd, lond in [(40.0, -75.0), (10.0, -40.0), (-20.0, -100.0)]:
            exact = azimuthal_equidistant_exact(
                np.radians(latd), np.radians(lond), lat0, lon0, e2=0.0
            )
            sph = azimuthal_equidistant(
                np.radians(latd), np.radians(lond), lat0, lon0, e2=0.0
            )
            np.testing.assert_allclose(exact.x, sph.x, rtol=1e-8)
            np.testing.assert_allclose(exact.y, sph.y, rtol=1e-8)
            np.testing.assert_allclose(exact.convergence, sph.convergence, atol=1e-8)

    def test_centre_maps_to_origin(self):
        r = azimuthal_equidistant_exact(self.LAT0, self.LON0, self.LAT0, self.LON0)
        assert r == (0.0, 0.0, 1.0, 0.0)

    def test_missing_geographiclib_is_loud(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "geographiclib", None)
        monkeypatch.setitem(sys.modules, "geographiclib.geodesic", None)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="geodesy"):
            azimuthal_equidistant_exact(0.1, 0.1, 0.0, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
