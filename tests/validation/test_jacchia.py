"""Validation of the Jacchia 1971 atmosphere port.

Two-layer oracle: the model physics is pinned bit-tight against MATLAB
fixtures captured with the solar geometry as explicit inputs (the
original's internal astro chain is unrunnable as shipped -- see
``scripts/matlab_capture/capture_jacchia.m``); the solar-geometry step
is validated separately against astropy. Physical sanity checks run
with no optional dependency.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.jacchia import (
    _jacchia_from_sun_geometry,
    _sun_ra_dec,
    jacchia_atmos_param,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "matlab"
REQUIRE = os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1"


class TestJacchiaMatlabFixtures:
    @pytest.fixture(scope="class")
    def rows(self):
        path = FIXTURES / "jacchia_atmos.csv"
        if not path.exists():
            if REQUIRE:
                pytest.fail("required MATLAB fixture missing: jacchia_atmos.csv")
            pytest.skip("MATLAB fixture not captured: jacchia_atmos.csv")
        return np.loadtxt(path, delimiter=",", ndmin=2)

    def test_physics_matches_matlab_bit_tight(self, rows):
        assert len(rows) >= 20
        for row in rows:
            (
                jul1,
                jul2,
                lat,
                _lon,
                alt,
                f10,
                f10b,
                kp,
                dec,
                lha,
                rho_ref,
                p_ref,
                t_ref,
                te_ref,
            ) = row
            rho, p, t, te = _jacchia_from_sun_geometry(
                jul1, jul2, lat, alt / 1000.0, f10, f10b, kp, dec, lha
            )
            # The bi-polynomial is summed vectorized here vs. MATLAB's
            # scalar loop; the ulp-level difference in log10(rho) is
            # amplified through 10**x to a few parts in 1e12.
            np.testing.assert_allclose(rho, rho_ref, rtol=5e-12)
            np.testing.assert_allclose(t, t_ref, rtol=1e-12)
            np.testing.assert_allclose(te, te_ref, rtol=1e-12)
            if p_ref > 0:
                # MATLAB's number is kPa (documented unit bug); this
                # port returns true Pa. The gas constant differs at
                # 4e-8 relative (CODATA 2018 vs the older value).
                np.testing.assert_allclose(p, p_ref * 1000.0, rtol=1e-6)
            else:
                # Where the molar-mass polynomial goes unphysical the
                # MATLAB pressure is negative; this port returns NaN.
                assert np.isnan(p)


class TestSunGeometry:
    def test_ra_dec_against_astropy(self):
        astropy_time = pytest.importorskip("astropy.time")
        astropy_coords = pytest.importorskip("astropy.coordinates")
        from astropy.utils import iers

        # Never reach the network for IERS tables; built-in precision
        # is far beyond this test's 0.02-degree tolerance.
        iers.conf.auto_download = False

        for jd in (2451545.0, 2455197.5, 2460310.5):
            t = astropy_time.Time(jd, format="jd", scale="tt")
            # Compare in the true-equator-true-equinox (of-date) frame:
            # get_sun returns GCRS (J2000 equinox), while the model
            # needs of-date RA for the LAST-based hour angle.
            sun = astropy_coords.get_sun(t).transform_to(astropy_coords.TETE(obstime=t))
            ra, dec = _sun_ra_dec(jd)
            ra_err = np.degrees(
                np.abs(np.mod(ra - sun.ra.rad + np.pi, 2 * np.pi) - np.pi)
            )
            dec_err = np.degrees(abs(dec - sun.dec.rad))
            assert ra_err < 0.02, f"RA error {ra_err:.4f} deg at JD {jd}"
            assert dec_err < 0.02, f"dec error {dec_err:.4f} deg at JD {jd}"


class TestPhysicalSanity:
    POINT = [np.radians(40.0), np.radians(-75.0), 400e3]

    def test_solar_activity_raises_density(self):
        lo = jacchia_atmos_param(2451545.0, 0.25, self.POINT, 90.0, 90.0, 1.0)
        hi = jacchia_atmos_param(2451545.0, 0.25, self.POINT, 220.0, 190.0, 7.0)
        assert hi.exospheric_temperature > lo.exospheric_temperature
        assert hi.density > lo.density

    def test_density_decreases_with_altitude(self):
        rhos = [
            jacchia_atmos_param(
                2451545.0,
                0.25,
                [np.radians(40.0), np.radians(-75.0), alt],
                150.0,
                150.0,
                3.0,
            ).density
            for alt in (150e3, 400e3, 800e3, 1500e3)
        ]
        assert all(a > b for a, b in zip(rhos, rhos[1:]))

    def test_low_altitude_matches_the_standard_atmosphere(self):
        # US76's tabulated extension gives rho(95 km) ~ 1.39e-6 kg/m^3
        # (pytcl's us_standard_atmosphere_1976 cannot serve as the
        # reference here: it clamps above its 86 km top). Jacchia's
        # 90-km boundary region should agree within a factor of two.
        state = jacchia_atmos_param(
            2451545.0,
            0.25,
            [np.radians(40.0), np.radians(-75.0), 95e3],
            150.0,
            150.0,
            3.0,
        )
        assert 0.7e-6 < state.density < 2.8e-6

    def test_domain_guards(self):
        with pytest.raises(ValueError, match="altitude"):
            jacchia_atmos_param(2451545.0, 0.25, [0.5, 0.0, 50e3], 150.0, 150.0, 3.0)
        with pytest.raises(ValueError, match="exospheric"):
            # Extreme flux drives Te far above the tabulated 1900 K.
            jacchia_atmos_param(2451545.0, 0.25, self.POINT, 500.0, 400.0, 9.0)

    def test_equator_is_finite(self):
        # MATLAB returns NaN at lat exactly 0 (0/0 in the
        # seasonal-latitude term); this port uses the limit.
        s = jacchia_atmos_param(2451545.0, 0.25, [0.0, 0.0, 300e3], 150.0, 150.0, 3.0)
        assert np.isfinite(s.density)

    def test_high_altitude_pressure_warns_nan(self):
        with pytest.warns(RuntimeWarning, match="mean-molecular-mass"):
            s = jacchia_atmos_param(2451545.0, 0.25, self.POINT, 150.0, 150.0, 3.0)
        assert np.isnan(s.pressure)
        assert np.isfinite(s.density)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
