"""What the barometric thermosphere model is, and is not (gh-79).

The module was named ``nrlmsise00`` and its class docstring called it "a
comprehensive thermosphere model covering altitudes from approximately -5 km
to 1000 km". It is not NRLMSISE-00: that model needs harmonic coefficient
tables from NOAA which are not distributed here. What it computes is a set of
per-species exponential profiles with floors.

Nothing warned a caller, and there were no validation tests, so a sea-level
density 44% below truth was invisible. These tests pin the limits now stated
in the docstring, so the documented numbers cannot drift away from behaviour
without something failing.
"""

import numpy as np
import pytest

from pytcl.atmosphere import (
    SimplifiedThermosphere,
    simplified_thermosphere,
    us_standard_atmosphere_1976,
)

MID_LAT = np.radians(45.0)


def _density(altitude_m, **kw):
    out = simplified_thermosphere(
        latitude=kw.pop("latitude", MID_LAT),
        longitude=kw.pop("longitude", 0.0),
        altitude=altitude_m,
        year=kw.pop("year", 2020),
        day_of_year=kw.pop("day_of_year", 100),
        seconds_in_day=kw.pop("seconds_in_day", 43200.0),
        **kw,
    )
    return float(np.atleast_1d(out.density)[0])


class TestThermosphereIsUsable:
    """Above ~200 km the model earns its place."""

    def test_matches_published_nrlmsise00_at_400km(self):
        """Published NRLMSISE-00 at 400 km, F10.7 = 150, is ~2-4e-12 kg/m^3.

        This is the claim the docstring makes for the model's own regime. If
        it fails, the docstring is overselling and must be corrected with it.
        """
        rho = _density(
            400e3,
            latitude=np.radians(40.0),
            longitude=np.radians(-75.0),
            year=2024,
            day_of_year=1,
            f107=150.0,
            f107a=150.0,
            ap=5.0,
        )
        assert 2e-12 <= rho <= 4e-12, f"400 km density {rho:.3e} outside 2-4e-12"

    def test_density_decreases_monotonically_through_the_thermosphere(self):
        alts = [200e3, 300e3, 400e3, 600e3, 800e3]
        rho = [_density(a) for a in alts]
        assert all(b < a for a, b in zip(rho, rho[1:])), rho

    def test_higher_solar_flux_raises_thermospheric_density(self):
        """The F10.7 coupling is the physical content that distinguishes this
        from a plain exponential atmosphere."""
        quiet = _density(400e3, f107=70.0, f107a=70.0)
        active = _density(400e3, f107=230.0, f107a=230.0)
        assert active > quiet, (quiet, active)


class TestLowAltitudeIsWrongAndSaysSo:
    """Below ~86 km the model is wrong, by a margin the docstring quotes.

    These assert the *error*, which is unusual but deliberate: the numbers are
    published in the class docstring to steer callers to
    us_standard_atmosphere_1976, so they have to stay true. If the model is
    ever fixed these fail, and the docstring must be updated with it.
    """

    @pytest.mark.parametrize(
        "altitude_km,expected_ratio,tol",
        [(0, 0.556, 0.02), (10, 0.612, 0.02), (30, 1.419, 0.02), (80, 0.020, 0.005)],
    )
    def test_ratio_to_us_standard_matches_the_documented_value(
        self, altitude_km, expected_ratio, tol
    ):
        mine = _density(altitude_km * 1000.0)
        ref = float(
            np.atleast_1d(us_standard_atmosphere_1976(altitude_km * 1000.0).density)[0]
        )
        assert mine / ref == pytest.approx(expected_ratio, abs=tol)

    def test_sea_level_density_is_far_enough_off_to_matter(self):
        """The headline of gh-79: 0.682 against a true 1.225."""
        assert _density(0.0) == pytest.approx(0.682, abs=0.01)
        ref = float(np.atleast_1d(us_standard_atmosphere_1976(0.0).density)[0])
        assert abs(_density(0.0) - ref) / ref > 0.4


class TestSpeciesFloorsAreVisible:
    """Several species densities are clamps, not measurements.

    Documented so that a caller reading h_density = 1e8 knows it is a floor
    rather than a computed number.
    """

    @staticmethod
    def _state(altitude_m):
        return simplified_thermosphere(
            latitude=MID_LAT,
            longitude=0.0,
            altitude=altitude_m,
            year=2020,
            day_of_year=100,
            seconds_in_day=43200.0,
        )

    def test_hydrogen_sits_on_its_floor_through_the_lower_atmosphere(self):
        for km in (0, 100, 200, 400):
            h = float(np.atleast_1d(self._state(km * 1000.0).h_density)[0])
            assert h == pytest.approx(1e8, rel=1e-9), f"{km} km: {h:.3e}"

    def test_hydrogen_leaves_its_floor_high_enough_up(self):
        """Guard the guard: if it were pinned everywhere the test above would
        pass for a model that computed nothing at all."""
        h = float(np.atleast_1d(self._state(800e3).h_density)[0])
        assert h > 1e8 * 10

    def test_helium_sits_on_its_floor_low_down(self):
        for km in (0, 100):
            he = float(np.atleast_1d(self._state(km * 1000.0).he_density)[0])
            assert he == pytest.approx(1e10, rel=1e-9), f"{km} km: {he:.3e}"


class TestTheNameNoLongerClaimsNRLMSISE00:
    """gh-79 was a naming defect as much as a numerical one."""

    def test_the_old_names_are_gone(self):
        import pytcl.atmosphere as atmosphere

        for gone in ("NRLMSISE00", "nrlmsise00", "NRLMSISE00Output"):
            assert not hasattr(atmosphere, gone), gone

    def test_the_docstring_disclaims_nrlmsise00(self):
        doc = SimplifiedThermosphere.__doc__ or ""
        assert "not NRLMSISE-00" in doc
        assert "us_standard_atmosphere_1976" in doc
