"""US Standard Atmosphere 1976 below sea level, and altitude_from_pressure's
domain (task 2.7).

``us_standard_atmosphere_1976`` clamped its geopotential altitude to
``[0, 84852]`` m, silently returning the sea-level state for anything below
0 m -- the Dead Sea (-430 m) came back as 101325.0 Pa rather than the
~106599 Pa the published troposphere formula gives, an 11% error at -1000 m.
The standard is actually defined down to a -5000 m floor.
``altitude_from_pressure`` inverts only the troposphere gradient layer, so a
pressure below the tropopause value (~22632 Pa, at 11000 m geopotential)
silently extrapolated that lapse rate through layers that do not use it: 30
km's pressure (1197 Pa) inverted to 25379.2 m, off by 4621 m.

Reference: NOAA/NASA/USAF, U.S. Standard Atmosphere, 1976, NOAA-S/T 76-1562,
Part 4 (troposphere pressure/temperature formulas and the geopotential
altitude relation).
"""

import warnings

import pytest

from pytcl.atmosphere.models import altitude_from_pressure, us_standard_atmosphere_1976

# The standard's own troposphere closed form, quoted rather than imported so
# this file checks the code against the standard instead of restating it.
T0 = 288.15  # K, sea-level temperature
P0 = 101325.0  # Pa, sea-level pressure
LAPSE_RATE = -0.0065  # K/m, tropospheric lapse rate
G0 = 9.80665  # m/s^2, standard gravity
MOLAR_MASS_AIR = 0.0289644  # kg/mol
R_UNIVERSAL = 8.31432  # J/(mol K)
R0 = 6356766.0  # m, the standard's own effective Earth radius for H = r0 h / (r0 + h)


def _geopotential(altitude_m: float) -> float:
    """H = r0 h / (r0 + h): geometric altitude to geopotential altitude."""
    return R0 * altitude_m / (R0 + altitude_m)


def _troposphere_pressure(altitude_m: float) -> float:
    """Closed-form troposphere pressure, Part 4 of the 1976 standard."""
    h = _geopotential(altitude_m)
    temperature = T0 + LAPSE_RATE * h
    exponent = -G0 * MOLAR_MASS_AIR / (R_UNIVERSAL * LAPSE_RATE)
    return P0 * (temperature / T0) ** exponent


US76_BELOW_SEA_LEVEL_PA = {
    -5000.0: _troposphere_pressure(-5000.0),
    -2000.0: _troposphere_pressure(-2000.0),
    -1000.0: _troposphere_pressure(-1000.0),
    -430.0: _troposphere_pressure(-430.0),  # Dead Sea
    0.0: _troposphere_pressure(0.0),
}


class TestUS76BelowSeaLevel:
    """The standard's troposphere formula, not a sea-level clamp."""

    @pytest.mark.parametrize(
        "altitude,expected", sorted(US76_BELOW_SEA_LEVEL_PA.items())
    )
    def test_pressure_matches_the_closed_form(self, altitude, expected):
        result = us_standard_atmosphere_1976(altitude)
        assert result.pressure == pytest.approx(expected, rel=1e-3)

    def test_dead_sea_is_not_the_sea_level_clamp(self):
        """The defect's headline number: -430 m read back as 101325.0 Pa,
        5274 Pa (5%) low."""
        result = us_standard_atmosphere_1976(-430.0)
        assert result.pressure != pytest.approx(101325.0, rel=1e-4)
        assert result.pressure == pytest.approx(106598.8, rel=1e-3)

    def test_pressure_increases_monotonically_below_sea_level(self):
        altitudes = [-5000.0, -2000.0, -1000.0, -430.0, 0.0]
        pressures = [us_standard_atmosphere_1976(a).pressure for a in altitudes]
        assert all(a > b for a, b in zip(pressures, pressures[1:]))

    def test_warns_below_the_five_km_floor(self):
        with pytest.warns(UserWarning, match="-5000"):
            us_standard_atmosphere_1976(-6000.0)

    def test_does_not_warn_within_the_defined_range(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            us_standard_atmosphere_1976(-430.0)

    def test_warns_above_the_86km_ceiling(self):
        """The pre-existing high-altitude clamp had the same defect --
        silent, not warned."""
        with pytest.warns(UserWarning, match="ceiling"):
            us_standard_atmosphere_1976(90000.0)


class TestAltitudeFromPressureDomain:
    """The inversion only reflects the troposphere layer."""

    def test_altitude_from_pressure_warns_outside_the_inverted_range(self):
        with pytest.warns(UserWarning, match="troposphere"):
            altitude_from_pressure(us_standard_atmosphere_1976(30000.0).pressure)

    def test_extrapolation_is_bounded_not_unbounded(self):
        """Before the fix, 30 km's pressure inverted to 25379.2 m -- 4621 m
        off from 30000. Clamping to the tropopause boundary must not
        silently return that unbounded extrapolation."""
        pressure_30km = us_standard_atmosphere_1976(30000.0).pressure
        with pytest.warns(UserWarning, match="troposphere"):
            altitude = altitude_from_pressure(pressure_30km)
        assert altitude != pytest.approx(25379.2, abs=1.0)
        assert altitude == pytest.approx(11019.07, abs=1.0)

    def test_round_trips_within_the_troposphere_without_warning(self):
        for h in [-5000.0, -1000.0, 0.0, 5000.0, 10000.0]:
            pressure = us_standard_atmosphere_1976(h).pressure
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                altitude = altitude_from_pressure(pressure)
            assert altitude == pytest.approx(h, abs=5.0)
