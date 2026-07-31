"""``true_airspeed_from_mach`` against the US Standard Atmosphere 1976 tables.

An exported conversion no test reached (gh-49). It is one multiplication --
Mach times the local speed of sound -- so the question a test has to answer is
not whether the arithmetic is right but whether the speed of sound it multiplies
by is the published one, at the altitude the caller meant.

That second clause is the whole substance here. US Standard Atmosphere 1976
defines its layers in **geopotential** altitude, while this function documents
its argument as **geometric** altitude. The two differ by about 0.17% at the
tropopause -- 11,019 m geometric is 11,000 m geopotential -- and a model that
confused them would be wrong by 0.08 m/s in the speed of sound there while
looking perfectly reasonable everywhere else. The reference points below are
chosen to pin that conversion down rather than to step around it.

Reference: NOAA/NASA/USAF, *U.S. Standard Atmosphere, 1976*, NOAA-S/T 76-1562,
Table I (temperature and speed of sound against geopotential altitude).
"""

import numpy as np
import pytest

from pytcl.atmosphere.models import true_airspeed_from_mach, us_standard_atmosphere_1976

# The defining constants of US Standard Atmosphere 1976, quoted rather than
# imported so this file states the standard instead of restating the code.
SEA_LEVEL_TEMPERATURE_K = 288.15
TROPOSPHERIC_LAPSE_K_PER_M = 0.0065
ISOTHERMAL_TEMPERATURE_K = 216.65  # 11 km to 20 km geopotential
SPECIFIC_HEAT_RATIO = 1.4
SPECIFIC_GAS_CONSTANT = 287.0528  # J/(kg K), = 8314.32 / 28.9644
EFFECTIVE_EARTH_RADIUS_M = 6_356_766.0  # the standard's own r0


def _geometric_altitude(geopotential_m: float) -> float:
    """The standard's altitude relation, ``z = r0 H / (r0 - H)``.

    Table I is indexed by geopotential altitude; this function takes geometric.
    Converting here is what lets the published temperatures below be quoted at
    the round altitudes they are actually published at.
    """
    return (
        EFFECTIVE_EARTH_RADIUS_M
        * geopotential_m
        / (EFFECTIVE_EARTH_RADIUS_M - geopotential_m)
    )


def _speed_of_sound(temperature_k: float) -> float:
    """``a = sqrt(gamma R T)``, the standard's definition."""
    return float(np.sqrt(SPECIFIC_HEAT_RATIO * SPECIFIC_GAS_CONSTANT * temperature_k))


class TestAgainstPublishedSpeedOfSound:
    """Mach 1 is the speed of sound, so this reads the table directly.

    Each entry is a geopotential altitude from Table I and the temperature the
    standard publishes there. Both the geometric altitude passed in and the
    speed of sound expected out are derived from the constants above, so a
    reader can check every number against the standard rather than taking a
    transcribed decimal on trust.
    """

    POINTS = [
        (0.0, SEA_LEVEL_TEMPERATURE_K, "sea level, where the two scales coincide"),
        (
            5_000.0,
            SEA_LEVEL_TEMPERATURE_K - TROPOSPHERIC_LAPSE_K_PER_M * 5_000.0,
            "mid-troposphere, on the lapse rate",
        ),
        (11_000.0, ISOTHERMAL_TEMPERATURE_K, "tropopause"),
        (20_000.0, ISOTHERMAL_TEMPERATURE_K, "top of the isothermal layer"),
    ]
    IDS = [f"{geopotential / 1000:.0f}km" for geopotential, _, _ in POINTS]

    @pytest.mark.parametrize("geopotential,temperature,description", POINTS, ids=IDS)
    def test_mach_one_equals_the_published_speed_of_sound(
        self, geopotential, temperature, description
    ):
        """Within 0.001 m/s of the standard.

        The tolerance is tight because both sides are exact: the standard
        defines the temperature profile by these constants, it does not measure
        it. Anything looser would tolerate a wrong gas constant.
        """
        altitude = _geometric_altitude(geopotential)
        expected = _speed_of_sound(temperature)
        speed = float(true_airspeed_from_mach(1.0, altitude))

        assert speed == pytest.approx(expected, abs=0.001), (
            f"at {geopotential:.0f} m geopotential ({altitude:.3f} m geometric, "
            f"{description}): got {speed:.4f} m/s, US Standard Atmosphere 1976 "
            f"gives {expected:.4f} m/s at {temperature:.2f} K"
        )

    @pytest.mark.parametrize("geopotential,temperature,description", POINTS, ids=IDS)
    def test_the_model_reproduces_the_published_temperature(
        self, geopotential, temperature, description
    ):
        """The speed of sound is only right if the temperature behind it is.

        Asserting the temperature separately means a compensating error -- a
        wrong temperature cancelled by a wrong gas constant -- cannot pass.
        """
        state = us_standard_atmosphere_1976(_geometric_altitude(geopotential))
        assert float(np.asarray(state.temperature)) == pytest.approx(
            temperature, abs=1e-3
        )

    def test_the_altitude_argument_is_geometric_not_geopotential(self):
        """The distinction this function's correctness turns on.

        At 11,000 m *geometric* the model is still in the troposphere and gives
        a speed of sound measurably above the tropopause value. If the argument
        were being read as geopotential, the two would agree. Asserting they
        differ is what makes the reference points above meaningful rather than
        accidentally satisfied.
        """
        at_geometric_11km = float(true_airspeed_from_mach(1.0, 11_000.0))
        at_geopotential_11km = float(
            true_airspeed_from_mach(1.0, _geometric_altitude(11_000.0))
        )

        assert at_geometric_11km > at_geopotential_11km, (
            "11,000 m geometric sits below the tropopause and must be warmer "
            "than 11,000 m geopotential"
        )
        assert at_geometric_11km - at_geopotential_11km == pytest.approx(
            0.084, abs=0.01
        ), (
            "the gap between the two altitude scales at the tropopause is not "
            "what the 1976 standard's geopotential conversion implies"
        )


class TestConversionProperties:
    """Behavior that must hold for any Mach number and altitude."""

    ALTITUDES = [0.0, 1_000.0, _geometric_altitude(11_000.0), 20_000.0, 30_000.0]

    @pytest.mark.parametrize("altitude", ALTITUDES)
    @pytest.mark.parametrize("mach", [0.0, 0.5, 0.8, 1.0, 2.0, 5.0])
    def test_airspeed_is_mach_times_the_local_speed_of_sound(self, mach, altitude):
        """The defining relation, checked against the atmosphere model itself.

        This is the one assertion here that is deliberately internal: it pins
        the function to the same atmosphere every other caller sees, so the two
        cannot drift apart. The published comparison above is what establishes
        that atmosphere is right in the first place.
        """
        state = us_standard_atmosphere_1976(altitude)
        expected = mach * float(np.asarray(state.speed_of_sound))
        assert float(true_airspeed_from_mach(mach, altitude)) == pytest.approx(
            expected, rel=1e-12
        )

    def test_zero_mach_is_zero_airspeed_at_every_altitude(self):
        result = true_airspeed_from_mach(0.0, np.array(self.ALTITUDES))
        np.testing.assert_array_equal(result, np.zeros(len(self.ALTITUDES)))

    def test_airspeed_scales_linearly_with_mach(self):
        """Doubling the Mach number doubles the airspeed at fixed altitude."""
        single = float(true_airspeed_from_mach(1.0, 10_000.0))
        double = float(true_airspeed_from_mach(2.0, 10_000.0))
        assert double == pytest.approx(2.0 * single, rel=1e-12)

    def test_the_speed_of_sound_falls_through_the_troposphere(self):
        """Temperature drops with altitude below the tropopause, so a does too.

        A model that lost its lapse rate would return a constant here.
        """
        speeds = [
            float(true_airspeed_from_mach(1.0, alt)) for alt in range(0, 11_000, 2_000)
        ]
        assert all(later < earlier for earlier, later in zip(speeds, speeds[1:])), (
            f"speed of sound does not decrease monotonically through the "
            f"troposphere: {speeds}"
        )

    def test_the_speed_of_sound_is_constant_through_the_isothermal_layer(self):
        """From 11 km to 20 km geopotential the temperature is fixed at 216.65 K.

        A lapse rate wrongly applied above the tropopause shows up here and
        nowhere else in this file.
        """
        speeds = [
            float(true_airspeed_from_mach(1.0, alt))
            for alt in (
                _geometric_altitude(11_000.0),
                14_000.0,
                17_000.0,
                _geometric_altitude(20_000.0),
            )
        ]
        assert max(speeds) - min(speeds) < 0.01, (
            f"speed of sound varies by {max(speeds) - min(speeds):.4f} m/s "
            f"across the isothermal layer, where it should be constant: {speeds}"
        )

    def test_arrays_are_handled_elementwise(self):
        """Both arguments broadcast, and the result matches the scalar calls."""
        machs = np.array([0.5, 0.8, 1.2])
        altitudes = np.array(
            [0.0, _geometric_altitude(5_000.0), _geometric_altitude(11_000.0)]
        )

        vectorized = true_airspeed_from_mach(machs, altitudes)
        assert vectorized.shape == (3,)
        for index, (mach, altitude) in enumerate(zip(machs, altitudes)):
            assert vectorized[index] == pytest.approx(
                float(true_airspeed_from_mach(mach, altitude)), rel=1e-12
            )
