"""Gas properties and gas-table speed of sound against MATLAB.

Fixtures captured from the MATLAB TCL's Constants.gasProp and
speedOfSoundInAir (algorithm 0) via headless R2026a. gasProp is pure
transcription and matches tightly; the speed of sound uses pytcl's
CODATA-2018 gas constants where MATLAB pins CODATA 2014 (~1.6e-8
relative on N_A), so its tolerance allows for that.

The MATLAB wrappers NRLMSISE00GasTemp/Alt4Pres are ported as
nrlmsise00_gas_temp/nrlmsise00_pressure_altitude with their upstream
defects fixed loudly (see the docstrings); the wrapper tests here
therefore validate against pytcl's own oracle-validated model, not
against MEX captures that would only echo the defects.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere._gas_properties import gas_properties, molar_mass
from pytcl.atmosphere.models import speed_of_sound_gas_table
from pytcl.atmosphere.nrlmsise00 import (
    nrlmsise00,
    nrlmsise00_alt_for_pressure,
    nrlmsise00_gas_temp,
    nrlmsise00_pressure_altitude,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# The compositions of the speed-of-sound fixture cases, mirrored from
# scripts in the capture driver. Case 3 includes species gasProp does
# not tabulate (atomic O, N, anomalous O*), which must be silently
# ignored; case 4 exercises the H2O closed-form virial correlation.
_COMPOSITIONS = {
    1: [("N2", 1.9e25 / 2.5), ("O2", 5.2e24 / 2.5), ("Ar", 2.3e23 / 2.5)],
    2: [
        ("N2", 1.6e25),
        ("O2", 4.3e24),
        ("Ar", 2.0e23),
        ("CO2", 1.0e22),
        ("He", 1.4e20),
    ],
    3: [
        ("N2", 1.6e25),
        ("O2", 4.3e24),
        ("O", 7.5e16),
        ("N", 3.1e14),
        ("O*", 5.0e10),
        ("Ar", 1.9e23),
    ],
    4: [
        ("N2", 1.5e25),
        ("O2", 4.1e24),
        ("Ar", 1.9e23),
        ("CO2", 9.0e21),
        ("H2O", 4.4e23),
    ],
}


class TestGasPropertiesAgainstMatlab:
    def test_all_gases_over_their_valid_ranges(self):
        by_gas = defaultdict(list)
        with open(FIXTURE_DIR / "gasprop.csv") as f:
            for row in csv.DictReader(f):
                by_gas[row["gas"]].append(row)
        assert len(by_gas) == 14
        for gas, rows in by_gas.items():
            for row in rows:
                t = float(row["T"])
                props = gas_properties(gas, t)
                np.testing.assert_allclose(
                    props.molar_mass, float(row["AMU"]), rtol=1e-13
                )
                np.testing.assert_allclose(props.c0p, float(row["C0p"]), rtol=1e-12)
                # The helium spline re-derives the same not-a-knot
                # piecewise polynomial through scipy; everything else
                # is closed-form.
                np.testing.assert_allclose(props.b, float(row["B"]), rtol=1e-9)
                np.testing.assert_allclose(props.db_dt, float(row["dBdT"]), rtol=1e-8)
                np.testing.assert_allclose(
                    props.d2b_dt2, float(row["d2BdT2"]), rtol=1e-7
                )

    def test_unknown_species_return_none(self):
        for name in ("O", "N", "H", "O*", "SF6"):
            assert molar_mass(name) is None
            assert gas_properties(name, 300.0) is None

    def test_out_of_range_temperature_warns(self):
        with pytest.warns(UserWarning, match="second virial"):
            gas_properties("NO", 500.0)
        with pytest.warns(UserWarning, match="specific heat"):
            gas_properties("N2", 100.0)


class TestSpeedOfSoundGasTableAgainstMatlab:
    def test_all_fixture_cases(self):
        with open(FIXTURE_DIR / "speed_of_sound_gas_table.csv") as f:
            for row in csv.DictReader(f):
                comp = _COMPOSITIONS[int(row["case"])]
                c = speed_of_sound_gas_table(float(row["T"]), float(row["P"]), comp)
                # 1e-6 comfortably covers the CODATA 2014 vs 2018
                # constant difference (~1e-8 relative).
                np.testing.assert_allclose(c, float(row["c"]), rtol=1e-6)

    def test_unknown_species_do_not_perturb_the_mixture(self):
        base = [("N2", 1.6e25), ("O2", 4.3e24), ("Ar", 1.9e23)]
        spiked = base + [("O", 7.5e16), ("N", 3.1e14), ("O*", 5.0e10)]
        c0 = speed_of_sound_gas_table(288.15, 101325.0, base)
        c1 = speed_of_sound_gas_table(288.15, 101325.0, spiked)
        assert c0 == c1

    def test_scale_invariance(self):
        # Only relative number densities matter.
        comp = _COMPOSITIONS[2]
        scaled = [(n, 7.3 * d) for n, d in comp]
        c0 = speed_of_sound_gas_table(300.0, 90000.0, comp)
        c1 = speed_of_sound_gas_table(300.0, 90000.0, scaled)
        np.testing.assert_allclose(c0, c1, rtol=1e-13)


class TestGasTempWrapper:
    LLA = [60 * math.pi / 180, -70 * math.pi / 180, 400e3]

    def test_matches_the_core_model(self):
        table, t, d = nrlmsise00_gas_temp(172, 29000.0, self.LLA)
        lst = 29000.0 / 3600.0 + (-70.0) / 15.0
        ref = nrlmsise00(172, 29000.0, 400.0, 60.0, -70.0, lst)
        np.testing.assert_allclose(d, ref.d, rtol=1e-14)
        np.testing.assert_allclose(t, ref.t, rtol=1e-14)

    def test_gas_table_layout(self):
        table, t, d = nrlmsise00_gas_temp(172, 29000.0, self.LLA)
        names = [name for name, _ in table]
        assert names == ["He", "O", "N2", "O2", "Ar", "H", "N", "O*"]
        # d[5] (total mass density) carries no label; every other d
        # entry appears in the table in order.
        expected = [d[i] for i in (0, 1, 2, 3, 4, 6, 7, 8)]
        np.testing.assert_allclose([v for _, v in table], expected, rtol=0)

    def test_lst_default_is_the_documented_formula(self):
        explicit = nrlmsise00_gas_temp(
            172, 29000.0, self.LLA, lst=29000.0 / 3600.0 - 70.0 / 15.0
        )
        default = nrlmsise00_gas_temp(172, 29000.0, self.LLA)
        np.testing.assert_allclose(default[2], explicit[2], rtol=0)
        # And it differs from the MATLAB MEX's hardcoded 16.
        matlab_like = nrlmsise00_gas_temp(172, 29000.0, self.LLA, lst=16.0)
        assert not np.allclose(default[2], matlab_like[2])

    def test_seconds_clip_at_86400(self):
        a = nrlmsise00_gas_temp(172, 90000.0, self.LLA)
        b = nrlmsise00_gas_temp(172, 86400.0, self.LLA)
        np.testing.assert_allclose(a[2], b[2], rtol=0)

    def test_solar_and_storm_parameters_are_honored(self):
        # The MATLAB MEX ignores these (nlhs guard defect); ours must
        # not.
        quiet = nrlmsise00_gas_temp(310, 50000.0, self.LLA)
        active = nrlmsise00_gas_temp(
            310, 50000.0, self.LLA, ap=48.8, f107=180.0, f107a=140.0
        )
        assert active[1][0] > quiet[1][0]
        storm = nrlmsise00_gas_temp(
            310,
            50000.0,
            self.LLA,
            ap=48.8,
            f107=180.0,
            f107a=140.0,
            ap_array=[148.8, 160.0, 139.0, 127.0, 118.0, 122.5, 115.4],
        )
        assert storm[1][0] > active[1][0]


class TestPressureAltitudeWrapper:
    LL = [60 * math.pi / 180, -70 * math.pi / 180]

    def test_matches_the_core_model_in_meters(self):
        alt_m, table, t, d = nrlmsise00_pressure_altitude(172, 29000.0, 1000.0, self.LL)
        lst = 29000.0 / 3600.0 - 70.0 / 15.0
        alt_km, ref = nrlmsise00_alt_for_pressure(
            172, 29000.0, 1000.0, 60.0, -70.0, lst
        )
        np.testing.assert_allclose(alt_m, alt_km * 1000.0, rtol=1e-14)
        np.testing.assert_allclose(d, ref.d, rtol=1e-14)
        assert [name for name, _ in table] == [
            "He",
            "O",
            "N2",
            "O2",
            "Ar",
            "H",
            "N",
            "O*",
        ]

    def test_end_to_end_with_speed_of_sound(self):
        # The intended composition workflow: model constituents at a
        # location, then the speed of sound from the mixture. Near sea
        # level the result must sit in the physical ballpark.
        alt_m, table, t, d = nrlmsise00_pressure_altitude(
            172, 29000.0, 101325.0, self.LL
        )
        c = speed_of_sound_gas_table(float(t[1]), 101325.0, table)
        assert 300.0 < c < 360.0
