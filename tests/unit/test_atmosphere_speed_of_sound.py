"""Tests for the speed-of-sound ports in pytcl.atmosphere.models.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_speed_of_sound.m.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.models import (
    STANDARD_SPEED_OF_SOUND,
    speed_of_sound_cramer,
    speed_of_sound_ideal_gas,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# The ideal-gas form multiplies pytcl's CODATA 2018 gas constant (MATLAB
# uses CODATA 2014, 3.4e-7 apart relative); Cramer's polynomial shares
# every literal with MATLAB.
RTOL_IDEAL = 1e-6
RTOL_EXACT = 1e-12


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


class TestIdealGasMatlab:
    def test_matches_matlab(self):
        for T, rh, c_ref in _load("sos_ideal_gas.csv"):
            c = speed_of_sound_ideal_gas(T, rh)
            assert c == pytest.approx(c_ref, rel=RTOL_IDEAL)

    def test_warns_outside_temperature_range(self):
        with pytest.warns(UserWarning, match="0-30"):
            speed_of_sound_ideal_gas(250.0)

    def test_humidity_raises_speed(self):
        dry = speed_of_sound_ideal_gas(293.15, 0.0)
        humid = speed_of_sound_ideal_gas(293.15, 1.0)
        assert humid > dry


class TestCramerMatlab:
    def test_matches_matlab(self):
        for T, P, xw, xc, c_ref in _load("sos_cramer.csv"):
            c = speed_of_sound_cramer(T, P, xw, xc)
            assert c == pytest.approx(c_ref, rel=RTOL_EXACT)

    def test_agrees_with_ideal_gas_for_moist_air(self):
        # Independent derivations should land within ~0.3 m/s.
        c1 = speed_of_sound_ideal_gas(293.15, 0.5)
        xw = 0.5 * 2338.8 / 101325.0  # RH * Psat(20C) / P as mole fraction
        c2 = speed_of_sound_cramer(293.15, 101325.0, xw)
        assert c1 == pytest.approx(c2, abs=0.3)

    def test_warnings_and_errors(self):
        with pytest.warns(UserWarning, match="pressure"):
            speed_of_sound_cramer(293.15, 60000.0)
        with pytest.warns(UserWarning, match="water-vapor"):
            speed_of_sound_cramer(293.15, 101325.0, 0.08)
        with pytest.warns(UserWarning, match="CO2"):
            speed_of_sound_cramer(293.15, 101325.0, 0.0, 0.02)
        with pytest.raises(ValueError, match="water"):
            speed_of_sound_cramer(293.15, 101325.0, -0.01)
        with pytest.raises(ValueError, match="carbon"):
            speed_of_sound_cramer(293.15, 101325.0, 0.0, -0.01)

    def test_standard_constant(self):
        # Smith & Harlow's STP reference, and Cramer's dry 0C value are
        # within half a meter per second of each other.
        assert STANDARD_SPEED_OF_SOUND == pytest.approx(
            speed_of_sound_cramer(273.15, 101325.0), abs=0.5
        )
