"""Tests for pytcl.atmosphere.humidity and the refractivity helpers.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_humidity_refrac.m.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.humidity import (
    abs_humid_to_number_density,
    abs_humid_to_rel_humid,
    abs_humid_to_spec_humid,
    dew_point_pressure,
    dew_point_temperature,
    number_density_to_abs_humid,
    rel_humid_to_abs_humid,
    rel_humid_to_spec_humid,
    spec_humid_to_abs_humid,
    spec_humid_to_rel_humid,
)
from pytcl.atmosphere.refraction import (
    approx_refractivity,
    atmos_exp_decay_const,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Formulas transcribed literal-for-literal match MATLAB to roundoff; the
# humidity conversions also involve the gas constant and atomic mass unit,
# where pytcl's CODATA 2018 values differ from MATLAB's CODATA 2014 ones
# at ~3.4e-7 relative.
RTOL_EXACT = 1e-12
RTOL_CONST = 1e-6


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


class TestDewPointMatlab:
    def test_pressure_and_temperature_match_matlab(self):
        rows = _load("humidity_dew_point.csv")
        for alg, T, p_ref, T_back_ref in rows:
            p = dew_point_pressure(T, int(alg))
            assert p == pytest.approx(p_ref, rel=RTOL_EXACT)
            T_back = dew_point_temperature(p, int(alg))
            assert T_back == pytest.approx(T_back_ref, rel=RTOL_EXACT)

    def test_round_trip_all_algorithms(self):
        temps = np.linspace(233.15, 313.15, 9)
        for alg in (0, 1, 2):
            p = dew_point_pressure(temps, alg)
            np.testing.assert_allclose(dew_point_temperature(p, alg), temps, rtol=1e-9)

    def test_pressure_monotone_in_temperature(self):
        temps = np.linspace(233.15, 313.15, 50)
        p = dew_point_pressure(temps)
        assert np.all(np.diff(p) > 0)

    def test_invalid_algorithm_raises(self):
        with pytest.raises(ValueError, match="algorithm"):
            dew_point_pressure(288.15, 3)
        with pytest.raises(ValueError, match="algorithm"):
            dew_point_temperature(1000.0, -1)

    def test_scalar_returns_float(self):
        assert isinstance(dew_point_pressure(288.15), float)
        assert isinstance(dew_point_temperature(1706.0), float)


class TestRelAbsHumidMatlab:
    def test_matches_matlab(self):
        rows = _load("humidity_rel_abs.csv")
        for alg, rh, T, ah_ref, rh_back_ref in rows:
            ah = rel_humid_to_abs_humid(rh, T, int(alg))
            assert ah == pytest.approx(ah_ref, rel=RTOL_CONST)
            rh_back = abs_humid_to_rel_humid(ah, T, int(alg))
            assert rh_back == pytest.approx(rh_back_ref, rel=RTOL_CONST)

    def test_round_trip_identity(self):
        rh = np.array([0.05, 0.3, 0.6, 0.95])
        ah = rel_humid_to_abs_humid(rh, 288.15)
        np.testing.assert_allclose(abs_humid_to_rel_humid(ah, 288.15), rh, rtol=1e-12)

    def test_vectorized_over_temperature(self):
        temps = np.array([263.15, 288.15, 308.15])
        ah = rel_humid_to_abs_humid(0.5, temps)
        assert ah.shape == temps.shape
        assert np.all(np.diff(ah) > 0)  # warmer air holds more water


class TestAbsSpecHumidMatlab:
    def test_matches_matlab(self):
        rows = _load("humidity_abs_spec.csv")
        for def_choice, ah, rho, sh_ref, ah_back_ref in rows:
            sh = abs_humid_to_spec_humid(ah, rho, int(def_choice))
            assert sh == pytest.approx(sh_ref, rel=RTOL_EXACT)
            ah_back = spec_humid_to_abs_humid(sh, rho, int(def_choice))
            assert ah_back == pytest.approx(ah_back_ref, rel=RTOL_EXACT)

    def test_round_trip_both_definitions(self):
        ah = np.array([1e-4, 5e-3, 0.03])
        for definition in (0, 1):
            sh = abs_humid_to_spec_humid(ah, 1.225, definition)
            np.testing.assert_allclose(
                spec_humid_to_abs_humid(sh, 1.225, definition), ah, rtol=1e-12
            )

    def test_definitions_differ(self):
        sh0 = abs_humid_to_spec_humid(0.01, 1.225, 0)
        sh1 = abs_humid_to_spec_humid(0.01, 1.225, 1)
        assert sh1 < sh0  # total-density denominator is larger


class TestRelSpecHumidMatlab:
    def test_matches_matlab(self):
        rows = _load("humidity_rel_spec.csv")
        for alg, def_choice, rh, T, rho, sh_ref, rh_back_ref in rows:
            sh = rel_humid_to_spec_humid(rh, T, rho, int(def_choice), int(alg))
            assert sh == pytest.approx(sh_ref, rel=RTOL_CONST)
            rh_back = spec_humid_to_rel_humid(sh, T, rho, int(def_choice), int(alg))
            assert rh_back == pytest.approx(rh_back_ref, rel=RTOL_CONST)

    def test_round_trip_identity(self):
        rh = 0.42
        for definition in (0, 1):
            sh = rel_humid_to_spec_humid(rh, 288.15, 1.225, definition)
            assert spec_humid_to_rel_humid(
                sh, 288.15, 1.225, definition
            ) == pytest.approx(rh, rel=1e-12)


class TestNumberDensityMatlab:
    def test_matches_matlab(self):
        rows = _load("humidity_number_density.csv")
        for ah, nd_ref, ah_back_ref in rows:
            nd = abs_humid_to_number_density(ah)
            assert nd == pytest.approx(nd_ref, rel=RTOL_CONST)
            assert number_density_to_abs_humid(nd) == pytest.approx(
                ah_back_ref, rel=RTOL_CONST
            )

    def test_round_trip_identity(self):
        ah = np.array([1e-5, 1e-3, 0.05])
        nd = abs_humid_to_number_density(ah)
        np.testing.assert_allclose(number_density_to_abs_humid(nd), ah, rtol=1e-15)


class TestApproxRefractivityMatlab:
    def test_matches_matlab(self):
        rows = _load("refrac_approx_refractivity.csv")
        for T, P, Pw, n_ref in rows:
            assert approx_refractivity(T, P, Pw) == pytest.approx(n_ref, rel=RTOL_EXACT)

    def test_dry_air_term_only_when_no_vapor(self):
        n = approx_refractivity(288.15, 101325.0, 0.0)
        assert n == pytest.approx(77.6 * (101325.0 / 100) / 288.15, rel=1e-12)

    def test_vectorized(self):
        temps = np.array([263.15, 288.15, 308.15])
        n = approx_refractivity(temps, 101325.0, 850.0)
        assert n.shape == temps.shape


class TestExpDecayConstMatlab:
    def test_matches_matlab(self):
        rows = _load("refrac_exp_decay_const.csv")
        for ns, ce_ref, dn_ref in rows:
            ce, delta_n = atmos_exp_decay_const(ns)
            assert ce == pytest.approx(ce_ref, rel=RTOL_EXACT)
            assert delta_n == pytest.approx(dn_ref, rel=RTOL_EXACT)

    def test_exponential_model_consistency(self):
        # ce is defined so that N at 1 km equals Ns + delta_n exactly.
        ns = 313.0
        ce, delta_n = atmos_exp_decay_const(ns)
        assert ns * np.exp(-ce * 1000.0) == pytest.approx(ns + delta_n, rel=1e-12)

    def test_vectorized(self):
        ns = np.array([250.0, 313.0, 400.0])
        ce, delta_n = atmos_exp_decay_const(ns)
        assert ce.shape == ns.shape
        assert delta_n.shape == ns.shape
        assert np.all(np.diff(ce) > 0)
