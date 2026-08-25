"""Tests for the astronomical-refraction functions in pytcl.atmosphere.refraction.

MATLAB reference values captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_astro_refraction.m,
with the simpAstroRefParam MEX compiled from the in-tree SOFA refco.c.
"""

from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.refraction import (
    add_astro_refraction,
    remove_astro_refraction,
    simple_astro_ref_params,
    sinclair_atmosphere,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# simple_astro_ref_params and algorithms 1-2 are literal-for-literal
# transcriptions: machine precision (measured max 3.9e-15).
RTOL_EXACT = 1e-12
# Algorithm 0 integrates with scipy quad where MATLAB uses integral();
# measured max relative disagreement on delta_z is 3.9e-8.
RTOL_ALG0 = 1e-6
# The Sinclair model feeds pytcl's CODATA 2018 gas constant through the
# barometric exponent (|exponent| up to ~8), amplifying the 3.4e-7
# constant difference to a measured max of 3.1e-6 on n-1, dndr and P.
RTOL_SINCLAIR = 1e-5


def _load(name):
    return np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)


class TestSimpleAstroRefParamsMatlab:
    def test_matches_matlab_mex(self):
        rows = _load("astro_ref_params.csv")
        for rh, P, T, wl, a_ref, b_ref in rows:
            a, b = simple_astro_ref_params(rh, P, T, wl)
            assert a == pytest.approx(a_ref, rel=RTOL_EXACT)
            assert b == pytest.approx(b_ref, rel=RTOL_EXACT)

    def test_zero_pressure_zeroes_water_vapor_term(self):
        # SOFA refco guards the water-vapor pressure against p == 0.
        a_dry, b_dry = simple_astro_ref_params(1.0, 0.0, 288.15)
        assert a_dry == 0.0
        assert b_dry == 0.0

    def test_radio_branch_differs_from_optical(self):
        opt = simple_astro_ref_params(0.5, 101325.0, 288.15, 0.574e-6)
        radio = simple_astro_ref_params(0.5, 101325.0, 288.15, 0.03)
        assert opt.a != pytest.approx(radio.a, rel=1e-3)

    def test_matches_pyerfa_when_available(self):
        erfa = pytest.importorskip("erfa")
        for rh, P, T, wl in [
            (0.0, 101325.0, 288.15, 0.574e-6),
            (0.8, 90000.0, 300.15, 1.0e-6),
        ]:
            a_ref, b_ref = erfa.refco(P / 100.0, T - 273.15, rh, wl * 1e6)
            a, b = simple_astro_ref_params(rh, P, T, wl)
            assert a == pytest.approx(a_ref, rel=1e-12)
            assert b == pytest.approx(b_ref, rel=1e-12)


class TestSinclairAtmosphereMatlab:
    def test_matches_matlab(self):
        rows = _load("astro_sinclair.csv")
        for lat, h0, h, rh, n_ref, dndr_ref, t_ref, p_ref in rows:
            res = sinclair_atmosphere(
                h, [lat, 0.25, h0], rh, 101325.0, 288.15, 0.574e-6, 11000.0
            )
            # Compare n-1, not n: n is ~1.0003, so rtol on n itself would
            # hide five orders of magnitude of error in the physics.
            assert res.n - 1 == pytest.approx(n_ref - 1, rel=RTOL_SINCLAIR)
            assert res.dndr == pytest.approx(dndr_ref, rel=RTOL_SINCLAIR)
            assert res.temperature == pytest.approx(t_ref, rel=RTOL_EXACT)
            assert res.pressure == pytest.approx(p_ref, rel=RTOL_SINCLAIR)

    def test_continuous_across_tropopause(self):
        obs = [0.61, 0.0, 100.0]
        below = sinclair_atmosphere(10999.999, obs, 0.5)
        above = sinclair_atmosphere(11000.001, obs, 0.5)
        assert below.n == pytest.approx(above.n, abs=1e-9)

    def test_vectorized_over_height(self):
        h = np.array([0.0, 5000.0, 20000.0])
        res = sinclair_atmosphere(h, [0.61, 0.0, 0.0], 0.5)
        assert res.n.shape == h.shape
        assert np.all(np.diff(res.n) < 0)  # thinner air, lower n


class TestRemoveAddAstroRefractionMatlab:
    def test_matches_matlab_all_algorithms(self):
        rows = _load("astro_remove_add.csv")
        obs = [0.61, 0.0, 100.0]
        for alg, z0, rh, T, zt_ref, dz_ref, zb_ref, dzb_ref in rows:
            rtol = RTOL_ALG0 if int(alg) == 0 else RTOL_EXACT
            zt, dz = remove_astro_refraction(int(alg), obs, z0, rh, 101325.0, T)
            assert dz == pytest.approx(dz_ref, rel=rtol)
            assert zt == pytest.approx(zt_ref, rel=1e-12, abs=1e-9)
            zb, dzb = add_astro_refraction(int(alg), obs, zt_ref, rh, 101325.0, T)
            assert zb == pytest.approx(zb_ref, rel=1e-12, abs=1e-9)
            assert dzb == pytest.approx(dzb_ref, rel=rtol)

    def test_matches_matlab_alt_observer(self):
        rows = _load("astro_remove_alt_observer.csv")
        obs = [-33 * np.pi / 180, 1.1, 2500.0]
        for z0, zt_ref, dz_ref in rows:
            zt, dz = remove_astro_refraction(0, obs, z0, 0.3, 75000.0, 278.15, 0.5e-6)
            assert dz == pytest.approx(dz_ref, rel=RTOL_ALG0)
            assert zt == pytest.approx(zt_ref, rel=1e-12, abs=1e-9)

    @pytest.mark.parametrize("algorithm", [0, 1, 2])
    def test_add_remove_round_trip(self, algorithm):
        obs = [0.61, 0.0, 100.0]
        z_true = np.array([0.1, 0.5, 1.0])
        z0, _ = add_astro_refraction(algorithm, obs, z_true, 0.5, 101325.0, 288.15)
        z_back, _ = remove_astro_refraction(algorithm, obs, z0, 0.5, 101325.0, 288.15)
        np.testing.assert_allclose(z_back, z_true, atol=1e-12)

    @pytest.mark.parametrize("algorithm", [0, 1, 2])
    def test_refraction_grows_with_zenith_distance(self, algorithm):
        obs = [0.61, 0.0, 100.0]
        z0 = np.array([0.2, 0.6, 1.0])
        _, dz = remove_astro_refraction(algorithm, obs, z0, 0.5, 101325.0, 288.15)
        assert np.all(np.diff(dz) > 0)
        assert np.all(dz > 0)

    def test_zero_zenith_distance_has_zero_refraction(self):
        zt, dz = remove_astro_refraction(0, [0.61, 0.0, 100.0], 0.0)
        assert zt == 0.0
        assert dz == 0.0

    def test_out_of_validity_returns_empty(self):
        obs = [0.61, 0.0, 100.0]
        # Algorithm 0: too far below the horizon.
        zt, dz = remove_astro_refraction(0, obs, 101 * np.pi / 180)
        assert zt.size == 0 and dz.size == 0
        # Algorithm 1: beyond 70 degrees.
        zt, dz = remove_astro_refraction(1, obs, 71 * np.pi / 180)
        assert zt.size == 0 and dz.size == 0
        # add propagates the empty result.
        zt, dz = add_astro_refraction(1, obs, 71 * np.pi / 180)
        assert zt.size == 0 and dz.size == 0

    def test_invalid_inputs_raise(self):
        obs = [0.61, 0.0, 100.0]
        with pytest.raises(ValueError, match="positive"):
            remove_astro_refraction(0, obs, -0.1)
        with pytest.raises(ValueError, match="troposphere"):
            remove_astro_refraction(0, [0.61, 0.0, 12000.0], 0.5)
        with pytest.raises(ValueError, match="algorithm"):
            remove_astro_refraction(3, obs, 0.5)

    def test_vectorized_input(self):
        obs = [0.61, 0.0, 100.0]
        z0 = np.array([0.3, 0.9])
        for alg in (0, 1, 2):
            zt, dz = remove_astro_refraction(alg, obs, z0)
            assert zt.shape == z0.shape
            assert dz.shape == z0.shape
