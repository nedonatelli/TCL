"""NRLMSISE-00 against the reference C implementation.

The oracle is the NRL-modified public-domain C reference vendored at
csrc/nrlmsise00 -- the same sources setup.py compiles into the runtime
extension. Fixtures (993 records at 17 significant digits, generated
by scripts/nrlmsise_capture/) cover every internal model boundary,
storm mode, gtd7d and seeded-random fills.

When the compiled backend is active the fixture comparison is bitwise
(same C code, same compiler family); the pure-Python transcription is
additionally cross-checked against the fixtures at its measured
1.1e-14 worst-case so a broken extension build cannot hide behind the
fallback (CI also sets PYTCL_REQUIRE_NRLMSISE00_C=1).
"""

import importlib
import warnings
from pathlib import Path

import numpy as np
import pytest

from pytcl.atmosphere.nrlmsise00 import (
    _run,
    nrlmsise00,
    nrlmsise00_alt_for_pressure,
    uses_compiled_backend,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "nrlmsise00"

# The pure-Python transcription's measured worst case is 1.1e-14
# relative; the compiled backend reproduces the oracle to a few ULP
# across compilers. One relative tolerance covers both backends.
RTOL = 1e-12


def _records():
    inputs = [
        ln.split()
        for ln in (FIXTURE_DIR / "grid_in.txt").read_text().splitlines()
        if ln.strip()
    ]
    outputs = [
        list(map(float, ln.split()))
        for ln in (FIXTURE_DIR / "grid_out.txt").read_text().splitlines()
        if ln.strip()
    ]
    assert len(inputs) == len(outputs)
    return list(zip(inputs, outputs))


def _run_record(rec, py_fallback=False):
    mode = int(rec[0])
    doy = int(rec[1])
    sec, alt, lat, lon, lst, f107a, f107, ap = map(float, rec[2:10])
    ap_a = list(map(float, rec[10:17])) if mode == 1 else None
    kind = "gtd7d" if mode == 2 else "gtd7"
    if py_fallback:
        # The package re-exports a function named like the submodule,
        # shadowing it on every attribute-based import form; only the
        # module registry hands back the module itself.
        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

        old = mod._c_ext
        mod._c_ext = None
        try:
            out = _run(doy, sec, alt, lat, lon, lst, f107a, f107, ap, ap_a, kind)
        finally:
            mod._c_ext = old
    else:
        out = _run(doy, sec, alt, lat, lon, lst, f107a, f107, ap, ap_a, kind)
    return list(out.d) + list(out.t)


class TestAgainstReferenceC:
    def test_all_993_fixture_records(self):
        worst = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # dnet edge warnings on extreme fills
            for rec, exp in _records():
                got = _run_record(rec)
                for g, e in zip(got, exp):
                    denom = abs(e) if e != 0 else 1.0
                    worst = max(worst, abs(g - e) / denom)
        assert worst < RTOL, f"worst relative error {worst:.3e}"

    def test_python_transcription_matches_fixtures(self):
        # The fallback is validated in its own right over the full
        # grid (the pure-Python model runs the 993 records in ~2 s).
        worst = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for rec, exp in _records():
                got = _run_record(rec, py_fallback=True)
                for g, e in zip(got, exp):
                    denom = abs(e) if e != 0 else 1.0
                    worst = max(worst, abs(g - e) / denom)
        assert worst < RTOL, f"worst relative error {worst:.3e}"

    def test_canonical_published_case(self):
        # The reference implementation's own test point 1 (doy 172,
        # 400 km, lst 16): published cgs values from the distributed
        # self-test, converted to the SI switch settings.
        out = nrlmsise00(172, 29000.0, 400.0, 60.0, -70.0, 16.0)
        np.testing.assert_allclose(out.d[0], 6.665177e5 * 1e6, rtol=1e-6)
        np.testing.assert_allclose(out.d[5], 4.074714e-15 * 1e3, rtol=1e-6)
        np.testing.assert_allclose(out.t[0], 1250.540, rtol=1e-6)
        np.testing.assert_allclose(out.t[1], 1241.416, rtol=1e-6)


class TestBackendContract:
    def test_backend_is_introspectable(self):
        assert isinstance(uses_compiled_backend(), bool)

    def test_backends_agree_when_both_available(self):
        if not uses_compiled_backend():
            pytest.skip("compiled backend unavailable; parity covered by fixtures")
        rec = ["0", "310", "50000", "250", "55", "10", "16", "140", "180", "48.8"] + [
            "0"
        ] * 7
        c_out = _run_record(rec)
        py_out = _run_record(rec, py_fallback=True)
        np.testing.assert_allclose(c_out, py_out, rtol=1e-12)


class TestPythonFallbackPaths:
    """The fallback must carry the full API, not just gtd7."""

    def _without_c(self):
        import contextlib

        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

        @contextlib.contextmanager
        def ctx():
            old = mod._c_ext
            mod._c_ext = None
            try:
                yield
            finally:
                mod._c_ext = old

        return ctx()

    def test_fallback_alt_for_pressure_matches_compiled(self):
        c_alt, c_out = nrlmsise00_alt_for_pressure(
            172, 29000.0, 1000.0, 60.0, -70.0, 16.0
        )
        with self._without_c():
            py_alt, py_out = nrlmsise00_alt_for_pressure(
                172, 29000.0, 1000.0, 60.0, -70.0, 16.0
            )
        np.testing.assert_allclose(py_alt, c_alt, rtol=1e-10)
        np.testing.assert_allclose(py_out.d, c_out.d, rtol=1e-10)

    def test_fallback_storm_mode_matches_compiled(self):
        aph = [148.8, 160.0, 139.0, 127.0, 118.0, 122.5, 115.4]
        c_out = nrlmsise00(
            310,
            50000.0,
            400.0,
            55.0,
            10.0,
            16.0,
            140.0,
            180.0,
            48.8,
            ap_array=aph,
        )
        with self._without_c():
            py_out = nrlmsise00(
                310,
                50000.0,
                400.0,
                55.0,
                10.0,
                16.0,
                140.0,
                180.0,
                48.8,
                ap_array=aph,
            )
        np.testing.assert_allclose(py_out.d, c_out.d, rtol=1e-12)
        np.testing.assert_allclose(py_out.t, c_out.t, rtol=1e-12)

    def test_unknown_kind_rejected_on_both_backends(self):
        from pytcl.atmosphere.nrlmsise00 import _run

        with pytest.raises(ValueError):
            _run(172, 0.0, 100.0, 0.0, 0.0, 16.0, 150.0, 150.0, 4.0, None, "nope")
        with self._without_c():
            with pytest.raises(ValueError):
                _run(172, 0.0, 100.0, 0.0, 0.0, 16.0, 150.0, 150.0, 4.0, None, "nope")


class TestAltForPressure:
    def test_round_trips_model_pressure(self):
        # The altitude ghp7 finds must reproduce the requested
        # pressure through the model's own n*k*T within its 0.043%
        # convergence tolerance.
        boltzmann_hpa = 1.3806e-19  # the reference's cgs-ish constant
        for press_pa in (101325.0, 1000.0, 1.0, 1e-4):
            alt, out = nrlmsise00_alt_for_pressure(
                172, 29000.0, press_pa, 60.0, -70.0, 16.0
            )
            xn = sum(out.d[i] for i in (0, 1, 2, 3, 4, 6, 7))
            p_model = boltzmann_hpa * xn * out.t[1] * 1e-6 * 100.0  # hPa -> Pa
            np.testing.assert_allclose(p_model, press_pa, rtol=1.5e-3)

    def test_sea_level_pressure_is_near_ground(self):
        alt, _ = nrlmsise00_alt_for_pressure(172, 29000.0, 101325.0, 60.0, -70.0, 16.0)
        assert -1.0 < alt < 1.0


class TestStormMode:
    def test_ap_array_changes_thermosphere(self):
        quiet = nrlmsise00(310, 50000.0, 400.0, 55.0, 10.0, 16.0, 140.0, 180.0, 4.0)
        storm = nrlmsise00(
            310,
            50000.0,
            400.0,
            55.0,
            10.0,
            16.0,
            140.0,
            180.0,
            48.8,
            ap_array=[148.8, 160.0, 139.0, 127.0, 118.0, 122.5, 115.4],
        )
        assert storm.t[0] > quiet.t[0]  # storms heat the exosphere

    def test_ap_array_must_have_seven_elements(self):
        with pytest.raises((ValueError, TypeError)):
            nrlmsise00(310, 50000.0, 400.0, 55.0, 10.0, 16.0, ap_array=[10.0, 20.0])


class TestPressureLadder:
    """ghp7's initial-guess ladder, exercised across its pressure bands.

    Each pressure lands in a different branch of the reference's
    piecewise first-guess polynomial (pl > 2.5 down to pl < -5), and
    doy 310 takes the second-half-of-year seasonal arm. Run on the
    Python fallback and cross-checked against the compiled oracle.
    """

    CASES = [  # (pressure_pa, doy)
        (101325.0, 172),  # pl ~ 3.0
        (50.0, 172),  # pl ~ -0.3
        (5.0, 172),  # pl ~ -1.3
        (0.05, 172),  # pl ~ -3.3
        (5e-3, 172),  # pl ~ -4.3
        (5e-4, 172),  # pl < -5: quadratic fallback guess
        (1000.0, 310),  # doy >= 182 seasonal arm
    ]

    @pytest.mark.parametrize("press_pa,doy", CASES)
    def test_fallback_matches_compiled_across_bands(self, press_pa, doy):
        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

        old = mod._c_ext
        mod._c_ext = None
        try:
            py_alt, py_out = nrlmsise00_alt_for_pressure(
                doy, 29000.0, press_pa, 60.0, -70.0, 16.0
            )
        finally:
            mod._c_ext = old
        assert np.isfinite(py_alt)
        if uses_compiled_backend():
            c_alt, c_out = nrlmsise00_alt_for_pressure(
                doy, 29000.0, press_pa, 60.0, -70.0, 16.0
            )
            np.testing.assert_allclose(py_alt, c_alt, rtol=1e-10)
            np.testing.assert_allclose(py_out.d, c_out.d, rtol=1e-10)

    def test_non_convergence_warns_after_twelve_iterations(self):
        # A model whose pressure never approaches the target forces the
        # full iteration budget: the post-5-iteration undamped step and
        # the ltest warning are otherwise unreachable (Newton converges
        # in <=4 steps everywhere in the physical envelope).
        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

        class _Stuck(mod._Model):
            def _gtd7(self, inp, flags):
                return mod.NRLMSISEOutput(np.full(9, 1.0e12), np.array([1000.0, 800.0]))

        for sw0 in (1.0, 0.0):  # metric and cgs unit arms
            model = _Stuck()
            model._glatf(60.0)
            inp = mod._Input(172, 29000.0, 0.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0)
            flags = mod._Flags([sw0] + [1.0] * 23)
            with pytest.warns(UserWarning, match="not converging"):
                z, out = model._ghp7(inp, flags, 1.0)
            assert np.isfinite(z)


class TestTranscriptionInternals:
    """Edge branches of the transcribed helpers.

    The fixture grid drives the physical envelope; these hit the
    reference's guard rails (exp clamps, degenerate-density recovery,
    natural-spline endpoints) that only fire on inputs the full model
    never produces. Expected values are the C expressions evaluated by
    hand: the guards are one-line early returns.
    """

    def _model(self):
        mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")
        m = mod._Model()
        m._glatf(45.0)
        return mod, m

    def test_ccor_saturates_at_large_negative_argument(self):
        from pytcl.atmosphere.nrlmsise00 import _ccor

        np.testing.assert_allclose(_ccor(0.0, 1.5, 1.0, 200.0), np.exp(1.5))

    def test_ccor2_saturates_both_directions(self):
        from pytcl.atmosphere.nrlmsise00 import _ccor2

        np.testing.assert_allclose(_ccor2(500.0, 1.5, 1.0, 200.0, 1.0), 1.0)
        np.testing.assert_allclose(_ccor2(0.0, 1.5, 1.0, 200.0, 1.0), np.exp(1.5))

    def test_dnet_degenerate_density_recovery(self):
        from pytcl.atmosphere.nrlmsise00 import _dnet

        with pytest.warns(UserWarning, match="dnet log error"):
            assert _dnet(0.0, 0.0, 28.0, 28.9, 16.0) == 1.0
        with pytest.warns(UserWarning, match="dnet log error"):
            assert _dnet(5.0, 0.0, 28.0, 28.9, 16.0) == 5.0
        with pytest.warns(UserWarning, match="dnet log error"):
            assert _dnet(0.0, 7.0, 28.0, 28.9, 16.0) == 7.0

    def test_dnet_returns_dominant_density_at_large_ratio(self):
        from pytcl.atmosphere.nrlmsise00 import _dnet

        assert _dnet(1.0, 1.0e30, 28.0, 28.9, 16.0) == 1.0e30

    def test_dnet_negative_density_propagates_nan(self):
        from pytcl.atmosphere.nrlmsise00 import _dnet

        # Negative (not zero) densities pass the zero-recovery ladder
        # and reach log(dm/dde) < 0: nan, exactly as the C's log(-x).
        with pytest.warns(UserWarning, match="dnet log error"):
            with np.errstate(invalid="ignore"):
                assert np.isnan(_dnet(2.0, -5.0, 28.0, 28.9, 16.0))

    def test_natural_spline_endpoints(self):
        from pytcl.atmosphere.nrlmsise00 import _spline

        # yp1/ypn above the 0.99e30 sentinel select natural boundary
        # conditions; a straight line then has zero second derivative.
        assert _spline([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], 1e31, 1e31) == [0, 0, 0]

    def test_splint_warns_on_degenerate_grid(self):
        from pytcl.atmosphere.nrlmsise00 import _splint

        # The C prints its diagnostic then divides by zero (inf); the
        # transcription warns identically, then Python's float division
        # raises where C would produce inf.
        with pytest.warns(UserWarning, match="bad XA"):
            with pytest.raises(ZeroDivisionError):
                _splint([0.0, 0.0], [1.0, 2.0], [0.0, 0.0], 0.0)

    _ZN3 = [32.5, 20.0, 15.0, 10.0, 0.0]
    _TN3 = [260.0, 240.0, 230.0, 220.0, 290.0]
    _TGN3 = [-2.0, -1.9]
    _ZN2 = [72.5, 55.0, 45.0, 32.5]
    _TN2 = [210.0, 250.0, 260.0, 260.0]
    _TGN2 = [-2.0, -2.0]
    _ZN1 = [120.0, 110.0, 100.0, 90.0, 72.5]

    def test_densm_above_grid_passes_through(self):
        _, m = self._model()
        args = (self._ZN3, self._TN3, self._TGN3, self._ZN2, self._TN2, self._TGN2)
        # Above zn2[0] there is nothing to do: temperature calls echo
        # tz, density calls echo d0.
        assert m._densm(80.0, 1.0, 0.0, 5.0, *args) == (5.0, 5.0)
        assert m._densm(80.0, 3.5, 28.0, 5.0, *args) == (3.5, 5.0)

    def test_densm_exponential_clamp_keeps_density_finite(self):
        _, m = self._model()
        args = (self._ZN3, self._TN3, self._TGN3, self._ZN2, self._TN2, self._TGN2)
        # A huge |xm| would overflow exp(-expl) without the expl=50
        # clamp; one altitude per grid segment.
        for alt in (50.0, 5.0):
            d, tz = m._densm(alt, 1.0, -1.0e6, 0.0, *args)
            assert np.isfinite(d) and d > 0
            assert np.isfinite(tz)

    def test_densu_guards_on_unphysical_temperature(self):
        _, m = self._model()
        # tinf < 0 drives both the expl > 50 clamp and the tt <= 0
        # guard on the Bates-profile branch.
        d, tz = m._densu(
            200.0,
            1.0,
            -100.0,
            -200.0,
            16.0,
            0.0,
            0.0,
            120.0,
            0.02,
            self._ZN1,
            [300.0] * 5,
            [0.0, 0.0],
        )
        assert np.isfinite(d)
        # Below ZA: huge |xm| hits the spline-integral clamp, negative
        # temperatures hit the tz <= 0 guard.
        d2, _ = m._densu(
            100.0,
            1.0,
            1000.0,
            900.0,
            -1.0e6,
            0.0,
            0.0,
            120.0,
            0.02,
            self._ZN1,
            [300.0, 280.0, 260.0, 250.0, 240.0],
            [0.0, -2.0],
        )
        assert np.isfinite(d2) and d2 > 0
        d3, tz3 = m._densu(
            100.0,
            1.0,
            -1000.0,
            -900.0,
            16.0,
            0.0,
            0.0,
            120.0,
            0.02,
            self._ZN1,
            [-300.0, -280.0, -260.0, -250.0, -240.0],
            [0.0, -2.0],
        )
        assert np.isfinite(d3)
        assert tz3 < 0

    def test_cgs_switch_scales_output_exactly(self):
        # switches[0] toggles cgs/SI output units in the reference;
        # nothing else may change, so the ratio is exactly 1e6 / 1e3.
        mod, _ = self._model()
        args = (172, 29000.0, 400.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0)
        scale = np.array([1e6] * 5 + [1e3] + [1e6] * 3)
        # Both the thermosphere-only and the lower-atmosphere branches
        # carry the units switch, as does gtd7d's effective density.
        for alt in (400.0, 30.0):
            args_alt = args[:2] + (alt,) + args[3:]
            si = mod._Model()._gtd7(mod._Input(*args_alt), mod._Flags([1.0] * 24))
            cgs = mod._Model()._gtd7(
                mod._Input(*args_alt), mod._Flags([0.0] + [1.0] * 23)
            )
            np.testing.assert_allclose(si.d, cgs.d * scale, rtol=1e-12)
            np.testing.assert_allclose(si.t, cgs.t, rtol=1e-13)
        si_d = mod._Model()._gtd7d(mod._Input(*args), mod._Flags([1.0] * 24))
        cgs_d = mod._Model()._gtd7d(mod._Input(*args), mod._Flags([0.0] + [1.0] * 23))
        np.testing.assert_allclose(si_d.d, cgs_d.d * scale, rtol=1e-12)

    def test_all_switches_off_runs_the_mean_model(self):
        # Every variation disabled leaves the global-mean atmosphere:
        # the off arms of the G(L) machinery must execute and yield a
        # finite, physically sane state.
        mod, _ = self._model()
        inp = mod._Input(172, 29000.0, 400.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0)
        out = mod._Model()._gtd7(inp, mod._Flags([0.0] * 24))
        assert np.all(np.isfinite(out.d))
        assert np.all(np.isfinite(out.t))
        assert 500.0 < out.t[0] < 2000.0

    def test_selective_switches_off_run_the_g_machinery_off_arms(self):
        # Disabling only the G(L) terms (diurnal/semidiurnal/terdiurnal,
        # magnetic activity, longitudinal/UT) keeps the mixing terms
        # live, so globe7/glob7s execute with those arms skipped and
        # the model still yields a finite reduced state.
        mod, _ = self._model()
        # Two configurations: everything off, and only the inner
        # longitudinal/UT/mixed terms off (sw[10] stays on so their
        # enclosing block executes with the inner arms skipped).
        for off in ((7, 8, 9, 10, 11, 12, 13, 14), (11, 12, 13)):
            switches = [1.0] * 24
            for k in off:
                switches[k] = 0.0
            for alt in (400.0, 100.0):
                inp = mod._Input(
                    172, 29000.0, alt, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0
                )
                out = mod._Model()._gtd7(inp, mod._Flags(switches))
                assert np.all(np.isfinite(out.d))
                assert np.all(np.isfinite(out.t))


class TestCompiledEntryPoints:
    """The raw extension entry points, called directly."""

    def test_gtd7_reproduces_the_oracle(self):
        if not uses_compiled_backend():
            pytest.skip("compiled backend unavailable")
        from pytcl.atmosphere import _nrlmsise00_c

        d, t = _nrlmsise00_c.gtd7(
            172, 29000.0, 400.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0, None, False
        )
        np.testing.assert_allclose(t[0], 1250.5399435607994, rtol=1e-12)
        np.testing.assert_allclose(d[0], 6.66517690495151978e11, rtol=1e-12)

    def test_ghp7_returns_altitude_and_output(self):
        if not uses_compiled_backend():
            pytest.skip("compiled backend unavailable")
        from pytcl.atmosphere import _nrlmsise00_c

        alt, (d, t) = _nrlmsise00_c.ghp7(
            172, 29000.0, 10.0, 60.0, -70.0, 16.0, 150.0, 150.0, 4.0, None
        )
        assert 15.0 < alt < 45.0
        assert len(d) == 9 and len(t) == 2
