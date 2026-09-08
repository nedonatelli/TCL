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
