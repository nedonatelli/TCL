"""ghp7 pressure-inversion honesty checks (compiled backend).

``nrlmsise00_alt_for_pressure``/``nrlmsise00_pressure_altitude`` invert
NRLMSISE-00's density/temperature model for altitude at a target
pressure via a bounded Newton iteration (the vendored ``ghp7``). The
iteration can land on an altitude that is finite but far outside the
model's validity range, or on a non-finite altitude, and previously did
so silently through the compiled backend (the default in every wheel)
-- the reference C only ``printf``s to native stdout on internal
non-convergence, which never reaches Python. See task-2.8-brief.md; the
brief's own reproduction number (3.19e+89 km) does not reproduce on
this tree -- the measured defect is a silent, plausible-looking
3706.76 km at 1e-10 Pa instead.

The validity range's lower bound is US76's own -5000 m floor (task
2.7, ``pytcl.atmosphere.models.US76_MIN_ALTITUDE_M``), not 0 km: an
ordinary high-pressure system (e.g. 105 kPa) converges to a legitimate
negative altitude, and a 0 km floor flagged every one of those as
non-convergence (review round 2, task 2.8 fix).
"""

import importlib
import warnings

import numpy as np
import pytest

from pytcl.atmosphere.nrlmsise00 import (
    NRLMSISEOutput,
    nrlmsise00_alt_for_pressure,
    nrlmsise00_pressure_altitude,
    uses_compiled_backend,
)

# pytcl.atmosphere's __init__ does `from .nrlmsise00 import nrlmsise00`,
# which shadows the `nrlmsise00` submodule attribute on the `atmosphere`
# package with that function -- `import pytcl.atmosphere.nrlmsise00 as x`
# would silently bind x to the function via that shadowed attribute, so
# the module is fetched through sys.modules instead.
nrlmsise00_mod = importlib.import_module("pytcl.atmosphere.nrlmsise00")

_KWARGS = dict(doy=172, sec=29000.0, g_lat_deg=60.0, g_long_deg=-70.0, lst=16.0)


def test_compiled_backend_is_active():
    assert uses_compiled_backend(), "this test must exercise the C path"


def test_out_of_range_solution_warns_with_a_converged_message():
    """press_pa=1e-10 converges (ghp7's own residual test is satisfied)
    to alt_km=3706.76, well past the model's 1000 km ceiling; measured
    directly, not from the brief's uncorroborated 3.19e+89 figure. This
    is a *converged* out-of-domain result, not a Newton failure, so it
    must not share wording with test_nonfinite_solution_warns_with_a_
    non_convergence_message below."""
    with pytest.warns(UserWarning, match="converged.*outside"):
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=1e-10, **_KWARGS)
    assert alt_km > nrlmsise00_mod._GHP7_MAX_VALID_ALT_KM


def test_nonfinite_solution_warns_with_a_non_convergence_message():
    with pytest.warns(UserWarning, match="did not converge"):
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=1e-30, **_KWARGS)
    assert not np.isfinite(alt_km)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_pressure_raises(bad):
    with pytest.raises(ValueError, match="pressure"):
        nrlmsise00_alt_for_pressure(press_pa=bad, **_KWARGS)


def test_in_range_solution_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=1000.0, **_KWARGS)
    assert 15.0 < alt_km < 35.0


def test_ordinary_high_pressure_negative_altitude_does_not_warn():
    """105 kPa is an unremarkable high-pressure system, not a defect
    case -- it converges to a small negative altitude (measured
    -0.24 km), well within the US76 floor. A 0 km lower bound flagged
    this as false-positive non-convergence; -5000 m does not."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=105_000.0, **_KWARGS)
    assert -1.0 < alt_km < 0.0


def test_pressure_below_the_us76_floor_still_warns():
    """200 kPa converges to alt_km=-5.32, genuinely below US76's
    -5000 m floor -- this must still be flagged, distinguishing a real
    below-floor case from the merely-negative 105 kPa case above."""
    with pytest.warns(UserWarning, match="converged.*outside"):
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=200_000.0, **_KWARGS)
    assert alt_km < nrlmsise00_mod._GHP7_MIN_VALID_ALT_KM


def test_boundary_altitude_exactly_at_ceiling_does_not_warn(monkeypatch):
    """Pins that the range check is inclusive at the 1000 km ceiling.
    Forcing ghp7 itself to land on exactly 1000.0 km from a real
    pressure input isn't practical, so this substitutes a canned
    result for the one call inside nrlmsise00_alt_for_pressure that
    talks to the model, leaving the range check itself untouched."""
    canned_output = NRLMSISEOutput(d=np.ones(9), t=np.array([1000.0, 1000.0]))
    monkeypatch.setattr(
        nrlmsise00_mod,
        "_run",
        lambda *args, **kwargs: (
            nrlmsise00_mod._GHP7_MAX_VALID_ALT_KM,
            canned_output,
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=1000.0, **_KWARGS)
    assert alt_km == nrlmsise00_mod._GHP7_MAX_VALID_ALT_KM


def test_pressure_altitude_wrapper_raises_on_non_positive_pressure():
    """nrlmsise00_pressure_altitude delegates to
    nrlmsise00_alt_for_pressure and must inherit its input validation."""
    with pytest.raises(ValueError, match="pressure"):
        nrlmsise00_pressure_altitude(
            172, 29000.0, -1.0, [np.radians(60.0), np.radians(-70.0)]
        )


def test_pressure_altitude_wrapper_warns_on_nonfinite_solution():
    """Same defect, exercised through the meters/radians entry point
    rather than nrlmsise00_alt_for_pressure directly."""
    with pytest.warns(UserWarning, match="did not converge"):
        alt_m, _, _, _ = nrlmsise00_pressure_altitude(
            172, 29000.0, 1e-30, [np.radians(60.0), np.radians(-70.0)]
        )
    assert not np.isfinite(alt_m)
