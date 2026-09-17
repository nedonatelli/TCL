"""ghp7 pressure-inversion honesty checks (compiled backend).

``nrlmsise00_alt_for_pressure``/``nrlmsise00_pressure_altitude`` invert
NRLMSISE-00's density/temperature model for altitude at a target
pressure via a bounded Newton iteration (the vendored ``ghp7``). The
iteration can land on an altitude that is finite but far outside the
model's documented 0-1000 km validity range, or on a non-finite
altitude, and previously did so silently through the compiled
backend (the default in every wheel) -- the reference C only
``printf``s to native stdout on internal non-convergence, which never
reaches Python. See task-2.8-brief.md; the brief's own reproduction
number (3.19e+89 km) does not reproduce on this tree -- the measured
defect is a silent, plausible-looking 3706.76 km at 1e-10 Pa instead.
"""

import warnings

import numpy as np
import pytest

from pytcl.atmosphere.nrlmsise00 import (
    nrlmsise00_alt_for_pressure,
    uses_compiled_backend,
)

_KWARGS = dict(doy=172, sec=29000.0, g_lat_deg=60.0, g_long_deg=-70.0, lst=16.0)


def test_compiled_backend_is_active():
    assert uses_compiled_backend(), "this test must exercise the C path"


def test_out_of_range_solution_warns_instead_of_returning_silently():
    """press_pa=1e-10 converges to alt_km=3706.76, four times past the
    model's documented 1000 km ceiling; measured directly, not from the
    brief's uncorroborated 3.19e+89 figure."""
    with pytest.warns(UserWarning, match="did not converge"):
        alt_km, _ = nrlmsise00_alt_for_pressure(press_pa=1e-10, **_KWARGS)
    assert alt_km > 1000.0


def test_nonfinite_solution_warns_instead_of_returning_silently():
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
