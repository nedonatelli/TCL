"""Round-trip PROPERTY tests for the classical orbital element <-> state
vector conversions at the equatorial-orbit degeneracy (i = 0, i = pi).

`state_to_orbital_elements` reports RAAN = 0 for equatorial orbits (the
node vector is undefined) and instead folds the periapsis longitude into
`omega`. That longitude must be measured with opposite sense for a
retrograde orbit (cos(i) < 0) versus a prograde one, matching MATLAB's
`state2OrbEls.m` (`state2OrbElsUniv`), which multiplies its equatorial
angle by `sign(r2h(3))` -- the sign of the angular-momentum z-component
-- before calling `atan2`. Using the prograde sign unconditionally is
the defect under test: the element -> state -> element -> state round
trip must return to the original state.
"""

import numpy as np
import pytest

from pytcl.astronomical.orbital_mechanics import (
    GM_EARTH,
    OrbitalElements,
    orbital_elements_to_state,
    state_to_orbital_elements,
)


def _roundtrip_errors(elements: OrbitalElements) -> tuple[float, float]:
    state = orbital_elements_to_state(elements, GM_EARTH)
    back = state_to_orbital_elements(state, GM_EARTH)
    state2 = orbital_elements_to_state(back, GM_EARTH)
    r_err = float(np.linalg.norm(state2.r - state.r))
    v_err = float(np.linalg.norm(state2.v - state.v))
    return r_err, v_err


@pytest.mark.parametrize(
    "inc",
    [0.0, 1e-12, np.pi - 1e-6, np.pi - 1e-12, np.pi],
    ids=[
        "prograde-equatorial",
        "prograde-equatorial-neighbor",
        "retrograde-equatorial-neighbor-1e-6",
        "retrograde-equatorial-neighbor-1e-12",
        "retrograde-equatorial",
    ],
)
def test_element_state_round_trip_at_equatorial_limits(inc):
    elements = OrbitalElements(a=8000.0, e=0.01, i=inc, raan=0.7, omega=1.0, nu=0.3)
    r_err, v_err = _roundtrip_errors(elements)
    assert r_err < 1e-6, f"i={inc}: position round-trip error {r_err} km"
    assert v_err < 1e-9, f"i={inc}: velocity round-trip error {v_err} km/s"


def test_element_state_round_trip_retrograde_equatorial_raan_zero():
    """Same degeneracy with raan=0.0 -- the 10995 km case from the audit."""
    elements = OrbitalElements(a=8000.0, e=0.01, i=np.pi, raan=0.0, omega=1.0, nu=0.3)
    r_err, v_err = _roundtrip_errors(elements)
    assert r_err < 1e-6, f"position round-trip error {r_err} km"
    assert v_err < 1e-9, f"velocity round-trip error {v_err} km/s"
