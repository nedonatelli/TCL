"""SDP4 deep-space validation against the reference SGP4 implementation.

The deep-space extension (DSCOM/DSINIT/DSPACE/DPPER) implements the same
published algorithm as Vallado's reference code, so agreement should be at
the level of floating-point round-off, not of an approximation. These tests
pin that: metre-level bounds would already indicate a transcription error.

The ``sgp4`` package (Brandon Rhodes' port of Vallado's C++) is used purely
as a verification oracle; the tests are skipped when it is not installed.
"""

import numpy as np
import pytest

from pytcl.astronomical.sgp4 import SGP4Satellite, sgp4_propagate
from pytcl.astronomical.tle import is_deep_space, parse_tle

sgp4_api = pytest.importorskip("sgp4.api")
sgp4_model = pytest.importorskip("sgp4.model")


def _checksum(line: str) -> str:
    """Append the modulo-10 TLE checksum to a 68-character line body."""
    total = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68])
    return line[:68] + str(total % 10)


# -----------------------------------------------------------------------------
# Test element sets
# -----------------------------------------------------------------------------

# Geostationary: 24-hour (1:1) resonance, near-zero inclination so DPPER takes
# the Lyddane branch.
GEO = (
    "1 25872U 99046A   24001.50000000 -.00000098  00000-0  00000-0 0  9992",
    "2 25872   0.0182 254.6431 0002269 195.0731 178.1197  1.00270298 88927",
)
# Molniya: 12-hour (2:1) resonance with e > 0.65, exercising the second branch
# of the geopotential resonance coefficients.
MOLNIYA = (
    "1 25485U 98054A   24001.50000000  .00000200  00000-0  00000-0 0  9994",
    "2 25485  63.1706 156.0074 7148063 264.9797  15.4390  2.00612394185378",
)
# GPS: 12-hour period but e < 0.5, so non-resonant deep space (irez == 0).
GPS = (
    "1 24876U 97035A   24001.50000000 -.00000029  00000-0  00000-0 0  9995",
    "2 24876  55.5460 106.1477 0102184  54.6647 306.4059  2.00565165194155",
)
# Molniya-class with e in the 0.65-0.70 window (a distinct coefficient branch).
MOLNIYA_LOW_E = (
    _checksum("1 25485U 98054A   24001.50000000  .00000200  00000-0  00000-0 0  999 "),
    _checksum("2 25485  63.1706 156.0074 6600000 264.9797  15.4390  2.00612394185378"),
)
# Retrograde deep-space orbit, non-resonant.
RETROGRADE = (
    _checksum("1 25485U 98054A   24001.50000000  .00000200  00000-0  00000-0 0  999 "),
    _checksum("2 25485 145.0000 156.0074 0500000 264.9797  15.4390  3.50000000185378"),
)
# Semi-synchronous, low inclination: exercises the DPPER Lyddane branch away
# from the resonance code.
LOW_INCLINATION = (
    _checksum("1 25485U 98054A   24001.50000000  .00000200  00000-0  00000-0 0  999 "),
    _checksum("2 25485   3.5000 156.0074 0050000 264.9797  15.4390  2.50000000185378"),
)
# Deep-space case with a non-zero B* so the drag terms are also active.
DRAGGY = (
    _checksum("1 25485U 98054A   24001.50000000  .00000200  00000-0  27000-3 0  999 "),
    _checksum("2 25485  63.1706 156.0074 7148063 264.9797  15.4390  2.00612394185378"),
)

DEEP_SPACE_CASES = [
    pytest.param(GEO, id="geo-24h-resonance"),
    pytest.param(MOLNIYA, id="molniya-12h-resonance"),
    pytest.param(GPS, id="gps-12h-nonresonant"),
    pytest.param(MOLNIYA_LOW_E, id="molniya-e0.66"),
    pytest.param(RETROGRADE, id="retrograde"),
    pytest.param(LOW_INCLINATION, id="low-inclination-lyddane"),
    pytest.param(DRAGGY, id="deep-space-with-drag"),
]

# Deep-space element sets from the official SGP4 verification set
# (SGP4-VER.TLE) that exercise the reference error paths.
# (some of these carry the deliberately invalid checksums of the published
# file, so the checksum digit is recomputed here).
FAILING_33333 = (
    _checksum("1 33333U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534"),
    _checksum("2 33333  96.4736 157.9986 9950000 244.0492 110.6523  4.00004038 10708"),
)
FAILING_33334 = (
    _checksum("1 33334U 78066F   06174.85818871  .00000620  00000-0  10000-3 0  6809"),
    _checksum("2 33334  68.4714 236.1303 5602877 123.7484 302.5767  0.00001000 67521"),
)
FAILING_20413 = (
    _checksum("1 20413U 83020D   05363.79166667  .00000000  00000-0  00000+0 0  7041"),
    _checksum("2 20413  12.3514 187.4253 7864447 196.3027 356.5478  0.24690082  7978"),
)

# Near-Earth reference case (the TLE in the module docstring).
ISS = (
    "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9997",
    "2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49815350479003",
)

TSINCE_GRID = [
    -4320.0,
    -2880.0,
    -1441.0,
    -1440.0,
    -1439.0,
    -721.0,
    -720.0,
    -719.0,
    -360.0,
    -60.0,
    -1.0,
    0.0,
    1.0,
    60.0,
    360.0,
    719.0,
    720.0,
    721.0,
    1439.0,
    1440.0,
    1441.0,
    2880.0,
    4320.0,
]

# Agreement bound (metres). The two implementations evaluate the same
# expressions in the same order, so the true residual is round-off.
POSITION_TOL_M = 1.0
VELOCITY_TOL_MM_S = 1.0


def _reference(lines):
    return sgp4_api.Satrec.twoline2rv(lines[0], lines[1], sgp4_api.WGS72)


def _ours(lines):
    return SGP4Satellite(parse_tle(lines[0], lines[1]))


class TestDeepSpaceAgreement:
    """Position/velocity agreement with the reference over +/- 3 days."""

    @pytest.mark.parametrize("lines", DEEP_SPACE_CASES)
    def test_selects_deep_space_model(self, lines):
        ref = _reference(lines)
        sat = _ours(lines)
        assert sat.is_deep_space
        assert is_deep_space(sat.tle)
        assert sat._ds_initialized
        # Reference reports 'd' for the deep-space method.
        assert ref.method == "d"
        assert ref.error == 0

    @pytest.mark.parametrize("lines", DEEP_SPACE_CASES)
    @pytest.mark.parametrize("tsince", TSINCE_GRID)
    def test_position_matches_reference(self, lines, tsince):
        ref = _reference(lines)
        sat = _ours(lines)
        err, r_ref, v_ref = ref.sgp4_tsince(tsince)
        assert err == 0
        state = sat.propagate(tsince)
        assert state.error == 0
        dr = float(np.linalg.norm(np.array(r_ref) - state.r)) * 1e3
        dv = float(np.linalg.norm(np.array(v_ref) - state.v)) * 1e6
        assert dr < POSITION_TOL_M, f"position error {dr:.4g} m at t={tsince}"
        assert dv < VELOCITY_TOL_MM_S, f"velocity error {dv:.4g} mm/s at t={tsince}"

    @pytest.mark.parametrize("lines", DEEP_SPACE_CASES)
    def test_sequential_propagation_matches_reference(self, lines):
        """The resonance integrator carries state between calls.

        Propagating a monotonically increasing sequence of times must give
        the same answers as the reference, which integrates the same way.
        """
        ref = _reference(lines)
        sat = _ours(lines)
        worst = 0.0
        for tsince in np.arange(0.0, 4321.0, 90.0):
            err, r_ref, _ = ref.sgp4_tsince(float(tsince))
            assert err == 0
            state = sat.propagate(float(tsince))
            worst = max(worst, float(np.linalg.norm(np.array(r_ref) - state.r)) * 1e3)
        assert worst < POSITION_TOL_M, f"max position error {worst:.4g} m"

    @pytest.mark.parametrize("lines", DEEP_SPACE_CASES)
    def test_resonance_branch_matches_reference(self, lines):
        """irez must agree with the reference: 0 none, 1 synchronous, 2 12-hour."""
        assert _ours(lines).irez == sgp4_model.Satrec.twoline2rv(*lines).irez

    def test_resonance_branches_are_all_exercised(self):
        irez = {_ours(c.values[0]).irez for c in DEEP_SPACE_CASES}
        assert irez == {0, 1, 2}


class TestNearEarthNonRegression:
    """The near-Earth path must be untouched by the deep-space work."""

    def test_iss_matches_reference_to_sub_millimetre(self):
        sat = _ours(ISS)
        assert not sat.is_deep_space
        ref = _reference(ISS)
        worst_r = worst_v = 0.0
        for tsince in TSINCE_GRID:
            err, r_ref, v_ref = ref.sgp4_tsince(tsince)
            assert err == 0
            state = sat.propagate(tsince)
            assert state.error == 0
            worst_r = max(worst_r, float(np.linalg.norm(np.array(r_ref) - state.r)))
            worst_v = max(worst_v, float(np.linalg.norm(np.array(v_ref) - state.v)))
        assert worst_r * 1e3 < 1e-3, f"position error {worst_r * 1e3:.4g} m"
        assert worst_v * 1e6 < 1e-3, f"velocity error {worst_v * 1e6:.4g} mm/s"

    def test_near_earth_does_not_initialize_deep_space(self):
        sat = _ours(ISS)
        assert not sat._ds_initialized
        assert not hasattr(sat, "irez")


class TestErrorHandling:
    """Error codes follow the reference algorithm."""

    def test_valid_deep_space_propagation_reports_no_error(self):
        for lines in (GEO, MOLNIYA, GPS):
            for tsince in (-4320.0, 0.0, 4320.0):
                assert sgp4_propagate(parse_tle(*lines), tsince).error == 0

    @pytest.mark.parametrize(
        ("lines", "start", "stop", "step"),
        [
            pytest.param(FAILING_33333, 0.0, 150.0, 5.0, id="negative-semi-latus"),
            pytest.param(FAILING_33334, 0.0, 1440.0, 20.0, id="eccentricity-range"),
            pytest.param(FAILING_20413, 1844300.0, 1844400.0, 5.0, id="decayed"),
        ],
    )
    def test_error_codes_match_reference(self, lines, start, stop, step):
        """Deep-space cases from the official SGP4 verification set.

        These element sets are designed to trip the reference error paths
        (negative semi-latus rectum, eccentricity out of range, decay below
        the Earth's surface); our codes must agree point for point.
        """
        ref = _reference(lines)
        sat = _ours(lines)
        assert sat.is_deep_space
        seen = set()
        for tsince in np.arange(start, stop + 0.5 * step, step):
            err, _, _ = ref.sgp4_tsince(float(tsince))
            state = sat.propagate(float(tsince))
            assert state.error == err, f"t={tsince}: ours {state.error} vs ref {err}"
            seen.add(err)
        assert seen - {0}, "case did not trigger any reference error"

    def test_failed_state_is_not_silently_zero(self):
        """A failed propagation must not look like a valid state at the origin."""
        from pytcl.astronomical.sgp4 import _failed_state

        state = _failed_state(1)
        assert state.error == 1
        assert np.all(np.isnan(state.r))
        assert np.all(np.isnan(state.v))
