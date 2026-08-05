"""The tracking stack against real aircraft.

Every other test in this suite drives one component with numbers this project
chose. This one runs the chain a user would actually assemble --
``geodetic2enu`` -> ``kf_predict``/``kf_update`` -> ``nis`` -- over 3600
position reports from 120 real aircraft, and scores it against a quantity the
filter is never given.

That quantity is the aircraft's own broadcast ground speed. The filter sees
positions only and estimates velocity from them; the aircraft reports its speed
independently. Agreement between the two is a reference measurement, not a
self-consistency check, which is what separates this from the synthetic
constant-velocity tests elsewhere.

Real traffic also supplies an innovation distribution that cannot be
manufactured: a very low median, because airliners in cruise are almost exactly
constant-velocity, with a heavy tail where they turn. Both halves are asserted
below, because a filter that produced only one of them would be wrong in a way
a synthetic test would never reveal.

The fixture is recorded, so this is deterministic and offline. See
``tests/fixtures/adsb/SOURCES.md``.
"""

import gzip
import json
import math
import pathlib

import numpy as np
import pytest

from pytcl.coordinate_systems.conversions import geodetic2enu
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_models import q_discrete_white_noise
from pytcl.mathematical_functions.statistics import nis

FIXTURE = (
    pathlib.Path(__file__).parent.parent / "fixtures" / "adsb" / "adsb_boston.json.gz"
)

KNOTS_TO_MS = 0.514444
FEET_TO_M = 0.3048

# ADS-B position accuracy is metres; 50 m is conservative for the NIC values
# these reports carry. The process noise covers ordinary course corrections,
# not deliberate manoeuvres -- the tail asserted below is exactly the cost of
# that choice, and is reported rather than tuned away.
POSITION_SIGMA_M = 50.0
PROCESS_VAR = 5.0


def _load():
    if not FIXTURE.exists():
        pytest.skip(f"ADS-B fixture not present: {FIXTURE.name}")
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def capture():
    return _load()


def _track(reports, ref):
    """Constant-velocity filter over one aircraft.

    Returns (final state, NIS per update). The state is [E, vE, N, vN] in
    metres and metres per second, in a local ENU frame about ``ref``.
    """
    lat0, lon0, alt0 = ref
    F = np.eye(4)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.diag([POSITION_SIGMA_M**2, POSITION_SIGMA_M**2])

    state = cov = previous = None
    scores = []

    for report in reports:
        east, north, _ = geodetic2enu(
            math.radians(report["lat"]),
            math.radians(report["lon"]),
            report["alt_ft"] * FEET_TO_M,
            lat0,
            lon0,
            alt0,
        )
        z = np.array([float(east), float(north)])

        if state is None:
            state = np.array([z[0], 0.0, z[1], 0.0])
            cov = np.diag([100.0**2, 300.0**2, 100.0**2, 300.0**2])
            previous = report["t"]
            continue

        dt = report["t"] - previous
        if not 0.5 < dt < 120.0:  # duplicate, or a gap too long to bridge
            previous = report["t"]
            continue
        previous = report["t"]

        F[0, 1] = F[2, 3] = dt
        Q = q_discrete_white_noise(dim=2, T=dt, var=PROCESS_VAR, block_size=2)

        prediction = kf_predict(state, cov, F, Q)
        innovation = z - H @ prediction.x
        S = H @ prediction.P @ H.T + R
        scores.append(float(nis(innovation, S)))

        update = kf_update(prediction.x, prediction.P, z, H, R)
        state, cov = update.x, update.P

    return state, scores


@pytest.fixture(scope="module")
def tracked(capture):
    """Run every aircraft once; the assertions below all read this."""
    ref = (
        math.radians(capture["centre"]["lat_deg"]),
        math.radians(capture["centre"]["lon_deg"]),
        0.0,
    )
    scores, speed_errors = [], []
    for reports in capture["aircraft"].values():
        if len(reports) < 5:
            continue
        state, track_scores = _track(reports, ref)
        if state is None or not track_scores:
            continue
        scores.extend(track_scores)
        broadcast = [r["gs_kt"] * KNOTS_TO_MS for r in reports[-3:] if r["gs_kt"]]
        if broadcast:
            estimated = math.hypot(state[1], state[3])
            speed_errors.append(estimated - float(np.mean(broadcast)))
    return np.array(scores), np.array(speed_errors)


class TestFixtureIsWhatItClaims:
    """Guard the guard: every assertion below is vacuous on an empty capture."""

    def test_capture_has_many_aircraft(self, capture):
        assert len(capture["aircraft"]) >= 100

    def test_capture_has_thousands_of_reports(self, capture):
        total = sum(len(v) for v in capture["aircraft"].values())
        assert total >= 3000

    def test_reports_carry_broadcast_ground_speed(self, capture):
        """The reference quantity. Without it this file proves nothing."""
        with_speed = sum(
            1
            for reports in capture["aircraft"].values()
            for r in reports
            if r.get("gs_kt")
        )
        assert with_speed >= 3000

    def test_aircraft_actually_move(self, capture):
        """A fixture of stationary points would pass a consistency check while
        saying nothing about tracking."""
        moved = 0
        for reports in capture["aircraft"].values():
            first, last = reports[0], reports[-1]
            if (
                math.hypot(last["lat"] - first["lat"], last["lon"] - first["lon"])
                > 0.01
            ):
                moved += 1
        assert moved >= 80, f"only {moved} aircraft moved appreciably"


class TestVelocityMatchesWhatTheAircraftBroadcast:
    """The reference test: position-only estimate vs the aircraft's own speed."""

    def test_most_aircraft_agree_closely(self, tracked):
        _, errors = tracked
        assert len(errors) >= 100
        within = np.mean(np.abs(errors) < 25.0)
        assert within > 0.90, f"only {within:.1%} of aircraft within 25 m/s"

    def test_median_error_is_small(self, tracked):
        """Measured at 6.3 m/s, about 12 knots, across 456 aircraft on a larger
        live sample. The bound is loose enough to survive a different capture
        and tight enough that a broken conversion or filter would breach it."""
        _, errors = tracked
        assert np.median(np.abs(errors)) < 20.0

    def test_no_systematic_bias(self, tracked):
        """A unit error or a wrong Earth radius would show up here as a
        consistent offset rather than as scatter."""
        _, errors = tracked
        assert abs(float(np.mean(errors))) < 15.0


class TestInnovationsLookLikeRealFlight:
    """The shape of the NIS distribution, which synthetic data cannot produce."""

    def test_mean_nis_is_near_the_consistent_value(self, tracked):
        """For a 2-D position measurement a consistent filter averages 2.0."""
        scores, _ = tracked
        assert len(scores) >= 2000
        assert 0.5 < float(np.mean(scores)) < 8.0

    def test_median_is_far_below_the_mean(self, tracked):
        """Cruise is almost exactly constant-velocity, so most innovations are
        tiny. A filter that had merely inflated its covariance would show mean
        and median together."""
        scores, _ = tracked
        assert float(np.median(scores)) < float(np.mean(scores)) / 2.0

    def test_there_is_a_manoeuvre_tail(self, tracked):
        """And a minority of updates must be large, or the aircraft never
        turned and the fixture is not real traffic."""
        scores, _ = tracked
        exceedance = float(np.mean(scores > 5.99))  # chi2(2) 95th percentile
        assert exceedance > 0.01, "no manoeuvres at all -- is this synthetic?"
        assert exceedance < 0.30, f"{exceedance:.1%} outliers: filter is inconsistent"
