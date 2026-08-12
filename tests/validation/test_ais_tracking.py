"""The tracking stack against real ship traffic.

Same shape as ``test_adsb_tracking.py``, over real, captured AIS position
reports for the constant-velocity chain a user would actually assemble --
``geodetic2enu`` -> ``kf_predict``/``kf_update`` -> ``nis`` -- scored against a
quantity the filter is never given.

That quantity is the ship's own broadcast speed over ground (SOG). The filter
sees positions only and estimates velocity from them; the ship reports its
speed independently (derived from its own GNSS receiver, not from the
positions the filter is fed). Agreement between the two is a reference
measurement, not a self-consistency check.

Ships also supply an innovation distribution synthetic data cannot
manufacture: a low median in open water or transit, with a heavy tail where a
ship manoeuvres -- fjords, harbour approaches, course changes to avoid other
traffic. Both halves are asserted below.

The fixture is a real capture from the Norwegian Coastal Administration's
open AIS relay, recorded once, so this test is deterministic and offline. See
``tests/fixtures/ais/SOURCES.md`` for capture provenance and the calibration
this test's envelope is derived from.
"""

import gzip
import math
import pathlib

import numpy as np
import pytest

pytest.importorskip("pyais")

from pytcl.coordinate_systems.conversions import geodetic2enu
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_models import q_discrete_white_noise
from pytcl.mathematical_functions.statistics import nis
from pytcl.transponders.ais import ais_position_reports, decode_ais

FIXTURE = (
    pathlib.Path(__file__).parent.parent / "fixtures" / "ais" / "ais_norway.nmea.gz"
)

# AIS positions are GNSS-derived (like ADS-B's), so the same order of
# accuracy applies: 15 m is conservative for ordinary differential-GNSS-free
# fixes.
#
# Process noise: this fixture's reports land roughly every 120 s (see
# SOURCES.md), ~24x the ADS-B fixture's polling interval, and the discrete
# white-noise-acceleration model's position term grows with dt**4 -- so the
# ADS-B test's PROCESS_VAR=5.0 (tuned for few-second gaps) would inject
# tens of millions of m^2 of position variance per 120 s step, swamping the
# 15 m measurement noise entirely (verified empirically: it drove mean NIS
# to ~0.002, three orders of magnitude below the consistent value).
# PROCESS_VAR=1e-5 was chosen by sweeping var and reading off the value
# whose mean NIS lands on 2.0, the value a perfectly-consistent 2-D filter
# averages -- see SOURCES.md's Calibration section for the sweep. Physically
# it means position grows by an extra ~23 m std (sqrt(dt**4/4 * var) at
# dt=120s) beyond the CV prediction per report interval, on top of the 15 m
# measurement noise -- a small, slow-maneuvering budget, consistent with
# ships (rudder authority, displacement-hull inertia) turning far more
# slowly than aircraft.
POSITION_SIGMA_M = 15.0
PROCESS_VAR = 1e-5

# A ship must contribute this many position reports before it is tracked --
# too few and a single-leg course gives no meaningful velocity estimate.
MIN_REPORTS = 15

# Calibrated envelope for the median |v_est - SOG| assertion below: 1.5x the
# measured median (0.013418 m/s), ceiled to 1 significant figure. See
# tests/fixtures/ais/SOURCES.md's Calibration section for the verbatim
# measurement and run date this is derived from.
_MEDIAN_ENVELOPE_MPS = 0.03


def _load():
    """Decode the fixture into per-ship (t, lat, lon, sog) rows.

    Position reports (types 1, 2, 3, 18, 19) are always single-fragment
    sentences per ITU-R M.1371 -- unlike type 5 (static/voyage) or type 24
    (Class B static), which span two -- so each line is decoded on its own
    and paired with that line's own receiver timestamp. This sidesteps
    needing to match `decode_ais`'s reassembled multi-line messages back to
    individual lines, which position reports never require.
    """
    if not FIXTURE.exists():
        pytest.skip(f"AIS fixture not present: {FIXTURE.name}")
    with gzip.open(FIXTURE, "rt", encoding="ascii") as handle:
        raw_lines = [line for line in handle.read().splitlines() if line.strip()]

    by_ship: dict[int, list[tuple[float, float, float, float]]] = {}
    for raw in raw_lines:
        t_str, sentence = raw.split("\t", 1)
        t = float(t_str)
        messages = decode_ais(sentence)
        if not messages:
            continue
        reports = ais_position_reports(messages, times=[t] * len(messages))
        for i in range(len(reports.mmsi)):
            if np.isnan(reports.lat[i]) or np.isnan(reports.lon[i]):
                continue
            by_ship.setdefault(int(reports.mmsi[i]), []).append(
                (
                    t,
                    float(reports.lat[i]),
                    float(reports.lon[i]),
                    float(reports.sog[i]),
                )
            )
    for rows in by_ship.values():
        rows.sort(key=lambda r: r[0])
    return by_ship


@pytest.fixture(scope="module")
def capture():
    return _load()


@pytest.fixture(scope="module")
def centroid(capture):
    lats = [r[1] for rows in capture.values() for r in rows]
    lons = [r[2] for rows in capture.values() for r in rows]
    return float(np.mean(lats)), float(np.mean(lons)), 0.0


def _track(rows, ref):
    """Constant-velocity filter over one ship's position reports.

    Returns (final state, NIS per update, broadcast SOG per update-adjacent
    report). State is [E, vE, N, vN] in metres and metres per second, in a
    local ENU frame about ``ref``.
    """
    lat0, lon0, alt0 = ref
    F = np.eye(4)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.diag([POSITION_SIGMA_M**2, POSITION_SIGMA_M**2])

    state = cov = previous_t = None
    scores = []
    sogs = []

    for t, lat, lon, sog in rows:
        east, north, _ = geodetic2enu(lat, lon, 0.0, lat0, lon0, alt0)
        z = np.array([float(east), float(north)])

        if state is None:
            state = np.array([z[0], 0.0, z[1], 0.0])
            cov = np.diag([50.0**2, 20.0**2, 50.0**2, 20.0**2])
            previous_t = t
            continue

        dt = t - previous_t
        if not 1.0 < dt < 900.0:
            # Duplicate report, or a gap too long to bridge (this fixture
            # concatenates several separately-connected capture windows, so
            # a ship seen in two windows has a multi-minute hole between
            # them). Re-initialize rather than predict across it: carrying
            # the stale pre-gap state forward would have the filter explain
            # the ship's real displacement over the gap as if it happened
            # within the next (short) dt, producing an unbounded velocity
            # estimate instead of just losing one update.
            state = np.array([z[0], 0.0, z[1], 0.0])
            cov = np.diag([50.0**2, 20.0**2, 50.0**2, 20.0**2])
            previous_t = t
            continue
        previous_t = t

        F[0, 1] = F[2, 3] = dt
        Q = q_discrete_white_noise(dim=2, T=dt, var=PROCESS_VAR, block_size=2)

        prediction = kf_predict(state, cov, F, Q)
        innovation = z - H @ prediction.x
        S = H @ prediction.P @ H.T + R
        scores.append(float(nis(innovation, S)))

        update = kf_update(prediction.x, prediction.P, z, H, R)
        state, cov = update.x, update.P
        if not np.isnan(sog):
            sogs.append(sog)

    return state, scores, sogs


@pytest.fixture(scope="module")
def tracked(capture, centroid):
    """Run every qualifying ship once; the assertions below all read this."""
    scores, speed_errors = [], []
    for rows in capture.values():
        if len(rows) < MIN_REPORTS:
            continue
        state, track_scores, sogs = _track(rows, centroid)
        if state is None or not track_scores:
            continue
        scores.extend(track_scores)
        broadcast = sogs[-3:]
        if broadcast:
            estimated = math.hypot(state[1], state[3])
            speed_errors.append(estimated - float(np.mean(broadcast)))
    return np.array(scores), np.array(speed_errors)


class TestFixtureIsWhatItClaims:
    """Guard the guard: every assertion below is vacuous on an empty capture."""

    def test_capture_has_enough_ships(self, capture):
        qualifying = [
            mmsi for mmsi, rows in capture.items() if len(rows) >= MIN_REPORTS
        ]
        assert len(qualifying) >= 50, (
            f"only {len(qualifying)} ships with >= {MIN_REPORTS} reports"
        )

    def test_capture_has_thousands_of_reports(self, capture):
        total = sum(len(rows) for rows in capture.values())
        assert total >= 1000

    def test_reports_carry_broadcast_speed(self, capture):
        """The reference quantity. Without it this file proves nothing."""
        with_speed = sum(
            1 for rows in capture.values() for r in rows if not math.isnan(r[3])
        )
        assert with_speed >= 500

    def test_ships_actually_move(self, capture):
        """A fixture of stationary points would pass a consistency check while
        saying nothing about tracking."""
        moved = 0
        for rows in capture.values():
            if len(rows) < MIN_REPORTS:
                continue
            first, last = rows[0], rows[-1]
            if math.hypot(last[1] - first[1], last[2] - first[2]) > 1e-4:
                moved += 1
        assert moved >= 20, f"only {moved} ships moved appreciably"


class TestVelocityMatchesWhatTheShipBroadcasts:
    """The reference test: position-only estimate vs. the ship's own SOG."""

    def test_most_ships_agree_closely(self, tracked):
        _, errors = tracked
        assert len(errors) >= 40
        within = np.mean(np.abs(errors) < 6.0)
        assert within > 0.80, f"only {within:.1%} of ships within 6 m/s"

    def test_median_error_is_small(self, tracked):
        """Calibrated envelope -- see tests/fixtures/ais/SOURCES.md's
        Calibration section for the measured value this bound is derived
        from (1.5x measured, ceiled to 1 significant figure)."""
        _, errors = tracked
        assert np.median(np.abs(errors)) < _MEDIAN_ENVELOPE_MPS

    def test_no_systematic_bias(self, tracked):
        """A unit error or a wrong Earth radius would show up here as a
        consistent offset rather than as scatter."""
        _, errors = tracked
        assert abs(float(np.mean(errors))) < 3.0


class TestInnovationsLookLikeRealShipTraffic:
    """The shape of the NIS distribution, which synthetic data cannot produce."""

    def test_mean_nis_is_near_the_consistent_value(self, tracked):
        """For a 2-D position measurement a consistent filter averages 2.0."""
        scores, _ = tracked
        assert len(scores) >= 100
        assert 0.2 < float(np.mean(scores)) < 10.0

    def test_median_is_far_below_the_mean(self, tracked):
        """Most legs are close to constant-velocity, so most innovations are
        small. A filter that had merely inflated its covariance would show
        mean and median together."""
        scores, _ = tracked
        assert float(np.median(scores)) < float(np.mean(scores)) / 1.5

    def test_there_is_a_manoeuvre_tail(self, tracked):
        """And a minority of updates must be large, or these ships never
        turned and the fixture is not real traffic -- ships turn in fjords,
        harbour approaches, and to avoid other traffic."""
        scores, _ = tracked
        exceedance = float(np.mean(scores > 5.99))  # chi2(2) 95th percentile
        assert exceedance > 0.01, "no manoeuvres at all -- is this synthetic?"
        assert exceedance < 0.40, f"{exceedance:.1%} outliers: filter is inconsistent"
