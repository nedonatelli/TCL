"""SGP4/SDP4 self-prediction against NORAD-fitted successor TLEs.

Each TLE in the vendored history is propagated to the epoch of the next TLE
for the same satellite; the successor, evaluated at its own epoch, is the
reference. Successor TLEs are fitted by 18 SPCS to real tracking
observations, so this scores the propagation against measurements of the
actual satellite -- the orbital-mechanics analog of the ADS-B test's
broadcast ground speed. Both states are TEME at the same instant, so they
are directly comparable.

Envelopes below are calibrated bounds on behavior measured at capture time
(rule: 1.5x the measured statistic, rounded up to one significant figure;
measured values in the comments and in tests/fixtures/tle/SOURCES.md).
They are regression rails, not accuracy claims.

Traceability note: the spec's horizon-growth check is phrased in terms of a
rank correlation between horizon and error, and its velocity check in terms
of per-regime rails; this module codifies both as simplifications --
`test_error_grows_with_horizon` compares a single early/late pair per
satellite rather than a full rank correlation, and `test_velocity_within_envelope`
uses one worst-case rail across all regimes rather than a rail per regime.
"""

import gzip
import json
import pathlib

import numpy as np
import pytest

from pytcl.astronomical.sgp4 import sgp4_propagate
from pytcl.astronomical.tle import is_deep_space, parse_tle, tle_epoch_to_jd

FIXTURE = (
    pathlib.Path(__file__).parent.parent / "fixtures" / "tle" / "tle_history.json.gz"
)

MINUTES_PER_DAY = 1440.0


def _load():
    if not FIXTURE.exists():
        pytest.skip(f"TLE fixture not present: {FIXTURE.name}")
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def history():
    return _load()


MIN_HORIZON_DAYS = 0.01  # ~14 minutes


def _pair_errors(sat):
    """Position/velocity error (km, km/s) and horizon (days) per TLE pair.

    Space-Track's `gp_history` occasionally carries revised fits whose epoch
    differs from the previous element set by a few tens of nanoseconds to
    microseconds of a day (a re-fit at essentially the same instant, not a
    new observation window). The fetch script dedups by exact EPOCH string,
    which misses these near-duplicates since the strings differ. Dividing a
    normal-sized position delta by a near-zero horizon blows up into a
    per-day rate of orbital-velocity magnitude (propagation error is fine;
    the horizon is degenerate), so such pairs are dropped here rather than
    at fetch time -- see tests/fixtures/tle/SOURCES.md for the measured
    counts.
    """
    tles = [parse_tle(t["line1"], t["line2"]) for t in sat["tles"]]
    jds = [tle_epoch_to_jd(t) for t in tles]
    rows = []
    for older, newer, jd0, jd1 in zip(tles, tles[1:], jds, jds[1:]):
        horizon_days = jd1 - jd0
        if horizon_days < MIN_HORIZON_DAYS:
            continue
        predicted = sgp4_propagate(older, horizon_days * MINUTES_PER_DAY)
        truth = sgp4_propagate(newer, 0.0)
        rows.append(
            (
                float(np.linalg.norm(predicted.r - truth.r)),
                float(np.linalg.norm(predicted.v - truth.v)),
                horizon_days,
            )
        )
    return rows


# Calibration probe (2026-08-08, re-run after adding the MIN_HORIZON_DAYS
# filter above -- see tests/fixtures/tle/SOURCES.md for the verbatim run):
# measured (median km/day, p95 km) per regime, this fixture, pairs with
# horizon_days < MIN_HORIZON_DAYS excluded --
#   25544 leo-high-drag        pairs 73  med 0.393  p95 0.521
#   45098 leo                  pairs 45  med 1.609  p95 6.813
#   28474 meo-deep-space       pairs 29  med 0.241  p95 0.578
#   41866 geo-deep-space       pairs 86  med 0.562  p95 0.543
#   25485 heo-high-eccentricity pairs 11 med 0.344  p95 2.977
#   69702 decaying              pairs 47 med 11.898 p95 41.019
# Envelope = 1.5x measured, rounded up to one significant figure.
#
# regime -> (median km/day envelope, p95 km envelope); derivation rule and
# measured basis in the module docstring and SOURCES.md.
ENVELOPES = {
    "leo-high-drag": (0.6, 0.8),
    "leo": (3.0, 20.0),
    "meo-deep-space": (0.4, 0.9),
    "geo-deep-space": (0.9, 0.9),
    "heo-high-eccentricity": (0.6, 5.0),
    "decaying": (20.0, 70.0),
}

EXPECTED_DEEP_SPACE = {
    "leo-high-drag": False,
    "leo": False,
    "meo-deep-space": True,
    "geo-deep-space": True,
    "heo-high-eccentricity": True,
    "decaying": False,
}


class TestSelfPrediction:
    def test_every_satellite_within_envelope(self, history):
        for norad, sat in history.items():
            rows = _pair_errors(sat)
            assert len(rows) >= 9, f"{norad}: too few TLE pairs to score"
            pos = np.array([r[0] for r in rows])
            horizon = np.array([r[2] for r in rows])
            med_env, p95_env = ENVELOPES[sat["regime"]]
            med = float(np.median(pos / np.maximum(horizon, 1e-6)))
            p95 = float(np.percentile(pos, 95))
            assert med < med_env, f"{norad} median {med:.2f} km/day"
            assert p95 < p95_env, f"{norad} p95 {p95:.2f} km"

    def test_error_grows_with_horizon(self, history):
        # Propagate each satellite's OLDEST TLE to successively later TLE
        # epochs; long-horizon error must exceed short-horizon error.
        for norad, sat in history.items():
            tles = [parse_tle(t["line1"], t["line2"]) for t in sat["tles"]]
            jds = [tle_epoch_to_jd(t) for t in tles]
            base, jd0 = tles[0], jds[0]

            def error_at(idx):
                pred = sgp4_propagate(base, (jds[idx] - jd0) * MINUTES_PER_DAY)
                truth = sgp4_propagate(tles[idx], 0.0)
                return float(np.linalg.norm(pred.r - truth.r))

            early = error_at(min(2, len(tles) - 1))
            late = error_at(len(tles) - 1)
            assert late > early, f"{norad}: {late:.2f} <= {early:.2f} km"

    def test_regime_exercises_intended_code_path(self, history):
        for norad, sat in history.items():
            tle = parse_tle(sat["tles"][0]["line1"], sat["tles"][0]["line2"])
            expected = EXPECTED_DEEP_SPACE[sat["regime"]]
            assert is_deep_space(tle) is expected, f"{norad} ({sat['regime']})"

    def test_velocity_within_envelope(self, history):
        # Velocity errors scale with position errors; a single generous rail
        # (calibrated the same way, worst regime) catches sign/frame bugs
        # which produce km/s-scale errors instantly.
        # Measured worst p95 (69702, decaying), post-filter: 0.04608 km/s ->
        # x1.5 -> 0.07.
        worst = 0.0
        for sat in history.values():
            rows = _pair_errors(sat)
            worst = max(worst, float(np.percentile([r[1] for r in rows], 95)))
        assert worst < 0.07
