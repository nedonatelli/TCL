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
`TestLongHorizon` below extends this with a full rank correlation and
binned 0.5-31 day envelopes; the binned tables in
tests/fixtures/tle/SOURCES.md ("Long-horizon calibration") are pytcl's
general SGP4-accuracy statement.
"""

import gzip
import json
import math
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


# --- Long-horizon accuracy envelopes -------------------------------------
#
# All ordered TLE pairs (i, j), i < j, at horizons of 0.5 day and above:
# TLE_i propagated to TLE_j's epoch, scored against TLE_j at its own epoch.
# Binned per regime; measured medians and the envelope derivation live in
# tests/fixtures/tle/SOURCES.md ("Long-horizon calibration"). For operated
# satellites these envelopes measure TLE predictability of the object
# (maneuvers included), not pure propagator fidelity.

LONG_HORIZON_MIN_DAYS = 0.5

LONG_HORIZON_BINS = [
    ("1d", 0.5, 1.5),
    ("3d", 1.5, 4.5),
    ("7d", 4.5, 9.5),
    ("14d", 9.5, 18.5),
    ("28d", 18.5, 31.0),
]

MIN_PAIRS_PER_BIN = 5


def _all_pair_errors(sat):
    """(position error km, horizon days) for every ordered pair >= 0.5 d."""
    tles = [parse_tle(t["line1"], t["line2"]) for t in sat["tles"]]
    jds = [tle_epoch_to_jd(t) for t in tles]
    truth_cache = {}
    rows = []
    for j in range(1, len(tles)):
        for i in range(j):
            horizon = jds[j] - jds[i]
            if horizon < LONG_HORIZON_MIN_DAYS:
                continue
            if j not in truth_cache:
                truth_cache[j] = sgp4_propagate(tles[j], 0.0).r
            predicted = sgp4_propagate(tles[i], horizon * MINUTES_PER_DAY)
            rows.append((float(np.linalg.norm(predicted.r - truth_cache[j])), horizon))
    return rows


def _binned_medians(rows):
    """{bin_label: (median_error_km, pair_count)} for one satellite."""
    out = {}
    for label, lo, hi in LONG_HORIZON_BINS:
        errs = [e for e, h in rows if lo <= h < hi]
        if errs:
            out[label] = (float(np.median(errs)), len(errs))
        else:
            out[label] = (None, 0)
    return out


# (regime, bin) -> envelope on the median position error (km). Cells absent
# here were measured with n < MIN_PAIRS_PER_BIN pairs and are tracked by
# EXPECTED_SPARSE_BINS instead. Derivation rule and measured basis:
# tests/fixtures/tle/SOURCES.md, "Long-horizon calibration".
#
# Vacuousness ceiling (controller ruling, 2026-08-08 --
# docs/superpowers/specs/2026-08-08-sgp4-long-horizon-design.md,
# "Vacuousness ceiling"): a cell is asserted here only if its derived
# envelope is below VACUOUSNESS_CEILING_KM -- above that, an assertion
# provably cannot fail against the ~13,100 km geometric maximum position
# error, so it would be a vacuous pass-always bound rather than a real
# regression rail. The decaying object's (69702) 14d/28d cells cross this
# line (measured medians saturate toward orbit-scale geometry as its
# perigee falls within the propagation window) and are recorded in
# SOURCES.md as measured-but-unasserted instead; see UNASSERTED_CELLS.
LONG_HORIZON_ENVELOPES = {
    "leo-high-drag": {"1d": 0.7, "3d": 5.0, "7d": 40.0, "14d": 300.0, "28d": 500.0},
    "leo": {"1d": 3.0, "3d": 20.0, "7d": 100.0, "14d": 700.0, "28d": 2000.0},
    "meo-deep-space": {"1d": 0.4, "3d": 2.0, "7d": 4.0, "14d": 20.0, "28d": 40.0},
    "geo-deep-space": {"1d": 0.5, "3d": 2.0, "7d": 8.0, "14d": 30.0, "28d": 60.0},
    "heo-high-eccentricity": {"3d": 2.0, "7d": 5.0, "14d": 20.0, "28d": 60.0},
    "decaying": {"1d": 30.0, "3d": 300.0, "7d": 2000.0},
}

EXPECTED_SPARSE_BINS = {
    ("25485", "1d"),  # Molniya: ~13 TLEs total, none paired within [0.5, 1.5) d
}

VACUOUSNESS_CEILING_KM = 5000.0  # controller ruling, see comment above

UNASSERTED_CELLS = {
    ("69702", "14d"),
    ("69702", "28d"),
}


def _derive_envelope(median_km):
    """1.5x measured median, rounded up to one significant figure.

    Mirrors the manual derivation used to fill LONG_HORIZON_ENVELOPES and
    tests/fixtures/tle/SOURCES.md's table, so the vacuousness-ceiling
    exemption below can be checked against a live re-measurement rather
    than a hardcoded number that could go stale.
    """
    scaled = median_km * 1.5
    magnitude = 10 ** math.floor(math.log10(scaled))
    return math.ceil(scaled / magnitude) * magnitude


_pair_error_cache = {}


def _cached_all_pair_errors(norad, sat):
    if norad not in _pair_error_cache:
        _pair_error_cache[norad] = _all_pair_errors(sat)
    return _pair_error_cache[norad]


class TestLongHorizon:
    def test_sparse_bins_are_exactly_the_expected_ones(self, history):
        # A wrong bin edge would silently empty a bin; assert the sparse
        # set matches the calibrated expectation exactly.
        sparse = set()
        for norad, sat in history.items():
            rows = _cached_all_pair_errors(norad, sat)
            for label, (_, n) in _binned_medians(rows).items():
                if n < MIN_PAIRS_PER_BIN:
                    sparse.add((norad, label))
        assert sparse == EXPECTED_SPARSE_BINS
        # An exempt cell that goes sparse would skip its saturation
        # self-check in test_binned_medians_within_envelope silently.
        assert UNASSERTED_CELLS.isdisjoint(sparse)

    def test_binned_medians_within_envelope(self, history):
        # A future recalibration must not be able to park a vacuous
        # envelope (>= the ceiling) in a cell that's actually asserted.
        for regime_envelopes in LONG_HORIZON_ENVELOPES.values():
            for bin_label, envelope in regime_envelopes.items():
                assert envelope < VACUOUSNESS_CEILING_KM, (
                    f"{bin_label}: envelope {envelope} km meets or exceeds "
                    f"the vacuousness ceiling ({VACUOUSNESS_CEILING_KM:.0f} "
                    f"km) -- move it to UNASSERTED_CELLS instead of "
                    f"asserting it"
                )
        for norad, sat in history.items():
            envelopes = LONG_HORIZON_ENVELOPES[sat["regime"]]
            rows = _cached_all_pair_errors(norad, sat)
            for label, (median, n) in _binned_medians(rows).items():
                if n < MIN_PAIRS_PER_BIN:
                    continue
                if (norad, label) in UNASSERTED_CELLS:
                    envelope = _derive_envelope(median)
                    assert envelope >= VACUOUSNESS_CEILING_KM, (
                        f"{norad} {label}: derived envelope {envelope:.0f} km "
                        f"no longer reaches the vacuousness ceiling "
                        f"({VACUOUSNESS_CEILING_KM:.0f} km) -- move it out of "
                        f"UNASSERTED_CELLS and into LONG_HORIZON_ENVELOPES"
                    )
                    continue
                assert median < envelopes[label], (
                    f"{norad} {label}: median {median:.2f} km, "
                    f"envelope {envelopes[label]} km over {n} pairs"
                )

    def test_error_rank_correlates_with_horizon(self, history):
        from scipy.stats import spearmanr

        for norad, sat in history.items():
            rows = _cached_all_pair_errors(norad, sat)
            errors = [e for e, _ in rows]
            horizons = [h for _, h in rows]
            rho = float(spearmanr(horizons, errors).correlation)
            assert rho > 0.5, f"{norad}: spearman rho {rho:.3f}"
