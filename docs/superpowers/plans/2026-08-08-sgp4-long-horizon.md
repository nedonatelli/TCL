# SGP4 Long-Horizon Accuracy Envelopes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the vendored TLE fixture's full 0.5–29-day horizon depth into calibrated, documented per-regime SGP4 accuracy envelopes with a rank-correlation growth assertion.

**Architecture:** One new test class (`TestLongHorizon`) appended to `tests/validation/test_tle_self_prediction.py`, an all-ordered-pairs harness with per-target caching, horizon bins at 1/3/7/14/28 days, envelopes from the existing fixed calibration rule, and a "Long-horizon calibration" section in `tests/fixtures/tle/SOURCES.md`. Spec: `docs/superpowers/specs/2026-08-08-sgp4-long-horizon-design.md`.

**Tech Stack:** numpy, scipy.stats.spearmanr (scipy is core), pytest, existing pytcl astronomical API. No new files, no network, no new dependencies.

## Global Constraints

- Branch: `test/sgp4-long-horizon` (stacked on `test/satellite-tle-validation`). If PR #103 has merged to main by execution time, rebase this branch onto main first.
- Calibration discipline unchanged and non-negotiable: measured values recorded VERBATIM with run date in SOURCES.md; every envelope = 1.5x measured median, rounded UP to one significant figure. No hand-tuning; a physically absurd measurement means a harness bug — STOP and report BLOCKED, do not write envelopes around it.
- Existing short-horizon tests and their envelopes are untouched.
- Full validation-file runtime stays under ~10 s.
- Style: ruff 88 cols; `uv run` from repo root; `export PATH="$HOME/.local/bin:$PATH"`; prek hook on commit; `git add` explicit paths only; commit messages end with the standard Co-Authored-By trailer (Claude Fable 5 <noreply@anthropic.com>).

## Verified facts the plan relies on

- Fixture: 6 satellites, TLE counts 79 (25544 leo-high-drag), 54 (45098 leo), 34 (28474 meo-deep-space), 90 (41866 geo-deep-space), 13 (25485 heo-high-eccentricity), 65 (69702 decaying); spans 27.1–29.7 days; epochs ascending and string-deduplicated, but near-duplicate epochs ~1e-8 day apart exist (38 sub-minute consecutive pairs) — the 0.5-day pair floor below excludes all of them from this harness.
- Existing module constants/helpers: `MINUTES_PER_DAY = 1440.0`, `_load()`, `history` module-scoped fixture, `parse_tle`, `tle_epoch_to_jd`, `sgp4_propagate` (tsince in MINUTES), regimes as listed above.
- `scipy.stats.spearmanr(a, b).correlation` returns the coefficient (float).
- Pair count across the fixture with a 0.5-day floor: roughly 8,000; two propagations per pair naively, ~half of that with per-target caching. The existing 4 tests run in ~0.7 s; the budget leaves ample room.

---

### Task 1: `TestLongHorizon` — harness, calibration, assertions

**Files:**
- Modify: `tests/validation/test_tle_self_prediction.py` (append)
- Modify: `tests/fixtures/tle/SOURCES.md` (new "Long-horizon calibration" section)

**Interfaces:**
- Consumes: fixture schema and module helpers listed above.
- Produces: `_all_pair_errors(sat) -> list[(error_km, horizon_days)]`, `LONG_HORIZON_BINS`, `LONG_HORIZON_ENVELOPES`, `EXPECTED_SPARSE_BINS`, class `TestLongHorizon` with three test methods.

- [ ] **Step 1: Append the harness and bin definitions**

```python
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
```

- [ ] **Step 2: Run the calibration probe and record it**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'tests/validation')
from test_tle_self_prediction import _load, _all_pair_errors, _binned_medians
for norad, sat in _load().items():
    rows = _all_pair_errors(sat)
    med = _binned_medians(rows)
    cells = ' '.join(
        f'{label}:med={m[0]:.3f},n={m[1]}' if m[0] is not None else f'{label}:EMPTY'
        for label, m in med.items())
    print(norad, sat['regime'], 'pairs', len(rows), cells)
"
```

Also time it (`time uv run pytest ... --collect-only` is not the point —
wrap the probe in `time`); if the probe alone exceeds ~8 s, report the
timing in your report (the runtime constraint applies to the final tests).

Record every printed line VERBATIM, with the run date, in a new
`## Long-horizon calibration` section of `tests/fixtures/tle/SOURCES.md`,
followed by a derived-envelope table: for every (regime, bin) cell with
`n >= 5`, envelope = 1.5x measured median rounded UP to one significant
figure (show measured -> x1.5 -> envelope in the table). Cells with
`n < 5` are listed in a sparse-bins line instead. Add the honesty
paragraph: operated satellites' long-horizon curves include maneuvers
(name Starlink/ISS/GOES); the decaying object's curve is drag-event
dominated; these are TLE-predictability envelopes, not pure propagator
fidelity.

Sanity gate before proceeding: medians must grow with the bin horizon for
most regimes and stay physically plausible (LEO tens of km at 28 d,
decaying possibly hundreds). A 28d LEO median SMALLER than its 1d median,
or thousands of km anywhere, means a harness bug (check the truth cache
and the minutes conversion) — STOP and report BLOCKED rather than
calibrating around it.

- [ ] **Step 3: Append the constants and the test class**

Fill `...` from the Step 2 derivation (every asserted cell), and
`EXPECTED_SPARSE_BINS` with exactly the (norad, bin_label) pairs measured
sparse:

```python
# (regime, bin) -> envelope on the median position error (km). Cells absent
# here were measured with n < MIN_PAIRS_PER_BIN pairs and are tracked by
# EXPECTED_SPARSE_BINS instead. Derivation rule and measured basis:
# tests/fixtures/tle/SOURCES.md, "Long-horizon calibration".
LONG_HORIZON_ENVELOPES = {
    "leo-high-drag": {"1d": ..., "3d": ..., "7d": ..., "14d": ..., "28d": ...},
    "leo": {...},
    "meo-deep-space": {...},
    "geo-deep-space": {...},
    "heo-high-eccentricity": {...},
    "decaying": {...},
}

EXPECTED_SPARSE_BINS = {
    # ("<norad>", "<bin_label>"), ... exactly as measured in Step 2
}


class TestLongHorizon:
    def test_sparse_bins_are_exactly_the_expected_ones(self, history):
        # A wrong bin edge would silently empty a bin; assert the sparse
        # set matches the calibrated expectation exactly.
        sparse = set()
        for norad, sat in history.items():
            for label, (_, n) in _binned_medians(_all_pair_errors(sat)).items():
                if n < MIN_PAIRS_PER_BIN:
                    sparse.add((norad, label))
        assert sparse == EXPECTED_SPARSE_BINS

    def test_binned_medians_within_envelope(self, history):
        for norad, sat in history.items():
            envelopes = LONG_HORIZON_ENVELOPES[sat["regime"]]
            for label, (median, n) in _binned_medians(_all_pair_errors(sat)).items():
                if n < MIN_PAIRS_PER_BIN:
                    continue
                assert median < envelopes[label], (
                    f"{norad} {label}: median {median:.2f} km, "
                    f"envelope {envelopes[label]} km over {n} pairs"
                )

    def test_error_rank_correlates_with_horizon(self, history):
        from scipy.stats import spearmanr

        for norad, sat in history.items():
            rows = _all_pair_errors(sat)
            errors = [e for e, _ in rows]
            horizons = [h for _, h in rows]
            rho = float(spearmanr(horizons, errors).correlation)
            assert rho > 0.5, f"{norad}: spearman rho {rho:.3f}"
```

If a regime's measured cells were ALL sparse for some bin label across the
table, drop that key from its dict entry rather than inventing a value —
`test_binned_medians_within_envelope` never reads keys for sparse cells.
If the Spearman assertion fails for the GEO satellite (station-keeping can
flatten growth), report the measured rho in your report and BLOCKED —
threshold changes are a controller decision, not an implementer one.

Also extend the module docstring's final paragraph with one sentence
pointing at the long-horizon tables in SOURCES.md as the general-accuracy
statement.

- [ ] **Step 4: Run and verify runtime**

Run: `uv run pytest tests/validation/test_tle_self_prediction.py -v --durations=5`
Expected: 7 tests pass (4 existing + 3 new); total wall time under ~10 s.
If over budget, cache `_all_pair_errors` per satellite across the three
new tests (module-scope dict keyed by norad) — do not reduce pair coverage.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check tests/validation/test_tle_self_prediction.py
uv run ruff format --check tests/validation/test_tle_self_prediction.py
git add tests/validation/test_tle_self_prediction.py tests/fixtures/tle/SOURCES.md
git commit -m "test: calibrated long-horizon SGP4 accuracy envelopes (1-28 days)"
```

---

### Task 2: docs, checks, PR

**Files:**
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: Task 1's tests and SOURCES.md tables.
- Produces: the PR.

- [ ] **Step 1: CHANGELOG**

Under `## [Unreleased]` / `### Added`, after the existing TLE
self-prediction bullet:

```markdown
- Long-horizon SGP4 accuracy envelopes: all ordered TLE pairs from the
  vendored history, binned at 1/3/7/14/28-day horizons, with calibrated
  per-regime median-error envelopes and a rank-correlation error-growth
  assertion (documented in `tests/fixtures/tle/SOURCES.md`).
```

- [ ] **Step 2: Full checks**

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check pytcl
uv run pytest tests/validation/test_tle_self_prediction.py -q
uv run pytest -q
```

All green; revert regenerated `docs/_static/images/examples/*.html` before
committing (`git checkout -- docs/_static/images/examples/`).

- [ ] **Step 3: Commit, push, PR**

```bash
git add -u && git commit -m "docs: changelog entry for long-horizon SGP4 envelopes"
git push -u origin test/sgp4-long-horizon
```

Then: if PR #103 has merged, open the PR against `main`; otherwise against
`test/satellite-tle-validation` (note the stack in the body):

```bash
gh pr create --base <resolved-base> --title "test: long-horizon SGP4 accuracy envelopes" \
  --body "Implements docs/superpowers/specs/2026-08-08-sgp4-long-horizon-design.md: all-pairs horizon analysis of the vendored TLE history (1/3/7/14/28-day bins), per-regime median-error envelopes under the existing fixed calibration rule, sparse-bin guard, and a Spearman error-growth assertion. Envelope basis and the operated-satellite caveat documented in tests/fixtures/tle/SOURCES.md."
```
