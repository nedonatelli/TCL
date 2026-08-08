# Real-Data Validation (ADS-B + Satellite TLE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land two REFERENCE-class real-data validations: the existing ADS-B tracking test (rebased and verified), and a new satellite TLE self-prediction test with a vendored Space-Track history fixture.

**Architecture:** Two independent PRs. Part A rebases the finished `test/adsb-real-data-validation` branch onto main and verifies it against the v2 codebase. Part B (branch `test/satellite-tle-validation`) adds a one-time Space-Track fetch script, a vendored ~30-day TLE-history fixture for 6 satellites, and `tests/validation/test_tle_self_prediction.py` which scores pytcl's SGP4/SDP4 propagation against each TLE's NORAD-fitted successor. Spec: `docs/superpowers/specs/2026-08-08-real-data-validation-design.md`.

**Tech Stack:** pytest, numpy, pytcl's own `parse_tle`/`tle_epoch_to_jd`/`sgp4_propagate`/`is_deep_space`, stdlib `urllib` for the fetch script (no new dependencies).

## Global Constraints

- Tests are deterministic and offline: absent fixture -> `pytest.skip`; malformed fixture -> failure. No network access from any test.
- The fetch script reads `SPACETRACK_USER`/`SPACETRACK_PASSWORD` from the environment, fails loudly on missing credentials or HTTP errors (no retries, no fallbacks), and is never executed by tests or CI.
- Assertion envelopes are calibrated from the captured data by a fixed documented rule (see Task 3) and recorded with their measured basis in both the test comments and SOURCES.md — never tuned iteratively to pass.
- ASCII-only in anything printed to the console (scripts included).
- Style: ruff (88 cols), NumPy docstrings; every commit passes the prek hook (ruff + ty).
- Run all commands with `uv run` from the repo root; prefix shells with `export PATH="$HOME/.local/bin:$PATH"`.
- Credentials never appear in files, commits, or command lines that persist (env vars only).

## Verified facts the plan relies on

- Branch `test/adsb-real-data-validation` holds exactly one commit `fe45811` adding `tests/validation/test_adsb_tracking.py`, `tests/fixtures/adsb/adsb_boston.json.gz`, `tests/fixtures/adsb/SOURCES.md`.
- Its imports: `pytcl.coordinate_systems.conversions.geodetic2enu`, `pytcl.dynamic_estimation.kalman.linear.kf_predict/kf_update`, `pytcl.dynamic_models.q_discrete_white_noise`, `pytcl.mathematical_functions.statistics.nis`.
- `parse_tle(l1, l2)` -> TLE; `tle_epoch_to_jd(tle)` -> float JD; `sgp4_propagate(tle, tsince_minutes)` -> state with `.r`/`.v` in km, km/s, TEME; `is_deep_space(tle)` -> bool (period > 225 min). All in `pytcl/astronomical/`.
- TEME is an of-date frame: TLE_k propagated to t_{k+1} and TLE_{k+1} at tsince=0 are both TEME at the same instant — directly comparable.
- The public-API coverage contract (`tests/contract/test_public_api_coverage.py`) tracks exported functions; new test files that only consume existing API require no ledger change.

---

### Task 1 (Part A, own branch): rebase, verify, and PR the ADS-B validation

**Files:**
- No new files; rebases existing commit `fe45811`.
- Modify (on that branch): `CHANGELOG.md`, `tests/validation/README.md`.

**Interfaces:**
- Consumes: existing branch `test/adsb-real-data-validation`; current main.
- Produces: a merged-ready PR; no APIs.

- [ ] **Step 1: Rebase onto main**

```bash
git checkout test/adsb-real-data-validation
git rebase main
```

Expected: clean rebase (the branch touches only new files). If conflicts appear, stop and report BLOCKED with the conflict list.

- [ ] **Step 2: Verify the imports still exist on v2**

```bash
uv run python -c "
from pytcl.coordinate_systems.conversions import geodetic2enu
from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_models import q_discrete_white_noise
from pytcl.mathematical_functions.statistics import nis
print('imports OK')"
```

Expected: `imports OK`. If any import fails, locate the moved symbol (`grep -rn "def <name>" pytcl/`) and update ONLY the import lines in `tests/validation/test_adsb_tracking.py`; do not restructure the test.

- [ ] **Step 3: Run the test**

```bash
uv run pytest tests/validation/test_adsb_tracking.py -v
```

Expected: all tests pass (they skip only if the fixture is missing — it is committed on the branch, so a skip here is a failure of this step). Investigate any failure; report BLOCKED with the assertion and values if the v2 stack genuinely changed filter behavior.

- [ ] **Step 4: Docs**

In `tests/validation/README.md`, extend the oracle list sentence to include the recorded air-traffic capture — after "IERS and IAU." append:

```markdown
Real-world recordings also serve as references: a vendored ADS-B air-traffic
capture (aircraft broadcast their own ground speed, which the filter never
sees).
```

In `CHANGELOG.md` under `## [Unreleased]` / `### Added` (create the subsection if absent):

```markdown
- REFERENCE-class validation of the tracking chain (`geodetic2enu` ->
  Kalman filter -> NIS) against 3,600 recorded ADS-B position reports from
  120 real aircraft, scored against each aircraft's self-broadcast ground
  speed — a quantity the filter is never given.
```

- [ ] **Step 5: Commit docs, full checks, push, PR**

```bash
git add -u && git commit -m "docs: changelog and oracle-list entries for the ADS-B validation"
uv run ruff check . && uv run ty check pytcl
uv run pytest tests/validation/test_adsb_tracking.py tests/contract/test_public_api_coverage.py -q
git push -u origin test/adsb-real-data-validation
gh pr create --base main --title "test: validate the tracking stack against real air traffic" \
  --body "Rebase of the pre-v2 ADS-B validation branch, verified against the v2 codebase. REFERENCE-class: the filter sees positions only and is scored against each aircraft's independently broadcast ground speed. Fixture is vendored and offline (120 aircraft, 3,600 reports; see tests/fixtures/adsb/SOURCES.md)."
```

Expected: PR URL printed. (End every commit with the repo's standard Co-Authored-By trailer.)

---

### Task 2 (Part B): Space-Track fetch script + fixture capture

**Files:**
- Create: `scripts/fetch_tle_history.py`
- Create: `tests/fixtures/tle/tle_history.json.gz` (generated by the script)
- Create: `tests/fixtures/tle/SOURCES.md`

**Interfaces:**
- Consumes: Space-Track gp_history API (one time, human-triggered).
- Produces: fixture JSON schema used by Task 3 — gzipped JSON `{ "<norad_id>": {"name": str, "regime": str, "tles": [{"epoch": str, "line1": str, "line2": str}, ...] } }`, epochs ascending, deduplicated.

- [ ] **Step 1: Write the fetch script**

```python
#!/usr/bin/env python3
"""One-time capture of TLE history from Space-Track for the validation fixture.

Requires SPACETRACK_USER and SPACETRACK_PASSWORD in the environment. Never run
by tests or CI; kept for provenance and reproducibility. See
tests/fixtures/tle/SOURCES.md.
"""

import gzip
import http.cookiejar
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

BASE = "https://www.space-track.org"
DAYS = 30

# One satellite per SGP4 regime. The decaying entry is chosen at capture time
# from Space-Track's decay list (perigee < 250 km, still being tracked) and
# recorded in SOURCES.md.
SATELLITES = {
    25544: ("ISS (ZARYA)", "leo-high-drag"),
    44713: ("STARLINK-1007", "leo"),
    28474: ("GPS BIIR-13", "meo-deep-space"),
    41866: ("GOES-16", "geo-deep-space"),
    25485: ("MOLNIYA 1-91", "heo-high-eccentricity"),
    # <NORAD_ID>: ("<NAME>", "decaying"),  # add at capture time
}

OUT = pathlib.Path(__file__).parent.parent / "tests" / "fixtures" / "tle"


def main() -> int:
    user = os.environ.get("SPACETRACK_USER")
    password = os.environ.get("SPACETRACK_PASSWORD")
    if not user or not password:
        print("SPACETRACK_USER and SPACETRACK_PASSWORD must be set")
        return 1

    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    login = urllib.parse.urlencode({"identity": user, "password": password})
    with opener.open(f"{BASE}/ajaxauth/login", login.encode()) as resp:
        if resp.status != 200:
            print(f"login failed: HTTP {resp.status}")
            return 1

    ids = ",".join(str(i) for i in SATELLITES)
    query = (
        f"{BASE}/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{ids}/EPOCH/%3Enow-{DAYS}/"
        "orderby/EPOCH%20asc/format/json"
    )
    with opener.open(query) as resp:
        records = json.load(resp)

    fixture: dict = {}
    for norad, (name, regime) in SATELLITES.items():
        rows = [r for r in records if int(r["NORAD_CAT_ID"]) == norad]
        seen: set = set()
        tles = []
        for r in rows:
            if r["EPOCH"] in seen:
                continue
            seen.add(r["EPOCH"])
            tles.append(
                {"epoch": r["EPOCH"], "line1": r["TLE_LINE1"], "line2": r["TLE_LINE2"]}
            )
        if len(tles) < 10:
            print(f"only {len(tles)} TLEs for {norad} ({name}) -- investigate")
            return 1
        fixture[str(norad)] = {"name": name, "regime": regime, "tles": tles}
        print(f"{norad} {name}: {len(tles)} TLEs")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "tle_history.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(fixture, handle)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Lint the script**

Run: `uv run ruff check scripts/fetch_tle_history.py && uv run ruff format --check scripts/fetch_tle_history.py`
Expected: clean.

- [ ] **Step 3: CHECKPOINT — human runs the capture**

This step needs the human partner's Space-Track credentials. Before running:
verify the five fixed NORAD IDs are still active/tracked and pick the
decaying-object ID (Space-Track decay_epoch queries or its website decay
list), adding it to `SATELLITES`. Then the human (or the controller with
credentials exported) runs:

```bash
export SPACETRACK_USER=... SPACETRACK_PASSWORD=...
uv run python scripts/fetch_tle_history.py
```

Expected: one line per satellite with TLE counts (>= 10 each), then
`wrote .../tle_history.json.gz`. If a satellite returns too few TLEs,
substitute another object of the same regime and record the substitution
in SOURCES.md.

- [ ] **Step 4: Sanity-check the fixture parses with pytcl**

```bash
uv run python -c "
import gzip, json
from pytcl.astronomical.tle import parse_tle, tle_epoch_to_jd
data = json.load(gzip.open('tests/fixtures/tle/tle_history.json.gz','rt'))
for norad, sat in data.items():
    tles = [parse_tle(t['line1'], t['line2']) for t in sat['tles']]
    jds = [tle_epoch_to_jd(t) for t in tles]
    assert all(b > a for a, b in zip(jds, jds[1:])), norad
    print(norad, sat['regime'], len(tles), 'span_days', round(jds[-1]-jds[0], 1))
"
```

Expected: one line per satellite, ascending epochs, span roughly 30 days.

- [ ] **Step 5: Write SOURCES.md**

Follow the structure of `tests/fixtures/adsb/SOURCES.md` (title, what/why,
table of files, field descriptions). Must include: capture UTC timestamp,
the exact gp_history query, the satellite table (NORAD ID, name, regime,
why chosen), the note that successor TLEs are NORAD-fitted to real tracking
observations (the independence argument), and a "Calibration" section left
with the heading plus "filled by test calibration (see
tests/validation/test_tle_self_prediction.py)" — Task 3 completes it.

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_tle_history.py tests/fixtures/tle/
git commit -m "test: vendor 30 days of Space-Track TLE history for six satellites"
```

(Standard Co-Authored-By trailer. The prek large-file hook allows files
under 1000 kB; the fixture should be well under — if it is not, reduce DAYS
and re-capture rather than raising the hook limit.)

---

### Task 3 (Part B): the self-prediction test

**Files:**
- Create: `tests/validation/test_tle_self_prediction.py`
- Modify: `tests/fixtures/tle/SOURCES.md` (fill the Calibration section)

**Interfaces:**
- Consumes: fixture schema from Task 2; `parse_tle`, `tle_epoch_to_jd`, `sgp4_propagate`, `is_deep_space` from `pytcl.astronomical`.
- Produces: the test module; no library API.

- [ ] **Step 1: Write the test skeleton with loader and pair harness (no envelopes yet)**

```python
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


def _pair_errors(sat):
    """Position/velocity error (km, km/s) and horizon (days) per TLE pair."""
    tles = [parse_tle(t["line1"], t["line2"]) for t in sat["tles"]]
    jds = [tle_epoch_to_jd(t) for t in tles]
    rows = []
    for older, newer, jd0, jd1 in zip(tles, tles[1:], jds, jds[1:]):
        horizon_days = jd1 - jd0
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
```

- [ ] **Step 2: Add the calibration probe and run it**

Append temporarily (or run as a snippet) to print the measured statistics:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'tests/validation')
from test_tle_self_prediction import _load, _pair_errors
import numpy as np
for norad, sat in _load().items():
    rows = _pair_errors(sat)
    pos = np.array([r[0] for r in rows]); h = np.array([r[2] for r in rows])
    per_day = pos / np.maximum(h, 1e-6)
    print(norad, sat['regime'], 'pairs', len(rows),
          'med_km_per_day', round(float(np.median(per_day)), 3),
          'p95_km', round(float(np.percentile(pos, 95)), 3))
"
```

Record every printed line verbatim in SOURCES.md's Calibration section
with the run date. Derive each envelope as **1.5x the measured value,
rounded UP to one significant figure** (e.g. med 1.8 -> 3.0; p95 6.2 ->
10.0). This rule is fixed in advance; if a measured statistic looks
physically absurd (LEO medians of hundreds of km/day), STOP and
investigate the harness before writing envelopes — that magnitude means a
units or epoch bug, not a loose rail.

- [ ] **Step 3: Write the assertions with the calibrated envelopes**

Add to the test module (values from Step 2 — the `...` markers below are
filled with the derived envelopes, one entry per satellite in the fixture):

```python
# regime -> (median km/day envelope, p95 km envelope); derivation rule and
# measured basis in the module docstring and SOURCES.md.
ENVELOPES = {
    "leo-high-drag": (..., ...),
    "leo": (..., ...),
    "meo-deep-space": (..., ...),
    "geo-deep-space": (..., ...),
    "heo-high-eccentricity": (..., ...),
    "decaying": (..., ...),
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
        worst = 0.0
        for sat in history.values():
            rows = _pair_errors(sat)
            worst = max(worst, float(np.percentile([r[1] for r in rows], 95)))
        assert worst < ...  # calibrated: 1.5x measured worst p95, 1 sig fig
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/validation/test_tle_self_prediction.py -v`
Expected: 4 tests pass. A failure after honest calibration means either the
harness has a bug (check units: tsince is MINUTES) or the fixture contains a
maneuver (a station-keeping burn breaks self-prediction for that pair) —
maneuvers are legitimate data; if one satellite's p95 is maneuver-dominated,
document it in SOURCES.md and calibrate the envelope over the measured data
as the rule prescribes, which absorbs it.

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_tle_self_prediction.py tests/fixtures/tle/SOURCES.md
git commit -m "test: score SGP4 self-prediction against NORAD-fitted successor TLEs"
```

(Standard Co-Authored-By trailer.)

---

### Task 4 (Part B): docs, checks, PR

**Files:**
- Modify: `CHANGELOG.md`, `tests/validation/README.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the satellite PR.

- [ ] **Step 1: Docs**

`tests/validation/README.md` — extend the real-world-recordings sentence
added by Part A (or add it if Part A has not merged yet) so it reads:

```markdown
Real-world recordings also serve as references: a vendored ADS-B air-traffic
capture (aircraft broadcast their own ground speed, which the filter never
sees), and a vendored Space-Track TLE history (each TLE is scored against
its NORAD-fitted successor, which was fitted to real tracking observations).
```

`CHANGELOG.md` under `## [Unreleased]` / `### Added`:

```markdown
- REFERENCE-class validation of SGP4/SDP4 propagation against real orbital
  data: 30 days of vendored Space-Track TLE history for six satellites
  spanning every SGP4 regime, each TLE scored against its NORAD-fitted
  successor at that successor's epoch.
```

- [ ] **Step 2: Full checks**

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check pytcl
uv run pytest tests/validation/test_tle_self_prediction.py tests/contract/test_public_api_coverage.py -q
uv run pytest -q
```

Expected: all green (full suite because nothing here touches `pytcl/`, but
the contract and docs gates must see the final tree). Revert any regenerated
`docs/_static/images/examples/*.html` before committing.

- [ ] **Step 3: Commit, push, PR**

```bash
git add -u && git commit -m "docs: changelog and oracle-list entries for TLE self-prediction"
git push -u origin test/satellite-tle-validation
gh pr create --base main --title "test: validate SGP4 propagation against real TLE history" \
  --body "Implements the satellite half of docs/superpowers/specs/2026-08-08-real-data-validation-design.md: 30 days of vendored Space-Track TLE history for six satellites across the SGP4 regimes; each TLE is propagated to its successor's epoch and scored against that NORAD-fitted successor. Envelopes are calibrated bounds on measured behavior (rule and values in SOURCES.md). Deterministic and offline."
```

(Standard Co-Authored-By trailer on the commit.)
