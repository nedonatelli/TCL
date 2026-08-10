# TLE history for six SGP4-regime satellites

Thirty days of real orbital element sets from Space-Track's `gp_history`
class, one satellite per SGP4 propagation regime, vendored so the validation
suite can check SGP4 self-prediction against the actual sequence of
NORAD-fitted elements without a network call or credentials.

These are the reference side of a REFERENCE-class test (see
`tests/validation/README.md`). The library never reads this fixture at
runtime; only `tests/validation/test_tle_self_prediction.py` does.

**Capture timestamp (UTC):** approx. `2026-08-08T01:58:21Z` (derived from
`tle_history.json.gz`'s file mtime, 2026-08-07 21:58 America/New_York
(EDT, UTC-4); Space-Track does not echo a server-side capture time in the
response body). Consistent with the fixture's most recent element, ISS's
`2026-08-07T20:38:15Z` epoch -- about 5 hours before capture, in line with
ISS's frequent element updates.

**Query:** the fetch script's Step-1 code with `DAYS = 30` and the final
`SATELLITES` keys below expands to:
```
https://www.space-track.org/basicspacedata/query/class/gp_history/NORAD_CAT_ID/25544,45098,28474,41866,25485,69702/EPOCH/%3Enow-30/orderby/EPOCH%20asc/format/json
```

| File | Contents |
|------|----------|
| `tle_history.json.gz` | Gzipped JSON, keyed by NORAD catalog ID as a string: `{"<norad_id>": {"name": str, "regime": str, "tles": [{"epoch": str, "line1": str, "line2": str}, ...]}}`. Epochs ascending, deduplicated by `EPOCH`. |

## Captured fixture contents

Verified by loading the fixture and parsing every TLE with
`pytcl.astronomical.tle.parse_tle`/`tle_epoch_to_jd`: all six satellites have
strictly ascending epochs, and every TLE count is above the brief's minimum
of 10.

| NORAD ID | Name | Regime | TLE count | Span (days) | Notes |
|----------|------|--------|-----------|-------------|-------|
| 25544 | ISS (ZARYA) | `leo-high-drag` | 79 | 29.7 | Near-daily elements, typical of ISS's high-drag orbit-maintenance cadence. |
| 45098 | STARLINK-1184 | `leo` | 54 | 28.6 | |
| 28474 | GPS BIIR-13 | `meo-deep-space` | 34 | 28.4 | |
| 41866 | GOES-16 | `geo-deep-space` | 90 | 29.2 | Most frequently updated object in the set. |
| 25485 | MOLNIYA 1-91 | `heo-high-eccentricity` | 13 | 29.2 | Full ~30-day span despite the low count -- Molniya-class objects simply receive far sparser element updates than the others; 13 TLEs was still comfortably above the brief's minimum of 10. |
| 69702 | ELECTRON R/B | `decaying` | 65 | 27.1 | Slightly short of 30 days: the object was only cataloged 2026-07-10 (its international designator is 2026-146B), so `gp_history` has no elements before that. |

## Satellites

Selected to cover one representative object per SGP4 propagation branch
(near-Earth vs. deep-space, low vs. high eccentricity, low vs. high drag).
IDs were verified against CelesTrak's public GP API
(`https://celestrak.org/NORAD/elements/gp.php?CATNR=<id>&FORMAT=json`)
immediately before capture, since that endpoint needs no credentials and
Space-Track does.

| NORAD ID | Name | Regime | Why chosen |
|----------|------|--------|------------|
| 25544 | ISS (ZARYA) | `leo-high-drag` | Low-altitude, high-drag near-Earth orbit; large BSTAR relative to other regimes here. |
| 45098 | STARLINK-1184 | `leo` | Ordinary low-eccentricity LEO orbit. Substituted for the brief's original pick, NORAD 44713 (STARLINK-1007) -- see Substitutions below. |
| 28474 | GPS BIIR-13 | `meo-deep-space` | MEO, ~2 rev/day, exercises SGP4's deep-space branch. CelesTrak's current object name for this ID is "NAVSTAR 56 (USA 180)"; both names refer to the same GPS BIIR-13 spacecraft. |
| 41866 | GOES-16 | `geo-deep-space` | Geostationary, ~1 rev/day, near-zero eccentricity; deep-space branch. |
| 25485 | MOLNIYA 1-91 | `heo-high-eccentricity` | Highly eccentric (e > 0.5) 12-hour orbit; exercises the high-eccentricity path through the deep-space branch. |
| 69702 | ELECTRON R/B | `decaying` | Rocket body from a recent Electron launch, perigee ~175 km and dropping fast (large BSTAR / mean-motion derivative); exercises SGP4 near-decay behavior. See Decaying-object selection below. |

### Substitutions

The brief's original LEO pick, NORAD 44713 (STARLINK-1007), no longer
returns GP data from CelesTrak (`No GP data found` for both `CATNR=44713`
and `NAME=STARLINK-1007` queries) -- it has deorbited. Replaced with NORAD
45098 (STARLINK-1184), an actively tracked Starlink satellite in the same
`leo` regime, verified at capture-prep time with mean motion 15.1686 rev/day
and eccentricity 0.0002129 (epoch 2026-08-07T14:33:00Z) -- consistent with
the rest of the constellation.

The other four fixed IDs from the brief (ISS, GPS BIIR-13, GOES-16, MOLNIYA
1-91) all still return current, regime-consistent elements; no other
substitutions were needed. See Verification below for the numbers.

### Decaying-object selection

Found via CelesTrak's `GROUP=last-30-days` GP query (recently launched or
recently cataloged objects), filtered to perigee below 250 km using
perigee = a(1 - e) - 6378.137 km, with the semi-major axis `a` derived from
`MEAN_MOTION` via Kepler's third law (mu = 398600.4418 km^3/s^2). Several
candidates in that group were rejected because their low perigee comes from
a highly eccentric GTO transfer orbit (e.g. CZ-7A R/B and CZ-3B R/B, both
with apogees above 34,000 km) rather than genuine atmospheric decay -- those
objects will not exhibit decaying-regime SGP4 behavior for a long time.

NORAD 69702 (ELECTRON R/B, international designator 2026-146B) was chosen
instead: circular-ish low orbit (perigee ~175 km, apogee ~228 km,
eccentricity 0.0040), consistent with genuine drag-dominated decay, and a
large `MEAN_MOTION_DOT` (0.0487 rev/day^2) confirming it is actively
decaying rather than merely low. Verified at capture-prep time
(epoch 2026-08-06T16:43:24Z): mean motion 16.2663 rev/day, eccentricity
0.0040444.

## Verification (CelesTrak public GP API, pre-capture)

Checked each fixed NORAD ID's current elements against its intended regime
immediately before running the fetch script:

| NORAD ID | CelesTrak name | Mean motion (rev/day) | Eccentricity | Epoch (UTC) | Regime check |
|----------|-----------------|------------------------|---------------|-------------|---------------|
| 25544 | ISS (ZARYA) | 15.4938 | 0.0007338 | 2026-08-07T20:38:15Z | OK, ~15.5 LEO |
| 45098 | STARLINK-1184 | 15.1686 | 0.0002129 | 2026-08-07T14:33:00Z | OK, ~15.0-15.2 Starlink shell |
| 28474 | NAVSTAR 56 (USA 180) | 2.0056 | 0.0170720 | 2026-08-07T04:19:11Z | OK, ~2.0 GPS |
| 41866 | GOES 16 | 1.0027 | 0.0001205 | 2026-08-07T20:19:08Z | OK, ~1.0 GEO, still station-kept (not drifting) |
| 25485 | MOLNIYA 1-91 | 2.3644 | 0.6644092 | 2026-08-07T09:39:51Z | OK, ~2.0 with e > 0.5 |
| 69702 | ELECTRON R/B | 16.2663 | 0.0040444 | 2026-08-06T16:43:24Z | OK, perigee ~175 km, decaying |

## Why these are independent of the code under test

The successor TLEs in this history are NORAD-fitted to real tracking
observations of each object -- not generated by this library's own SGP4
implementation. Comparing this library's SGP4 propagation of TLE `N` against
the independently-fitted TLE `N+1` (or against the raw element drift across
the 30-day span) tests the propagator against ground truth the code under
test had no part in producing.

## Near-duplicate epochs

Space-Track's `gp_history` contains, for some objects, revised fits whose
epoch differs from the immediately preceding element set by a few tens of
nanoseconds to a few hundred microseconds of a day (e.g. `2.98e-8` days,
about 2.6 ms) -- a re-fit at essentially the same instant rather than a new
observation window. `scripts/fetch_tle_history.py` dedups by exact `EPOCH`
string, which does not catch these because the strings differ.

Feeding such a pair into `_pair_errors` divides a normal, sub-km propagation
delta by a near-zero horizon and produces a per-day rate of orbital-velocity
magnitude -- an artifact of the degenerate horizon, not a propagation
error. Across this fixture, 38 of the 329 raw TLE pairs (11.6%) have
`horizon_days < 0.01` (~14 minutes):

| NORAD ID | Regime | Total pairs | Degenerate pairs (<0.01 day) |
|----------|--------|-------------:|------------------------------:|
| 25544 | leo-high-drag | 78 | 5 |
| 45098 | leo | 53 | 8 |
| 28474 | meo-deep-space | 33 | 4 |
| 41866 | geo-deep-space | 89 | 3 |
| 25485 | heo-high-eccentricity | 12 | 1 |
| 69702 | decaying | 64 | 17 |

Worst offenders (implied per-day rate, position delta, horizon), all from
`69702` (decaying) except one `25544` pair:

| Implied km/day | NORAD ID | Position delta (km) | Horizon (days) |
|----------------:|----------|---------------------:|------------------:|
| 12,613,343 | 69702 | 0.376 | 2.98e-8 |
| 1,263,977 | 69702 | 0.025 | 2.00e-8 |
| 1,115,433 | 69702 | 0.112 | 1.00e-7 |
| 1,000,851 | 69702 | 0.300 | 3.00e-7 |
| 879,721 | 69702 | 0.202 | 2.30e-7 |
| 693,971 | 25544 | 0.305 | 4.40e-7 |

`tests/validation/test_tle_self_prediction.py`'s `_pair_errors` now skips
any pair with `horizon_days < MIN_HORIZON_DAYS` (0.01 day, ~14 minutes)
before computing statistics, both for calibration and at test time. The
`decaying` regime, which had the most degenerate pairs (17 of 64), shows
the largest calibration shift as a result -- see below.

## Calibration

**Run date (UTC):** `2026-08-08T03:25:10Z`, against this fixture, via the
brief's Step 2 probe, re-run after adding the `MIN_HORIZON_DAYS` filter
described above:

```
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

Verbatim output:

```
25544 leo-high-drag pairs 73 med_km_per_day 0.393 p95_km 0.521
45098 leo pairs 45 med_km_per_day 1.609 p95_km 6.813
28474 meo-deep-space pairs 29 med_km_per_day 0.241 p95_km 0.578
41866 geo-deep-space pairs 86 med_km_per_day 0.562 p95_km 0.543
25485 heo-high-eccentricity pairs 11 med_km_per_day 0.344 p95_km 2.977
69702 decaying pairs 47 med_km_per_day 11.898 p95_km 41.019
```

A companion probe (same harness, adding the 95th percentile of per-pair
velocity error) gave the single worst velocity p95 across all six
satellites: `69702 decaying vel_p95_km_s 0.04608` (all others well below
this).

All post-filter pair counts are still comfortably above the
`assert len(rows) >= 9` floor in `test_every_satellite_within_envelope`
(minimum is 11, for `25485`/MOLNIYA, which started at 12 raw pairs and lost
1 to the filter); the floor was left unchanged.

All statistics remain within the expected physical ballpark from the task
brief: LEO medians in the 0.4-2 km/day range, deep-space regimes below
1 km/day, and the decaying object (69702, ELECTRON R/B, perigee ~175 km
and dropping) is still the clear worst case, now ~12 km/day median / 41 km
p95 (down from ~22 km/day median pre-filter, as expected since the filtered
pairs were inflating exactly this regime's per-day statistic; the p95 rose
slightly because removing many small-position degenerate pairs shifts the
percentile rank of the remaining, genuinely larger errors) -- consistent
with drag mismodeling near decay, not a units or epoch bug (tsince is
minutes; both states are TEME at the same instant per `_pair_errors`). No
step-change affecting the calibrated statistics was observed in any
regime's error sequence; a single 26.022 km outlier pair exists in the
Starlink (`45098`, leo) history (horizon 1.186 days, versus 6.813 km at the
95th percentile for that regime), but it is an isolated pair, not a
step-shaped shift in the sequence, and does not dominate the regime's
median.

**Envelope derivation** (fixed rule: 1.5x the measured statistic, rounded
up to one significant figure):

| Regime | Measured med (km/day) | x1.5 | Envelope | Measured p95 (km) | x1.5 | Envelope |
|--------|------------------------|------|----------|--------------------|------|----------|
| `leo-high-drag` (25544) | 0.393 | 0.590 | 0.6 | 0.521 | 0.782 | 0.8 |
| `leo` (45098) | 1.609 | 2.414 | 3.0 | 6.813 | 10.220 | 20.0 |
| `meo-deep-space` (28474) | 0.241 | 0.362 | 0.4 | 0.578 | 0.867 | 0.9 |
| `geo-deep-space` (41866) | 0.562 | 0.843 | 0.9 | 0.543 | 0.815 | 0.9 |
| `heo-high-eccentricity` (25485) | 0.344 | 0.516 | 0.6 | 2.977 | 4.466 | 5.0 |
| `decaying` (69702) | 11.898 | 17.847 | 20.0 | 41.019 | 61.529 | 70.0 |

The `leo` p95 envelope loosens (10.0 -> 20.0) rather than tightens: the
filtered measured p95 (6.813 km) x1.5 = 10.22, which crosses the 10 km
decade boundary, so the ceiling-to-one-significant-figure rule carries it
up to 20.0. This is the fixed, mechanical rule applied verbatim, not
hand-tuning -- most regimes tighten (as expected, since the pollution was
upward), but this one measured statistic happened to move just past a
rounding boundary.

Velocity envelope (single rail, worst regime across all six): measured
worst p95 `0.04608` km/s (69702, decaying) x1.5 = `0.06912` -> rounded up
to one significant figure -> `0.07` km/s (unchanged from the pre-filter
calibration).

These envelopes and the measured basis are also recorded as a comment in
`tests/validation/test_tle_self_prediction.py`.

## Long-horizon calibration

**Run date (UTC):** `2026-08-09T01:47:39Z`, against this fixture, via the
Task 1 brief's Step 2 probe (`tests/validation/test_tle_self_prediction.py`'s
`_all_pair_errors`/`_binned_medians`, all-pairs harness at horizons >= 0.5
day, binned into `1d [0.5,1.5)`, `3d [1.5,4.5)`, `7d [4.5,9.5)`,
`14d [9.5,18.5)`, `28d [18.5,31.0)`):

```
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

Verbatim output:

```
25544 leo-high-drag pairs 3010 1d:med=0.410,n=214 3d:med=2.946,n=572 7d:med=25.735,n=805 14d:med=171.537,n=985 28d:med=276.188,n=434
45098 leo pairs 1394 1d:med=1.462,n=99 3d:med=7.708,n=296 7d:med=61.285,n=395 14d:med=410.870,n=439 28d:med=723.109,n=165
28474 meo-deep-space pairs 546 1d:med=0.250,n=38 3d:med=0.675,n=112 7d:med=2.234,n=112 14d:med=9.353,n=175 28d:med=24.575,n=109
41866 geo-deep-space pairs 3893 1d:med=0.283,n=282 3d:med=0.934,n=770 7d:med=4.782,n=1034 14d:med=16.713,n=1288 28d:med=33.597,n=519
25485 heo-high-eccentricity pairs 76 1d:EMPTY 3d:med=0.874,n=13 7d:med=2.934,n=23 14d:med=11.646,n=26 28d:med=36.299,n=14
69702 decaying pairs 2015 1d:med=16.499,n=151 3d:med=188.210,n=370 7d:med=1161.768,n=434 14d:med=6441.801,n=617 28d:med=8467.683,n=443
```

A companion probe (same harness, Spearman rank correlation between horizon
and per-pair position error, `scipy.stats.spearmanr`) gave, per satellite:

```
25544 leo-high-drag n 3010 rho 0.8532
45098 leo n 1394 rho 0.8764
28474 meo-deep-space n 546 rho 0.9723
41866 geo-deep-space n 3893 rho 0.9001
25485 heo-high-eccentricity n 76 rho 0.9393
69702 decaying n 2015 rho 0.7768
```

None of the six satellites' rho is anywhere near the brief's GEO gate
(`rho <= 0.5`); GOES-16 (41866, geo-deep-space) measured `0.9001`, so
station-keeping does not flatten its horizon/error relationship in this
fixture. That gate is not the blocker below.

**Envelope derivation** (fixed rule: 1.5x the measured median, rounded up
to one significant figure; cells with `n < MIN_PAIRS_PER_BIN` (5) are
listed as sparse, not asserted):

| Regime (NORAD) | Bin | Measured median (km) | n | x1.5 | Envelope (km) |
|---|---|---:|---:|---:|---:|
| `leo-high-drag` (25544) | 1d | 0.410 | 214 | 0.615 | 0.7 |
| `leo-high-drag` (25544) | 3d | 2.946 | 572 | 4.419 | 5 |
| `leo-high-drag` (25544) | 7d | 25.735 | 805 | 38.603 | 40 |
| `leo-high-drag` (25544) | 14d | 171.537 | 985 | 257.306 | 300 |
| `leo-high-drag` (25544) | 28d | 276.188 | 434 | 414.282 | 500 |
| `leo` (45098) | 1d | 1.462 | 99 | 2.193 | 3 |
| `leo` (45098) | 3d | 7.708 | 296 | 11.562 | 20 |
| `leo` (45098) | 7d | 61.285 | 395 | 91.928 | 100 |
| `leo` (45098) | 14d | 410.870 | 439 | 616.305 | 700 |
| `leo` (45098) | 28d | 723.109 | 165 | 1084.664 | 2000 |
| `meo-deep-space` (28474) | 1d | 0.250 | 38 | 0.375 | 0.4 |
| `meo-deep-space` (28474) | 3d | 0.675 | 112 | 1.013 | 2 |
| `meo-deep-space` (28474) | 7d | 2.234 | 112 | 3.351 | 4 |
| `meo-deep-space` (28474) | 14d | 9.353 | 175 | 14.030 | 20 |
| `meo-deep-space` (28474) | 28d | 24.575 | 109 | 36.863 | 40 |
| `geo-deep-space` (41866) | 1d | 0.283 | 282 | 0.425 | 0.5 |
| `geo-deep-space` (41866) | 3d | 0.934 | 770 | 1.401 | 2 |
| `geo-deep-space` (41866) | 7d | 4.782 | 1034 | 7.173 | 8 |
| `geo-deep-space` (41866) | 14d | 16.713 | 1288 | 25.070 | 30 |
| `geo-deep-space` (41866) | 28d | 33.597 | 519 | 50.396 | 60 |
| `heo-high-eccentricity` (25485) | 1d | -- | 0 | -- | SPARSE |
| `heo-high-eccentricity` (25485) | 3d | 0.874 | 13 | 1.311 | 2 |
| `heo-high-eccentricity` (25485) | 7d | 2.934 | 23 | 4.401 | 5 |
| `heo-high-eccentricity` (25485) | 14d | 11.646 | 26 | 17.469 | 20 |
| `heo-high-eccentricity` (25485) | 28d | 36.299 | 14 | 54.449 | 60 |
| `decaying` (69702) | 1d | 16.499 | 151 | 24.749 | 30 |
| `decaying` (69702) | 3d | 188.210 | 370 | 282.315 | 300 |
| `decaying` (69702) | 7d | 1161.768 | 434 | 1742.652 | 2000 |
| `decaying` (69702) | 14d | 6441.801 | 617 | 9662.702 | measured, **unasserted** (>= 5,000 km ceiling) |
| `decaying` (69702) | 28d | 8467.683 | 443 | 12701.525 | measured, **unasserted** (>= 5,000 km ceiling) |

Sparse bins (n < 5, recorded not asserted): `("25485", "1d")` (0 pairs --
Molniya's ~13-TLE, sparsely-updated history has no pair with a horizon in
`[0.5, 1.5)` days).

**Honesty paragraph.** These are TLE-predictability envelopes, not pure
propagator fidelity: for operated satellites the long-horizon curve folds
in real maneuvers over the window (station-keeping burns for GOES-16
(`41866`, geo-deep-space) and the Molniya (`25485`,
heo-high-eccentricity); orbit-maintenance reboosts for the ISS (`25544`,
leo-high-drag); constellation-keeping adjustments for the Starlink
(`45098`, leo)), and the decaying object's curve (`69702`, ELECTRON R/B) is
drag-event dominated -- SGP4's static B* term cannot track a perigee that
is itself falling over the 28-day window, so the propagator's own error
compounds with the real, accelerating decay of the object. Separately, the
per-bin `n` counts are pair counts, not independent samples: the all-pairs
harness means every pair shares its base or target TLE with many other
pairs in the same bin (and, since near-duplicate epochs are not filtered
out of this harness, some pairs share both), so `n` overstates the
effective independent sample size relative to a naive reading of the
table.

**Sanity gate result at initial measurement: BLOCKED, then resolved by
controller ruling.** The Step 2 brief's original hard gate flagged any cell
at "thousands of km anywhere" as a probable harness bug rather than a value
to calibrate around. Three `decaying` (69702) cells crossed that line: 7d
median 1161.768 km (n=434), 14d median 6441.801 km (n=617), and 28d median
8467.683 km (n=443) -- the last exceeds LEO's own orbital radius
(~6900 km). Per satellite, all five bins grow monotonically with horizon
(no bin's median is smaller than an earlier bin's, so the other half of
the hard gate did not fire), and the harness was checked against the
brief's code verbatim: `truth_cache` is keyed and populated once per `j`
before the inner loop over `i` (`sgp4_propagate(tles[j], 0.0).r`, the
successor's own epoch), and the tsince argument passed to the predictor is
`horizon * MINUTES_PER_DAY` with `MINUTES_PER_DAY = 1440.0` -- no
truth-cache aliasing or unit-conversion bug found. The measured Spearman
rho for every satellite, including `69702` (`0.7768`), comfortably clears
the brief's separate GEO gate (`rho > 0.5`), so growth-with-horizon was not
in question -- only the absolute magnitude of the `decaying` regime's 7d,
14d, and 28d medians. This was reported BLOCKED with the measured numbers
(`.superpowers/sdd/2026-08-08-sgp4-long-horizon/task-1-report.md`) rather
than calibrated around, per the calibration discipline.

**Controller ruling (2026-08-08,
`docs/superpowers/specs/2026-08-08-sgp4-long-horizon-design.md`,
"Vacuousness ceiling" paragraph):** the measurements are accepted as
genuine physics, not a harness bug -- the "thousands of km means a bug"
heuristic was wrong for a regime whose whole point is measuring
near-decay predictability collapse. The ruling replaces that heuristic
with a **vacuousness ceiling**: a cell is asserted only if its derived
envelope is below 5,000 km, comfortably under the ~13,100 km geometric
maximum position error above which an assertion provably cannot fail
(a bound that wide would always pass, regardless of what SGP4 actually
does). Under this rule:

- `decaying` (69702) 1d/3d/7d are asserted with the unchanged derivation
  rule (30.0 / 300.0 / 2000.0 km, all below the ceiling).
- `decaying` (69702) 14d/28d are recorded here as **measured, not
  asserted** -- their derived envelopes (9,662.702 -> 10,000 km and
  12,701.525 -> 20,000 km respectively) both exceed the 5,000 km ceiling.
  The saturation is physical: ELECTRON R/B's perigee (~175 km per the
  Satellites section above) is itself falling within the 14-28 day
  propagation window, so predicted and true along-track position phase
  apart by a sizeable fraction of one orbit -- the error is approaching
  orbit-scale geometry, not diverging from a units or harness bug.
  `tests/validation/test_tle_self_prediction.py` tracks this exemption as
  `UNASSERTED_CELLS = {("69702", "14d"), ("69702", "28d")}` plus a
  `VACUOUSNESS_CEILING_KM = 5000.0` constant, and
  `test_binned_medians_within_envelope` re-derives each unasserted cell's
  envelope at test time and asserts it is still >= the ceiling, so the
  exemption self-documents and cannot silently grow to cover a
  regression.
- All other regimes' cells are unaffected: every other derived envelope
  in the table above is well under the ceiling (worst case is `leo`
  (45098) 28d at 2,000 km).
