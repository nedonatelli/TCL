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

## Calibration

<!-- TODO(Task 3): fill in once tests/validation/test_tle_self_prediction.py
     runs against the captured fixture. -->
Filled by test calibration (see `tests/validation/test_tle_self_prediction.py`).
