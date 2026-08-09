# Real-Data Validation: ADS-B and Satellite TLE Self-Prediction

**Date:** 2026-08-08
**Status:** Approved

## Problem

The validation suite's REFERENCE tests compare pytcl against other software
and published constants. Only one test scores the library against physical
reality recorded from the outside world (the ADS-B capture on the unmerged
`test/adsb-real-data-validation` branch), and nothing does so for the
astronomical/SGP4 stack. Real data supplies two things synthetic tests
cannot: an independent reference quantity the code under test never sees,
and error structure (maneuver tails, drag mismodeling) nobody chose.

## Shape: two PRs

The ADS-B work is finished and merely unmerged; the satellite work is new.
They land as separate PRs with separate reviews.

## Part A — ADS-B tracking validation (existing branch)

Rebase `test/adsb-real-data-validation` (single commit fe45811, 2026-08-04)
onto current main. Verify against the v2 codebase: the test's imports
(`geodetic2enu`, `kf_predict`/`kf_update`, `q_discrete_white_noise`, `nis`)
and the public-API coverage contract. Fresh review pass, then the normal PR
flow. No rewrites unless verification finds real breakage; the fixture
(`adsb_boston.json.gz`, 120 aircraft / 3600 reports) and its SOURCES.md are
kept verbatim.

## Part B — satellite TLE self-prediction (new)

### Truth model

For each satellite, take its real TLE history. Propagate TLE_k from its
epoch to TLE_{k+1}'s epoch with pytcl's SGP4/SDP4; evaluate TLE_{k+1} at
its own epoch (tsince = 0) as truth. The successor TLE was fitted by
NORAD/18 SPCS to real tracking observations, so it is an independent
reference for a state the propagation never saw — the orbital-mechanics
analog of the ADS-B broadcast ground speed.

### Fixture

- ~30 days of TLE history for ~6 satellites spanning the SGP4 regimes:
  ISS (LEO, high drag), one Starlink (LEO), one GPS (MEO, deep-space),
  one GOES (GEO, deep-space), one Molniya (high eccentricity, deep-space),
  one low-perigee/decaying object. Exact NORAD IDs chosen at capture time
  and recorded in SOURCES.md.
- Stored gzipped under `tests/fixtures/tle/` with `SOURCES.md` documenting
  capture date, the Space-Track query used, per-satellite rationale, and
  the calibration numbers derived from the capture (see Assertions).
- One-time acquisition script `scripts/fetch_tle_history.py`: reads
  `SPACETRACK_USER` / `SPACETRACK_PASSWORD` from the environment, queries
  the Space-Track gp_history API, writes the fixture. Credentials are never
  stored; the script stays in-repo for reproducibility and is not executed
  by any test or CI job.

### Test: `tests/validation/test_tle_self_prediction.py`

REFERENCE class. Deterministic and offline; skips only if the fixture file
is absent (same convention as the ADS-B test).

Assertions:

- Per orbit class, median and 95th-percentile TEME position error over all
  consecutive pairs, against envelopes calibrated once from the captured
  data and documented (with the calibration values and date) in the test:
  LEO at km-per-day scale, GEO at sub-km scale. Envelopes are recorded
  bounds on observed behavior, not tuned-to-pass thresholds; the test
  comments state the measured values they were derived from.
- Velocity error scored the same way.
- Error grows with prediction horizon (rank correlation over pairs, per
  satellite class) — guards against a propagator that ignores tsince.
- Each satellite exercises its intended code path: `is_deep_space` matches
  the regime the satellite was chosen for.

## Out of scope

- SP3/precise-ephemeris readers.
- Filter-on-orbit tracking scenarios (synthetic measurements from real
  orbits) — candidate follow-up.
- Any live-network access from tests or CI.
- Automated fixture refresh.

## Error handling

The acquisition script fails loudly on missing credentials or HTTP errors
(no retries, no fallbacks). Tests raise nothing new: absent fixture skips,
malformed fixture fails.

## Documentation

- CHANGELOG entries (one per PR) under Unreleased/Added.
- `tests/validation/README.md` oracle list gains "NORAD-fitted successor
  TLEs (real tracking data)" and the ADS-B entry if not already present.
- Both SOURCES.md files carry the full provenance story.
