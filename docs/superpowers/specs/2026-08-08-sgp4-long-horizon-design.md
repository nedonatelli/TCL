# SGP4 Long-Horizon Accuracy Envelopes

**Date:** 2026-08-08
**Status:** Approved
**Builds on:** `test/satellite-tle-validation` (PR #103) — extends
`tests/validation/test_tle_self_prediction.py` and its fixture in place.

## Problem

The self-prediction test scores ~1-day horizons (consecutive TLE pairs), so
its envelopes are regression rails, not general SGP4 accuracy statements.
The fixture already contains every horizon from hours to ~29 days; nothing
uses that depth.

## Deliverable

One new test class in `tests/validation/test_tle_self_prediction.py`
(`TestLongHorizon`) plus a calibration extension in
`tests/fixtures/tle/SOURCES.md`. No new files, no recapture, no new
dependencies, no network.

### Pair generation

For each satellite: all ordered TLE pairs (i, j), i < j, propagating TLE_i
to TLE_j's epoch and scoring against TLE_j at tsince = 0 (same TEME-of-date
comparability as the existing test). The existing near-duplicate filter
(`horizon < 0.01 d`) applies to the base and target selection. Roughly
8,000 pairs across the fixture; runtime must stay in single-digit seconds.

### Binning

Horizon bins (days): [0.5, 1.5), [1.5, 4.5), [4.5, 9.5), [9.5, 18.5),
[18.5, 31). Labelled 1d / 3d / 7d / 14d / 28d. Pairs below 0.5 d are the
existing short-horizon test's territory and are excluded here.

### Assertions

1. **Per-regime, per-bin envelopes on the median position error (km)**,
   calibrated by the unchanged fixed rule: 1.5x the measured median,
   rounded UP to one significant figure. Bins with fewer than 5 pairs for
   a satellite are recorded in SOURCES.md but not asserted (small-sample
   noise), and the test asserts the *expected* sparse bins stay sparse
   rather than silently skipping (a wrong bin edge would otherwise
   silently empty a bin).

   **Vacuousness ceiling (ruling, 2026-08-08):** a cell is asserted only
   if its derived envelope is below 5,000 km — comfortably under the
   ~13,100 km geometric maximum position error, above which an assertion
   provably cannot fail. Measured during implementation: the decaying
   object's 14d/28d medians (6,442/8,468 km) saturate toward orbit-scale
   geometry, so those cells are recorded in SOURCES.md as
   measured-but-unasserted with the saturation explanation; its 1d/3d/7d
   cells remain asserted, and the Spearman growth test covers all
   horizons for every satellite.
2. **Rank correlation (Spearman) between horizon and position error > 0.5
   per satellite** over all its pairs — the spec-original growth assertion,
   now statistically meaningful with 0.5-29-day spread. scipy.stats is
   already a core dependency.
3. The existing short-horizon tests are untouched.

### Honesty requirements

- SOURCES.md's new "Long-horizon calibration" section records the verbatim
  per-regime, per-bin measured medians and pair counts with run date, and
  states explicitly that long-horizon error for operated satellites
  (Starlink, ISS, GOES) includes maneuvers — these envelopes measure TLE
  predictability of the object, not pure propagator fidelity; the decaying
  object's curve is drag-event dominated.
- The binned tables are the documented "general SGP4 accuracy" statement
  for pytcl; the test module docstring points to them.

## Out of scope

- Wider/longer captures, population-scale fixtures, official AIAA test
  vectors (separate candidates).
- Any change to the short-horizon envelopes or filter.

## Testing/verification

The new class must pass alongside the existing four tests; full validation
file runtime stays under ~10 s. Calibration discipline identical to the
existing sections; a reviewer must be able to reproduce every number from
the committed fixture.
