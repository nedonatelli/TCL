# Real AIS traffic off the Norwegian coast

A capture of live AIS position reports, vendored so the tracking stack can be
checked against real ship traffic without a network call. Captured with
`scripts/capture_ais.py` (stdlib socket client, no credentials).

These are the reference side of a REFERENCE-class test (see
`tests/validation/README.md`). Nothing in the library reads this fixture at
runtime; only `tests/validation/test_ais_tracking.py` does.

| File | Contents |
|------|----------|
| `ais_norway.nmea.gz` | 300 ships, 6,831 NMEA lines (position reports only), min 22 / max 24 reports per ship. See "Fixture trimming" below. |

## Endpoint

Norwegian Coastal Administration (Kystverket) open AIS relay:
`153.44.253.27:5631`, raw `!BSVDM`/`!AIVDM` sentences aggregated from over 50
national base stations, covering 40-60 nm off the Norwegian coast. Public,
unauthenticated access under the Norwegian Licence for Open Government Data
(NLOD) -- confirmed via <https://www.kystverket.no/en/sea-transport-and-ports/ais/access-to-ais-data/>.
Per that page, the unauthenticated feed excludes fishing vessels under 15 m
and recreational craft under 45 m; registered access can additionally
request custom update-rate filters, which this capture did not use.

## Capture windows (UTC)

Connecting and capturing is a genuinely one-time, sanctioned network step;
run in four separate TCP sessions rather than one because the busiest
per-vessel report rate this endpoint delivers (see "Independence and the
per-vessel report interval" below) meant the first session fell well short
of the fixture's sanity bar and the sessions afterward needed to add
capture time before that bar was cleared:

| Session | Start | End | Duration |
|---|---|---|---|
| 1 | 2026-08-11T19:22:42Z | 2026-08-11T19:27:42Z | 299.5 s |
| 2 | 2026-08-11T19:52:31Z | 2026-08-11T20:07:31Z | 899.4 s |
| 3 | 2026-08-11T20:10:28Z | 2026-08-11T20:20:25Z | 597.0 s (session ended early; environment/session boundary, not an endpoint failure -- see below) |
| 4 | 2026-08-11T21:28:28Z | 2026-08-11T21:43:27Z | 899.2 s |

Total connected capture time: 2,695.1 s (~44.9 min) across the four
sessions, spanning a wall-clock window of 2026-08-11T19:22:42Z to
2026-08-11T21:43:27Z. Session 3 stopped short of its requested 2,700 s
duration -- the capture process was killed by an environment/session
recycling event unrelated to `scripts/capture_ais.py` or the endpoint (it
was still receiving data at the last line written); a fourth session was
run afterward to make up the shortfall. The raw per-session outputs were
concatenated in chronological order (sessions do not overlap in time) to
build the working capture this fixture and its calibration are drawn from.

## Message-type histogram (full four-session capture, pre-trim)

109,770 raw lines decoded to 71,619 AIS messages via `decode_ais`
(38,151 lines did not decode to a complete message -- either an
undecodable/malformed sentence, or one half of a multipart message whose
other fragment fell outside the batch; both are expected and are the same
"skip, don't fail" behavior `decode_ais` documents):

| Type | Count | Meaning |
|---|---:|---|
| 1 | 38,991 | Class A position report |
| 3 | 16,569 | Class A position report (response to interrogation) |
| 18 | 5,901 | Class B position report |
| 19 | 2 | Class B extended position report |
| 21 | 2,077 | Aid-to-navigation report |
| 24 | 7,774 | Class B static data (2-part) |
| 8 | 303 | Binary broadcast message |
| 6 | 2 | Binary addressed message |

4,145 distinct MMSIs carried at least one position report (types 1, 2, 3,
18, 19; no type-2 reports were seen in this capture); 61,263 total position
reports with non-sentinel lat/lon across those MMSIs. 1,617 MMSIs had 20 or
more such reports -- the fixture's ship selection (below) draws from this
pool.

## Fixture trimming

Following `tests/fixtures/adsb/SOURCES.md`'s precedent ("trimmed to the 120
aircraft with the most reports, to keep the file small"): `ais_norway.nmea.gz`
keeps only the 300 MMSIs with the most position reports across the full
capture, and only their position-report lines (types 1, 3, 18, 19; static/
voyage and other non-position message types are dropped -- the validation
test never reads them, and dropping them also sidesteps needing to preserve
multipart-message line pairing under the trim). Kept-ship report counts
range from 22 to 24 (of a possible 24, since 4 sessions x ~1 report/session
is this endpoint's ceiling for the busiest vessels over the capture window --
see below). This reduces the fixture from a 10.6 MB raw concatenation (four sessions,
all message types) to a 192 KB gzip.

One kept MMSI (257432500) has all 23 of its reports carrying the ITU-R
M.1371 "position not available" sentinel (lat 91 deg / lon 181 deg) --
`ais_position_reports` correctly converts these to NaN, and
`test_ais_tracking.py`'s loader drops rows with NaN lat/lon, so this ship
contributes 0 usable reports and is silently excluded from tracking. Left
in rather than reselected around: it is a real, unremarkable edge case (a
Class A station broadcasting without a GPS fix), and the other 299 kept
ships comfortably clear the sanity bar on their own.

## Sanity bar (from the task brief: >= 50 MMSIs with >= 20 position reports)

Measured against the committed, trimmed fixture, via
`tests/validation/test_ais_tracking.py`'s own `_load()`:

```
n ships (position-report MMSIs, NaN-filtered): 299
n ships with >= 20 reports: 299
max reports for one ship: 24   mean: 22.77   median: 23
total usable position reports: 6,808
```

299 ships clear the >= 50-with->= 20-reports bar by close to 6x.

## Independence and the per-vessel report interval

Each ship's speed over ground (SOG) is broadcast by the ship's own
transponder, derived from its GNSS receiver's own velocity solution (many
units report a Doppler-derived SOG independent of position differencing).
The tracking test's Kalman filter sees positions only and estimates
velocity purely from consecutive `geodetic2enu` fixes -- it is never given
SOG. Agreement between the filter's position-only estimate and the ship's
independently-broadcast SOG is therefore a reference measurement, the same
argument `tests/fixtures/adsb/SOURCES.md` makes for ADS-B ground speed.

A characteristic of this specific endpoint shaped how the capture was run:
across all four sessions, no single MMSI produced more than one position
report roughly every 112-130 seconds, regardless of session length (the
5-minute session topped out at 3 reports per ship, the 15-minute session at
8, and the four-session total at 24 -- consistent with a report ceiling of
roughly (connected seconds) / 120 per vessel, not with organic Class A
reporting rates of 2-10 s while underway). This was verified as a genuine
property of the endpoint, not a decoding bug: raw sentences differ
byte-for-byte between consecutive reports for the same MMSI, their
NMEA-v4 tag-block `c:` timestamps (see below) advance in step, and the
aggregate line rate across the whole capture (~40 lines/s) matches
independently-reported throughput for this endpoint (a third-party blog
using the same endpoint recorded ~32 msg/s over a live hour). The Kystverket
access page confirms registered users can request a custom "update rate"
filter, implying the free/unauthenticated tier applies its own -- most
likely to bound bandwidth for a national, 50-base-station, unauthenticated
public feed. This is why the capture ran in four sessions rather than one
5-minute pull: the fixture's `>= 20 reports/ship` bar needed roughly 20 x
120 s = 2,400 s of connected time, not the ~15 minutes a naive
back-of-envelope estimate (assuming ordinary Class A reporting rates) would
have suggested.

## Tag-block timestamps

Every line from this endpoint carries an NMEA-v4 tag block before the
sentence, e.g. `\s:2573145,c:1786475628*0D\!BSVDM,...` -- `s:` is the
source (base station) ID and `c:` is that station's own UTC capture
timestamp (Unix epoch seconds). `decode_ais` passes these through to pyais
transparently (verified: `pyais.decode`/`IterMessages` strip the tag block
before parsing the six-bit payload, so tag-block-prefixed lines decode
identically to bare sentences).

Per the task brief, `scripts/capture_ais.py` stamps each line with this
script's own receiver time (`time.time()` on receipt), not the tag block's
`c:` field, and `tests/validation/test_ais_tracking.py` uses that receiver
timestamp throughout (for report ordering, `dt` between updates, and the
capture-window arithmetic above). The two are close in practice --
spot-checking the first captured line, receiver time and `c:` differ by
under 2 seconds -- but the `c:` field is not read by any code in this repo;
it is documented here only because it is present in every raw line and a
future reader may wonder whether it is the timestamp in use.

## Calibration

**Run date (UTC):** `2026-08-11T22:23:49Z`, against the committed fixture,
via `tests/validation/test_ais_tracking.py`'s own `_load()`/`_track()`
functions (`POSITION_SIGMA_M=15.0`, `MIN_REPORTS=15`):

```python
import sys; sys.path.insert(0, "tests/validation")
import numpy as np, math
import test_ais_tracking as t

by_ship = t._load()
lats = [r[1] for rows in by_ship.values() for r in rows]
lons = [r[2] for rows in by_ship.values() for r in rows]
centroid = (float(np.mean(lats)), float(np.mean(lons)), 0.0)

scores, speed_errors = [], []
for mmsi, rows in by_ship.items():
    if len(rows) < t.MIN_REPORTS:
        continue
    state, track_scores, sogs = t._track(rows, centroid)
    if state is None or not track_scores:
        continue
    scores.extend(track_scores)
    broadcast = sogs[-3:]
    if broadcast:
        estimated = math.hypot(state[1], state[3])
        speed_errors.append(estimated - float(np.mean(broadcast)))

scores, speed_errors = np.array(scores), np.array(speed_errors)
print("n tracked ships", ...); print("median abs error", float(np.median(np.abs(speed_errors))))
# ... (mean/median NIS, exceedance, etc.)
```

Verbatim output:

```
n tracked ships 299
n scores (updates) 5911
n speed_errors 298
median abs error (m/s) 0.013417792359355304
mean error / bias (m/s) 0.020748700575871974
fraction within 6 m/s 1.0
fraction within 25 m/s 1.0
mean nis 1.9915826232126583
median nis 0.0023515451429340856
exceedance >5.99 0.03265098968025715
abs error percentiles [50,75,90,95,99] [0.01341779 0.0308706  0.07636811 0.1170376  1.19824741]
```

## PROCESS_VAR calibration

Before the run above, `PROCESS_VAR` itself was calibrated by sweeping it
and reading off the value whose mean NIS lands nearest the value a
perfectly-consistent 2-D filter averages, 2.0 (same run date, same `_load`,
`POSITION_SIGMA_M=15.0`):

```
var=1.0e-06 mean_nis=  8.5100 median_nis=0.003900 exceed5.99=0.0599 median_abs_err=0.0078
var=1.0e-05 mean_nis=  1.9916 median_nis=0.002352 exceed5.99=0.0327 median_abs_err=0.0134
var=3.0e-05 mean_nis=  0.9855 median_nis=0.001641 exceed5.99=0.0230 median_abs_err=0.0180
var=1.0e-04 mean_nis=  0.4456 median_nis=0.001021 exceed5.99=0.0135 median_abs_err=0.0256
var=3.0e-04 mean_nis=  0.2101 median_nis=0.000571 exceed5.99=0.0059 median_abs_err=0.0330
var=1.0e-03 mean_nis=  0.0865 median_nis=0.000241 exceed5.99=0.0020 median_abs_err=0.0409
var=3.0e-03 mean_nis=  0.0357 median_nis=0.000101 exceed5.99=0.0010 median_abs_err=0.0442
var=1.0e-02 mean_nis=  0.0137 median_nis=0.000035 exceed5.99=0.0002 median_abs_err=0.0472
```

`var=1e-5` was the closest match to mean NIS 2.0 in this sweep and is what
`PROCESS_VAR` is set to in `test_ais_tracking.py`. This is standard Kalman
filter tuning practice (choosing process noise so the filter's own
predicted uncertainty matches its actual error), not tuning to the
assertions -- the assertions were derived from the resulting distribution
afterward, per the fixed 1.5x-ceil-to-1-sig-fig rule below, not the other
way around.

An earlier, un-calibrated choice (`PROCESS_VAR=0.5`, copied from nothing in
particular as a starting guess) produced a catastrophic result: mean error
55.2 m/s with individual ship errors up to 368 m/s (about 715 knots --
physically impossible for any real vessel). Root-caused before any
calibration was attempted: this fixture concatenates four separately
-connected capture sessions, so a ship seen in two sessions has a multi
-minute hole in its report sequence; `_track`'s original gap-handling
skipped the Kalman update across such a gap but left the pre-gap state and
covariance in place, so the very next (short, ordinary) interval's predict
step explained the ship's real displacement over the whole unobserved gap
as if it had happened within that short interval -- an unbounded velocity
artifact of the multi-session merge, not of the endpoint, the decoder, or
the filter's process-noise value. Fixed by reinitializing state and
covariance on any gap outside `1 s < dt < 900 s` (matching
`test_adsb_tracking.py`'s analogous duplicate/gap filter) instead of
carrying stale state across it. This is a harness bug in the strict sense
of the calibration discipline (a value indicating a defect, not real
physics) and was fixed rather than calibrated around; the `PROCESS_VAR`
sweep above and the envelope below are both post-fix.

## Envelope derivation

**Fixed rule:** 1.5x the measured statistic, rounded up to one significant
figure.

| Statistic | Measured | x1.5 | Envelope |
|---|---:|---:|---:|
| Median \|v_est - SOG\| (m/s) | 0.013418 | 0.020127 | **0.03** |

This is an unusually tight envelope by this project's standards (compare
the ADS-B fixture's 20 m/s). It is tight because the underlying data is
tight, not because of a smaller measurement-noise assumption doing the
work: at this fixture's ~120 s report interval and ~5-8 m/s typical ship
speed, a ship travels roughly 700-900 m between consecutive reports against
15 m of assumed GNSS position noise -- a signal-to-noise ratio around 50:1
per leg, far better than ADS-B's ~5 s polling interval gives at aircraft
cruise speed. Most kept ships are also in steady coastal transit (the
capture's centroid is well offshore of any port), so the constant-velocity
model fits almost exactly leg to leg. The honesty check for this envelope
is the innovation-structure assertions alongside it, not the tightness of
the number itself: mean NIS 1.99 (a textbook-consistent value) with a
non-trivial 3.3% manoeuvre tail confirms the filter is neither
overconfident nor merely coasting on an inflated covariance -- see
`TestInnovationsLookLikeRealShipTraffic` in `test_ais_tracking.py`.

**Caveat for future recaptures.** Because this envelope is tight, a future
recapture of this fixture (different vessels, different transit geometry,
maybe including ships that are anchored or maneuvering more often within
the kept-ship pool) could plausibly measure a somewhat larger median error
without anything being wrong with the tracker -- 0.03 m/s has little slack
to absorb sampling variation across different real traffic. If a recapture
trips `test_median_error_is_small`, re-run the Calibration section above
against the new fixture before assuming a regression.

## Licence

Kystverket AIS data are released under the Norwegian Licence for Open
Government Data (NLOD), which permits redistribution with attribution
(satisfied by this file). See
<https://www.kystverket.no/en/sea-transport-and-ports/ais/access-to-ais-data/>.
If this file is ever removed, `test_ais_tracking.py` reads it through a
helper that skips cleanly when absent, so nothing breaks.
