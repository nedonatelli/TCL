# v2.2.0 — Results I/O (polars + msgspec + AIS + ASDF)

**Date:** 2026-08-11
**Status:** Approved
**Seed:** `docs/superpowers/specs/2026-08-06-modernization-campaign-design.md`
("v2.2.0 — Results I/O"), expanded by decision to include AIS decoding and
ASDF support. This spec supersedes and details the seed.

## Scope decisions (approved)

- AIS decoding **wraps pyais** — paralleling the MATLAB TCL, whose
  `decodeAISString` is itself a MEX wrapper around libais. No native
  decoder.
- **msgspec becomes a core dependency** (tiny, zero-dep; v2.3.0's typed
  configs need it core regardless). polars, pyais, and asdf land as
  granular extras: `[dataframe]`, `[ais]`, `[asdf]`, all added to `[all]`.
- A **real AIS capture** is vendored and validated, mirroring the ADS-B
  pattern.
- Ride-alongs: HDF5 compression work, ADS-B ingest dogfooding, the GEBCO
  synthetic `.nc` fixture.
- **Documentation and references are a first-class deliverable** with an
  explicit page inventory (below), not a rebuild-only step.

## Boundary rule

The compute core stays numpy/scipy/numba. polars appears only at the I/O
boundary: no algorithm takes or returns a DataFrame, and no module outside
`pytcl/io/` and `pytcl/transponders/` imports polars. Reviews enforce this.

## Deliverables

### 1. Ingest readers — `pytcl/io/readers.py`

Load measurement/detection files into tracker-ready numpy: CSV and Parquet
in, `(times, measurements-per-scan, optional ids/metadata)` out. polars is
the parsing engine; the public API returns numpy arrays and plain Python
structures only. Column-mapping is explicit (caller names the columns for
time/position/etc.); no schema guessing.

### 2. DataFrame accessors — `pytcl/io/dataframes.py`

`to_polars()` for track histories and performance-evaluation outputs.

Track-history schema: **long format**, one row per (track, scan):
`track_id`, `t`, `status`, `state` (List[Float64]), `covariance`
(List[Float64], row-major flattened). A helper explodes position/velocity
components into wide columns when the caller states the layout (e.g.
`[x, vx, y, vy]`). Performance-evaluation outputs map to natural flat
tables. Being Arrow-native, this delivers the roadmap's Parquet and Arrow
bullets (`df.write_parquet`, Arrow interchange) without further code.

### 3. msgspec serialization — `pytcl/io/serialize.py`

Track histories, filter states, and covariances to/from JSON and
MessagePack via msgspec-typed structures, with validated decode.

Fidelity contract (tested, not aspirational):
- MessagePack: bitwise float64 round-trip, including NaN/inf.
- JSON: bitwise for finite float64; encoding non-finite values **raises**
  with a clear message (never silently nulled).
- Covariances round-trip symmetric because the full matrix is stored.
- Malformed/mistyped input fails loudly at decode (msgspec validation).

### 4. ASDF — `pytcl/io/asdf_io.py` (`[asdf]` extra)

`save_tracks_asdf` / `load_tracks_asdf` covering the same track-history and
state surface as the msgspec layer. Round-trip tested. Skips with
`DependencyError` when asdf is absent.

### 5. AIS — `pytcl/transponders/ais.py` (`[ais]` extra)

New `pytcl/transponders/` package (named for the MATLAB `Transponders/`
folder). Two functions:
- `decode_ais(nmea_text)` — thin wrapper over pyais: NMEA checksum,
  multi-part assembly, message decode; returns normalized records
  (message type, MMSI, fields) for the tracking-relevant types pyais
  supports.
- `ais_position_reports(nmea_text_or_records)` — the
  `decodeAISPosReports2Mat` analog: extracts position-report messages
  (types 1-3, 18, 19) into measurement arrays — MMSI, timestamp (receiver
  time), lat/lon (radians at the API boundary per repo convention),
  SOG, COG, heading — ready for the ingest layer and `to_polars`.

### 6. HDF5 compression (ride-along)

Implement states-only chunking and/or a covariance transform in the HDF5
track storage; **measure** the compression ratio on the existing benchmark
scenario. The deliverable is the measured figure documented wherever the
ratio is claimed, whatever it turns out to be — target the once-claimed
5-10x, but a smaller honest number with the mechanism documented also
closes the roadmap item. No unmeasured claims (see the vacuous-claims
lesson in the project's audit history).

### 7. Fixtures and real-data validation

- **ADS-B dogfood (REFERENCE):** the existing vendored ADS-B fixture is
  loaded through the new reader; the same tracking chain must reproduce
  the existing `test_adsb_tracking.py` results (same estimates, same
  reference agreement).
- **AIS capture (REFERENCE):** a few minutes of live NMEA from the
  Norwegian Coastal Administration's open AIS stream (public TCP feed, no
  credentials), captured during implementation, vendored gzipped under
  `tests/fixtures/ais/` with a SOURCES.md (capture time, endpoint,
  provenance, field notes). Validation mirrors the ADS-B trick: the filter
  sees decoded positions only and its recovered velocity is scored against
  the ships' self-broadcast SOG/COG — an independent reference. Assertion
  envelopes calibrated from the capture by the established fixed rule
  (1.5x measured, ceil to 1 significant figure, basis recorded).
- **GEBCO synthetic `.nc` fixture:** small generated NetCDF checked into
  test fixtures so the GEBCO loader's diagnostics/progress path runs
  end-to-end in CI (closes the roadmap's standing test-debt item).
  Requires netCDF4 only at generation time; generation script kept like
  `fetch_tle_history.py`.

### 8. Documentation and reference sweep (first-class deliverable)

Every page below is updated in the release PRs, and re-read (not just
rebuilt) before the release tag per CONTRIBUTING step 2b:

- `README.md` — extras list gains `[dataframe]`/`[ais]`/`[asdf]`; feature
  overview gains Results I/O; metrics refreshed at release.
- `CLAUDE.md` — extras list updated; version bumped at release.
- `CONTRIBUTING.md` — metrics block at release.
- `ROADMAP.md` — v2.2.0 marked released at release; AIS removed from the
  measured-backlog line; the "Format Support Expansion" bullets
  (Parquet, Arrow, ASDF) marked delivered; next milestone advances to
  v2.3.0.
- `docs/roadmap.rst` — the hand-mirror updated in the same PR as
  ROADMAP.md (mirrors drift; see the post-2.1.0 audit).
- `docs/matlab_parity_inventory.rst` — Transponders/AIS row updated to
  ported-via-pyais with the MATLAB-wraps-libais context.
- `docs/index.rst` — toctree gains the new page; front-page line updated
  at release.
- New `docs/results_io.rst` — readers, schemas, fidelity contract, AIS,
  ASDF, HDF5 compression figures; all code blocks pass the docs gate on a
  machine without optional extras or data files (skip via DependencyError,
  the diagnostics-page lesson).
- `CHANGELOG.md` — entries land with each PR, not at release.
- `pyproject.toml` extras comments; `examples/README.md` if an example is
  added (an ingest-to-tracking demo example is included in scope).

## Out of scope

- `msgspec.Struct` configs and tracker save/restore (v2.3.0's identity).
- Streaming/real-time ingest; live AIS connectivity in the library
  (capture is a one-time script, like the TLE fetch).
- Writing Parquet from inside algorithms; any polars type in a public
  algorithm signature.
- AIS message types beyond pyais's support; AIS transmission.

## Testing summary

Per the REFERENCE/PROPERTY bar: bitwise round-trip property tests
(msgspec, ASDF); reader dogfood equivalence against the ADS-B validation
results; AIS decode against pyais-documented vectors plus the real
capture's end-to-end tracking validation with calibrated envelopes; HDF5
compression ratio asserted against its measured, documented figure;
coverage contract green; full suite under PYTCL_REQUIRE_MLX=1 before each
merge that touches storage or trackers.
