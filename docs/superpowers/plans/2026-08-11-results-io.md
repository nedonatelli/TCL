# v2.2.0 Results I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Results I/O — polars ingest/accessors, msgspec serialization, AIS decoding (pyais), ASDF export, HDF5 compression with a measured figure, three fixtures, and the full documentation sweep.

**Architecture:** New modules under `pytcl/io/` (`serialize.py`, `dataframes.py`, `readers.py`, `asdf_io.py`) and a new `pytcl/transponders/` package (`ais.py`). msgspec joins core deps; polars/pyais/asdf are granular extras. polars never crosses the I/O boundary. Spec: `docs/superpowers/specs/2026-08-11-results-io-design.md`.

**Tech Stack:** msgspec (core), polars `[dataframe]`, pyais `[ais]`, asdf `[asdf]`, h5py (existing core), numpy, pytest.

## Global Constraints

- Branch: `feat/results-io`.
- Boundary rule: no module outside `pytcl/io/` and `pytcl/transponders/` imports polars; no public algorithm signature takes or returns a DataFrame. Reviewers check this by grep.
- Units at API boundaries: angles in radians; AIS speeds converted to m/s (broadcast knots noted in docstrings).
- Optional deps degrade via `DependencyError` (`pytcl.core.exceptions`), following the existing guarded-import pattern (see `pytcl/terrain/loaders.py` for the model); tests for optional features use the repo's `_data_skip`/importorskip conventions, never bare `except Exception`.
- Fidelity contract (tested): MessagePack bitwise float64 incl. NaN/inf; JSON bitwise for finite, RAISES on non-finite; decode validates types and fails loudly.
- No unmeasured performance/compression claims anywhere — a number appears in docs only as the output of a command that the docs/tests can reproduce.
- Style: ruff 88, NumPy docstrings with platform-robust doctests; `uv run` from repo root with `export PATH="$HOME/.local/bin:$PATH"`; prek hook on commit; explicit `git add` paths, never `-A`; commits end with the Co-Authored-By trailer (Claude Fable 5 <noreply@anthropic.com>).
- Background/watcher shells must never `git checkout` (shared-worktree hazard; see workflow memory).
- New public functions land at REFERENCE or PROPERTY class; run the coverage contract after adding exports and follow its messages.

## Verified anchors

- `Track` NamedTuple: `pytcl/trackers/multi_target.py` (~25) — fields `id`, `state`, `covariance`, `status` (TrackStatus enum with `.value`).
- HDF5 track storage: `pytcl/io/hdf5_track_storage.py` — already takes `compression`/`compression_level` (gzip default); the compression work happens here. Generic array storage: `pytcl/io/hdf5_storage.py`.
- ADS-B fixture: `tests/fixtures/adsb/adsb_boston.json.gz` — gzipped JSON, list of aircraft each with position reports (`t`, lat/lon, alt, broadcast ground speed); consumed by `tests/validation/test_adsb_tracking.py` (`geodetic2enu` -> `kf_predict`/`kf_update` chain, `POSITION_SIGMA_M = 50.0`, `PROCESS_VAR = 5.0`).
- Norwegian Coastal Administration open AIS stream: TCP `153.44.253.27:5631`, raw NMEA `!BSVDM`/`!AIVDM` sentences, no credentials (verify liveness at capture; if the endpoint has moved, kystverket.no documents the current one).
- GEBCO loader diagnostics path: `pytcl/terrain/loaders.py` (`_find_gebco_file`, `parse_gebco_netcdf` start/finish DEBUG pair); needs a tiny synthetic `.nc` to execute in CI.
- Extras pattern in `pyproject.toml`: `[project.optional-dependencies]` with commented rationale; `all` aggregates user-facing extras.
- The `examples` CI job runs `uv sync --locked --extra all`, so anything the new example imports must be reachable from `[all]` or the dev group.

---

### Task 1: Dependencies + msgspec serialization (`pytcl/io/serialize.py`)

**Files:**
- Modify: `pyproject.toml` (msgspec>=0.18 to `[project].dependencies`; extras `dataframe = ["polars>=1.0"]`, `ais = ["pyais>=2.5"]`, `asdf = ["asdf>=3.0"]`; add all three to `all`), `uv.lock`
- Create: `pytcl/io/serialize.py`, `tests/unit/test_io_serialize.py`
- Modify: `pytcl/io/__init__.py` (exports, following its existing pattern)

**Interfaces:**
- Produces (consumed by Tasks 4/6/9):

```python
class TrackRecord(msgspec.Struct):
    track_id: int
    t: float
    status: str            # TrackStatus .value
    state: list[float]
    covariance: list[float]  # row-major, len == len(state)**2

def encode_tracks(history, times, fmt="msgpack") -> bytes
    # history: list of per-scan lists of Track-like objects (id/state/
    # covariance/status attrs); times: sequence of scan timestamps.
def decode_tracks(data: bytes, fmt="msgpack") -> tuple[list[float], list[list[SimpleTrack]]]
    # SimpleTrack: NamedTuple(id, state: NDArray, covariance: NDArray (n,n), status: str)
def encode_states(x, P, fmt="msgpack") -> bytes
def decode_states(data, fmt="msgpack") -> tuple[NDArray, NDArray]
```

- [ ] **Step 1: Deps.** Edit pyproject as above with one-line comments per extra (what it enables); `uv lock && uv sync --extra all --quiet`. (polars/pyais/asdf install now so later tasks and the examples CI path have them; msgspec is core.)

- [ ] **Step 2: Failing tests.** `tests/unit/test_io_serialize.py`:

```python
"""msgspec serialization: bitwise round-trips and loud failures."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from pytcl.io.serialize import (
    decode_states,
    decode_tracks,
    encode_states,
    encode_tracks,
)


def _history():
    from pytcl.trackers import MultiTargetTracker

    rng = np.random.default_rng(3)
    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=np.array(
            [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float
        ),
        H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
        Q=np.eye(4) * 0.01,
        R=np.eye(2) * 1.0,
        confirm_hits=1,
    )
    history, times = [], []
    for k in range(6):
        z = [np.array([k + rng.normal(0, 0.3), k + rng.normal(0, 0.3)])]
        history.append(tracker.process(z, dt=1.0))
        times.append(float(k))
    return history, times


class TestTrackRoundTrip:
    @pytest.mark.parametrize("fmt", ["msgpack", "json"])
    def test_bitwise_round_trip(self, fmt):
        history, times = _history()
        blob = encode_tracks(history, times, fmt=fmt)
        times2, history2 = decode_tracks(blob, fmt=fmt)
        assert times2 == times
        assert len(history2) == len(history)
        for scan, scan2 in zip(history, history2):
            for tr, tr2 in zip(scan, scan2):
                assert tr2.id == tr.id
                assert tr2.status == tr.status.value
                assert_array_equal(tr2.state, np.asarray(tr.state, dtype=np.float64))
                assert_array_equal(tr2.covariance, np.asarray(tr.covariance))
                assert tr2.covariance.shape == (len(tr2.state), len(tr2.state))

    def test_msgpack_preserves_non_finite(self):
        # tobytes() compares raw float64 bit patterns, so NaN == NaN here.
        x = np.array([1.0, np.nan, np.inf, -np.inf])
        P = np.eye(4)
        x2, P2 = decode_states(encode_states(x, P, fmt="msgpack"), fmt="msgpack")
        assert x2.tobytes() == x.tobytes()
        assert P2.tobytes() == P.tobytes()

    def test_json_raises_on_non_finite(self):
        with pytest.raises(ValueError, match="non-finite"):
            encode_states(np.array([np.nan]), np.eye(1), fmt="json")

    def test_json_finite_bitwise(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=8)
        P = rng.normal(size=(8, 8))
        x2, P2 = decode_states(encode_states(x, P, fmt="json"), fmt="json")
        assert x.tobytes() == x2.tobytes()
        assert P.tobytes() == P2.tobytes()

    def test_malformed_decode_fails_loudly(self):
        with pytest.raises(Exception) as excinfo:
            decode_tracks(b'{"nonsense": true}', fmt="json")
        assert (
            "nonsense" in str(excinfo.value)
            or "Object" in str(excinfo.value)
            or "missing" in str(excinfo.value)
        )

    def test_unknown_fmt_raises(self):
        with pytest.raises(ValueError, match="fmt"):
            encode_states(np.zeros(2), np.eye(2), fmt="pickle")
```

- [ ] **Step 3: RED** (`uv run pytest tests/unit/test_io_serialize.py -x -q` → ImportError).

- [ ] **Step 4: Implement.** msgspec Structs (`TrackRecord`, `TrackSet` with `times` + `scans: list[list[TrackRecord]]`, `StateRecord` with `x`/`p_flat`); `fmt` dispatch to `msgspec.json`/`msgspec.msgpack` encoders/decoders. JSON non-finite policy: pre-scan arrays with `np.isfinite`; raise `ValueError("... non-finite ...")` before encoding. Decode rebuilds numpy arrays (`np.asarray(..., dtype=np.float64)`, covariance reshaped `(n, n)`), returns `SimpleTrack` NamedTuples. NumPy docstrings with a doctest each (round-trip a 1-track history; `round()` outputs).

- [ ] **Step 5: GREEN + doctests** (`uv run pytest tests/unit/test_io_serialize.py -q && uv run pytest --doctest-modules pytcl/io/serialize.py -q`), lint, commit (`feat: msgspec serialization for tracks and filter states`).

---

### Task 2: DataFrame accessors (`pytcl/io/dataframes.py`)

**Files:**
- Create: `pytcl/io/dataframes.py`, `tests/unit/test_io_dataframes.py`
- Modify: `pytcl/io/__init__.py`

**Interfaces:**
- Produces: `tracks_to_polars(history, times) -> pl.DataFrame` (columns: `track_id` i64, `t` f64, `status` str, `state` List[f64], `covariance` List[f64]); `explode_state_columns(df, layout: Sequence[str]) -> pl.DataFrame` (adds one f64 column per layout name; layout length must equal state dim, ValueError otherwise); `metrics_to_polars(times, **series) -> pl.DataFrame` (flat table; each series 1-D, length-checked).
- polars imported inside a guarded helper; absence raises `DependencyError` naming the `[dataframe]` extra.

- [ ] **Step 1: Failing tests** (same `_history` helper pattern as Task 1 — import it or duplicate the small builder):

```python
import numpy as np
import pytest

pl = pytest.importorskip("polars")

from pytcl.io.dataframes import (
    explode_state_columns,
    metrics_to_polars,
    tracks_to_polars,
)


class TestTracksToPolars:
    def test_long_schema(self):
        history, times = _history()
        df = tracks_to_polars(history, times)
        assert df.columns == ["track_id", "t", "status", "state", "covariance"]
        assert df["t"].dtype == pl.Float64
        n_rows = sum(len(scan) for scan in history)
        assert df.height == n_rows
        first = df.row(0, named=True)
        dim = len(first["state"])
        assert len(first["covariance"]) == dim * dim

    def test_values_match_source_bitwise(self):
        history, times = _history()
        df = tracks_to_polars(history, times)
        tr = history[-1][0]
        row = df.filter((pl.col("t") == times[-1]) & (pl.col("track_id") == tr.id)).row(
            0, named=True
        )
        assert (
            np.asarray(row["state"]).tobytes()
            == np.asarray(tr.state, dtype=np.float64).tobytes()
        )

    def test_explode_layout(self):
        history, times = _history()
        df = explode_state_columns(
            tracks_to_polars(history, times), ["x", "vx", "y", "vy"]
        )
        assert {"x", "vx", "y", "vy"}.issubset(df.columns)
        row = df.row(0, named=True)
        assert row["x"] == row["state"][0] and row["vy"] == row["state"][3]

    def test_explode_wrong_layout_raises(self):
        history, times = _history()
        with pytest.raises(ValueError, match="layout"):
            explode_state_columns(tracks_to_polars(history, times), ["x", "y"])

    def test_metrics_table_and_parquet_round_trip(self, tmp_path):
        t = np.arange(5.0)
        ospa = np.array([1.0, 0.5, 0.25, 0.2, 0.1])
        df = metrics_to_polars(t, ospa=ospa)
        assert df.columns == ["t", "ospa"]
        p = tmp_path / "m.parquet"
        df.write_parquet(p)
        assert pl.read_parquet(p)["ospa"].to_list() == ospa.tolist()

    def test_dependency_error_without_polars(self, monkeypatch):
        import pytcl.io.dataframes as mod

        monkeypatch.setattr(mod, "_import_polars", mod._raise_missing)
        from pytcl.core.exceptions import DependencyError

        with pytest.raises(DependencyError, match="dataframe"):
            tracks_to_polars([], [])
```

(Adapt the last test to the guard mechanism you implement — the assertion that matters is `DependencyError` naming the extra.)

- [ ] **Step 2: RED.** **Step 3: Implement** (guarded `_import_polars()`; construction via `pl.DataFrame({...})` with explicit dtypes; no polars in any signature annotation — return type documented as "polars.DataFrame" in the docstring, annotated `Any`). **Step 4: GREEN + doctests + lint.** **Step 5: Commit** (`feat: polars accessors for track histories and metrics`).

---

### Task 3: Ingest readers (`pytcl/io/readers.py`) + ADS-B dogfood

**Files:**
- Create: `pytcl/io/readers.py`, `tests/unit/test_io_readers.py`, `tests/validation/test_reader_adsb_dogfood.py`
- Modify: `pytcl/io/__init__.py`

**Interfaces:**
- Produces:

```python
class MeasurementSet(NamedTuple):
    times: NDArray[np.float64]          # unique scan times, ascending
    scans: list[NDArray[np.float64]]    # scans[k]: (n_k, n_cols) measurements at times[k]
    ids: list[NDArray] | None           # per-scan id arrays when id_column given

def read_measurements_csv(path, *, time_column, measurement_columns, id_column=None) -> MeasurementSet
def read_measurements_parquet(path, *, time_column, measurement_columns, id_column=None) -> MeasurementSet
```

- Rows grouped into scans by exact `time_column` value, ascending; column mapping explicit; missing column -> ValueError listing available columns.

- [ ] **Step 1: Unit tests** (synthetic): write a small CSV and a Parquet (via polars) in `tmp_path` with known rows incl. two rows sharing a timestamp; assert grouping, ordering, dtype float64, ids threading, missing-column ValueError, `DependencyError` without polars (same guard pattern as Task 2).

- [ ] **Step 2: Dogfood REFERENCE test** (`tests/validation/test_reader_adsb_dogfood.py`): load `tests/fixtures/adsb/adsb_boston.json.gz` exactly as `tests/validation/test_adsb_tracking.py` does (reuse its module by import — its loader and chain are importable since it's a test module; if importing a test module proves fragile, lift the minimal loader into the new test with a comment naming the source), materialize the reports to a Parquet file in `tmp_path` (columns: t, east, north — after the same `geodetic2enu` conversion), read it back with `read_measurements_parquet`, run the same `kf_predict`/`kf_update` chain with the same constants, and assert the recovered per-aircraft velocity agreement with broadcast speed matches the original test's aggregate statistic to float tolerance (same median within 1e-9). The reader must be a transparent pipe.

- [ ] **Step 3: RED, implement, GREEN** (unit + dogfood + doctests + lint).

- [ ] **Step 4: Commit** (`feat: CSV/Parquet measurement readers with ADS-B dogfood validation`).

---

### Task 4: AIS decoding (`pytcl/transponders/ais.py`)

**Files:**
- Create: `pytcl/transponders/__init__.py`, `pytcl/transponders/ais.py`, `tests/unit/test_ais.py`
- Modify: `pytcl/__init__.py` (subpackage import, following the existing pattern)

**Interfaces:**
- Produces:

```python
class AISMessage(NamedTuple):
    msg_type: int
    mmsi: int
    fields: dict          # normalized pyais payload (asdict)

class PositionReports(NamedTuple):
    mmsi: NDArray[np.int64]
    t: NDArray[np.float64]        # receiver timestamps if given, else NaN
    lat: NDArray[np.float64]      # radians
    lon: NDArray[np.float64]      # radians
    sog: NDArray[np.float64]      # m/s (broadcast knots converted)
    cog: NDArray[np.float64]      # radians
    heading: NDArray[np.float64]  # radians, NaN when unavailable (511)

def decode_ais(nmea_text: str) -> list[AISMessage]
def ais_position_reports(nmea_text_or_messages, times=None) -> PositionReports
```

- Position-report types: 1, 2, 3, 18, 19. Sentinel handling per ITU-R M.1371: lat/lon 91/181 degrees -> NaN; SOG 1023 -> NaN; COG 3600 -> NaN; heading 511 -> NaN. pyais guarded by `DependencyError` naming `[ais]`.

- [ ] **Step 1: Failing tests** with documented NMEA vectors (these are standard published test sentences; verify expected values against pyais itself in the test so the assertion is anchored, plus hand-checked MMSI/type):

```python
import numpy as np
import pytest

pytest.importorskip("pyais")

from pytcl.transponders.ais import ais_position_reports, decode_ais

# Standard type-1 position report test sentence (widely published vector).
VDM_TYPE1 = "!AIVDM,1,1,,A,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"
# Two-part type-5 static/voyage message.
# COPY A VERIFIED TWO-PART PAIR FROM pyais's own test suite (pyais/tests/)
# and cite the source file -- widely reprinted web vectors are often
# transcription-mangled (the one previously here contained a mailing-list
# "address@hidden" artifact).
VDM_TYPE5_1 = "<verified part 1 from pyais tests>"
VDM_TYPE5_2 = "<verified part 2 from pyais tests>"


class TestDecode:
    def test_type1_position(self):
        msgs = decode_ais(VDM_TYPE1)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 1
        assert msgs[0].mmsi > 0

    def test_multipart_assembly(self):
        msgs = decode_ais(VDM_TYPE5_1 + "\n" + VDM_TYPE5_2)
        assert len(msgs) == 1
        assert msgs[0].msg_type == 5

    def test_garbage_lines_skipped_not_fatal(self):
        msgs = decode_ais("not an nmea line\n" + VDM_TYPE1 + "\nanother bad line")
        assert len(msgs) == 1


class TestPositionReports:
    def test_arrays_and_units(self):
        rep = ais_position_reports(VDM_TYPE1)
        assert rep.lat.dtype == np.float64
        assert abs(rep.lat[0]) < np.pi / 2 + 1e-9  # radians, not degrees
        assert abs(rep.lon[0]) < np.pi + 1e-9
        # cross-check against pyais directly: same sentence, degrees->radians
        import pyais

        decoded = pyais.decode(VDM_TYPE1)
        assert rep.lat[0] == pytest.approx(np.radians(decoded.lat), abs=0)
        assert rep.sog[0] == pytest.approx(decoded.speed * 0.514444, rel=1e-9)

    def test_non_position_messages_excluded(self):
        rep = ais_position_reports(VDM_TYPE5_1 + "\n" + VDM_TYPE5_2)
        assert len(rep.mmsi) == 0
```

If the type-5 vector above fails pyais checksum validation (it is a
commonly reprinted pair but transcriptions vary), substitute a two-part
vector from pyais's own test suite (`pyais/tests`) and cite it — do not
weaken the multipart assertion.

- [ ] **Step 2: RED, implement, GREEN** (+ doctests with the type-1 vector, lint). Sentinel and unit conversions exactly per the Interfaces block. `decode_ais` collects decodable messages and skips undecodable lines (counting them; the count is available via `logger.bind(site="transponders")` DEBUG when diagnostics are enabled — consistent with the diagnostics taxonomy).

- [ ] **Step 3: Commit** (`feat: AIS decoding via pyais with position-report extraction`).

---

### Task 5: AIS capture fixture + maritime REFERENCE validation

**Files:**
- Create: `scripts/capture_ais.py`, `tests/fixtures/ais/ais_norway.nmea.gz` (captured), `tests/fixtures/ais/SOURCES.md`, `tests/validation/test_ais_tracking.py`

**Interfaces:**
- Consumes: Task 4's decode + Task 3's conventions; the ADS-B validation test's structure as the template.

- [ ] **Step 1: Capture script** (`scripts/capture_ais.py`): stdlib socket client for the Norwegian Coastal Administration open stream (`153.44.253.27:5631`; verify liveness first with a 5-second probe; if moved, find the current endpoint on kystverket.no and record it). Arguments: duration (default 300 s), output path. Each line stamped with receiver time: output format `<unix_time>\t<nmea_sentence>` so the validation has timestamps. Fails loudly on connect failure. ASCII output only.

- [ ] **Step 2: Capture.** Run the script for ~5 minutes (no credentials needed — run it directly; this is the plan's only network step and it is one-time). Gzip to the fixture path. Sanity: decode with Task 4's `decode_ais`; require >= 50 distinct MMSIs with >= 20 position reports each across the capture, else capture longer. Write SOURCES.md (ADS-B file's structure as template): endpoint, capture UTC window, message-type histogram, the independence argument (ships broadcast SOG/COG; the filter sees positions only), and a Calibration section placeholder filled in Step 3.

- [ ] **Step 3: Validation test** (`tests/validation/test_ais_tracking.py`), mirroring `test_adsb_tracking.py`'s shape: decode fixture -> select ships with enough reports -> project lat/lon to local ENU (`geodetic2enu`, anchor at capture-area centroid) -> per-ship constant-velocity KF on positions only (ship-appropriate noise: measurement sigma ~15 m — AIS positions are GNSS-derived; process variance sized for slow maneuvering, documented) -> compare recovered speed against broadcast SOG. Assertions: median |v_est - SOG| within a calibrated envelope (fixed rule: 1.5x measured, ceil to 1 sig fig, measured value recorded verbatim in SOURCES.md with date), plus an innovation-distribution structure assertion (low median; tail present — ships turn in fjords). Skip if fixture absent. Deterministic, offline.

- [ ] **Step 4: GREEN, lint, commit** (`test: validate the tracking stack against real ship traffic (AIS)`).

---

### Task 6: ASDF export (`pytcl/io/asdf_io.py`)

**Files:**
- Create: `pytcl/io/asdf_io.py`, `tests/unit/test_io_asdf.py`
- Modify: `pytcl/io/__init__.py`

**Interfaces:**
- Produces: `save_tracks_asdf(path, history, times) -> None`, `load_tracks_asdf(path) -> tuple[times, history]` (same in-memory shapes as Task 1's decode: `SimpleTrack` per record — import it from `pytcl.io.serialize`); `save_states_asdf(path, x, P)` / `load_states_asdf(path)`. ASDF tree layout: `{"pytcl": {"schema_version": 1, "times": ndarray, "tracks": {"track_id": ndarray, "scan_index": ndarray, "status": list[str], "states": 2-D ndarray padded? NO — ragged states stored as per-scan groups}}}` — states may differ in dimension per tracker config but are uniform within one history; store `states` as a single `(n_records, dim)` ndarray plus `track_id`/`scan_index`/`status` parallel arrays, `covariances` as `(n_records, dim, dim)`. asdf guarded by `DependencyError` naming `[asdf]`.

- [ ] **Step 1: Failing tests**: round-trip bitwise (`assert_array_equal` on every state/covariance; times exact; ids/status preserved); `pytest.importorskip("asdf")`; `DependencyError` path; a schema_version field present in the file (open with asdf and check the tree) so future format evolution has a hook.

- [ ] **Step 2: RED, implement, GREEN** (+ doctests guarded per repo convention, lint).

- [ ] **Step 3: Commit** (`feat: ASDF export/import for track histories and states`).

---

### Task 7: HDF5 compression — measure, improve, document honestly

**Files:**
- Modify: `pytcl/io/hdf5_track_storage.py`
- Create: `tests/unit/test_hdf5_compression.py` (or extend the existing storage tests file if one covers this module — check `grep -rln hdf5_track_storage tests/`)
- Modify: whatever docs/docstrings currently state a ratio (locate with `grep -rn "1.3\|4.3x\|5-10x" docs/ pytcl/ README.md CONTRIBUTING.md`)

- [ ] **Step 1: Baseline measurement.** Build a benchmark scenario in the test: 100 tracks x 500 scans, 6-D states, realistic covariances (not random noise — run a CV filter so covariances converge and correlate, which is what makes them compressible). Measure current on-disk size vs raw `8 * n_values` bytes. Record the number in the task report.

- [ ] **Step 2: Improvements.** In order, measuring after each: (a) chunk shape aligned to per-track time series (chunks spanning time, not tracks — gzip then sees smooth trajectories); (b) `shuffle=True` byte-shuffle filter (h5py built-in, big win for slowly-varying float64); (c) optional `states_only=True` storage mode that drops per-scan covariances in favor of the filter config needed to regenerate them (the "covariance transform" idea — store the Cholesky of the steady-state P once when covariances have converged, plus per-scan deviations only if above a tolerance; if this proves intricate, ship (a)+(b) and record that (c) was evaluated and deferred, with the measured numbers motivating the decision).

- [ ] **Step 3: Assert and document.** The test asserts the achieved ratio on the benchmark scenario with ~20% headroom below the measured figure (regression rail, not aspiration). Update every located claim site with the measured figure and the command that reproduces it. If the result reaches the 5-10x band, say so plainly with the configuration that achieves it; if not, the docs state the achieved figure and the roadmap item is updated rather than left claiming the target.

- [ ] **Step 4: Full storage-suite regression** (`uv run pytest tests/unit -k "hdf5 or storage" -q` and the io validation files), lint, commit (`perf: HDF5 track storage compression -- measured, chunk-aligned, shuffled`).

---

### Task 8: GEBCO synthetic fixture

**Files:**
- Create: `scripts/make_synthetic_gebco.py`, `tests/fixtures/terrain/synthetic_gebco.nc` (tiny: 2-degree tile at coarse resolution, < 100 kB), `tests/fixtures/terrain/SOURCES.md`
- Modify: `tests/unit/test_terrain_loaders.py` or `tests/unit/test_diagnostics.py` (one new test: with `PYTCL_DATA_DIR` pointed at the fixture dir and the file named as the loader expects, `load_gebco` succeeds and — with diagnostics enabled — emits the found + start/finish DEBUG records; requires netCDF4, `importorskip`)

- [ ] Generate deterministically (fixed seed, documented grid), verify `load_gebco` reads it, write the test, SOURCES.md (synthetic, generation command, why it exists — the diagnostics-path debt), lint, commit (`test: synthetic GEBCO fixture exercises the loader diagnostics path in CI`).

---

### Task 9: Docs sweep, example, exports, verification, PR

**Files:**
- Create: `docs/results_io.rst`, `examples/measurement_ingest.py`
- Modify: `docs/index.rst` (toctree), `docs/matlab_parity_inventory.rst` (Transponders row), `README.md` (extras + overview bullet), `CLAUDE.md` (extras list), `CHANGELOG.md` (consolidated check — each prior task added its entry; verify), `examples/README.md`, `docs/architecture.rst` (module/package counts per its contract), coverage-contract ledger if its failure message asks

- [ ] **Step 1: Docs page** (`docs/results_io.rst`): readers with column mapping, the long schema and explode helper, the fidelity contract table (MessagePack vs JSON), AIS decode + position reports (radians/SI units called out), ASDF, the HDF5 compression figures WITH their reproduction command. Every code block must pass the docs gate on a machine with no optional extras and no data files: guard imports so absence raises DependencyError (the gate's skip marker) or use synthetic in-page data; never call a network or a real data file (the diagnostics-page lesson).
- [ ] **Step 2: Example** (`examples/measurement_ingest.py`): synthesize a small CSV of detections, read via `read_measurements_csv`, run the GNN tracker, `tracks_to_polars` -> print summary table (ASCII), write Parquet to `examples/output/`. PYTCL_SHOW_PLOTS conventions n/a (no plots); ASCII-only prints; runs under the examples contract (the venv has `[all]` in CI, which now includes polars).
- [ ] **Step 3: Reference sweep** per the spec's inventory (release-time items — version bumps, ROADMAP tick, roadmap.rst mirror — are explicitly deferred to the release PR; this task updates the content pages: parity inventory, README extras/overview, CLAUDE.md extras, docs page/toctree, examples README).
- [ ] **Step 4: Verification**: `uv run ruff check . && uv run ruff format --check .`; `uv run ty check pytcl`; new unit files + coverage contract + docs-architecture contract; `uv run pytest tests/contract/test_docs_code_blocks.py -m examples -q` with `PYTCL_DATA_DIR=$(mktemp -d)`; full `PYTCL_REQUIRE_MLX=1 uv run pytest -q` (storage + trackers touched); revert example-HTML churn.
- [ ] **Step 5: Push, PR** (`gh pr create --base main --title "feat: Results I/O -- polars ingest/accessors, msgspec, AIS, ASDF, HDF5 compression"` with a body summarizing per-deliverable status including the MEASURED HDF5 figure).
