# Modernization Campaign: Tooling + v2.1–v2.3 Feature Series

**Date:** 2026-08-06
**Status:** Approved design, pre-implementation
**Scope:** uv, ty, loguru, rich, msgspec, polars, xy

## Summary

Post-v2.0.0 modernization in two tracks. A tooling track (uv, ty) lands
immediately as unversioned chore PRs — contributor and CI facing only. A
feature track ships as three staged minor releases: v2.1.0 diagnostics
(loguru + rich), v2.2.0 results I/O (polars + msgspec export), v2.3.0 typed
configs and session save/restore (msgspec). An xy visualization extra is
planned but unversioned, gated on xy leaving alpha.

ROADMAP.md is reconciled as part of this work: the aspirational items
currently pinned to v2.1/v2.2/v2.3 (RAPIDS, ROS 2, quantum-inspired
algorithms) move to the unversioned Long-Term Vision section, and this
campaign claims those version numbers.

## Tooling track (no release)

### uv (level 2: project-managed)

- Add `uv.lock`, committed.
- Move the `dev` extra out of `[project.optional-dependencies]` into
  `[dependency-groups]` — it stops being published as an installable extra
  of the PyPI package. The `all` extra drops `dev` from its list.
- The `benchmark` extra's contents fold into dependency groups as
  appropriate; published extras remain only those meaningful to end users
  (astronomy, geodesy, terrain, visualization, signal, gpu, gpu-apple, all).
- CI switches to `astral-sh/setup-uv` with `uv sync` (locked) replacing
  `pip install` across ci.yml, docs.yml, benchmark-*.yml, gpu.yml,
  publish.yml where applicable.
- CONTRIBUTING.md and ROADMAP.md development-setup instructions change to
  `uv sync` / `uv run`.
- **Out of scope:** the build backend. setuptools continues to produce the
  PyPI wheels. Swapping to `uv_build` is a separate future decision, taken
  only after this campaign has settled.

### ty (replace mypy, with probation)

- ty becomes the blocking type-check gate in CI, configured in
  pyproject.toml — the single source of typing truth.
- mypy continues as a **non-blocking** CI job during probation, to reveal
  anything ty misses relative to `mypy --strict`.
- This retires the current split-brain: CI runs `mypy pytcl --strict
  --ignore-missing-imports` while mypy.ini holds a much looser config.
  Both go away; mypy.ini is deleted when the probation ends.
- **Probation end is scheduled, not open-ended:** the non-blocking mypy job
  and mypy.ini are removed in the v2.1.0 release PR.

## Feature track

### v2.1.0 — Diagnostics (loguru + rich)

New module `pytcl.diagnostics`.

- **Library-safe logging:** loguru with `logger.disable("pytcl")` executed at
  import — the library is completely silent by default. Users opt in with
  `pytcl.enable_debug_logging(level=...)`, which enables the `pytcl` logger
  and installs a rich-formatted handler. A corresponding
  `disable_debug_logging()` returns to silence.
- **Instrumentation sites:** the opaque decision points where tracking goes
  wrong without visible cause — gating rejections (which measurements were
  excluded and why), association decisions (chosen hypothesis vs.
  alternatives), filter health indicators (innovation/covariance symptoms of
  divergence), and external data-file resolution (which path was tried,
  found, or missing).
- **Progress reporting:** rich progress bars for long-running batch
  operations — batch tracking runs, benchmark suites, large terrain-file
  loads. Off by default; enabled per-call via a `progress=True` parameter
  on the operations that support it.
- **Results display:** rich table rendering for track summaries in the
  terminal.
- **Constraint:** all emitted output must pass the existing console-encoding
  contract test (`tests/contract/test_console_encoding.py`) — ASCII-safe
  under redirected stdout on Windows (cp1252). No box-drawing or Unicode
  glyph output paths that bypass it.
- Note: v2.0.0 removed `pytcl.logging_config` outright. This is its
  deliberate, redesigned successor — no compatibility with the removed
  module is provided or implied.

### v2.2.0 — Results I/O (polars + msgspec export)

- **Ingest:** readers that load measurement/detection files (CSV, Parquet)
  via polars into tracker-ready numpy arrays.
- **Results as DataFrames:** `to_polars()` accessors on track histories and
  performance-evaluation outputs, for filtering, joining, and export.
- **msgspec export:** track histories, filter states, and covariances
  serialize to JSON and MessagePack for downstream consumers, with
  msgspec-validated decode on the way back in.
- Polars being Arrow-native, this release also delivers the roadmap's
  existing "Parquet format" and "Apache Arrow integration" bullets.
- The compute core stays numpy/scipy/numba; polars appears only at the I/O
  boundary, never inside algorithms.

### v2.3.0 — Typed configs + save/restore (msgspec)

- **Typed configs:** filter/tracker configuration objects as
  `msgspec.Struct`s — validated at construction, self-documenting,
  serializable. Constructors accept these alongside existing kwargs; no
  existing call sites break.
- **Session save/restore:** snapshot a tracker's full state (filter states,
  covariances, track table, configuration) to disk via msgspec, restore and
  continue. The config Structs define the schema that save/restore reuses —
  which is why these ship together.

### xy — unversioned, gated

- New `[visualization-xy]` extra alongside the existing plotly-based
  `[visualization]`; nothing is ported away from plotly.
- Target use cases: large-dataset plots — dense measurement clouds, long
  track histories — where xy's screen-bounded density rendering wins.
- **Gate:** work does not start until xy has a stable (non-alpha) release.
  It sits in ROADMAP.md without a version number until then.

## Dependency policy

| Package | Placement | Rationale |
|---------|-----------|-----------|
| msgspec | Core dependency | Typed configs are core API; small, wheels on all platforms |
| loguru  | Core dependency | Diagnostics must work with zero setup; lightweight pure Python |
| rich    | Core dependency | Same; pairs with loguru for handler/progress/tables |
| polars  | New `[dataframe]` extra | ~30 MB binary; too heavy to impose on every install |
| xy      | New `[visualization-xy]` extra | Alpha; optional by definition |
| uv, ty  | Dev tooling only | Never touch the published package |

## ROADMAP.md reconciliation

- RAPIDS/distributed (currently "v2.1"), ROS 2/ecosystem ("v2.2"),
  quantum-inspired ("v2.3"), adaptive learning ("v2.4+") move to the
  unversioned Long-Term Vision section. No content is deleted; only version
  labels detach.
- The near-term sections describe this campaign: tooling track, v2.1.0,
  v2.2.0, v2.3.0, and the xy gate.
- The measured backlog (MATLAB parity inventory) stays as-is — it remains
  the candidate pool for parity work, which continues independently of this
  campaign and is not assigned versions here.
- The "Limitations vs MATLAB" table's "v2.1 with RAPIDS" cell updates to
  match.

## Testing

- Every feature release follows existing conventions: pytest, hypothesis
  where property-based testing fits, `_data_skip` for external-data tests,
  fake timers per the global rule, coverage of every new exported function
  (enforced automatically by `tests/contract/test_public_api_coverage.py`).
- Diagnostics: assert silence-by-default (no output, no handler side
  effects on import), assert enable/disable round-trip, and run all
  emitting paths through the console-encoding contract test.
- Save/restore: round-trip property tests — save, restore, continue, and
  compare against an uninterrupted run to numerical tolerance.
- msgspec export: round-trip JSON and MessagePack; schema-validation
  failures raise `pytcl.TCLError` subclasses, not raw msgspec errors.
- Polars ingest: golden-file tests on small fixtures; the polars import
  guards follow the same optional-dependency pattern as existing extras
  (`DependencyError` when absent).
- CI: the uv/ty switch must leave the existing matrix (3 OS × 3 Python +
  Windows arm), doctests, docs build, and benchmark jobs green before any
  feature work starts on top of it.

## Release mechanics

Each of v2.1.0/v2.2.0/v2.3.0 follows the standing release rule: update
CHANGELOG.md, README.md, ROADMAP.md, rebuild the Sphinx docs so
`docs/_build/index.html` reflects the release, then tag. All work lands via
feature branches and PRs — never direct to main.

## Non-goals

- No `uv_build` backend swap in this campaign.
- No porting of plotly visualization to xy.
- No polars inside the compute core.
- No pydantic; msgspec is the single serialization/validation library.
- No compatibility layer for the removed v1.x `pytcl.logging_config`.
