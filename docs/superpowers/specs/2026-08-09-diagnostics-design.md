# pytcl.diagnostics — Logging, Instrumentation, Progress (v2.1.0 headline)

**Date:** 2026-08-09
**Status:** Approved
**Seed:** `docs/superpowers/specs/2026-08-06-modernization-campaign-design.md`,
"v2.1.0 — Diagnostics (loguru + rich)". This spec supersedes and details it.

## Problem

Tracking failures are opaque: gating silently drops measurements,
association silently picks hypotheses, filters diverge without symptoms
surfacing, and data-file loaders fail without saying which paths they
tried. v2.0.0 removed the ad-hoc `pytcl.logging_config`; this is its
deliberate, redesigned successor (no compatibility provided or implied).

## Decisions (approved)

- **loguru and rich become core dependencies** (`loguru>=0.7`,
  `rich>=13`) alongside numpy/scipy/numba/h5py.
- **All four instrumentation families ship in v2.1.0.**
- **Hook mechanism: direct logger calls** at the Python orchestration
  layer, guarded by a cheap enabled-flag for hot loops. No event/observer
  subsystem (YAGNI until a second consumer exists).

## Module and public API

`pytcl/diagnostics/` package; these names are also re-exported from
`pytcl` top level:

| API | Behavior |
|-----|----------|
| `enable_debug_logging(level="DEBUG")` | Enables the `pytcl` loguru logger and installs a rich-backed stderr handler. Idempotent; calling twice replaces, not stacks. Returns None. |
| `disable_debug_logging()` | Returns the library to complete silence. Idempotent. |
| `diagnostics_enabled() -> bool` | Cheap flag; hot paths consult it before constructing any log payload. |
| `track_table(tracks) -> None` | Renders a rich table (id, status, position, velocity, NIS if available) to the console. |
| `progress_bar(iterable, description)` | Wraps an iterable in a rich progress bar; the helper behind `progress=True` parameters and usable directly by users around scan loops. |

Import-time contract: `import pytcl` executes `logger.disable("pytcl")`
and installs **no** handlers — the library is completely silent by
default, and importing it never mutates the root/global logging state of
the host application.

## ASCII discipline (hard constraint)

All rendered output must satisfy the existing console-encoding contract
(`tests/contract/test_console_encoding.py`; cp1252-safe under redirected
stdout on Windows) **by construction**:

- rich tables: `box=box.ASCII`.
- progress bars: ASCII bar characters, no spinner columns, no glyphs.
- log format strings: ASCII only (`->` not arrows).

## Instrumentation sites

Python orchestration layer only. numba kernels and the pure functional
filter API (`kf_predict`, `ckf_update`, ...) are untouched — no signature
changes, no log calls inside them.

1. **Data-file resolution** — `pytcl/core/paths.py` and the terrain /
   magnetism / gravity loaders: every candidate path tried, found, or
   missing, at DEBUG, including the `PYTCL_DATA_DIR` override when active.
2. **Gating** — `MultiTargetTracker`'s gating step and the module-level
   helpers in `assignment_algorithms/gating.py` that the trackers call:
   excluded measurement indices with distance vs. threshold. Lazily
   formatted; nothing is built when disabled.
3. **Association** — GNN: chosen assignment and total cost per scan;
   JPDA: per-track marginal association probability summary; MHT:
   hypothesis count, pruned count, and best-hypothesis score per scan.
4. **Filter health** — computed at tracker level from quantities the
   update already produced: per-track NIS with a windowed outlier flag
   (window and threshold are module constants with documented defaults,
   not new API), and covariance condition number. Emitted at WARNING when
   symptomatic, DEBUG otherwise. No new estimation mathematics.

Hot-loop discipline: every per-measurement or per-track log site is
either wrapped in `if diagnostics_enabled():` or uses loguru lazy
formatting such that the disabled path constructs no strings and no
intermediate arrays.

## Progress parameters

`progress: bool = False` added to the terrain loaders (the genuinely
long-running operations; GEBCO subsetting and Earth2014 reads). No other
signatures gain the parameter in this release; users wrap their own scan
loops with `progress_bar`.

## Out of scope

- Event/observer systems, structured log export, log-file management.
- Any change to functional filter signatures or numba kernels.
- Progress parameters beyond the terrain loaders.
- Compatibility with the removed `pytcl.logging_config`.

## Testing

- **Silence by default:** importing pytcl and running a tracking scenario
  emits nothing to stdout/stderr and installs no handlers (capsys/capfd
  asserted empty; loguru handler count unchanged).
- **Round-trip:** enable -> emit -> disable -> silence again; enable twice
  does not duplicate handlers.
- **Per-family emission tests** via a loguru test sink: each of the four
  families produces expected records when enabled and exactly zero when
  disabled. The disabled case also asserts zero payload construction where
  guard flags are used (spy on the summary builder).
- **ASCII rendering:** `track_table` and `progress_bar` output captured
  and asserted to contain only cp1252-encodable characters; the existing
  console-encoding contract test stays green.
- **Behavioral neutrality:** a fixed tracking scenario produces bitwise
  identical estimates with diagnostics enabled vs. disabled (logging must
  observe, never perturb).

## Documentation

- New docs page (`docs/diagnostics.rst`): enable/disable, what each
  family logs, progress usage, the silence guarantee.
- CHANGELOG under Unreleased/Added; ROADMAP v2.1.0 item ticked when
  released.
- Note in the docs page: successor to the removed `pytcl.logging_config`,
  no compatibility.
