# pytcl.diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `pytcl.diagnostics` — silent-by-default loguru logging, four instrumentation families, rich progress/tables with hard ASCII discipline — as v2.1.0's headline feature.

**Architecture:** New package `pytcl/diagnostics/` (core in `__init__.py`, rendering in `render.py`); direct logger calls at the Python orchestration layer guarded by `diagnostics_enabled()`; loguru+rich become core deps. Spec: `docs/superpowers/specs/2026-08-09-diagnostics-design.md`.

**Tech Stack:** loguru>=0.7, rich>=13 (new core deps), pytest with loguru test sinks and capsys.

## Global Constraints

- Branch: `feat/diagnostics`.
- Import-time contract: `import pytcl` executes `logger.disable("pytcl")`, installs NO handlers, prints nothing, and never mutates host-app logging state.
- ASCII discipline by construction: rich tables `box=box.ASCII`; progress bars ASCII bar chars, no spinner columns; log formats use `->` not arrows. Everything must keep `tests/contract/test_console_encoding.py` green.
- Hot-loop discipline: every per-measurement/per-track site is wrapped in `if diagnostics_enabled():` (or loguru lazy formatting) so the disabled path builds no strings and no intermediate arrays.
- numba kernels and functional filter APIs (`kf_predict`, `ckf_update`, ...) are untouched — no signature changes, no log calls inside them.
- Behavioral neutrality: identical numerical results with diagnostics enabled vs disabled (asserted in Task 5).
- Style: ruff 88 cols, NumPy docstrings with platform-robust doctests; `uv run` from repo root with `export PATH="$HOME/.local/bin:$PATH"`; prek hook on commit; explicit `git add` paths; commits end with the Co-Authored-By trailer (Claude Fable 5 <noreply@anthropic.com>).

## Verified anchors (from the current tree)

- Gating helpers: `pytcl/assignment_algorithms/gating.py` — `gate_measurements` (line ~218), `ellipsoidal_gate` (~99).
- GNN tracker: `pytcl/trackers/multi_target.py` — `MultiTargetTracker.process` (~157); internal `_InternalTrack` (~380); `Track` NamedTuple has `covariance`.
- MHT: `pytcl/trackers/mht.py` — `MHTTracker.process` (~178), `MHTConfig`/`MHTResult`.
- JPDA: `pytcl/assignment_algorithms/jpda.py` — `jpda_probabilities` (~181).
- Terrain: `pytcl/terrain/loaders.py` — `load_gebco` (~516), `load_earth2014` (~584).
- Paths: `pytcl/core/paths.py` — `get_data_dir`, `ensure_data_dir` (reads `PYTCL_DATA_DIR`).
- loguru testing idiom: `logger.add(sink_list.append, format="{message}")` after `logger.enable("pytcl")`, remove in teardown.
- The public-API coverage contract will see new exports; follow its failure message (register the diagnostics API — it is behavioral, tested by the unit tests below; nothing STRUCTURAL).

---

### Task 1: Core module — silence, enable/disable, guard flag

**Files:**
- Create: `pytcl/diagnostics/__init__.py`
- Create: `tests/unit/test_diagnostics.py`
- Modify: `pyproject.toml` (core deps), `uv.lock`, `pytcl/__init__.py` (import + re-export)

**Interfaces:**
- Produces: `enable_debug_logging(level="DEBUG") -> None`, `disable_debug_logging() -> None`, `diagnostics_enabled() -> bool`, module-internal `logger` (loguru logger bound to the pytcl namespace) that later tasks import as `from pytcl.diagnostics import diagnostics_enabled, logger`.

- [ ] **Step 1: Add deps**

In `pyproject.toml` `[project].dependencies`, append `"loguru>=0.7"` and `"rich>=13"`. Run `uv lock && uv sync --quiet`.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_diagnostics.py
"""pytcl.diagnostics: silence by default, enable/disable, instrumentation."""

import subprocess
import sys

import pytest
from loguru import logger as _loguru_logger

import pytcl
from pytcl.diagnostics import (
    diagnostics_enabled,
    disable_debug_logging,
    enable_debug_logging,
)


@pytest.fixture(autouse=True)
def _always_silent_after():
    yield
    disable_debug_logging()


class TestSilenceByDefault:
    def test_import_emits_nothing_and_installs_no_handlers(self):
        # Fresh interpreter: importing pytcl and running a filter step must
        # print nothing and leave loguru handler state untouched.
        code = (
            "from loguru import logger; before = len(logger._core.handlers); "
            "import numpy as np; import pytcl; "
            "from pytcl.dynamic_estimation.kalman.linear import kf_predict; "
            "kf_predict(np.zeros(2), np.eye(2), np.eye(2), np.eye(2)); "
            "after = len(logger._core.handlers); "
            "assert after == before, f'{before} -> {after} handlers'; "
            "print('SILENT-OK', end='')"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout == "SILENT-OK"
        assert result.stderr == ""

    def test_disabled_flag_by_default(self):
        assert diagnostics_enabled() is False


class TestEnableDisable:
    def test_round_trip(self, capsys):
        records = []
        enable_debug_logging()
        assert diagnostics_enabled() is True
        handle = _loguru_logger.add(records.append, format="{message}")
        _loguru_logger.bind(name="pytcl").debug("probe")
        _loguru_logger.remove(handle)
        disable_debug_logging()
        assert diagnostics_enabled() is False

    def test_enable_twice_does_not_stack_handlers(self):
        enable_debug_logging()
        n1 = len(_loguru_logger._core.handlers)
        enable_debug_logging()
        n2 = len(_loguru_logger._core.handlers)
        assert n2 == n1
        disable_debug_logging()

    def test_reexported_from_top_level(self):
        assert pytcl.enable_debug_logging is enable_debug_logging
        assert pytcl.disable_debug_logging is disable_debug_logging
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_diagnostics.py -x -q`
Expected: FAIL (ImportError: no module `pytcl.diagnostics`)

- [ ] **Step 4: Implement the core module**

```python
# pytcl/diagnostics/__init__.py
"""
Diagnostics: opt-in logging, instrumentation, and progress reporting.

pytcl is completely silent by default: importing it disables the ``pytcl``
loguru namespace and installs no handlers. Call :func:`enable_debug_logging`
to see gating rejections, association decisions, filter-health symptoms,
and data-file resolution at DEBUG level; :func:`disable_debug_logging`
returns to silence. This module is the redesigned successor to the
``pytcl.logging_config`` module removed in v2.0.0; no compatibility is
provided.

Examples
--------
>>> import pytcl
>>> pytcl.diagnostics.diagnostics_enabled()
False
"""

import sys
from typing import Any, Optional

from loguru import logger as _logger

# The library never speaks unless spoken to.
_logger.disable("pytcl")

logger = _logger  # instrumentation sites import this and pass name="pytcl"

_handler_id: Optional[int] = None
_enabled: bool = False


def diagnostics_enabled() -> bool:
    """Whether diagnostic logging is currently enabled.

    Hot paths consult this before constructing log payloads, so the
    disabled path costs one boolean check.
    """
    return _enabled


def enable_debug_logging(level: str = "DEBUG") -> None:
    """
    Enable pytcl's diagnostic logging with a rich-formatted handler.

    Parameters
    ----------
    level : str, optional
        Minimum level to emit ("DEBUG", "INFO", "WARNING", ...).

    Notes
    -----
    Idempotent: calling again replaces the previous handler rather than
    stacking a second one. Output goes to stderr, ASCII-safe.
    """
    global _handler_id, _enabled
    if _handler_id is not None:
        _logger.remove(_handler_id)
    _logger.enable("pytcl")
    _handler_id = _logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[site]}</cyan> - {message}"
        ),
        filter=lambda record: record["name"].startswith("pytcl"),
    )
    _enabled = True


def disable_debug_logging() -> None:
    """Return the library to complete silence. Idempotent."""
    global _handler_id, _enabled
    if _handler_id is not None:
        _logger.remove(_handler_id)
        _handler_id = None
    _logger.disable("pytcl")
    _enabled = False
```

Note on the format string: sites log via
`logger.bind(site="gating").debug(...)`; the `{extra[site]}` field keys
that. If binding `site` everywhere proves awkward during implementation,
switch the format to `{name}` (module path) — the tests below assert on
messages, not the header — and note the deviation in your report.

In `pytcl/__init__.py`, follow the existing import pattern to add:
`from pytcl import diagnostics` and re-export `enable_debug_logging`,
`disable_debug_logging` (add to `__all__` alongside existing entries).

- [ ] **Step 5: Run tests, lint, commit**

Run: `uv run pytest tests/unit/test_diagnostics.py -q` (expected: all pass), `uv run ruff check .`, then:

```bash
git add pytcl/diagnostics/__init__.py pytcl/__init__.py tests/unit/test_diagnostics.py pyproject.toml uv.lock
git commit -m "feat: pytcl.diagnostics core -- silent-by-default opt-in logging"
```

---

### Task 2: Rendering — `track_table`, `progress_bar`, terrain `progress=`

**Files:**
- Create: `pytcl/diagnostics/render.py`
- Modify: `pytcl/diagnostics/__init__.py` (re-export), `pytcl/terrain/loaders.py` (`load_gebco`, `load_earth2014` gain `progress: bool = False`), `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: `Track` NamedTuple (`id`, `status`, `state`, `covariance`) from `pytcl.trackers`.
- Produces: `track_table(tracks, console=None) -> None`; `progress_bar(iterable, description="working", total=None)` generator wrapping any iterable; both re-exported from `pytcl.diagnostics`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_diagnostics.py`:

```python
class TestRendering:
    def _tracks(self):
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.eye(4),
            H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 2.0,
        )
        tracks = tracker.process([np.array([1.0, 2.0])], dt=1.0)
        return tracks

    def test_track_table_is_ascii_only(self):
        import io

        from rich.console import Console

        from pytcl.diagnostics import track_table

        buf = io.StringIO()
        track_table(self._tracks(), console=Console(file=buf, width=100))
        out = buf.getvalue()
        assert len(out) > 0
        out.encode("cp1252")  # raises UnicodeEncodeError on any unsafe char

    def test_progress_bar_yields_all_items_and_is_ascii(self, capsys):
        from pytcl.diagnostics import progress_bar

        items = list(progress_bar(range(5), description="test"))
        assert items == [0, 1, 2, 3, 4]
        err = capsys.readouterr().err
        err.encode("cp1252")

    def test_terrain_progress_param_accepted(self):
        import inspect

        from pytcl.terrain.loaders import load_earth2014, load_gebco

        assert "progress" in inspect.signature(load_gebco).parameters
        assert "progress" in inspect.signature(load_earth2014).parameters
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_diagnostics.py::TestRendering -x -q`
Expected: FAIL (ImportError: `track_table`)

- [ ] **Step 3: Implement render.py**

```python
# pytcl/diagnostics/render.py
"""Rich-based rendering: track tables and progress bars, ASCII-safe."""

from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn
from rich.table import Table


def track_table(tracks: Sequence[Any], console: Optional[Console] = None) -> None:
    """
    Render a summary table of tracks to the console.

    Parameters
    ----------
    tracks : sequence
        Track objects with ``id``, ``status``, ``state`` (and optionally
        ``covariance``) attributes, e.g. from ``MultiTargetTracker.process``.
    console : rich.console.Console, optional
        Target console; defaults to stderr. Output is ASCII-only
        (``box.ASCII``) to satisfy the console-encoding contract.
    """
    console = console or Console(stderr=True)
    table = Table(title="Tracks", box=box.ASCII, safe_box=True)
    table.add_column("id", justify="right")
    table.add_column("status")
    table.add_column("position")
    table.add_column("speed", justify="right")
    for t in tracks:
        state = np.asarray(t.state, dtype=float).ravel()
        # Convention: interleaved [x, vx, y, vy, ...]; fall back to halves.
        pos = state[0::2] if len(state) % 2 == 0 else state
        vel = state[1::2] if len(state) % 2 == 0 else np.zeros(1)
        table.add_row(
            str(t.id),
            getattr(t.status, "value", str(t.status)),
            "(" + ", ".join(f"{p:.1f}" for p in pos) + ")",
            f"{float(np.linalg.norm(vel)):.2f}",
        )
    console.print(table)


def progress_bar(
    iterable: Iterable[Any],
    description: str = "working",
    total: Optional[int] = None,
) -> Iterator[Any]:
    """
    Wrap an iterable in an ASCII progress bar on stderr.

    Yields the items unchanged; the bar completes when iteration ends.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(complete_style="none", finished_style="none"),
        TaskProgressColumn(),
        console=Console(stderr=True, no_color=False),
    )
    with progress:
        task = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task)
```

Re-export both from `pytcl/diagnostics/__init__.py` (`from pytcl.diagnostics.render import progress_bar, track_table` at the bottom, added to the module's `__all__`), and add `track_table`/`progress_bar` to the `pytcl` top-level re-exports next to `enable_debug_logging` (the spec's API table promises all five names at top level).

In `pytcl/terrain/loaders.py`: add `progress: bool = False` keyword to `load_gebco` and `load_earth2014`; when True, wrap the dominant read loop / chunk iteration in `progress_bar(...)` (read the function bodies to find the natural iteration; if a loader reads in one shot with no loop, wrap the per-band or per-tile step if present, otherwise emit a start/finish DEBUG log pair and note in your report that a single-shot read has no meaningful progress to show). BarColumn note: rich's default bar uses Unicode block characters — verify the captured output passes the cp1252 test; if it does not, use `Progress(TextColumn(...), TextColumn("{task.completed}/{task.total}"), ...)` (pure-text progress) instead and note it.

- [ ] **Step 4: Run tests (rendering + console contract)**

Run: `uv run pytest tests/unit/test_diagnostics.py tests/contract/test_console_encoding.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add pytcl/diagnostics/ pytcl/terrain/loaders.py tests/unit/test_diagnostics.py
git commit -m "feat: diagnostics rendering -- ASCII track tables and progress bars"
```

---

### Task 3: Data-file resolution instrumentation

**Files:**
- Modify: `pytcl/core/paths.py`, `pytcl/terrain/loaders.py`, the magnetism and gravity coefficient loaders (locate with `grep -rn "get_data_dir()" pytcl/ --include='*.py'` — instrument each call site's resolution outcome), `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: `logger`, `diagnostics_enabled` from Task 1.
- Produces: DEBUG records for every data-path resolution: the directory resolved (and whether `PYTCL_DATA_DIR` overrode), each candidate file tried, found or missing.

- [ ] **Step 1: Write the failing test**

```python
class TestDataFileInstrumentation:
    def test_missing_file_resolution_is_logged(self, tmp_path, monkeypatch):
        from loguru import logger as _l

        from pytcl.terrain.loaders import load_gebco

        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            with pytest.raises((FileNotFoundError, Exception)):
                load_gebco()
        finally:
            _l.remove(handle)
            disable_debug_logging()
        text = " ".join(str(r) for r in records)
        assert str(tmp_path) in text          # the resolved directory
        assert "PYTCL_DATA_DIR" in text       # the override was named
        assert "missing" in text or "not found" in text

    def test_silent_when_disabled(self, tmp_path, monkeypatch):
        from loguru import logger as _l

        from pytcl.terrain.loaders import load_gebco

        monkeypatch.setenv("PYTCL_DATA_DIR", str(tmp_path))
        records = []
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            with pytest.raises((FileNotFoundError, Exception)):
                load_gebco()
        finally:
            _l.remove(handle)
        assert records == []  # disabled namespace emits nothing
```

- [ ] **Step 2: Run to verify failure** (`assert str(tmp_path) in text` fails)

- [ ] **Step 3: Implement**

In `paths.py`, log inside `get_data_dir()`:

```python
from pytcl.diagnostics import logger

# inside get_data_dir(), after resolving:
logger.bind(site="data-files").debug(
    "data dir resolved to {} (PYTCL_DATA_DIR {})",
    path,
    "override active" if env_override else "not set",
)
```

(Adapt names to the function's actual locals; the message MUST contain the
resolved path and the string "PYTCL_DATA_DIR".) In each loader that opens a
data file, log the candidate path and outcome ("found"/"missing") before
raising or proceeding. These are cold paths — no guard flag needed; plain
`logger.bind(site="data-files").debug(...)` calls are fine.

Import note: `pytcl.core.paths` importing `pytcl.diagnostics` must not
create an import cycle — `pytcl/diagnostics/__init__.py` imports only
loguru/rich/numpy/sys, never pytcl modules. Keep it that way.

- [ ] **Step 4: Run tests** (`uv run pytest tests/unit/test_diagnostics.py -q`), plus the terrain unit tests (`uv run pytest tests/unit/test_terrain_loaders.py -q`) to confirm no behavior change.

- [ ] **Step 5: Commit** (`git add -u && git commit -m "feat: instrument data-file resolution"`)

---

### Task 4: Gating and association instrumentation

**Files:**
- Modify: `pytcl/trackers/multi_target.py` (`MultiTargetTracker.process`, gating + GNN assignment sections), `pytcl/trackers/mht.py` (`MHTTracker.process`), `pytcl/assignment_algorithms/jpda.py` (`jpda_probabilities`), `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: `logger`, `diagnostics_enabled`.
- Produces: per-scan DEBUG records — gating: excluded measurement indices with distance vs threshold; GNN: assignment pairs + total cost; JPDA: per-track top marginal probability summary; MHT: hypothesis count, pruned count, best score.

- [ ] **Step 1: Write the failing tests**

```python
class TestGatingAssociationInstrumentation:
    def _run_scenario(self, capture):
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=float),
            H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
            Q=np.eye(4) * 0.01,
            R=np.eye(2) * 1.0,
            confirm_hits=1,
        )
        # Scan 1 starts a track at origin; scan 2 has one near measurement
        # and one far measurement that MUST be gated out.
        tracker.process([np.array([0.0, 0.0])], dt=1.0)
        tracker.process([np.array([0.1, 0.1]), np.array([500.0, 500.0])], dt=1.0)

    def test_gating_rejection_logged_when_enabled(self):
        from loguru import logger as _l

        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_scenario(records)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        text = " ".join(str(r) for r in records)
        assert "gate" in text.lower()
        assert "assign" in text.lower() or "association" in text.lower()

    def test_zero_records_and_zero_payload_when_disabled(self, monkeypatch):
        from loguru import logger as _l

        from pytcl.diagnostics import diagnostics_enabled as real

        calls = []
        # Patch the CONSUMING module's imported name -- the tracker holds a
        # direct reference, so patching pytcl.diagnostics itself would miss.
        monkeypatch.setattr(
            "pytcl.trackers.multi_target.diagnostics_enabled",
            lambda: (calls.append(1), real())[1],
        )
        records = []
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            self._run_scenario(records)
        finally:
            _l.remove(handle)
        assert records == []
        assert len(calls) > 0  # the guard IS consulted on the hot path

    def test_jpda_summary_logged(self):
        import numpy as np

        from loguru import logger as _l

        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        records = []
        enable_debug_logging()
        handle = _l.add(records.append, format="{message}", level="DEBUG")
        try:
            likelihood = np.array([[0.8, 0.1], [0.2, 0.7]])
            jpda_probabilities(likelihood, pd=0.9, clutter_density=1e-6)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        assert any("jpda" in str(r).lower() for r in records)
```

(Adapt `jpda_probabilities`' exact signature from its docstring when
writing the test — keyword names may differ; the assertion is the point.)

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

Pattern for every site (this is the hot-loop discipline made concrete):

```python
from pytcl.diagnostics import diagnostics_enabled, logger

# inside MultiTargetTracker.process, after gating computes distances:
if diagnostics_enabled():
    rejected = [
        (j, float(d)) for j, d in enumerate(distances) if d > threshold
    ]
    if rejected:
        logger.bind(site="gating").debug(
            "track {}: gated out {} of {} measurements: {}",
            track.id,
            len(rejected),
            len(distances),
            "; ".join(f"m{j} d={d:.2f}>thr={threshold:.2f}" for j, d in rejected),
        )
```

Adapt variable names to each site's actual locals. GNN: after the
assignment solve, log pairs + total cost. MHT: after hypothesis
management, log `len(hypotheses)`, pruned count, best score. JPDA: at the
end of `jpda_probabilities`, log the per-track max marginal (guarded).
Never compute a summary outside the `if diagnostics_enabled():` block.

- [ ] **Step 4: Run tests + tracker regressions**

Run: `uv run pytest tests/unit/test_diagnostics.py -q && uv run pytest tests/unit -k "tracker or multi_target or mht or jpda" -q`
Expected: all pass, no behavior change.

- [ ] **Step 5: Commit** (`git add -u && git commit -m "feat: instrument gating and association decisions"`)

---

### Task 5: Filter-health instrumentation + behavioral neutrality

**Files:**
- Modify: `pytcl/diagnostics/__init__.py` (health helper), `pytcl/trackers/multi_target.py` (call it per track update), `tests/unit/test_diagnostics.py`

**Interfaces:**
- Consumes: innovation `y` and innovation covariance `S` already computed in the tracker's update path; track covariance `P`.
- Produces: `pytcl/diagnostics` module constants `NIS_WINDOW = 20`, `NIS_OUTLIER_FACTOR = 3.0`, `CONDITION_WARN = 1e12`; function `log_filter_health(track_id, nis_value, nis_window, cov_condition) -> None` (DEBUG normally, WARNING when symptomatic).

- [ ] **Step 1: Write the failing tests**

```python
class TestFilterHealth:
    def test_health_logged_and_warning_on_symptoms(self):
        from loguru import logger as _l

        from pytcl.diagnostics import log_filter_health

        records = []
        enable_debug_logging()
        handle = _l.add(
            lambda m: records.append((m.record["level"].name, m.record["message"])),
            level="DEBUG",
        )
        try:
            log_filter_health(1, nis_value=1.0, nis_window=[1.0] * 5, cov_condition=10.0)
            log_filter_health(2, nis_value=99.0, nis_window=[50.0] * 5, cov_condition=1e15)
        finally:
            _l.remove(handle)
            disable_debug_logging()
        levels = [lvl for lvl, _ in records]
        assert "DEBUG" in levels and "WARNING" in levels

    def test_behavioral_neutrality(self):
        # Identical numerical results with diagnostics enabled vs disabled.
        import numpy as np

        from pytcl.trackers import MultiTargetTracker

        def run():
            rng = np.random.default_rng(11)
            tracker = MultiTargetTracker(
                state_dim=4,
                meas_dim=2,
                F=np.array(
                    [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                    dtype=float,
                ),
                H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
                Q=np.eye(4) * 0.01,
                R=np.eye(2) * 1.0,
            )
            out = []
            for k in range(20):
                z = [np.array([k + rng.normal(0, 0.5), k + rng.normal(0, 0.5)])]
                out.append(tracker.process(z, dt=1.0))
            return out

        baseline = run()
        enable_debug_logging()
        try:
            with_diag = run()
        finally:
            disable_debug_logging()
        for a, b in zip(baseline, with_diag):
            for ta, tb in zip(a, b):
                np.testing.assert_array_equal(ta.state, tb.state)
                np.testing.assert_array_equal(ta.covariance, tb.covariance)
```

- [ ] **Step 2: Run to verify failure**

- [ ] **Step 3: Implement**

`log_filter_health` in the diagnostics package (plain function, guarded
internally by `diagnostics_enabled()` so callers may call it bare);
symptomatic = `nis_value > NIS_OUTLIER_FACTOR * mean(nis_window)` or
`cov_condition > CONDITION_WARN`. In `MultiTargetTracker`'s per-track
update: maintain a `deque(maxlen=NIS_WINDOW)` of NIS values per internal
track (only when diagnostics are enabled — the deque itself must not be
built on the disabled path), compute NIS from the innovation and S the
update already produced, condition number via `np.linalg.cond(P)` (also
only when enabled), and call `log_filter_health`.

- [ ] **Step 4: Run** the new tests + full tracker tests + `uv run pytest tests/unit/test_diagnostics.py -q`.

- [ ] **Step 5: Commit** (`git add -u && git commit -m "feat: filter-health instrumentation with behavioral-neutrality guarantee"`)

---

### Task 6: Docs, exports ledger, full verification, PR

**Files:**
- Create: `docs/diagnostics.rst` (+ add to the docs toctree where peer pages live)
- Modify: `CHANGELOG.md`, possibly `docs/architecture.rst` (module count), possibly the coverage-contract ledger (follow its failure message)

- [ ] **Step 1: Docs page**

`docs/diagnostics.rst`: enable/disable usage, the silence guarantee, what
each family logs (with a short sample of output), progress usage,
`track_table`, the ASCII/cp1252 note, and the "successor to the removed
``pytcl.logging_config``, no compatibility" note. Follow the structure of
an existing mid-size docs page (e.g. `docs/gpu_acceleration.rst`).

- [ ] **Step 2: CHANGELOG** under Unreleased/Added:

```markdown
- `pytcl.diagnostics`: opt-in diagnostic logging (silent by default;
  `enable_debug_logging()`/`disable_debug_logging()`), instrumentation of
  gating rejections, association decisions, filter-health symptoms, and
  data-file resolution; ASCII-safe rich progress bars (`progress_bar`,
  `progress=True` on terrain loaders) and track tables (`track_table`).
  loguru and rich join the core dependencies. Successor to the
  `pytcl.logging_config` module removed in v2.0.0 (no compatibility).
```

- [ ] **Step 3: Full verification**

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check pytcl
uv run pytest tests/unit/test_diagnostics.py tests/contract/test_console_encoding.py tests/contract/test_public_api_coverage.py -q
uv run pytest --doctest-modules pytcl/diagnostics/ -q
PYTCL_REQUIRE_MLX=1 uv run pytest -q
```

All green (full suite because trackers were touched). Fix the
architecture-count / coverage-ledger fallout per their failure messages
if they fire. Revert regenerated `docs/_static/images/examples/*.html`.

- [ ] **Step 4: Commit, push, PR**

```bash
git add -u docs/diagnostics.rst && git commit -m "docs: diagnostics guide, changelog"
git push -u origin feat/diagnostics
gh pr create --base main --title "feat: pytcl.diagnostics -- opt-in logging, instrumentation, progress" \
  --body "Implements docs/superpowers/specs/2026-08-09-diagnostics-design.md: silent-by-default loguru logging with enable/disable, four instrumentation families (gating, association, filter health, data files), ASCII-safe rich progress bars and track tables, behavioral-neutrality and silence guarantees asserted by tests. loguru+rich join core dependencies. v2.1.0 headline feature."
```
