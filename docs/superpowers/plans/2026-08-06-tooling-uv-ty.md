# Tooling Track: uv + ty Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate contributor/CI workflow to uv (lockfile + dependency groups) and make ty the blocking type-check gate with mypy on scheduled non-blocking probation.

**Architecture:** Two-layer change. Layer 1: `pyproject.toml` moves dev tooling out of published extras into PEP 735 `[dependency-groups]` and gains a committed `uv.lock`; every CI workflow (except publish.yml, deliberately) switches from setup-python+pip to setup-uv+`uv sync --locked`. Layer 2: ty is configured in pyproject, made clean against `pytcl/`, and replaces mypy as the CI gate; mypy survives as a non-blocking probation job scheduled for deletion in the v2.1.0 release PR.

**Tech Stack:** uv 0.12.x, `astral-sh/setup-uv@v9`, ty 0.0.69 (beta, exact-pinned by uv.lock), existing ruff/pytest toolchain.

**Spec:** `docs/superpowers/specs/2026-08-06-modernization-campaign-design.md`

## Global Constraints

- setuptools stays the build backend; `publish.yml` is NOT modified (wheel production untouched).
- Published extras after this change: astronomy, geodesy, terrain, visualization, signal, gpu, gpu-apple, all. `dev` and `benchmark` extras are deleted from `[project.optional-dependencies]`; `all` no longer includes dev.
- The CI test matrix currently installs only `[dev]` — extras-dependent tests skip by design. Preserve this: matrix jobs use `uv sync --locked` (no extras). Do not add `--extra all` to the matrix.
- mypy.ini is NOT deleted in this plan — it feeds the probation job and dies in the v2.1.0 release PR.
- All work on branch `chore/uv-ty-migration`, PR to main, never direct to main.
- Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Any console output added must stay ASCII-safe (cp1252 rule).

---

### Task 1: pyproject restructure + uv.lock

**Files:**
- Modify: `pyproject.toml` (lines 49-100, the `[project.optional-dependencies]` block)
- Create: `.python-version`, `uv.lock`

**Interfaces:**
- Produces: dependency group `dev` (installed by `uv sync` by default) containing all former dev-extra packages plus `ty`; extra `all` = astronomy,geodesy,terrain,visualization,signal. Every later task's CI/docs commands assume `uv sync --locked` gives the dev toolchain and `--extra all` adds the user-facing extras.

- [ ] **Step 1: Create the branch**

```bash
cd "/Users/nedonatelli/Documents/Local Repositories/TCL"
git checkout main && git pull && git checkout -b chore/uv-ty-migration
```

- [ ] **Step 2: Edit pyproject.toml**

Delete the `dev = [...]` and `benchmark = [...]` entries from `[project.optional-dependencies]`, and change `all`:

```toml
all = [
    "nrl-tracker[astronomy,geodesy,terrain,visualization,signal]",
]
```

Append after the `[project.urls]` section (top-level table, exact content):

```toml
[dependency-groups]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "pytest-xdist>=3.0.0",
    "pytest-benchmark>=4.0.0",
    "pytest-timeout>=2.0.0",
    "nbval>=0.10.0",
    "hypothesis>=6.0.0",
    "ruff>=0.16.0",
    "mypy>=1.0.0",
    "ty>=0.0.69",
    "pre-commit>=3.0.0",
    "sphinx>=6.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "myst-parser>=1.0.0",
    "nbsphinx>=0.9.0",
    "sphinxcontrib-mermaid>=0.9.2",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
]
```

(`pytest-benchmark` was already in dev, which is why deleting the `benchmark` extra loses nothing.)

- [ ] **Step 3: Create `.python-version`**

File content, one line: `3.11`

- [ ] **Step 4: Verify the metadata is what the spec requires**

```bash
python3 -c "
import tomllib
d = tomllib.load(open('pyproject.toml','rb'))
extras = set(d['project']['optional-dependencies'])
assert extras == {'astronomy','geodesy','terrain','visualization','signal','gpu','gpu-apple','all'}, extras
assert 'dev' in d['dependency-groups']
assert not any('dev' in s for s in d['project']['optional-dependencies']['all'])
print('OK')
"
```

Expected: `OK`

- [ ] **Step 5: Generate the lock and sync**

```bash
uv lock && uv sync --locked
```

Expected: `uv.lock` created; `.venv` rebuilt with the dev group. If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh` first.

- [ ] **Step 6: Verify the toolchain works from the synced env**

```bash
uv run ruff check . && uv run pytest tests/ -n auto --benchmark-skip -m "not examples" -q
```

Expected: ruff clean; test suite passes (~6000 tests, extras-dependent tests skip — same as before).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version
git commit -m "chore: move dev tooling to dependency-groups, add uv.lock

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: ty configuration, clean against pytcl

**Files:**
- Modify: `pyproject.toml` (append `[tool.ty]` sections; replace the trailing `# Note: mypy config lives in mypy.ini` comment)
- Possibly modify: files under `pytcl/` (targeted `# ty: ignore[rule]` comments or real fixes)

**Interfaces:**
- Consumes: dev group from Task 1 (ty is installed by `uv sync`).
- Produces: `uv run ty check pytcl` exits 0. This exact command is the CI gate in Task 3 and the pre-commit hook in Task 5.

- [ ] **Step 1: Add ty config to pyproject.toml**

Replace the final comment line `# Note: mypy config lives in mypy.ini` with:

```toml
# mypy config lives in mypy.ini; both are retired when the ty probation
# ends in the v2.1.0 release PR (see the 2026-08-06 modernization spec).

[tool.ty.environment]
python-version = "3.10"

[tool.ty.src]
include = ["pytcl"]
```

- [ ] **Step 2: Run ty and capture the baseline**

```bash
uv run ty check pytcl 2>&1 | tail -5
```

Expected: either clean, or a diagnostic count. Record the count.

- [ ] **Step 3: Triage to zero**

Policy, in priority order:
1. Real defects ty found: fix the code.
2. A rule firing many times (>20) purely from numpy/numba dynamic typing: set it project-wide in `[tool.ty.rules]` with a one-line comment naming why, e.g.

```toml
[tool.ty.rules]
# numpy ArrayLike widening; same class of noise mypy.ini suppressed
possibly-unbound-attribute = "ignore"
```

3. Isolated false positives: per-site `# ty: ignore[rule-name]`.

Never blanket-ignore with `all = "ignore"`; never suppress a rule that flagged a genuine bug.

- [ ] **Step 4: Verify clean**

```bash
uv run ty check pytcl
```

Expected: exit 0.

- [ ] **Step 5: Verify tests still pass (if any code was changed in Step 3)**

```bash
uv run pytest tests/ -n auto --benchmark-skip -m "not examples" -q
```

Expected: PASS. Skip this step only if Step 3 touched zero files under `pytcl/`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml pytcl
git commit -m "chore: adopt ty as the type checker, clean against pytcl

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: ci.yml — uv everywhere, ty gates, mypy probation

**Files:**
- Modify: `.github/workflows/ci.yml` (all six jobs: lint, doctests, test, docs, examples, notebooks; add job mypy-probation)

**Interfaces:**
- Consumes: `uv.lock` + dev group (Task 1), `uv run ty check pytcl` (Task 2).
- Produces: the setup-uv step pattern reused verbatim in Task 4.

In every job, replace this pair of steps:

```yaml
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "X.Y"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install <whatever>
```

with:

```yaml
      - name: Install uv
        uses: astral-sh/setup-uv@v9
        with:
          python-version: "X.Y"
          enable-cache: true

      - name: Install dependencies
        run: uv sync --locked<flags per job, below>
```

keeping each job's original Python version, and prefix every `pytest`/`ruff`/`mypy`/`sphinx-build` invocation with `uv run `.

- [ ] **Step 1: lint job**

Sync flags: none (`uv sync --locked`). Delete the `pip install ruff==0.16.0 mypy==2.3.0 numpy` line — versions now come from uv.lock. Replace the mypy step (keep the existing comment block above it by moving it to the probation job):

```yaml
      - name: Type check with ty
        run: uv run ty check pytcl
```

- [ ] **Step 2: add mypy-probation job**

After the lint job, add:

```yaml
  # Non-blocking while ty is the gate: reveals anything ty misses relative
  # to mypy --strict. This job and mypy.ini are deleted in the v2.1.0
  # release PR (see docs/superpowers/specs/2026-08-06-modernization-campaign-design.md).
  # --strict, not just --ignore-missing-imports: the looser command had
  # been passing while 12 strict errors accumulated in pytcl/io.
  mypy-probation:
    runs-on: ubuntu-latest
    continue-on-error: true
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v9
        with:
          python-version: "3.11"
          enable-cache: true

      - name: Install dependencies
        run: uv sync --locked

      - name: Type check with mypy (non-blocking)
        run: uv run mypy pytcl --strict --ignore-missing-imports
```

- [ ] **Step 3: doctests job**

Sync flags: `--extra all` (was `pip install -e ".[all]" pytest pytest-timeout`; pytest/pytest-timeout are in the dev group). Keep the network-isolation comment and `--timeout=60`.

- [ ] **Step 4: test matrix job**

Sync flags: none — `[dev]` was the old install; extras-dependent tests must keep skipping (Global Constraints). Both pytest invocations (parallel and coverage) get the `uv run` prefix; everything else (matrix, if-conditions, Codecov upload) is unchanged.

- [ ] **Step 5: docs job**

Sync flags: none. Delete the extra `pip install sphinx-rtd-theme nbsphinx sphinxcontrib-mermaid` line — all three are in the dev group. `sphinx-build` becomes `uv run sphinx-build`; the ERROR/WARNING gating logic is untouched.

- [ ] **Step 6: examples job**

Sync flags: `--extra all` (was `.[all]`; the old `all` included dev, the new sync adds dev by default — same closure). Keep the comment about exercising the real dependency set.

- [ ] **Step 7: notebooks job**

Sync flags: `--extra visualization`, then add matplotlib into the env:

```yaml
      - name: Install dependencies
        run: |
          uv sync --locked --extra visualization
          uv pip install matplotlib
```

`python -m ipykernel ...` and `python examples/data/generate_datasets.py` and the inline `python -c` become `uv run python ...`; the nbval invocation becomes `uv run pytest --nbval-lax ...` (keep the do-not-swallow-exit-code comment).

- [ ] **Step 8: Validate YAML parses**

```bash
uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: run ci.yml through uv, gate types on ty with mypy probation

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: remaining workflows (docs, benchmarks, gpu)

**Files:**
- Modify: `.github/workflows/docs.yml`, `.github/workflows/benchmark-light.yml`, `.github/workflows/benchmark-full.yml`, `.github/workflows/gpu.yml`
- NOT modified: `.github/workflows/publish.yml` — it produces the PyPI artifacts; the spec keeps wheel production untouched.

**Interfaces:**
- Consumes: `uv.lock` + dev group (Task 1).

In each file, replace the `actions/setup-python@v5` step and the `pip install` step with (keeping that job's original Python version):

```yaml
      - name: Install uv
        uses: astral-sh/setup-uv@v9
        with:
          python-version: "X.Y"
          enable-cache: true

      - name: Install dependencies
        run: uv sync --locked
```

and prefix every `pytest`/`python`/`sphinx-build` run step with `uv run `.

- [ ] **Step 1: docs.yml**

Apply the pattern (python 3.11). Also drop the extra `pip install sphinx-rtd-theme nbsphinx` line — both are in the dev group.

- [ ] **Step 2: benchmark-light.yml and benchmark-full.yml**

Both currently `pip install -e ".[dev,benchmark]"` on Python 3.11. Replace with setup-uv (python 3.11) + `uv sync --locked` (pytest-benchmark is in the dev group). Prefix every `pytest`/`python` run step with `uv run`.

- [ ] **Step 3: gpu.yml**

Replace setup-python/pip block with setup-uv (python 3.12) + install, preserving the dispatch-selectable wheel:

```yaml
      - name: Install dependencies
        run: |
          uv sync --locked
          uv pip install "${{ github.event.inputs.cuda_wheel || 'cupy-cuda13x' }}"
```

Prefix run steps with `uv run`. Note in the PR description: gpu.yml is manual-dispatch on GPU hardware, so this file is verified on its next manual run, not by this PR's checks.

- [ ] **Step 4: Validate all four parse**

```bash
for f in docs benchmark-light benchmark-full gpu; do uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/$f.yml'))" || echo "FAIL $f"; done; echo done
```

Expected: `done` with no `FAIL` lines.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows
git commit -m "ci: run docs, benchmark, and gpu workflows through uv

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: pre-commit — ty replaces the mypy hook

**Files:**
- Modify: `.pre-commit-config.yaml`

**Interfaces:**
- Consumes: `uv run ty check pytcl` (Task 2).

- [ ] **Step 1: Replace the mirrors-mypy block**

Delete the entire `repo: https://github.com/pre-commit/mirrors-mypy` entry and append:

```yaml
  - repo: local
    hooks:
      - id: ty
        name: ty check
        entry: uv run ty check pytcl
        language: system
        types: [python]
        pass_filenames: false
```

- [ ] **Step 2: Verify the hook runs**

```bash
uv run pre-commit run ty --all-files
```

Expected: `ty check` passes.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "chore: replace mypy pre-commit hook with ty

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: contributor docs, README, CLAUDE.md, CHANGELOG

**Files:**
- Modify: `CONTRIBUTING.md` (Development Setup steps 2-3; the "PR with green CI" line ~144), `README.md` (extras list lines ~37-68), `ROADMAP.md` (setup block lines ~332-338), `CLAUDE.md` (extras list), `CHANGELOG.md` (new Unreleased section)

- [ ] **Step 1: CONTRIBUTING.md setup**

Replace steps 2-3 of Development Setup with:

````markdown
2. **Install uv** (once): https://docs.astral.sh/uv/getting-started/installation/

3. **Install development dependencies:**
   ```bash
   uv sync
   ```
   This creates `.venv` with the locked dev toolchain. Prefix commands with
   `uv run` (e.g. `uv run pytest`) or activate `.venv` as before.
````

Update the PR checklist line: `types (mypy, pinned)` becomes `types (ty, locked; mypy non-blocking during probation)`.

- [ ] **Step 2: README.md extras**

Delete the `pip install nrl-tracker[dev]` line and its description. Update the `[all]` description to "every user-facing extra except gpu (dev tooling is no longer a published extra — contributors use `uv sync`)".

- [ ] **Step 3: ROADMAP.md dev setup**

Replace the `python -m venv .venv && source .venv/bin/activate` / `pip install -e ".[dev]"` lines in "Getting Started for Potential Contributors" with `uv sync` / `uv run pytest`.

- [ ] **Step 4: CLAUDE.md**

Update the Quick Reference commands to `uv sync` + `uv run pytest` (keep `.venv/bin/python` forms working — uv's venv lives at `.venv`, so mention both). In Optional Dependency Extras, note dev moved to `[dependency-groups]` and `all` no longer includes it.

- [ ] **Step 5: Sweep for stragglers**

```bash
grep -rn "pip install -e" --include="*.md" --include="*.rst" . | grep -v _build | grep -v node_modules
```

Update any hit that instructs contributors (leave end-user `pip install nrl-tracker[...]` instructions alone — the package still installs fine with pip).

- [ ] **Step 6: CHANGELOG.md**

Add at the top:

```markdown
## [Unreleased]

### Changed
- Dev tooling moved from the published `dev`/`benchmark` extras to PEP 735
  dependency groups; `pip install nrl-tracker[dev]` no longer exists and
  `[all]` now contains only user-facing extras. Contributors: `uv sync`.
- Type checking is gated on ty; mypy runs non-blocking during a probation
  period ending at v2.1.0.
```

- [ ] **Step 7: Commit**

```bash
git add CONTRIBUTING.md README.md ROADMAP.md CLAUDE.md CHANGELOG.md docs
git commit -m "docs: document uv workflow and ty gate

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: ROADMAP reconciliation (campaign versions)

**Files:**
- Modify: `ROADMAP.md` (v2.1 section header/intro, Long-Term Vision version labels, Limitations table)

Per the spec: aspirational items lose their version pins; this campaign claims v2.1.0-v2.3.0. No content is deleted.

- [ ] **Step 1: Insert the campaign section**

After the "Session-identified, held deliberately out of 2.0.0" list, insert:

```markdown
### Modernization campaign (versioned; see docs/superpowers/specs/2026-08-06-modernization-campaign-design.md)

- **Tooling (no release):** uv-managed workflow (uv.lock, dependency
  groups, CI on uv) and ty as the type-check gate with a mypy probation
  ending at v2.1.0.
- **v2.1.0 — Diagnostics:** `pytcl.diagnostics` — loguru logging (silent by
  default, `enable_debug_logging()` opt-in), rich progress bars and track
  tables, instrumentation at gating/association/filter-health decision
  points.
- **v2.2.0 — Results I/O:** polars ingest (CSV/Parquet) and `to_polars()`
  results accessors (new `[dataframe]` extra); msgspec export of track
  histories/states to JSON and MessagePack. Delivers the Parquet/Arrow
  bullets below.
- **v2.3.0 — Typed configs + save/restore:** filter/tracker configs as
  `msgspec.Struct`s; full tracker state snapshot/resume.
- **Unversioned, gated:** `[visualization-xy]` extra for large-dataset
  plotting, once xy has a stable release.
```

- [ ] **Step 2: De-version the aspirational items**

- "Core Tracking Evolution": drop the `(v2.2+)`, `(v2.3+)`, `(v2.4+)` tags.
- "Infrastructure Maturation": reword the three `**v2.1**:/**v2.2**:/**v2.3**:` bullets to unversioned bullets ("RAPIDS, distributed tracking, advanced diagnostics", "Extended ecosystem (ROS 2, autonomous systems)", "Emerging tech (quantum, federated learning)") with a lead-in line "Unscheduled, in rough order:".
- Limitations table: `v2.1 with RAPIDS` becomes `Planned (RAPIDS; unscheduled)`.
- Footer: `**Next Milestone:** v2.1 (see the measured backlog)` becomes `**Next Milestone:** v2.1.0 (diagnostics; see the modernization campaign)`.

- [ ] **Step 3: Verify no stale version pins remain**

```bash
grep -n "v2\.[1-9]" ROADMAP.md
```

Expected: hits only in the campaign section, the breaking-changes-for-v2.0.0 section, and prose that references the campaign. No aspirational item keeps a version.

- [ ] **Step 4: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: reassign roadmap versions to the modernization campaign

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: PR and CI verification

- [ ] **Step 1: Push and open the PR**

```bash
git push -u origin chore/uv-ty-migration
gh pr create --title "chore: migrate to uv and ty" --body "$(cat <<'EOF'
Tooling track of the modernization campaign
(docs/superpowers/specs/2026-08-06-modernization-campaign-design.md).

- uv: committed uv.lock; dev/benchmark extras moved to PEP 735 dependency
  groups; all workflows except publish.yml run through uv sync --locked.
- ty: blocking type gate; mypy runs non-blocking (probation ends in the
  v2.1.0 release PR, which also deletes mypy.ini).
- publish.yml untouched: wheel production stays on setuptools/pip until
  the separate uv_build decision.
- gpu.yml is manual-dispatch on GPU hardware; its uv conversion is
  verified on its next manual run.
- ROADMAP: aspirational items de-versioned; campaign claims v2.1.0-v2.3.0.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Watch checks**

```bash
gh pr checks --watch
```

Expected: every job green — lint (ruff + ty), mypy-probation (may show findings; must not block), doctests, 10-job test matrix, docs, examples, notebooks, benchmark-light. If a job fails, fix on the branch and re-push; do not merge red.

- [ ] **Step 3: Confirm probation visibility**

Open the mypy-probation job log; record any mypy findings ty did not flag in a PR comment (they are the probation's whole point — the v2.1.0 decision data).
