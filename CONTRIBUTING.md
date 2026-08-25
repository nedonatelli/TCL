# Contributing to Tracker Component Library (Python)

Thank you for your interest in contributing! This document provides guidelines for contributing to the Python port of the Tracker Component Library.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   ```

2. **Install uv** (once): https://docs.astral.sh/uv/getting-started/installation/

3. **Install development dependencies:**
   ```bash
   uv sync
   ```
   This creates `.venv` with the locked dev toolchain. Prefix commands with
   `uv run` (e.g. `uv run pytest`) or activate `.venv` as before.

   **Apple Silicon:** `uv sync` alone does not install MLX — it lives in the
   `gpu-apple` extra. Run `uv sync --extra gpu-apple` so the MLX-gated GPU
   tests (see `PYTCL_REQUIRE_MLX=1` in the pipeline section below) actually
   execute instead of silently skipping.

4. **Install git hooks** (prek reads `.pre-commit-config.yaml` directly —
   it is a drop-in Rust replacement for the pre-commit framework, installed
   by `uv sync` as part of the dev group):
   ```bash
   uv run prek install
   ```

## Code Style

We follow these conventions:

- **Code formatting:** [Ruff](https://docs.astral.sh/ruff/) (`ruff format`, black-compatible style)
- **Linting & import sorting:** [Ruff](https://docs.astral.sh/ruff/) (`ruff check`)
- **Type hints:** Required for all public functions
- **Docstrings:** NumPy style
- **Printed output must be ASCII** — see below

### Text encoding

**Anything printed to the console must be ASCII.** No emoji, box-drawing
characters, arrows, or Greek letters in `print()` strings, banners, or table
borders.

On Windows, Python encodes stdout using the locale codepage (cp1252) whenever
stdout is a pipe or a file rather than a console. A character outside that
codepage raises `UnicodeEncodeError` and kills the script, so a demo that looks
fine in a terminal dies the moment someone redirects it to a log. Eight of the
example scripts used to fail this way, most on their opening banner.

Use `=` and `-` for rules, `->` for arrows, and spell out `pi`, `sigma`,
`sqrt`, `inf`. `°` and `±` are inside cp1252 and are safe to print.

Non-ASCII is fine in comments, docstring prose, and plot titles or axis
labels — those are read from UTF-8 source or written into a figure, and never
pass through the console encoder.

Reading files has the mirror-image problem: `Path.read_text()` and `open()`
default to the same locale encoding, so **always pass `encoding="utf-8"`**
explicitly. This caused a Windows CI failure reading the notebooks.

CI runs the examples on Ubuntu only, so it will not catch an encoding
regression. `tests/contract/test_console_encoding.py` guards it, and you can reproduce
the Windows behavior anywhere with:

```bash
PYTHONIOENCODING=cp1252 python examples/some_demo.py > /dev/null
```

### Example Function

```python
def cart2sphere(
    cart_points: ArrayLike,
    system_type: int = 0,
) -> NDArray[np.floating[Any]]:
    """
    Convert Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    cart_points : array_like
        Cartesian points with shape (3,) or (3, n) where each column is [x, y, z].
    system_type : int, optional
        Spherical coordinate system convention. Default is 0.
        - 0: [range, azimuth, elevation] with azimuth from x-axis in xy-plane
        - 1: [range, azimuth, elevation] with azimuth from y-axis

    Returns
    -------
    NDArray
        Spherical coordinates with shape (3,) or (3, n).
        Each column is [range, azimuth, elevation].

    Examples
    --------
    >>> cart2sphere([1, 0, 0])
    array([1.        , 0.        , 0.        ])

    >>> cart2sphere([[1, 0], [0, 1], [0, 0]])
    array([[1.        , 1.        ],
           [0.        , 1.57079633],
           [0.        , 0.        ]])

    Notes
    -----
    This is a port of Cart2Sphere.m from the MATLAB library.

    References
    ----------
    .. [1] D. F. Crouse, "The Tracker Component Library..."
    """
    ...
```

## Development Process

Every change to this codebase follows the same pipeline. These rules exist
because structural tests once passed while the magnetic field model was ~180°
wrong — passing tests are necessary, not sufficient.

### The pipeline

1. **Branch** — never commit to `main`. One feature branch per unit of work.
2. **Validate** — new or changed numerical code needs a *validation-class*
   test (see below), not just a smoke test.
3. **Run the full suite locally if you touch `pytcl/gpu`** — on Apple
   Silicon:

   ```bash
   PYTCL_REQUIRE_MLX=1 .venv/bin/python -m pytest
   ```

   149 tests do not execute without MLX, and **no CI runner has it**, so they
   skip on every runner and the build is green whatever their state. That is
   how 36 broken tests once reached `main`: a refactor missed three files
   under `tests/unit/`, every runner skipped them, and CI reported 14/14.
   `PYTCL_REQUIRE_MLX=1` turns "MLX is absent" from a silent skip into an
   error, so the command cannot be satisfied by accident. Every run also
   prints which backends were missing.

   The CuPy device layer has the same shape (`PYTCL_REQUIRE_CUPY=1`) and needs
   an NVIDIA machine; the `GPU` workflow covers it on demand.
4. **PR with green CI** — lint (ruff, pinned), types (ty, locked), tests
   (Ubuntu + macOS × 3 Python, plus Windows on 3.12 — 7 combinations, not a
   full 3×3 matrix), docstring examples (`--doctest-modules`), docs build
   (zero docutils errors), benchmarks (SLO enforcement on main). Green CI
   means the checks that ran passed; it does not mean the GPU layers were
   exercised.
5. **Merge, then release deliberately** — releases follow the checklist in
   the Release Process section; every user-facing fix gets a CHANGELOG entry
   when it lands, not at release time.

### Test validation classes

Every public function belongs to one of these classes, enforced by
`tests/contract/test_public_api_coverage.py` (the per-package ledger that
tracked the pre-2.0.0 audit, AUDIT.md, was removed when the audit closed;
it survives in git history). New code must land at REFERENCE or PROPERTY class;
STRUCTURAL-only tests are not accepted for numerical code.

| Class | Meaning | Example |
|-------|---------|---------|
| **REFERENCE** | Output compared against an independent implementation or published values | WMM vs. NOAA test values; UTM vs. pyproj; geodesics vs. geographiclib |
| **PROPERTY** | Mathematical invariants verified | round-trips, orthogonality, moment preservation, quadrature exactness |
| **STRUCTURAL** | Only shape/type/no-crash checked | `assert result.F > 0` |
| **UNTESTED** | No test exercises it | — |

Rules of thumb:
- If scipy/numpy/pyproj/geographiclib/astropy/sklearn implements it, compare
  against it (mind convention differences — document the mapping in the test).
- If no reference exists, test invariants the math guarantees.
- Reference values in tests must state their source in a comment.
- Docstring examples are executable documentation — they run in CI; keep
  outputs platform-robust (`round()`, `bool()`, seeds).
- Known-broken behavior is never silently skipped: mark it
  `# doctest: +SKIP (known bug: gh-NNN)` with a tracking issue.

### Property-based testing (Hypothesis)

`tests/property/` holds PROPERTY-class tests written against generated
inputs instead of fixed values — `hypothesis` draws the inputs and shrinks
any failure to a minimal reproducing example. Each module has a README-level
description of its target and invariants in `tests/property/README.md`;
this section is the durable policy (planning artifacts under
`docs/superpowers/` are gitignored, so nothing there is a substitute for
this).

**When to reach for a generated test versus an example-based one.** An
example-based test (`tests/unit/`, `tests/validation/`) fixes one input and
one expected output — necessary, but it only proves the function is right
*there*. A property test proves an invariant holds over a whole input space:
a round trip returns the original, a covariance stays PSD, an optimizer's
result matches a brute-force oracle. Reach for a property test when the
claim in a docstring or PR description is universal ("round-trips for any
finite state", "always PSD") — an example-based test cannot support that
claim, no matter how many examples you add by hand. Reach for example-based
instead when there is no independent oracle and no algebraic invariant to
check (most REFERENCE-class tests), or when a property test found a
counterexample and you are pinning it (see below) — the pin *is* an
example-based test.

**Profiles.** `tests/property/conftest.py` registers two Hypothesis
profiles and loads one based on the `CI` environment variable — GitHub
Actions sets `CI=true` on every runner automatically, so no workflow
configuration is needed for this to activate:

| Profile | Examples | Randomization | When |
|---------|----------|----------------|------|
| `ci` | 100 | `derandomize=True` | `CI` is set |
| `dev` | 500 | Hypothesis's own random seed, example DB on | `CI` is unset (local) |

`ci`'s derandomization means a red build is reproducible from the commit
alone — the same seed produces the same sequence of generated examples and
the same shrunk failure on every run, on every machine. `dev` explores
harder locally, where a slow or stalled run costs you time rather than
leaving a teammate unable to reproduce a CI failure. Verified by running
`CI=true uv run pytest tests/property/ -q --hypothesis-show-statistics`
twice in a row: identical pass/fail/invalid counts and identical
invalid-example percentages per test both times, and a deliberately broken
assertion shrinks to a byte-identical failing example and traceback (differing
only in Python object addresses) on repeated runs.

`.hypothesis/` (the local example database and cache Hypothesis writes) is
gitignored — it is regenerated by any run and has nothing to say about
another machine's or another commit's history.

**The narrowing rule is absolute.** Shrinking the input domain — excluding
the failing region from a strategy, tightening a `st.floats()` bound, adding
an `assume()` that quietly rules out the case that broke — is forbidden,
full stop, regardless of which outcome below applies. **A generator narrowed
until the failure disappears is a test that verifies nothing**: it still
runs, still passes, still looks like coverage in a diff, but it no longer
exercises the input that broke the code. That is strictly worse than having
no test, because a missing test is visibly absent and a narrowed one is not.

When a generator finds a genuine counterexample, decide which of these it
is, then respond accordingly:

- **Library defect.** The code is wrong. Fix it. If the fix can't land yet,
  pin the counterexample with `@pytest.mark.xfail(strict=True)` plus a
  tracking issue and a ROADMAP Known-Issues row.
- **Float64 conditioning limit, not a defect.** The assertion demanded more
  precision than IEEE 754 double-precision arithmetic can deliver in some
  input regime — the algorithm is correct, the number format isn't infinite
  precision. There is no bug to fix and no issue to open. Split the
  *assertion* by regime (tighter tolerance away from the ill-conditioned
  region, a looser or different check within it), pin the counterexample as
  a permanent **passing** regression test, and document the mechanism in a
  comment or docstring so the next reader sees the numerical argument, not
  just a magic tolerance. `tests/property/test_coordinate_properties.py` is
  the worked example: both of this suite's real counterexamples were
  float64-conditioning limits at the coordinate poles, not defects — the fix
  split the assertion by regime while leaving the pole- and
  subnormal-generating strategies untouched, and no issue was opened because
  there was nothing to track.

The choosing criterion: if a different, correct implementation could pass
the *original* assertion for this input, it's a defect — fix or xfail. If no
float64 implementation could ever pass the original assertion for this input
because the arithmetic itself is the limit, it's a conditioning limit —
split, pin, document.

**Counterexample promotion.** Every genuine counterexample a property test
finds gets promoted to a permanent example-based regression test beside the
property that found it, so the specific failing input stays pinned under
`ci`'s derandomized run even after whatever made the seed land there stops
recurring on its own.

### Tooling discipline

- CI tool versions are **pinned** (ruff, ty); bump deliberately in a PR
  that also fixes whatever the new version flags.
- Generated artifacts (benchmark history, plots) never mix into feature PRs.
- **Planning artifacts are not tracked.** Design specs, implementation
  plans, and agent-workflow scratch (`docs/superpowers/`, `.superpowers/`)
  are gitignored: they are scaffolding for one campaign, and nothing in the
  code, CI, or published docs reads them. Anything from a plan that must
  outlive it belongs somewhere a reader actually reaches -- a docstring, a
  `SOURCES.md`, the CHANGELOG, or a ROADMAP entry. Do not add cross-
  references to these paths from tracked files; they dangle for anyone who
  clones the repo.

## Testing

### Test layout

Tests are grouped by what they establish, not by which module they cover. Each
directory has a README stating what belongs in it.

File counts below are as of 2026-08-18 (`find tests/<dir> -maxdepth 1 -name
"test_*.py" | wc -l`); this table drifts as the suite grows, so treat the
counts as a snapshot rather than a maintained total.

| Directory | Holds | Files |
|-----------|-------|-------|
| `tests/unit/` | One function or class, expected values derived independently | 101 |
| `tests/validation/` | Checked against an outside implementation or published data | 45 |
| `tests/integration/` | More than one subsystem, composed as a caller would | 3 |
| `tests/contract/` | Assertions about the repository: examples run, notebooks execute, documented imports resolve | 12 |
| `tests/api/` | Public surface: exports, signatures, error contracts | 1 |
| `tests/property/` | Invariants over generated inputs (`hypothesis`, two profiles — see below) | 5 |
| `tests/characterization/` | Pins existing behavior where correctness is not established (empty by design) | 0 |

A new test goes in `unit/` unless one of the narrower directories clearly fits.
The distinction that matters most is `unit/` versus `validation/`: if the
expected value came from running something other than this codebase, it is
validation. That distinction exists because structural tests passed while WMM
magnetism was roughly 180 degrees wrong.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pytcl --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_core.py

# Run one category
uv run pytest tests/validation/
uv run pytest tests/contract/

# Run tests matching a pattern
uv run pytest -k "test_wrap"

# Run only fast tests
uv run pytest -m "not slow"
```

### Writing Tests

1. **Place tests in the appropriate directory:**
   - `tests/unit/` - Unit tests for individual functions
   - `tests/integration/` - Integration tests
   - `tests/fixtures/` - Test data files

2. **Use descriptive test names:**
   ```python
   def test_cart2sphere_single_point(): ...


   def test_cart2sphere_multiple_points(): ...


   def test_cart2sphere_raises_on_invalid_input(): ...
   ```

3. **Test against MATLAB reference values:**
   ```python
   @pytest.mark.matlab_validated
   def test_cart2sphere_matches_matlab():
       # Load reference values generated from MATLAB
       ref = np.load("tests/fixtures/cart2sphere_reference.npz")
       result = cart2sphere(ref["input"])
       np.testing.assert_allclose(result, ref["expected"], rtol=1e-12)
   ```

### Generating MATLAB Reference Data

For functions ported from MATLAB, generate reference test data:

```matlab
% In MATLAB with TrackerComponentLibrary loaded
input = randn(3, 100);
output = Cart2Sphere(input);
save('cart2sphere_reference.mat', 'input', 'output');
```

Then convert to NumPy format:
```python
from scipy.io import loadmat
import numpy as np

data = loadmat("cart2sphere_reference.mat")
np.savez("cart2sphere_reference.npz", input=data["input"], expected=data["output"])
```

## Porting Functions from MATLAB

When porting a function from the original MATLAB library:

1. **Study the original:**
   - Read the MATLAB code and comments thoroughly
   - Understand the algorithm and edge cases
   - Note any MATLAB-specific behaviors

2. **Follow naming conventions:**
   - MATLAB: `Cart2Sphere`, `FPolyKal`, `KalmanUpdate`
   - Python: `cart2sphere`, `poly_kalman_F`, `kalman_update`

3. **Handle array conventions:**
   - MATLAB uses column vectors by default
   - NumPy is row-major but we follow MATLAB conventions for state vectors
   - Document clearly when shapes differ

4. **Add the reference:**
   ```python
   Notes
   -----
   This is a port of Cart2Sphere.m from the MATLAB Tracker Component Library.

   References
   ----------
   .. [1] Original implementation:
          https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary
   ```

5. **Test thoroughly:**
   - Generate MATLAB reference values
   - Test edge cases (empty arrays, single values, etc.)
   - Test numerical accuracy

## Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/add-coordinated-turn-model
   ```

2. **Make your changes:**
   - Write code following style guidelines
   - Add tests
   - Update documentation

3. **Run quality checks:**
   ```bash
   # Format code
   uv run ruff format .

   # Lint (includes import sorting)
   uv run ruff check .

   # Type check (gate)
   uv run ty check pytcl

   # Run tests
   uv run pytest
   ```

4. **Submit the PR:**
   - Write a clear description
   - Reference any related issues
   - Ensure CI passes

## Current Development Status

**Version:** v2.6.0 (released). See "Current metrics (v2.5.0)" under
[Verify Current Metrics](#2-verify-current-metrics) below for up-to-date
function/module/test/coverage numbers -- this section used to duplicate
those and drift out of sync, so it now just points there.

The modernization campaign (uv-managed workflow, ty as the type-check gate,
diagnostics, results I/O, typed configs + session save/restore) is complete
as of v2.3.0, except for one unversioned, gated item: the
`[visualization-xy]` extra for large-dataset plotting, waiting on an
upstream `xy` stable release. See ROADMAP.md for details and for the
measured backlog of what remains.

## Priority Areas for Contributors

If you're looking for ways to contribute:

### High Priority
- Items in the measured backlog (see ROADMAP.md's "Measured backlog"
  section, derived from `docs/matlab_parity_inventory.rst`)
- Performance profiling and benchmarking
- Algorithm optimization and refactoring

### Medium Priority
- New test cases (especially edge cases)
- Documentation improvements
- Example script enhancements
- Bug fixes and code review

### Other Areas
- Astronomical code (consider using astropy)
- Gravity/magnetism models
- Terrain models
- Domain-specific optimizations

## Release Process

When preparing a new release, follow these steps:

### 1. Update Version Numbers

Update the version in these files:
- `pyproject.toml` - `version = "X.Y.Z"`
- `pytcl/__init__.py` - `__version__ = "X.Y.Z"`
- `CHANGELOG.md` - Add new version entry at top
- `ROADMAP.md` - Update version references if applicable

### 2. Verify Current Metrics

Before release, verify these metrics:
```bash
# Count functions
grep -r "^def " pytcl/ | wc -l

# Count modules
find pytcl -name "*.py" -type f | wc -l

# Run tests with collection
pytest --collect-only -q | tail -1

# Get coverage
pytest --cov=pytcl --cov-report=term
```

Current metrics (v2.6.0):
- **Functions:** 1,200+ (top-level `def`; measured 1,227 via
  `grep -r "^def " pytcl/ | wc -l`)
- **Modules:** 189
- **Tests:** 8,000+ (all passing; measured 8,035 via
  `pytest --collect-only -q`)
- **Coverage:** 88.80% as CI's gate measures it (`NUMBA_DISABLE_JIT=1`,
  `--cov-branch`, no MLX -- run 32393664562); 93.8% locally where the MLX
  layer is traceable. CI gate is `--cov-fail-under=85` (measurement minus
  the house 3-point headroom).

An earlier version of this list cited "100% MATLAB Parity (NRLMSISE-00, CEKF,
RBPF verified)". The NRLMSISE-00 entry was wrong -- the model was a barometric
approximation shipped under that name, found and corrected in gh-79 -- so the
parity line is retired here rather than repeated.
docs/matlab_parity_inventory.rst carries the honest per-component account.
- **GPU Backends:** 2 (CuPy + MLX)
- **Performance:** Numba JIT + lru_cache optimizations

### 2b. Re-read the claim-bearing docs pages

Rebuilding the docs does not re-read them. Before tagging, open and read
the pages that make claims about the project's current state --
`docs/index.rst` (the front-page badge line), `docs/roadmap.rst` (a
hand-maintained mirror of ROADMAP.md; mirrors drift), this file's metrics
block above, and any page naming the toolchain. The v2.1.0 release
shipped with the front page still claiming a type checker that release
itself had retired; this step exists because the checklist's "rebuild
docs" line did not catch that.

### 3. Verify Example References

Do NOT copy `examples/*.py` into `docs/examples/` -- an earlier version of
this step said to, and the copies drifted 238 lines from the canonical
scripts before being removed. `tests/contract/test_docs_references.py` now
fails the build if a second copy appears; every docs reference resolves to
the root `examples/` directory. This step is just:

```bash
uv run pytest tests/contract/test_docs_references.py -q
```

### 3b. Check the diff against the stability registry

```bash
uv run python scripts/check_release_stability.py <last-release-tag>
```

Lists every changed `pytcl` module with its registered maturity level, most
binding first. STABLE modules promise API frozen until a major bump; for each
one listed, decide -- and record in the CHANGELOG -- whether the change is
non-breaking, needs a major bump, or justifies reclassification. Added after
the post-v2.5.0 audit (2026-08) found a breaking change to a STABLE
module queued for a minor
release, with nothing positioned to notice.

### 4. Run Quality Checks

```bash
# Format code (also sorts imports via ruff check --fix)
uv run ruff format .

# Lint (includes import sorting)
uv run ruff check .

# Type check (gate)
uv run ty check pytcl

# Run full test suite with coverage
uv run pytest tests/ --cov=pytcl --cov-report=term-missing

# Run benchmark tests
uv run pytest benchmarks/ -v

# Verify all pass
echo "All checks complete!"
```

### 5. Verify Examples Run

Run each example script to ensure they execute without errors:

```bash
for f in examples/*.py; do echo "Running $f..."; python "$f" || exit 1; done
```

### 6. Update Roadmap Files

Update both `ROADMAP.md` and `docs/roadmap.rst`:

**In `ROADMAP.md`:**
- Check off completed items with `[x]` or ~~strikethrough~~
- Update phase status (e.g., "✅ Completed in vX.Y.Z")
- Add any new planned features discovered during development
- Update version targets for upcoming phases if needed

**In `docs/roadmap.rst`:**
- Update "Current State (vX.Y.Z)" section header with new version
- Update statistics (functions, tests, coverage)
- Add new completed phase under "Completed Phases"
- Update the "Version Targets" table with new release

### 7. Update Documentation

- Ensure all new features are documented
- Rebuild docs locally to verify: `cd docs && make html`

### 8. Create Release Commit and Tag

```bash
# Stage all changes
git add -A

# Create commit with comprehensive message
git commit -m "vX.Y.Z: Release description

- Feature 1
- Feature 2
- Bug fix 1
- Documentation updates

Quality metrics:
- Tests: #### passed
- Coverage: ##%
- MATLAB Parity: 100%"

# Create annotated tag with release notes
git tag -a vX.Y.Z -m "vX.Y.Z - Release Title

Release description and highlights"

# Push commits and tags
git push origin main
git push origin vX.Y.Z
```

### 9. Create GitHub Release

```bash
# Use GitHub CLI to create release
gh release create vX.Y.Z --title "vX.Y.Z - Release Title" --notes-file release_notes.md
```

### 10. Publish to PyPI (Optional)

```bash
# Build distribution
python -m build

# Upload to PyPI (requires credentials)
twine upload dist/*
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
