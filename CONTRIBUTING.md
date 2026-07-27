# Contributing to Tracker Component Library (Python)

Thank you for your interest in contributing! This document provides guidelines for contributing to the Python port of the Tracker Component Library.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/nedonatelli/TCL.git
   cd TCL
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

## Code Style

We follow these conventions:

- **Code formatting:** [Ruff](https://docs.astral.sh/ruff/) (`ruff format`, black-compatible style)
- **Linting & import sorting:** [Ruff](https://docs.astral.sh/ruff/) (`ruff check`)
- **Type hints:** Required for all public functions
- **Docstrings:** NumPy style

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
3. **PR with green CI** — lint (ruff, pinned), types (mypy, pinned), tests
   (3 OS × 3 Python), docstring examples (`--doctest-modules`), docs build
   (zero docutils errors), benchmarks (SLO enforcement on main).
4. **Merge, then release deliberately** — releases follow the checklist in
   the Release Process section; every user-facing fix gets a CHANGELOG entry
   when it lands, not at release time.

### Test validation classes

Every public function belongs to one of these classes, tracked in
[AUDIT.md](AUDIT.md). New code must land at REFERENCE or PROPERTY class;
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

### Tooling discipline

- CI tool versions are **pinned** (ruff, mypy); bump deliberately in a PR
  that also fixes whatever the new version flags.
- Generated artifacts (benchmark history, plots) never mix into feature PRs.

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pytcl --cov-report=html

# Run specific test file
pytest tests/unit/test_core.py

# Run tests matching a pattern
pytest -k "test_wrap"

# Run only fast tests
pytest -m "not slow"
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
   ruff format .

   # Lint (includes import sorting)
   ruff check .

   # Type check
   mypy pytcl

   # Run tests
   pytest
   ```

4. **Submit the PR:**
   - Write a clear description
   - Reference any related issues
   - Ensure CI passes

## Current Development Status

**Version:** v1.16.0
**MATLAB Parity:** 100% ✅ (all tier 1-2 missing components verified)
**Test Suite:** 3,322 tests passing (docstring examples also run in CI)
**Code Coverage:** 80% (target 80%+ in v2.0.0) ✅
**Quality:** 100% compliance (ruff check, ruff format, mypy --strict)
**GPU Acceleration:** CuPy (NVIDIA) + MLX (Apple Silicon)
**Performance Optimization:** Numba JIT, lru_cache, sparse matrix support

## v2.0.0 Roadmap Progress

### Completed Phases
- **Phase 1** ✅: Network flow performance (50-100x faster)
- **Phase 2** ✅: API standardization (exceptions, spatial indexes, optional deps)
- **Phase 5** ✅: GPU acceleration (CuPy + MLX, 5-15x speedup)
- **Phase 6** ✅: Test expansion
- **Phase 7** ✅: Performance optimization (Numba JIT, lru_cache, sparse matrices)

- **Phase 3** ✅: Documentation expansion
- **Phase 4** ✅: Jupyter notebooks (9 verified in CI)
- **Phase 8** ✅: Track management (SQL + HDF5, migration tools)

### In Progress
- **Phase 9**: Release preparation (alpha → beta → RC → v2.0.0)

## Priority Areas for Contributors

If you're looking for ways to contribute:

### High Priority
- v2.0.0 release preparation and validation (see ROADMAP.md Phase 9)
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

Current metrics (v1.16.0):
- **Functions:** 1,048+
- **Modules:** 133
- **Tests:** 3,322 (all passing)
- **Coverage:** 80%
- **MATLAB Parity:** 100% (NRLMSISE-00, CEKF, RBPF verified)
- **GPU Backends:** 2 (CuPy + MLX)
- **Performance:** Numba JIT + lru_cache optimizations

### 3. Sync Examples

Copy examples from the root `examples/` directory to `docs/examples/`:

```bash
cp examples/*.py docs/examples/
```

### 4. Run Quality Checks

```bash
# Format code (also sorts imports via ruff check --fix)
ruff format .

# Lint (includes import sorting)
ruff check .

# Type check (strict mode)
mypy --strict pytcl

# Run full test suite with coverage
pytest tests/ --cov=pytcl --cov-report=term-missing

# Run benchmark tests
pytest benchmarks/ -v

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
