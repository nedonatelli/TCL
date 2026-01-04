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

- **Code formatting:** [Black](https://black.readthedocs.io/) with default settings
- **Linting:** [Flake8](https://flake8.pycqa.org/)
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
   def test_cart2sphere_single_point():
       ...

   def test_cart2sphere_multiple_points():
       ...

   def test_cart2sphere_raises_on_invalid_input():
       ...
   ```

3. **Test against MATLAB reference values:**
   ```python
   @pytest.mark.matlab_validated
   def test_cart2sphere_matches_matlab():
       # Load reference values generated from MATLAB
       ref = np.load('tests/fixtures/cart2sphere_reference.npz')
       result = cart2sphere(ref['input'])
       np.testing.assert_allclose(result, ref['expected'], rtol=1e-12)
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

data = loadmat('cart2sphere_reference.mat')
np.savez('cart2sphere_reference.npz', 
         input=data['input'], 
         expected=data['output'])
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
   black .
   
   # Lint
   flake8 pytcl tests
   
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

**Version:** v1.9.0
**MATLAB Parity:** 100% ✅
**Test Suite:** 2,133 tests passing
**Code Coverage:** 76% (target 80%+ in v2.0.0)
**Quality:** 100% compliance (black, isort, flake8, mypy --strict)

## v2.0.0 Roadmap - 8 Phases Over 18 Months

### Phase 1: Network Flow Performance (NEXT)
**Focus:** Replace Bellman-Ford O(VE²) with network simplex O(VE log V)
- **Impact:** 50-100x faster network flow computations
- **Location:** `pytcl/assignment/network_flow.py`
- **Tests:** Will re-enable 13 currently skipped tests
- **Estimated:** 2-3 weeks

### Phases 2-5: Algorithm Optimization
- Phase 2: Kalman filter performance
- Phase 3: Tracking algorithm improvements
- Phase 4: Signal processing optimization
- Phase 5: Advanced estimation methods

### Phase 6: Test Expansion
- **Goal:** +50 new tests, target 80%+ coverage
- **Focus:** Edge cases, numerical stability, batch operations

### Phases 7-8: Documentation & Final Polish
- Phase 7: Documentation updates and examples
- Phase 8: Performance tuning and optimization

## Priority Areas for Contributors

If you're looking for ways to contribute:

### High Priority (v2.0.0 Phase 1)
- Network flow algorithm optimization (`assignment/network_flow.py`)
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

Current metrics (v1.9.0):
- **Functions:** 1,070+
- **Modules:** 150+
- **Tests:** 2,133 (all passing)
- **Coverage:** 76%
- **MATLAB Parity:** 100%

### 3. Sync Examples

Copy examples from the root `examples/` directory to `docs/examples/`:

```bash
cp examples/*.py docs/examples/
```

### 4. Run Quality Checks

```bash
# Sort imports
isort pytcl tests examples docs/examples scripts

# Format code
black .

# Lint
flake8 pytcl tests examples docs/examples scripts

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
