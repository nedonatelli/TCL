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

## Priority Areas

If you're looking for ways to contribute, these areas are priorities:

### High Priority
- Coordinate system conversions (`coordinate_systems/conversions/`)
- Basic Kalman filter implementation (`dynamic_estimation/kalman/`)
- Constant velocity/acceleration models (`dynamic_models/discrete_time/`)
- Hungarian algorithm for assignment (`assignment_algorithms/`)

### Medium Priority  
- UKF and EKF implementations
- Coordinated turn models
- JPDA data association
- OSPA metric

### Lower Priority
- Astronomical code (consider using astropy)
- Gravity/magnetism models
- Terrain models

## Release Process

When preparing a new release, follow these steps:

### 1. Update Version Numbers

Update the version in these files:
- `pyproject.toml` - `version = "X.Y.Z"`
- `docs/conf.py` - `release = "X.Y.Z"`
- `docs/_static/landing.html` - Update the status badge version

### 2. Update Landing Page Statistics

In `docs/_static/landing.html`, update the statistics to reflect current metrics:
- **Version badge** (line ~730): `vX.Y.Z — 720+ Functions`
- **Test count** (line ~766): Run `pytest --collect-only` to get current test count
- **Other stats** as needed (functions, modules, etc.)

### 3. Sync Examples

Copy examples from the root `examples/` directory to `docs/examples/`:

```bash
cp examples/*.py docs/examples/
```

### 4. Run Quality Checks

```bash
# Sort imports
isort pytcl tests examples docs/examples

# Format code
black .

# Lint
flake8 pytcl tests examples docs/examples

# Type check
mypy pytcl examples docs/examples

# Run full test suite with coverage
pytest --cov=pytcl --cov-report=term-missing
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

### 8. Create Release Commit

```bash
git add -A
git commit -m "Release vX.Y.Z: <brief description>"
git tag vX.Y.Z
git push origin main --tags
```

### 9. Publish to PyPI

```bash
python -m build
twine upload dist/*
```

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas

Thank you for contributing!
