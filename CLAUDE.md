# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

pyTCL (Tracker Component Library) is a Python port of the U.S. Naval Research Laboratory's MATLAB library for target tracking algorithms. It provides 1,070+ functions across 150+ modules covering estimation, coordinate systems, assignment algorithms, navigation, and more.

**Package name on PyPI:** `nrl-tracker`

## Common Commands

```bash
# Install for development
pip install -e ".[dev]"

# Run all tests
pytest tests/ -x -q

# Run specific test file
pytest tests/test_<module>.py -v

# Run tests matching a pattern
pytest -k "test_kalman"

# Run only fast tests (skip slow ones)
pytest -m "not slow"

# Code quality (run before committing)
isort pytcl/ tests/ --check-only --diff
black pytcl/ tests/ --check
flake8 pytcl/ tests/

# Auto-fix formatting
isort pytcl/ tests/ && black pytcl/ tests/

# Type checking
mypy pytcl

# Build documentation
python -m sphinx docs docs/_build/html
```

## Code Style

- **Formatting:** black (line-length 88)
- **Import sorting:** isort (profile: black)
- **Linting:** flake8 (max-line-length 100)
- **Docstrings:** NumPy style
- **Type hints:** Required for public APIs (mypy --strict)

## Architecture

### Core Design Principles

1. **NumPy-native API:** All functions accept and return NumPy arrays
2. **Numba acceleration:** Performance-critical paths use `@njit(cache=True)`
3. **LRU caching:** Expensive computations (Legendre polynomials, precession matrices, geodesic calculations) use `@lru_cache` with quantized keys
4. **Lazy loading:** Large data files (EGM coefficients, ephemerides) load on first use
5. **Minimal core dependencies:** Only NumPy, SciPy, Numba required

### Module Organization

Key subpackages and their responsibilities:

- **`dynamic_estimation/`**: Kalman filters (KF, EKF, UKF, CKF, SR-KF, UD, IMM, H-infinity), particle filters, smoothers (RTS, fixed-lag), information filters
- **`dynamic_models/`**: State transition matrices for motion models (constant velocity, coordinated turn, Singer)
- **`coordinate_systems/`**: Conversions (Cartesian/spherical/geodetic), rotations (quaternions, Euler, DCM), map projections (UTM, Mercator, Lambert)
- **`assignment_algorithms/`**: Hungarian algorithm, auction, 3D/ND assignment, k-best (Murty), JPDA, MHT, gating
- **`navigation/`**: INS mechanization, GNSS utilities, geodesy (great circle, rhumb line), INS/GNSS integration
- **`astronomical/`**: Reference frames (GCRF, ITRF, TEME, TOD, MOD), SGP4/SDP4 propagation, ephemerides, relativity corrections
- **`gravity/`, `magnetism/`, `atmosphere/`**: Geophysical models (EGM96/2008, WMM, IGRF, ionosphere)

### Performance Patterns

**Numba-optimized hot paths:**
- `assignment_algorithms.gating`: Mahalanobis distance
- `assignment_algorithms.hungarian`: Hungarian algorithm core
- `coordinate_systems.rotations`: Rotation operations
- `mathematical_functions.signal_processing`: CFAR detection

**Cached computations (with key quantization):**
- `gravity.spherical_harmonics.associated_legendre` - 8 decimal precision
- `astronomical.reference_frames.precession_matrix` - 0.001 day precision
- `navigation.great_circle` - 10 decimal precision
- `magnetism.wmm.magnetic_field_spherical` - 6 decimal precision

### MATLAB Port Conventions

- Function names: `PascalCase` (MATLAB) → `snake_case` (Python)
- Arrays follow MATLAB column-vector conventions for state vectors
- Reference original function in docstring Notes section

## Release Preparation Checklist

When asked to "prepare for a new release", follow these steps in order:

### 1. Code Quality Checks (REQUIRED FIRST)

```bash
isort pytcl/ tests/ --check-only --diff
black pytcl/ tests/ --check
flake8 pytcl/ tests/
```

If checks fail: run tools without `--check` flags to auto-fix, commit fixes separately.

### 2. Run Tests

```bash
python -m pytest tests/ -x -q
```

All tests must pass before proceeding.

### 3. Version Bump

Update version in:
- `pyproject.toml` - `version = "X.Y.Z"`
- `pytcl/__init__.py` - `__version__ = "X.Y.Z"`
- `docs/conf.py` - `release = "X.Y.Z"`

### 4. Update ROADMAP.md

- Update "Current State" header to new version
- Add release to Version Targets table
- Mark newly completed features as done

### 5. Commit and Tag

```bash
git add -A
git commit -m "Release vX.Y.Z: <brief description>"
git tag -a vX.Y.Z -m "Release vX.Y.Z: <description>"
git push && git push origin vX.Y.Z
```

### 6. Create GitHub Release

Use `gh release create` with tag name, title, and release notes.
