# Claude Code Instructions for pyTCL

This file contains project-specific instructions for Claude Code when working on the pyTCL (Tracker Component Library) project.

## Release Preparation Checklist

When asked to "prepare for a new release", follow these steps in order:

### 1. Code Quality Checks (REQUIRED FIRST)

Run all code quality tools and fix any issues before proceeding:

```bash
# Import sorting
isort pytcl/ tests/ --check-only --diff

# Code formatting
black pytcl/ tests/ --check

# Linting
flake8 pytcl/ tests/
```

If any checks fail:
1. Run the tools without `--check` flags to auto-fix
2. Review and commit the fixes separately before the release commit
3. Re-run checks to verify all issues are resolved

### 2. Run Tests

Ensure all tests pass:

```bash
python -m pytest tests/ -x -q
```

All tests must pass before proceeding with the release.

### 3. Version Bump

Update version in these files (use semantic versioning):
- `pyproject.toml` - `version = "X.Y.Z"`
- `pytcl/__init__.py` - `__version__ = "X.Y.Z"`
- `docs/conf.py` - `release = "X.Y.Z"`

### 4. Update ROADMAP.md

- Update "Current State" header to new version
- Add release to Version Targets table
- Mark any newly completed features as done

### 5. Commit and Tag

```bash
git add -A
git commit -m "Release vX.Y.Z: <brief description>"
git tag -a vX.Y.Z -m "Release vX.Y.Z: <description>"
git push && git push origin vX.Y.Z
```

### 6. Create GitHub Release

Use `gh release create` with:
- Tag name
- Title
- Release notes summarizing changes since last release

## Code Style

- **Formatting**: black (line-length 88)
- **Import sorting**: isort (profile: black)
- **Linting**: flake8 (max-line-length 100)
- **Docstrings**: NumPy style
- **Type hints**: Required for public APIs

## Project Structure

```
pytcl/                    # Main package
├── dynamic_estimation/   # Kalman filters, particle filters, IMM
├── dynamic_models/       # Motion models (CV, CA, Singer, CT)
├── coordinate_systems/   # Spherical, geodetic, rotations
├── assignment_algorithms/# Hungarian, JPDA, MHT
├── navigation/           # INS, GNSS, geodesy
├── mathematical_functions/# Signal processing, transforms
└── ...

tests/                    # Test files
docs/                     # Sphinx documentation
```

## Common Tasks

### Adding a New Module
1. Create module in appropriate subpackage
2. Add exports to `__init__.py`
3. Create tests in `tests/`
4. Add API documentation in `docs/api/`

### Running Documentation Build
```bash
python -m sphinx docs docs/_build/html
```

### Running Specific Test Files
```bash
python -m pytest tests/test_<module>.py -v
```
