# CLAUDE.md

## Project Overview

Python port of the U.S. Naval Research Laboratory's Tracker Component Library (TCL).
**Version:** 2.0.0 | **Package name:** `nrl-tracker` | **Source:** `pytcl/`

## Quick Reference

```bash
# Activate venv (required — system python lacks test deps)
source .venv/bin/activate

# Run all tests
.venv/bin/python -m pytest

# Before merging anything that touches pytcl/gpu (Apple Silicon only).
# 149 tests are MLX-gated and no CI runner has MLX, so they skip everywhere
# and the build stays green regardless. This makes their absence an error.
PYTCL_REQUIRE_MLX=1 .venv/bin/python -m pytest

# Run specific test file
.venv/bin/python -m pytest tests/test_terrain_loaders.py -x -q

# Lint and format
.venv/bin/ruff check . --fix
.venv/bin/ruff format .
```

## Architecture

- `pytcl/core/` — Shared utilities: constants, exceptions, validation, array helpers, `paths.py` (data directory)
- `pytcl/core/paths.py` — Single source of truth for `get_data_dir()` / `ensure_data_dir()`. Used by terrain, magnetism, and gravity modules.
- All external data files live in `~/.pytcl/data/` (override with `PYTCL_DATA_DIR` env var)

## External Data Files

These are too large for the repo. Tests skip gracefully when files are absent (via `FileNotFoundError`/`DependencyError`).

| Dataset | File(s) in `~/.pytcl/data/` | Install Extra |
|---------|----------------------------|---------------|
| GEBCO 2025 | `GEBCO_2025.nc` (~7 GB) | `pip install nrl-tracker[terrain]` |
| Earth2014 | `Earth2014.SUR2014.1min.geod.bin` etc (~445 MB each) | `pip install nrl-tracker[terrain]` |
| WMMHR2025 | `WMMHR2025.COF` (~521 KB) | N/A (no extra dep) |

## Optional Dependency Extras

**Core deps:** numpy, scipy, numba, h5py

**Extras:** `astronomy` (astropy, jplephem), `geodesy` (pyproj, geographiclib), `terrain` (netCDF4), `visualization` (plotly), `optimization` (cvxpy), `signal` (pywavelets), `gpu` (cupy), `gpu-apple` (mlx), `dev` (test/lint/docs tooling), `all` (everything except gpu)

## Code Conventions

- **Units:** All angles in radians (not degrees) at API boundaries
- **Style:** ruff (format + lint + import sorting), line length 88, config in pyproject.toml
- **Docstrings:** NumPy style
- **Naming:** `snake_case` functions, `PascalCase` classes
- **Tests:** pytest, use `_data_skip = (FileNotFoundError, DependencyError)` for tests needing external data — never bare `except Exception`
- **Type annotations:** Use `Union[float, NDArray[np.floating]]` for scalar-or-array returns

## Key Decisions

- GEBCO 2025 is the default version (changed from 2024 in v1.15.0)
- EMM/WMMHR functions accept both scalar and array inputs
- `get_data_dir()` is centralized in `pytcl/core/paths.py` — do not duplicate
