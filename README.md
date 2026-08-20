# Tracker Component Library (Python)

[![PyPI version](https://img.shields.io/pypi/v/nrl-tracker.svg)](https://pypi.org/project/nrl-tracker/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Public Domain](https://img.shields.io/badge/License-Public%20Domain-brightgreen.svg)](https://en.wikipedia.org/wiki/Public_domain)
[![Linted and formatted with Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-7000%2B%20passing-success.svg)](https://github.com/nedonatelli/TCL)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](https://github.com/nedonatelli/TCL/actions)
[![Type Checking](https://img.shields.io/badge/types-ty-blue.svg)](pyproject.toml)

A Python port of the [U.S. Naval Research Laboratory's Tracker Component Library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary), a comprehensive collection of algorithms for target tracking, estimation, coordinate systems, and related mathematical functions.

**1,150+ functions** | **189 modules** | **7,000+ tests** | **85% coverage**

## Overview

The Tracker Component Library provides building blocks for developing target tracking algorithms, including:

- **Coordinate Systems**: Conversions between Cartesian, spherical, geodetic, and other coordinate systems
- **Dynamic Models**: State transition matrices for constant velocity, coordinated turn, and other motion models
- **Estimation Algorithms**: Kalman filters (KF, EKF, UKF, CKF, *CEKF*, H-infinity), particle filters (bootstrap, *RBPF*), smoothers, and batch estimation
- **Assignment Algorithms**: Hungarian algorithm, auction algorithms, 3D/ND assignment, k-best assignments
- **Data Association**: Global Nearest Neighbor, JPDA, MHT for multi-target tracking
- **Mathematical Functions**: Special functions, statistics, numerical integration (Gaussian quadrature, cubature rules including the full 10-variant seventh-order algorithm surface, Genz-Keister and Smolyak sparse grids), and more
- **Astronomical Code**: SGP4/SDP4 propagation, TLE parsing, special orbits (parabolic/hyperbolic), ephemerides, relativistic corrections
- **Reference Frames**: GCRF, ITRF, TEME, TOD, MOD with full transformation chains
- **Navigation**: Geodetic calculations, INS mechanization, GNSS utilities, INS/GNSS integration
- **Geophysical Models**: Gravity (WGS84, EGM96/2008), magnetism (WMM, IGRF, EMM, WMMHR2025), atmosphere (US Standard 1976/ISA, simplified thermosphere, ionosphere), tides, terrain (GEBCO 2025, Earth2014)
- **Signal Processing**: Digital filters, matched filtering, CFAR detection, transforms (FFT, STFT, wavelets)
- **GPU Acceleration**: CuPy (NVIDIA CUDA) and MLX (Apple Silicon) backends for batch Kalman filtering and particle filters
- **Results I/O**: CSV/Parquet measurement readers, polars DataFrame accessors for track histories, msgspec (MessagePack/JSON) serialization, ASDF archival export, HDF5 compression (measured 4.73x), and AIS/NMEA transponder decoding
- **Typed Configs & Sessions**: `msgspec.Struct` configs (`IMMConfig`, `GaussianSumConfig`, `RBPFConfig`, `SingleTargetConfig`, `MultiTargetConfig`) accepted via a keyword-only `config=` on the matching constructor, plus full state snapshot/resume (`pytcl.io.save_session`/`load_session`) for six tracker and filter classes, bit-exact resume for the four deterministic ones

## Installation

### Basic Installation

```bash
pip install nrl-tracker
```

### With Optional Dependencies

```bash
# For astronomy features (ephemerides, celestial mechanics)
pip install nrl-tracker[astronomy]

# For geodesy features (coordinate transforms, map projections)
pip install nrl-tracker[geodesy]

# For terrain data (GEBCO, Earth2014 via NetCDF)
pip install nrl-tracker[terrain]

# For visualization
pip install nrl-tracker[visualization]

# For signal processing (wavelets)
pip install nrl-tracker[signal]

# For polars DataFrame accessors on track histories and metrics
pip install nrl-tracker[dataframe]

# For AIS/NMEA transponder decoding
pip install nrl-tracker[ais]

# For ASDF archival export/import of tracks and states
pip install nrl-tracker[asdf]

# For GPU acceleration (NVIDIA CUDA)
pip install nrl-tracker[gpu]

# For GPU acceleration (Apple Silicon M1/M2/M3)
pip install nrl-tracker[gpu-apple]

# Install every user-facing extra except gpu (dev tooling is no longer a
# published extra — contributors use `uv sync`)
pip install nrl-tracker[all]
```

### From Source

```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
pip install -e .
```

## Quick Start

### Coordinate Conversions

```python
import numpy as np
from pytcl.coordinate_systems import cart2sphere, sphere2cart

# Convert Cartesian to spherical coordinates
cart_point = np.array([1.0, 1.0, 1.0])
r, az, el = cart2sphere(cart_point, system_type="az-el")  # tracking convention
print(
    f"Range: {r:.3f}, Azimuth: {np.degrees(az):.1f}°, Elevation: {np.degrees(el):.1f}°"
)

# Convert back
cart_recovered = sphere2cart(r, az, el, system_type="az-el")
```

### Kalman Filter

```python
import numpy as np
from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity

# Constant-velocity model, 2D state [x, vx, y, vy]
dt = 0.1
F = f_constant_velocity(dt, num_dims=2)
Q = q_constant_velocity(dt, sigma_a=1.0, num_dims=2)
H = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]])  # measure position only
R = np.eye(2) * 10.0

x = np.zeros(4)
P = np.eye(4) * 100.0
measurement = np.array([1.2, -0.7])

pred = kf_predict(x, P, F, Q)
upd = kf_update(pred.x, pred.P, measurement, H, R)
print(upd.x)  # updated state; upd.P, upd.y, upd.S, upd.K, upd.likelihood
```

### Assignment Problem

```python
import numpy as np
from pytcl.assignment_algorithms import hungarian

cost_matrix = np.array(
    [
        [10.0, 5.0, 13.0],
        [3.0, 15.0, 8.0],
        [7.0, 9.0, 12.0],
    ]
)

row_ind, col_ind, total_cost = hungarian(cost_matrix)
print(f"rows {row_ind} -> columns {col_ind}, total cost {total_cost}")  # cost 20.0
```

### GPU Acceleration

The library supports GPU acceleration for batch processing of multiple tracks:

```python
import numpy as np

from pytcl.gpu import is_gpu_available, get_backend, to_gpu, to_cpu

# 100 tracks with a 4-dimensional state each
states = np.random.randn(100, 4)
covariances = np.tile(np.eye(4), (100, 1, 1))
F = np.eye(4)
Q = np.eye(4) * 0.01

# Check GPU availability (auto-detects CUDA or Apple Silicon)
if is_gpu_available():
    print(f"GPU available, using {get_backend()} backend")

    # Transfer data to GPU
    x_gpu = to_gpu(states)  # (n_tracks, state_dim)
    P_gpu = to_gpu(covariances)  # (n_tracks, state_dim, state_dim)

    # Use batch Kalman filter operations
    from pytcl.gpu import batch_kf_predict

    x_pred, P_pred = batch_kf_predict(x_gpu, P_gpu, F, Q)

    # Transfer results back to CPU
    x_pred_cpu = to_cpu(x_pred)
```

**Supported backends:**
- **NVIDIA CUDA**: Via CuPy (`pip install nrl-tracker[gpu]`) — float64
- **Apple Silicon**: Via MLX (`pip install nrl-tracker[gpu-apple]`) — float32

The backend is automatically selected based on your platform. Batch Kalman,
EKF, UKF, particle-filter, and matrix operations all run on either backend.

Measured on Apple Silicon (MLX), batch linear Kalman predict+update versus a
per-track CPU loop, end-to-end including host-device transfers and result
materialization, after warm-up: **1.6x at 100 tracks, 13x at 1,000, 40x at
20,000** (August 2026).

> **Precision note:** MLX computes in float32 (it raises on float64 GPU
> operations), so results match the CPU implementations to ~1e-7 relative
> rather than machine epsilon. The unscented filter is especially sensitive:
> its default `alpha=1e-3` yields sigma-point weights of order 1e6, which
> float32 cannot resolve — use `alpha >= 0.1` on MLX (the library warns).

## Module Structure

```
pytcl/
├── core/                    # Foundation utilities and constants
├── mathematical_functions/  # Basic math, statistics, special functions
├── coordinate_systems/      # Coordinate conversions and transforms
├── dynamic_models/          # State transition and process noise models
├── dynamic_estimation/      # Kalman filters, particle filters
├── static_estimation/       # ML, least squares estimation
├── assignment_algorithms/   # 2D and multi-dimensional assignment
├── clustering/              # Mixture reduction, clustering
├── performance_evaluation/  # OSPA, track metrics
├── astronomical/            # Ephemerides, time systems
├── navigation/              # Geodetic, INS, GNSS
├── atmosphere/              # Standard atmosphere, thermosphere, ionosphere
├── gravity/                 # Gravity models
├── magnetism/               # Magnetic field models
├── terrain/                 # Terrain elevation models
├── containers/              # Spatial indexes, track/measurement containers
├── trackers/                # Multi-target trackers (GNN, JPDA, MHT)
├── io/                      # SQL/HDF5 storage, CSV/Parquet/DataFrame/ASDF/msgspec I/O
├── diagnostics/             # Opt-in loguru logging, ASCII-safe rich progress/tables
├── transponders/            # AIS/NMEA decoding (maritime)
├── gpu/                     # GPU acceleration (CuPy/MLX)
└── plotting/                # Covariance ellipses, tracks, metrics plots
```

## Examples & Tutorials

The library includes 42 runnable code examples demonstrating all major features:

### Examples (32 files in `/examples/`)

Comprehensive demonstrations of library functionality:
- **Tracking & Estimation**: Kalman filters, particle filters, smoothers
- **Assignment**: Hungarian algorithm, k-best assignments, 3D assignment
- **Coordinates**: Frame conversions, transformations, geodetic calculations
- **Dynamics**: State models, motion models, dynamic systems
- **Filtering**: Uncertainty visualization, multi-target tracking
- **Astronomy**: Ephemerides, orbital mechanics, relativistic corrections
- **Navigation**: INS/GNSS integration, geophysical modeling
- **Signal Processing**: Detection, filtering, transforms
- **Terrain & Atmosphere**: Elevation models, atmospheric properties

**Status**: ✅ All 32 examples run in CI on every push

### Tutorials (10 modules in `/docs/tutorials/`)

Interactive learning modules with visualizations:
- Assignment algorithms and 3D assignment problems
- Atmospheric and geophysical models
- Dynamical systems and reference frames
- Filtering and smoothing techniques
- Sensor fusion and advanced filtering
- Special functions and mathematical tools

**Status**: ✅ All 10 tutorials validated and passing (100% execution success)

## Documentation

- [API Reference](https://nedonatelli.github.io/TCL/api/)
- [User Guides](https://nedonatelli.github.io/TCL/user_guide/)
- [Examples](examples/) - 32 validated example scripts
- [Tutorials](docs/tutorials/) - 10 interactive tutorial modules

## Comparison with Original MATLAB Library

The core tracking workflow is fully ported and validated against independent
references; the complete function-level accounting is in
[docs/matlab_parity_inventory.rst](docs/matlab_parity_inventory.rst), and the
explicit name mappings plus calling-convention differences are in
[docs/matlab_migration_map.rst](docs/matlab_migration_map.rst). A taste:

| MATLAB | Python |
|--------|--------|
| `Cart2Sphere(cartPoints)` | `cart2sphere(cart_points)` |
| `discKalPred(x, P, F, Q)` | `kf_predict(x, P, F, Q)` |
| `KalmanUpdate(x, P, z, R, H)` | `kf_update(x, P, z, H, R)` — note `H`/`R` order |
| `FPolyKal(T, xDim, 1)` | `f_constant_velocity(dt, dim)` |

Key differences:
- Function names use `snake_case`; multiple return values become NamedTuples
- States are 1-D arrays and batches are `(N, dim)` row-major (coordinate
  conversions also accept MATLAB-style column layouts)
- 0-based indexing, explicit `unassigned_rows` instead of 0-sentinels

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=pytcl

# Run only fast tests
uv run pytest -m "not slow"

# Run tests validated against MATLAB
uv run pytest -m matlab_validated
```

A bare local `pytest` run picks up `tests/property/`'s Hypothesis-generated
property tests too, at the 500-example `dev` profile (CI runs a
derandomized 100-example profile instead); see
[`tests/property/README.md`](tests/property/README.md) for what's covered
and the profile policy.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
uv sync
uv run prek install
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full setup (installing uv,
`uv run` vs. activating `.venv`, the `gpu-apple` extra for Apple Silicon).

### Running Quality Checks

```bash
# Format code
uv run ruff format .

# Lint (includes import sorting)
uv run ruff check .

# Type check (gate)
uv run ty check pytcl

# Run all checks
uv run prek run --all-files
```

## Citation

If you use this library in your research, please cite the original MATLAB library:

```bibtex
@article{crouse2017tracker,
  title={The Tracker Component Library: Free Routines for Rapid Prototyping},
  author={Crouse, David F.},
  journal={IEEE Aerospace and Electronic Systems Magazine},
  volume={32},
  number={5},
  pages={18--27},
  year={2017},
  publisher={IEEE}
}
```

## License

This project is in the public domain, following the original MATLAB library's license. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original MATLAB library by David F. Crouse at the U.S. Naval Research Laboratory
- This port follows the Federal Source Code Policy (OMB M-16-21)

## Related Projects

- [FilterPy](https://github.com/rlabbe/filterpy) - Kalman filtering library
- [Stone Soup](https://github.com/dstl/Stone-Soup) - Framework for tracking algorithms
- [Astropy](https://www.astropy.org/) - Astronomy library for Python
