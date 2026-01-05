# Tracker Component Library (Python)

[![PyPI version](https://img.shields.io/badge/pypi-v1.9.2-blue.svg)](https://pypi.org/project/nrl-tracker/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Public Domain](https://img.shields.io/badge/License-Public%20Domain-brightgreen.svg)](https://en.wikipedia.org/wiki/Public_domain)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://img.shields.io/badge/tests-2133%20passing-success.svg)](https://github.com/nedonatelli/TCL)
[![MATLAB Parity](https://img.shields.io/badge/MATLAB%20Parity-100%25-brightgreen.svg)](docs/gap_analysis.rst)
[![Type Checking](https://img.shields.io/badge/mypy--strict-passing-brightgreen.svg)](mypy.ini)

A Python port of the [U.S. Naval Research Laboratory's Tracker Component Library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary), a comprehensive collection of algorithms for target tracking, estimation, coordinate systems, and related mathematical functions.

**1,070+ functions** | **153 modules** | **2,133 tests** | **100% MATLAB parity**

## Overview

The Tracker Component Library provides building blocks for developing target tracking algorithms, including:

- **Coordinate Systems**: Conversions between Cartesian, spherical, geodetic, and other coordinate systems
- **Dynamic Models**: State transition matrices for constant velocity, coordinated turn, and other motion models
- **Estimation Algorithms**: Kalman filters (EKF, UKF, CKF, H-infinity), particle filters, smoothers, and batch estimation
- **Assignment Algorithms**: Hungarian algorithm, auction algorithms, 3D/ND assignment, k-best assignments
- **Data Association**: Global Nearest Neighbor, JPDA, MHT for multi-target tracking
- **Mathematical Functions**: Special functions, statistics, numerical integration, and more
- **Astronomical Code**: SGP4/SDP4 propagation, TLE parsing, special orbits (parabolic/hyperbolic), ephemerides, relativistic corrections
- **Reference Frames**: GCRF, ITRF, TEME, TOD, MOD with full transformation chains
- **Navigation**: Geodetic calculations, INS mechanization, GNSS utilities, INS/GNSS integration
- **Geophysical Models**: Gravity (WGS84, EGM96/2008), magnetism (WMM, IGRF), atmosphere, tides, terrain
- **Signal Processing**: Digital filters, matched filtering, CFAR detection, transforms (FFT, STFT, wavelets)

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

# For visualization
pip install nrl-tracker[visualization]

# For development
pip install nrl-tracker[dev]

# Install everything
pip install nrl-tracker[all]
```

### From Source

```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
pip install -e ".[dev]"
```

## Quick Start

### Coordinate Conversions

```python
import numpy as np
from pytcl.coordinate_systems import cart2sphere, sphere2cart

# Convert Cartesian to spherical coordinates
cart_point = np.array([1.0, 1.0, 1.0])
r, az, el = cart2sphere(cart_point)
print(f"Range: {r:.3f}, Azimuth: {np.degrees(az):.1f}°, Elevation: {np.degrees(el):.1f}°")

# Convert back
cart_recovered = sphere2cart(r, az, el)
```

### Kalman Filter

```python
from pytcl.dynamic_estimation.kalman import KalmanFilter
from pytcl.dynamic_models import constant_velocity_model

# Create a constant velocity model
dt = 0.1
F, Q = constant_velocity_model(dt, dimension=2, process_noise_intensity=1.0)

# Initialize filter
kf = KalmanFilter(
    F=F,  # State transition matrix
    H=np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),  # Measurement matrix
    Q=Q,  # Process noise
    R=np.eye(2) * 10,  # Measurement noise
)

# Run filter
x_est, P_est = kf.predict()
x_est, P_est = kf.update(measurement)
```

### Assignment Problem

```python
from pytcl.assignment_algorithms import hungarian

# Cost matrix (tracks x measurements)
cost_matrix = np.array([
    [10, 5, 13],
    [3, 15, 8],
    [7, 9, 12],
])

# Solve assignment
assignment, total_cost = hungarian(cost_matrix)
print(f"Optimal assignment: {assignment}, Total cost: {total_cost}")
```

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
├── atmosphere/              # Atmosphere models, refraction
├── gravity/                 # Gravity models
├── magnetism/               # Magnetic field models
├── terrain/                 # Terrain elevation models
└── misc/                    # Utilities, visualization
```

## Documentation

- [API Reference](https://pytcl.readthedocs.io/en/latest/api/)
- [User Guides](https://pytcl.readthedocs.io/en/latest/user_guide/)
- [Examples](examples/) - 29 validated example scripts
- [Tutorials](docs/tutorials/) - 10 interactive tutorial modules

## Examples & Tutorials

The library includes 39 runnable code examples demonstrating all major features:

### Examples (29 files in `/examples/`)

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

**Status**: ✅ All 29 examples validated and passing (100% execution success)

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

- [API Reference](https://pytcl.readthedocs.io/en/latest/api/)
- [User Guides](https://pytcl.readthedocs.io/en/latest/user_guide/)
- [Examples](examples/) - 29 validated example scripts
- [Tutorials](docs/tutorials/) - 10 interactive tutorial modules

## Comparison with Original MATLAB Library

This library aims to provide equivalent functionality to the original MATLAB library with Pythonic APIs:

| MATLAB | Python |
|--------|--------|
| `Cart2Sphere(cartPoints)` | `cart2sphere(cart_points)` |
| `FPolyKal(T, xDim, order)` | `poly_kalman_F(dt, dim, order)` |
| `KalmanUpdate(...)` | `KalmanFilter.update(...)` |

Key differences:
- Function names use `snake_case` instead of `PascalCase`
- Arrays are NumPy arrays (row-major) vs MATLAB matrices (column-major)
- 0-based indexing vs 1-based indexing
- Object-oriented APIs where appropriate

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pytcl

# Run only fast tests
pytest -m "not slow"

# Run tests validated against MATLAB
pytest -m matlab_validated
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
pip install -e ".[dev]"
pre-commit install
```

### Running Quality Checks

```bash
# Format code
black .

# Lint
flake8 pytcl

# Type check
mypy pytcl

# Run all checks
pre-commit run --all-files
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
