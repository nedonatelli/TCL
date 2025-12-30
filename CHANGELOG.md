# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-12-30

### Added
- Comprehensive example scripts demonstrating library capabilities:
  - `coordinate_systems.py`: Coordinate transforms (spherical, geodetic, ENU/NED, rotations, quaternions)
  - `kalman_filter_comparison.py`: KF vs EKF vs UKF comparison with NEES/NIS metrics
  - `navigation_geodesy.py`: Geodetic conversions, distance calculations, waypoint navigation
  - `performance_evaluation.py`: OSPA metric, filter consistency testing, Monte Carlo evaluation

### Changed
- Switched visualization from matplotlib to plotly for interactive plots
- Updated all existing examples to use plotly

## [0.1.1] - 2025-12-29

### Added
- Read the Docs configuration
- Package prepared for PyPI publishing
- CI workflow using flake8 for linting

### Changed
- Renamed package from `tracker_component_library` to `tcl` for simpler imports
- Updated all imports across the codebase

## [0.1.0] - 2025-12-28

### Added
- Initial release of the Python port
- Core mathematical functions and utilities
- Coordinate system transformations (Cartesian, spherical, geodetic, ECEF, ENU/NED)
- Dynamic models (constant velocity, constant acceleration, coordinated turn)
- Kalman filters (linear, extended, unscented)
- Assignment algorithms (Hungarian, auction, GNN)
- Multi-target tracking with track management
- Navigation utilities (geodetic calculations, INS algorithms)
- Astronomical functions (ephemerides, celestial mechanics)
- Atmospheric models
- Performance evaluation metrics (OSPA, NEES, NIS)
