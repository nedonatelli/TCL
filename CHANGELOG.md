# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-12-30

### Added
- New `pytcl.plotting` module with 30 visualization functions:
  - `ellipses.py`: Covariance ellipse/ellipsoid utilities (`covariance_ellipse_points`, `plot_covariance_ellipse`, etc.)
  - `tracks.py`: Trajectory visualization (`plot_trajectory_2d/3d`, `plot_tracking_result`, `create_animated_tracking`)
  - `coordinates.py`: Coordinate system visualization (`plot_coordinate_axes_3d`, `plot_euler_angles`, `plot_quaternion_interpolation`)
  - `metrics.py`: Performance metric plots (`plot_nees_sequence`, `plot_ospa_over_time`, `plot_error_histogram`)
- Interactive plotting examples:
  - `coordinate_visualization.py`: 3D rotation and coordinate system visualization
  - `filter_uncertainty_visualization.py`: Kalman filter covariance ellipse animations
- Comprehensive test suite with 170+ new tests
  - `test_coordinate_systems.py`: 53 tests for coordinate transforms and rotations
  - `test_dynamic_models.py`: 35 tests for state transition and process noise
  - `test_kalman_filters.py`: 33 tests for KF, EKF, UKF, CKF
  - `test_mathematical_functions.py`: 49 tests for matrix operations and geometry
  - `test_plotting.py`: 35 tests for plotting module

### Changed
- Test coverage increased from 29% to 58%

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
