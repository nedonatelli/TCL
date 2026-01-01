Development Roadmap
===================

This document outlines the development phases for the Tracker Component Library.

Current State (v0.21.5)
-----------------------

* **800+ functions** implemented across 144 Python files
* **1,530 tests** with comprehensive coverage
* **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA, MHT), multi-target tracking
* **Advanced assignment algorithms**: 3D assignment (Lagrangian relaxation, auction, greedy), k-best 2D (Murty's algorithm)
* **Gaussian mixture operations**: moment matching, Runnalls/West reduction algorithms
* **Complete clustering module**: K-means, DBSCAN, hierarchical clustering
* **Static estimation**: Least squares (OLS, WLS, TLS, GLS, RLS), robust M-estimators (Huber, Tukey), RANSAC, maximum likelihood estimation, Fisher information, Cramer-Rao bounds
* **Spatial data structures**: K-D tree, Ball tree, R-tree, VP-tree, Cover tree for efficient nearest neighbor queries
* **Tracking containers**: TrackList, MeasurementSet, ClusterSet for managing tracking data
* **Geophysical models**: Gravity (spherical harmonics, WGS84, J2, EGM96/EGM2008), Magnetism (WMM2020, IGRF-13, EMM, WMMHR)
* **Tidal effects**: Solid Earth tides, ocean tide loading, atmospheric pressure loading, pole tide
* **Terrain models**: DEM interface, GEBCO/Earth2014 loaders, line-of-sight, viewshed analysis
* **Map projections**: Mercator, Transverse Mercator, UTM, Stereographic, Lambert Conformal Conic, Azimuthal Equidistant
* **Astronomical code**: Orbital mechanics, Kepler propagation, Lambert problem, reference frame transformations
* **INS/Navigation**: Strapdown INS mechanization, coning/sculling corrections, alignment algorithms, error state model
* **INS/GNSS Integration**: Loosely-coupled and tightly-coupled integration, DOP computation, fault detection
* **Signal Processing**: Digital filter design (IIR/FIR), matched filtering, CFAR detection
* **Transforms**: FFT utilities, STFT/spectrogram, wavelet transforms (CWT, DWT)
* **Smoothers**: RTS smoother, fixed-lag, fixed-interval, two-filter smoothers
* **Information filters**: Standard and square-root information filters (SRIF)
* **Published on PyPI** as ``nrl-tracker``

Completed Phases
----------------

Summary of major phases completed through v0.21.5. See ROADMAP.md for detailed file listings and implementation notes.

- **Phase 1 (v0.3.0)**: Square-root filters, JPDA, IMM estimator
- **Phase 2 (v0.4.0)**: Gaussian mixture operations, clustering, MHT
- **Phase 3 (v0.5.0)**: Static estimation, spatial data structures
- **Phase 4 (v0.5.1)**: Container data structures (KD-tree, Ball tree, R-tree, VP-tree, Cover tree)
- **Phase 5.1-5.2 (v0.6.0)**: Geophysical models (gravity, magnetism)
- **Phase 5.7 (v0.7.1)**: EGM96/EGM2008 gravity models
- **Phase 5.3-5.4 (v0.8.0)**: Advanced magnetic models, terrain analysis
- **Phase 5.5 (v0.9.0)**: Map projections
- **Phase 5.6 (v0.10.0)**: Tidal effects
- **Phase 6.1-6.2 (v0.7.0)**: Orbital mechanics, reference frames
- **Phase 6.3 (v0.11.0)**: INS mechanization
- **Phase 6.4 (v0.12.0)**: INS/GNSS integration
- **Phase 7 (v0.13.0)**: Signal processing & transforms
- **Phase 8 (v0.16.0)**: Tracking containers
- **Phase 9 (v0.17.0)**: Advanced assignment algorithms
- **Phase 10 (v0.18.0)**: Batch estimation & smoothing
- **Phase 11 (v0.20.0)**: Navigation utilities
- **Phase 12 (v0.21.0)**: Special mathematical functions
- **Documentation & Testing**: Complete API documentation, 1,530+ tests, example scripts

Planned
-------

- **v1.0.0**: Full MATLAB TCL parity, 80%+ test coverage

Contributing
------------

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality
5. Submit a pull request

See the `original MATLAB library <https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_ for reference implementations.
