Development Roadmap
===================

This document outlines the development phases for the Tracker Component Library.

Current State (v0.14.1)
-----------------------

* **720+ functions** implemented across 130 Python files
* **1,255 tests** with comprehensive coverage
* **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN, JPDA, MHT), multi-target tracking
* **Gaussian mixture operations**: moment matching, Runnalls/West reduction algorithms
* **Complete clustering module**: K-means, DBSCAN, hierarchical clustering
* **Static estimation**: Least squares (OLS, WLS, TLS, GLS, RLS), robust M-estimators (Huber, Tukey), RANSAC, maximum likelihood estimation, Fisher information, Cramer-Rao bounds
* **Spatial data structures**: K-D tree, Ball tree, R-tree, VP-tree, Cover tree
* **Geophysical models**: Gravity (spherical harmonics, WGS84, J2, EGM96/EGM2008), Magnetism (WMM2020, IGRF-13, EMM, WMMHR)
* **Tidal effects**: Solid Earth tides, ocean tide loading, atmospheric pressure loading, pole tide
* **Terrain models**: DEM interface, GEBCO/Earth2014 loaders, line-of-sight, viewshed analysis
* **Map projections**: Mercator, Transverse Mercator, UTM, Stereographic, Lambert Conformal Conic, Azimuthal Equidistant
* **Astronomical code**: Orbital mechanics, Kepler propagation, Lambert problem, reference frame transformations
* **INS/Navigation**: Strapdown INS mechanization, coning/sculling corrections, alignment algorithms, error state model
* **INS/GNSS Integration**: Loosely-coupled and tightly-coupled integration, DOP computation, fault detection
* **Signal Processing**: Digital filter design (IIR/FIR), matched filtering, CFAR detection
* **Transforms**: FFT utilities, STFT/spectrogram, wavelet transforms (CWT, DWT)
* **Published on PyPI** as ``nrl-tracker``

Completed Phases
----------------

Phase 1: Advanced Estimation & Data Association (v0.3.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Square-root Kalman filter (Cholesky-based)
* U-D factorization filter
* Square-root UKF variants
* JPDA for multi-target tracking
* IMM estimator with model mixing

Phase 2: Clustering & Mixture Reduction (v0.4.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Gaussian mixture representation and moment matching
* Runnalls' and West's mixture reduction algorithms
* K-means with K-means++ initialization
* DBSCAN clustering
* Hierarchical clustering
* Multiple Hypothesis Tracking (MHT)

Phase 3: Static Estimation (v0.5.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Least squares variants (OLS, WLS, TLS, GLS, RLS)
* Robust M-estimators (Huber, Tukey) and RANSAC
* Maximum likelihood estimation framework
* Fisher information and Cramer-Rao bounds

Phase 4: Container Data Structures (v0.5.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* K-D tree and Ball tree
* R-tree for bounding boxes
* VP-tree (vantage point tree)
* Cover tree with O(c^12 log n) guarantee

Phase 5.1-5.2: Geophysical Models (v0.6.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Spherical harmonic evaluation
* WGS84/GRS80 gravity models
* Normal gravity (Somigliana formula)
* World Magnetic Model (WMM2020)
* International Geomagnetic Reference Field (IGRF-13)

Phase 6.1-6.2: Astronomical Code (v0.7.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Two-body orbit propagation
* Kepler's equation solvers
* Lambert problem solver (universal and Izzo)
* GCRF/ITRF reference frame transformations
* Precession/nutation models (IAU 1976/1980)

Phase 5.7: EGM High-Degree Gravity (v0.7.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* EGM96 model support (degree 360)
* EGM2008 model support (degree 2190)
* Clenshaw summation for numerical stability
* Geoid height computation
* Gravity disturbance/anomaly

Phase 5.3-5.4: Advanced Magnetic & Terrain (v0.8.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Enhanced Magnetic Model (EMM2017)
* High-resolution WMM (WMMHR) - degree 790
* Digital elevation model interface
* GEBCO bathymetry/topography loader
* Earth2014 terrain model loader
* Line-of-sight and viewshed analysis
* Terrain masking for radar coverage

Phase 5.5: Map Projections (v0.9.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Mercator projection (forward/inverse)
* Transverse Mercator projection
* UTM with zone handling and Norway/Svalbard exceptions
* Stereographic projection (oblique and polar)
* Lambert Conformal Conic projection
* Azimuthal Equidistant projection

Phase 5.6: Tidal Effects (v0.10.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Solid Earth tide displacement and gravity
* Ocean tide loading
* Atmospheric pressure loading
* Pole tide effects
* Love/Shida numbers (IERS 2010)

Phase 6.3: INS Mechanization (v0.11.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* INS state representation (position, velocity, attitude)
* Gravity computation (Somigliana formula)
* Earth/transport rates
* Coning/sculling corrections
* Strapdown mechanization (NED frame)
* Coarse and gyrocompass alignment
* 15-state error model

Phase 6.4: INS/GNSS Integration (v0.12.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* GNSS measurement models
* Satellite geometry and DOP computation
* Loosely-coupled integration
* Tightly-coupled integration with pseudoranges
* GNSS outage detection

Phase 7: Signal Processing & Transforms (v0.13.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Digital filter design (Butterworth, Chebyshev, elliptic, Bessel, FIR)
* Matched filtering and pulse compression
* CFAR detection (CA, GO, SO, OS, 2D)
* FFT utilities and power spectrum
* Short-time Fourier transform and spectrogram
* Wavelet transforms (CWT, DWT)

Phase 7.1: Performance Optimization (v0.13.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Numba JIT for CFAR detection algorithms
* Numba JIT for ambiguity function computation
* Numba JIT for batch Mahalanobis distance
* Numba JIT for rotation matrix utilities

Phase 8: Documentation (v0.14.0)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Complete API documentation for all modules
* Tutorials: Kalman filtering, nonlinear filtering, signal processing, radar detection, INS/GNSS, multi-target tracking
* Example scripts for common use cases
* Custom landing page with dark theme
* Sphinx RTD dark theme CSS

Phase 8.1: Test Coverage & Process (v0.14.1)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Test coverage boost: geodetic conversions, Jacobians, matrix decompositions
* Additional tests for process noise models, interpolation, statistics
* Release process documentation in CONTRIBUTING.md
* 1,255 tests (up from 1,109)

Planned Phases
--------------

Phase 8 (Remaining): Performance & Infrastructure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Profile and optimize additional bottlenecks
* Consider Cython for hot spots
* Increase test coverage to 80%+
* MATLAB validation tests for new functions
* Performance regression tests

Version Targets
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 65 20

   * - Version
     - Focus
     - Status
   * - **v0.3.0**
     - Square-root filters, JPDA, IMM estimator
     - Released
   * - **v0.4.0**
     - Clustering module, Gaussian mixture reduction, MHT
     - Released
   * - **v0.5.0**
     - Static estimation, spatial data structures
     - Released
   * - **v0.6.0**
     - Gravity and magnetic models (WGS84, WMM, IGRF)
     - Released
   * - **v0.7.0**
     - Complete astronomical code (orbit, Lambert, frames)
     - Released
   * - **v0.7.1**
     - EGM96/EGM2008 gravity models, Clenshaw summation
     - Released
   * - **v0.8.0**
     - EMM/WMMHR, terrain models, visibility
     - Released
   * - **v0.9.0**
     - Map projections
     - Released
   * - **v0.10.0**
     - Tidal effects (solid Earth, ocean, atmospheric)
     - Released
   * - **v0.11.0**
     - INS mechanization and navigation
     - Released
   * - **v0.12.0**
     - INS/GNSS integration
     - Released
   * - **v0.13.0**
     - Signal processing & transforms
     - Released
   * - **v0.13.1**
     - Numba JIT performance optimization
     - Released
   * - **v0.13.2**
     - Cross-platform scipy compatibility fix
     - Released
   * - **v0.14.0**
     - Documentation overhaul
     - Released
   * - **v0.14.1**
     - Test coverage boost, release process documentation
     - Released
   * - **v1.0.0**
     - Full feature parity, 80%+ test coverage
     - Planned

Contributing
------------

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality
5. Submit a pull request

See the `original MATLAB library <https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_ for reference implementations.
