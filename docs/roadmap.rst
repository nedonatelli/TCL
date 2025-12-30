Development Roadmap
===================

This document outlines the planned development phases for the Tracker Component Library.

Current State (v0.2.2)
----------------------

* **380 functions** implemented across 90 Python files
* **58% test coverage** with 355 tests
* **Core tracking functionality complete**: Kalman filters (KF, EKF, UKF, CKF), particle filters, coordinate systems, dynamic models, data association (GNN), multi-target tracking
* **Published on PyPI** as ``nrl-tracker``

Phase 1: Advanced Estimation & Data Association
-----------------------------------------------

Square-Root Filters
^^^^^^^^^^^^^^^^^^^
* Square-root Kalman filter (Cholesky-based)
* U-D factorization filter
* Square-root UKF and CKF variants

Joint Probabilistic Data Association (JPDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* JPDA for multi-target tracking
* Association probability computation
* Combined update with association probabilities

Multiple Hypothesis Tracking (MHT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Hypothesis tree management
* N-scan pruning
* Track-oriented MHT

Interacting Multiple Model (IMM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* IMM estimator
* Model probability mixing
* Markov transition matrix handling

Phase 2: Clustering & Mixture Reduction
---------------------------------------

Gaussian Mixture Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Gaussian mixture representation class
* Mixture moment matching
* Runnalls' mixture reduction algorithm
* West's algorithm

Clustering Algorithms
^^^^^^^^^^^^^^^^^^^^^
* K-means clustering
* DBSCAN
* Hierarchical clustering for track fusion

Phase 3: Static Estimation
--------------------------

Maximum Likelihood Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* ML estimator framework
* Fisher information computation
* Cramer-Rao bounds

Least Squares & Robust Estimation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* Weighted least squares
* Total least squares
* Robust M-estimators (Huber, Tukey)
* RANSAC

Phase 4: Container Data Structures
----------------------------------

Spatial Search Structures
^^^^^^^^^^^^^^^^^^^^^^^^^
* k-d tree implementation
* Ball tree
* R-tree for bounding boxes

Metric Trees
^^^^^^^^^^^^
* VP-tree (vantage point tree)
* Cover tree
* Efficient nearest neighbor queries

Phase 5: Geophysical Models
---------------------------

Gravity Models
^^^^^^^^^^^^^^
* Spherical harmonic evaluation
* EGM96 model support
* EGM2008 model support
* Tidal effects

Magnetic Field Models
^^^^^^^^^^^^^^^^^^^^^
* World Magnetic Model (WMM)
* International Geomagnetic Reference Field (IGRF)
* Enhanced Magnetic Model (EMM)

Terrain Models
^^^^^^^^^^^^^^
* Digital elevation model interface
* GEBCO integration
* Earth2014 support
* Terrain masking for visibility

Phase 6: Advanced Astronomical & Navigation
-------------------------------------------

Celestial Mechanics
^^^^^^^^^^^^^^^^^^^
* Two-body orbit propagation
* Kepler's equation solvers
* Orbital element conversions
* Lambert problem solver

Reference Frame Transformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
* GCRF/ITRF conversions
* Precession/nutation models
* Earth orientation parameters (EOP)
* Polar motion corrections

INS Mechanization
^^^^^^^^^^^^^^^^^
* Complete strapdown INS
* Coning/sculling corrections
* Error state models
* INS/GNSS integration

Phase 7: Signal Processing & Transforms
---------------------------------------

Signal Processing
^^^^^^^^^^^^^^^^^
* Digital filter design
* Matched filtering
* Detection algorithms (CFAR)

Transforms
^^^^^^^^^^
* Discrete Fourier transform utilities
* Short-time Fourier transform
* Wavelet transforms

Phase 8: Performance & Infrastructure
-------------------------------------

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^
* Expand Numba JIT coverage to critical paths
* Profile and optimize bottlenecks
* Consider Cython for hot spots

Documentation
^^^^^^^^^^^^^
* Complete API documentation
* Add tutorials and examples for new features
* Improve Read the Docs integration

Testing
^^^^^^^
* Increase test coverage to 80%+
* Add MATLAB validation tests for new functions
* Performance regression tests

Version Targets
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Version
     - Focus
   * - **v0.3.0**
     - Square-root filters, JPDA, IMM estimator
   * - **v0.4.0**
     - Clustering module, Gaussian mixture reduction
   * - **v0.5.0**
     - Static estimation, spatial data structures
   * - **v0.6.0**
     - Gravity and magnetic models
   * - **v0.7.0**
     - Complete astronomical code
   * - **v1.0.0**
     - Full feature parity, 80%+ test coverage

Contributing
------------

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality
5. Submit a pull request

See the `original MATLAB library <https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_ for reference implementations.
