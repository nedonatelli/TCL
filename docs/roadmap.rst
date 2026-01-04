Development Roadmap
===================

This document outlines the development phases for the Tracker Component Library.

For comprehensive details including v2.0.0 planning, see `ROADMAP.md <../ROADMAP.md>`_.

Current State (v1.9.0) - Infrastructure Improvements Release
-----------------------------------------------------------

* **1,070+ functions** implemented across 150+ Python modules
* **2,133 tests** with 100% pass rate - fully production-ready
* **76% line coverage** across 16,209 lines (target: 80%+ in v2.0.0)
* **100% code quality** compliance with isort, black, flake8, mypy --strict
* **10-50x performance improvement** on network flow solver (Phase 1 complete)
* **42 interactive HTML visualizations** with Git LFS tracking
* **23 comprehensive example scripts** with Plotly-based interactive plots
* **100% MATLAB TCL parity** - all core features implemented
* **Benchmarking infrastructure**: Session-scoped fixtures, CI workflows, SLO tracking
* **Logging framework**: Hierarchical logging with performance instrumentation
* **Performance optimization**: 3-8x speedup on critical paths via Numba JIT, vectorization, caching
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
* **Astronomical code**: Orbital mechanics, Kepler propagation, Lambert problem, reference frame transformations, JPL ephemerides, relativistic corrections
* **INS/Navigation**: Strapdown INS mechanization, coning/sculling corrections, alignment algorithms, error state model
* **INS/GNSS Integration**: Loosely-coupled and tightly-coupled integration, DOP computation, fault detection
* **Signal Processing**: Digital filter design (IIR/FIR), matched filtering, CFAR detection
* **Transforms**: FFT utilities, STFT/spectrogram, wavelet transforms (CWT, DWT)
* **Smoothers**: RTS smoother, fixed-lag, fixed-interval, two-filter smoothers
* **Information filters**: Standard and square-root information filters (SRIF)
* **Documentation**: Interactive visualization system with 42 HTML plots
* **Code Quality**: 100% compliance with isort, black, flake8, mypy
* **Published on PyPI** as ``nrl-tracker``

Completed Phases
----------------

For detailed version history and implementation notes, see `ROADMAP.md <../ROADMAP.md>`_.

**Achieved Milestones:**

* v1.0.0 (Jan 1, 2026): Full MATLAB TCL parity with 830+ functions across 146 modules
* v1.1.0-v1.3.0 (Jan 2, 2026): Performance optimization phases with 3-8x speedups
* v1.6.0 (Jan 2, 2026): H-infinity filters, TOD/MOD frames, SGP4/SDP4 satellite propagation
* v1.7.2-v1.7.3 (Jan 4, 2026): Repository maintenance, Git LFS, test coverage analysis

Phase 15 (v1.1.0): Performance Infrastructure ✅
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Session-scoped pytest fixtures (30-40% runtime reduction)
* Benchmark SLO definitions with trend detection
* Two-tier CI benchmarking (light for PRs, full for main)
* Logging framework with performance instrumentation
* Unified module documentation template

Phase 16 (v1.3.0): Comprehensive Refactoring & Optimization ✅
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Three concurrent optimization tracks:**

**Track A: Mathematical Functions & Performance** ✅
  * Numba JIT: CFAR detection, matched filter, Debye functions
  * Vectorization: Matrix operations (2-5x improvement)
  * Caching: Common function inputs for 25-40% speedup

**Track B: Containers & Maintainability** ✅
  * Modularized Kalman filters: split sr_kalman.py into focused submodules
  * Input validation decorator system
  * Test coverage: 65%+ with parametrized tests
  * Improved container operations

**Track C: Geophysical Models & Architecture** ✅
  * LRU caching for magnetism, navigation, gravity lookups (2-3x speedup)
  * Lazy-load high-resolution models (EGM2008, Earth2014)
  * Architecture Decision Records (ADRs) for module patterns
  * Performance instrumentation with trend tracking

Planned: v2.0.0 Release (18 Months)
-----------------------------------

Comprehensive architectural upgrade targeting critical fixes, API standardization, GPU acceleration, and test expansion.

**Release Timeline:** Months 1-18 (Q1 2026 - Q4 2027)

**Key Objectives:**

* **Phase 1 (Months 1-3)**: Network flow performance [BLOCKER] - Replace Bellman-Ford with network simplex (50-100x faster)
* **Phase 2 (Months 2-4)**: API standardization - Unify spatial indexes, exception handling, optional dependencies
* **Phase 3 (Months 3-6)**: Documentation expansion - Complete module docstrings, function examples, module graduation
* **Phase 4 (Months 4-8)**: 8 Jupyter interactive notebooks covering Kalman, particle filters, tracking, GPU, networking
* **Phase 5 (Months 6-10)**: GPU acceleration Tier-1 - CuPy Kalman filters (5-10x), particle filters (8-15x)
* **Phase 6 (Months 7-12)**: Test expansion - Add 50+ new tests, increase coverage 76% → 80%+
* **Phase 7 (Months 8-12)**: Performance optimization - Numba JIT expansion, systematic caching, sparse matrices
* **Phase 8 (Months 13-18)**: Release preparation - alpha → beta → RC → v2.0.0 final

**v2.0.0 Success Metrics:**

+--------------------------------------+---------------+----------+
| Metric                               | Current       | Target   |
+======================================+===============+==========+
| Network flow tests passing           | 2,044/2,057   | 2,057/2,057 |
| Kalman duplicate code                | ~300 lines    | 0 lines  |
| Test coverage                        | 76%           | 80%+     |
| Unit tests                           | 2,057         | 2,100+   |
| GPU speedup (Kalman)                 | N/A           | 5-10x    |
| GPU speedup (particles)              | N/A           | 8-15x    |
| Jupyter tutorials                    | 0             | 8        |
| Documentation quality                | 85%           | 95%+     |
+--------------------------------------+---------------+----------+

**Phase 6 Test Expansion Details:**

+--------------------------+----------+--------+----------+-----+
| Module                   | Current  | Target | New Tests| Effort |
+==========================+==========+========+==========+=====+
| sr_ukf.py                | 6%       | 50%+   | 20       | 3w  |
| ud_filter.py             | 11%      | 60%+   | 15       | 2w  |
| square_root.py           | 19%      | 70%+   | 15       | 2w  |
| imm.py                   | 21%      | 60%+   | 15       | 2w  |
| detection.py             | 34%      | 65%+   | 20       | 3w  |
| filters.py               | 61%      | 75%+   | 10       | 2w  |
| loaders.py               | 60%      | 80%+   | 15       | 2w  |
| Integration/Property     | -        | -      | 10       | 2w  |
+--------------------------+----------+--------+----------+-----+
| **Total**                | **76%**  | **80%+**| **50+** | **15w** |
+--------------------------+----------+--------+----------+-----+

See `ROADMAP.md <../ROADMAP.md>`_ for comprehensive v2.0.0 planning, effort estimates, risk analysis, and detailed implementation strategies for all 8 phases.

Contributing
------------

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality
5. Submit a pull request

See the `original MATLAB library <https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary>`_ for reference implementations.
