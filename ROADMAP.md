# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v1.7.4 (Released January 4, 2026)  
**Current Test Suite:** 2,057 tests passing, 13 skipped, 76% line coverage  
**Production Status:** Feature-complete MATLAB TCL parity achieved

---

## Table of Contents

1. [Current State](#current-state)
2. [Release History](#release-history)
3. [Performance Optimization (v1.1.0-v1.6.0)](#performance-optimization)
4. [v2.0.0 Comprehensive Roadmap](#v200-comprehensive-roadmap)
5. [Contributing](#contributing)

---

## Current State

### v1.7.4 - Documentation & Roadmap Updates (January 4, 2026)

**Status:** ✅ Released with production-quality code

- **1,070+ functions** implemented across 150+ Python modules
- **2,057 tests** with 100% pass rate (13 skipped network flow tests)
- **76% line coverage** (16,209 lines, 3,292 missing, 4,014 partial)
- **100% MATLAB TCL parity** achieved
- **100% code quality compliance:** isort, black, flake8, mypy --strict
- **42 interactive HTML visualizations** with Git LFS tracking
- **23 example scripts** with Plotly renderings
- **Published on PyPI** as `nrl-tracker`

#### New in v1.7.x Series

**v1.6.0 - H-infinity & Satellite Propagation**
- H-infinity filter: Robust minimax filtering for systems with model uncertainty
- TOD/MOD reference frames: Legacy True of Date and Mean of Date transformations
- SGP4/SDP4 satellite propagation: Full TLE-based propagation with TEME support

**v1.7.0 - Advanced Optimizations**
- Domain-specific optimization opportunities identified
- Performance caching infrastructure expanded

**v1.7.2 - Repository Maintenance**
- Git LFS cleanup: 4.2GB terrain_demo.html file removed
- Test consolidation: Merged redundant test files

**v1.7.3 - Test Framework Updates**
- HTML visualization system regenerated (11 interactive files)
- Test coverage analysis: Identified 50+ test expansion opportunities
- Code quality verification: 100% compliance across all tools

#### Core Features (Complete)

- **Performance SLO compliance reporting**: Automated reports with markdown/JSON output
- **Unified architecture documentation**: PERFORMANCE.md and ARCHITECTURE.md
- **Performance caching infrastructure**: LRU caching for 16+ functions
- **Modular Kalman filters**: KF, EKF, UKF, CKF, SR-KF, UD, SR-UKF, IMM, H-infinity
- **Advanced data association**: GNN, JPDA, MHT with full tracking pipelines
- **Advanced assignment algorithms**: Hungarian, auction, 3D assignment, k-best 2D (Murty)
- **Clustering**: K-means, DBSCAN, hierarchical, Gaussian mixture operations
- **Static estimation**: OLS, WLS, TLS, GLS, RLS, M-estimators, RANSAC, MLE
- **Spatial data structures**: K-D tree, Ball tree, R-tree, VP-tree, Cover tree
- **Tracking containers**: TrackList, MeasurementSet, ClusterSet for data management
- **Geophysical models**: Gravity (WGS84, J2, EGM96/EGM2008), Magnetism (WMM, IGRF, EMM, WMMHR)
- **Tidal effects**: Solid Earth tides, ocean loading, atmospheric pressure, pole tide
- **Terrain models**: DEM interface, GEBCO/Earth2014, line-of-sight, viewshed analysis
- **Map projections**: Mercator, Transverse Mercator, UTM, Stereographic, Lambert Conic, Azimuthal
- **Astronomical code**: Orbits, Lambert, reference frames (GCRF, ITRF, TEME, TOD, MOD), ephemerides, relativity
- **INS/Navigation**: Strapdown mechanization, coning/sculling, alignment, great circle, rhumb line
- **INS/GNSS Integration**: Loosely/tightly-coupled, DOP, fault detection
- **Signal Processing**: IIR/FIR filters, matched filtering, CFAR detection
- **Transforms**: FFT, STFT, spectrogram, CWT/DWT wavelets
- **Smoothers**: RTS, fixed-lag, fixed-interval, two-filter smoothers
- **Information filters**: Standard and square-root information filters (SRIF)
- **Special Functions**: Marcum Q, Lambert W, Debye, hypergeometric, Bessel
- **Code Quality**: 100% compliance with isort, black, flake8, mypy

---

## Release History

### v1.0.0 - Full MATLAB TCL Parity (January 1, 2026) ✅

**Milestone Achieved:** 100% feature parity with original MATLAB TCL

- 830+ functions across 146 modules
- 1,598 comprehensive tests (100% pass rate)
- 100% code quality compliance
- 42 interactive visualizations
- 23 example scripts
- Complete API documentation

### v0.22.0 - Advanced Astronomical (December 31, 2025) ✅

**New Features:**
- JPL Development Ephemeris (DE405, DE430, DE432s, DE440) with automatic kernel download
- Relativistic corrections: time dilation, Shapiro delay, precession effects
- 31 ephemeris tests + 37 relativity tests
- 68 new functions across astronomical module

**Statistics:**
- 830+ total functions (30 new)
- 147 total modules (3 new)
- 1,598 tests (68 new)
- 802 lines of example code

### v0.21.0 - Special Functions (December 31, 2025) ✅

**New Functions:**
- Marcum Q function with variants (Q1, log, inverse) and Swerling detection
- Lambert W function with real-valued and Wright omega variants
- Debye functions (Debye 1-4) with heat capacity and entropy
- Hypergeometric functions (0F1, 1F1, 2F1, 1F1U) with regularized forms
- Advanced Bessel: ratio, derivatives, zeros, Struve H/L, Kelvin functions

### v0.20.0 - Navigation Utilities (January 1, 2026) ✅

**New Features:**
- **Great circle:** Distance, azimuth, waypoints, intersections, cross-track distance
- **Rhumb line:** Spherical and ellipsoidal distance, navigation, intersections, waypoints
- **TDOA localization** using great circle geometry
- **Path comparison** utility for great circle vs rhumb line analysis

### v0.18.0 - Batch Estimation & Smoothing (December 31, 2025) ✅

**New Features:**
- Rauch-Tung-Striebel (RTS) smoother with time-varying parameters
- Fixed-lag smoother for real-time applications
- Fixed-interval smoother
- Two-filter smoother (Fraser-Potter form)
- Information filter and Square-root Information Filter (SRIF)
- Multi-sensor information fusion

### v0.17.0 - Advanced Assignment (December 31, 2025) ✅

**New Features:**
- 3D Assignment: Lagrangian relaxation, auction algorithm, greedy, 2D decomposition
- k-Best 2D Assignment: Murty's algorithm, ranked enumeration
- `assign3d()` unified interface with method selection

### v0.16.0 - Tracking Containers (December 31, 2025) ✅

**New Features:**
- **TrackList:** Track collection with filtering, querying, batch operations
- **MeasurementSet:** Time-indexed measurements with spatial queries
- **ClusterSet:** Track clustering with DBSCAN/K-means support

### Earlier Releases

| Version | Focus | Released |
|---------|-------|----------|
| **v0.15.0** | New example scripts, visualization system | Dec 31, 2025 |
| **v0.14.0** | Documentation overhaul, landing page | Dec 31, 2025 |
| **v0.13.0** | Signal processing & transforms (filters, matched filter, FFT, STFT, wavelets) | Dec 31, 2025 |
| **v0.12.0** | INS/GNSS integration (loosely/tightly-coupled, DOP, fault detection) | Dec 31, 2025 |
| **v0.11.0** | INS mechanization (strapdown, coning/sculling, alignment) | Dec 30, 2025 |
| **v0.10.0** | Tidal effects (solid Earth, ocean, atmospheric, pole tide) | Dec 30, 2025 |
| **v0.9.0** | Map projections (Mercator, UTM, Stereographic, LCC, Azimuthal) | Dec 30, 2025 |
| **v0.8.0** | EMM/WMMHR magnetic models (degree 790), terrain visibility | Dec 30, 2025 |
| **v0.7.0** | Orbital mechanics, Lambert, reference frames (GCRF, ITRF, TEME) | Dec 30, 2025 |
| **v0.6.0** | Gravity (WGS84, J2, EGM96/EGM2008), magnetism (WMM, IGRF) | Dec 30, 2025 |
| **v0.5.1** | ML estimation, Fisher info, R-tree, VP-tree, Cover tree | Dec 30, 2025 |
| **v0.5.0** | Static estimation (OLS, WLS, TLS, GLS, RLS), K-D/Ball trees | Dec 30, 2025 |
| **v0.4.0** | Gaussian mixtures (moment matching, reduction), clustering (K-means, DBSCAN, hierarchical), MHT | Dec 30, 2025 |
| **v0.3.0** | Square-root filters (SR-KF, UD), JPDA, IMM | Dec 30, 2025 |

---

## Performance Optimization

### Phase 15: Infrastructure Setup ✅ (v1.1.0)

**Benchmarking Framework**
- Session-scoped fixture caching (30-40% reduction in test runtime)
- Performance SLO definitions in `.benchmarks/slos.json`
- Trend detection and SLO violation reporting

**Performance Monitoring**
- `scripts/track_performance.py` - Commit-level performance history
- `scripts/detect_regressions.py` - Trend detection
- `.benchmarks/history.jsonl` - Time-series tracking

**CI/CD Benchmarking**
- Light benchmarking for PRs (2 min on hot-path functions)
- Full benchmarking for main/develop (10 min with SLO enforcement)

**Module Logging Framework**
- `pytcl/logging_config.py` with hierarchical logger setup
- Performance instrumentation decorators (`@timed`)
- Context managers for timing critical sections
- `PerformanceTracker` for cumulative statistics

**Module Documentation Template**
- Standardized across 146 modules
- Architecture, validation contract, logging spec, performance characteristics

### Phase 16: Parallel Refactoring ✅ (v1.3.0)

**Track A: Mathematical Functions & Performance**

*Modules:* `special_functions/`, `signal_processing/`, `transforms/`

- [x] Comprehensive benchmarks for special functions, signal processing, transforms
- [x] Numba JIT: CFAR detection, matched filter, Debye functions
- [x] Vectorization: Matrix operations in transforms
- [x] SLO Tracking: Performance SLOs defined and monitored

**Track B: Containers & Maintainability**

*Modules:* `containers/`, `dynamic_estimation/`

- [x] Modularization: `square_root.py` split into `ud_filter.py`, `sr_ukf.py`
- [x] RTree API Compatibility: `from_points()`, `query()`, `query_radius()`
- [x] Input Validation: `@validate_inputs` decorator framework

**Track C: Geophysical Models & Architecture**

*Modules:* `atmosphere/`, `magnetism/`, `navigation/`

- [x] Ionosphere module: Klobuchar, dual-frequency TEC, simplified IRI, scintillation
- [x] Magnetism caching: LRU caching for WMM/IGRF with quantized precision
- [x] Architecture documentation: ADR-001 (caching), ADR-002 (lazy-loading)

**Performance Results**
- Special Functions: 5-10x speedup via Numba JIT
- Signal Processing: 2-5x speedup via vectorization
- Geophysical: 2-3x speedup via caching
- Benchmark Setup: 30-40% reduction via fixture caching
- Overall: 3-8x performance improvement on critical paths

### Code Quality Infrastructure

**Status:** ✅ 100% Compliance

- **isort:** 243+ files organized, 1 fix applied in v1.7.3
- **black:** 242 files verified compliant
- **flake8:** 0 errors
- **mypy --strict:** 160 files, 0 type errors

---

## v2.0.0 Comprehensive Roadmap

**Release Target:** 18 months (Months 1-18, starting Q1 2026)  
**Status:** Planning Phase  
**Last Updated:** January 4, 2026

### Executive Summary

v2.0.0 targets architectural improvements, performance optimization, GPU acceleration, and documentation enhancement. The release focuses on:

- Resolving critical bottlenecks (network flow performance - Phase 1)
- Standardizing APIs (spatial indexes, exceptions - Phase 2)
- Expanding documentation (8 Jupyter notebooks - Phase 4)
- Implementing GPU acceleration (5-15x speedup - Phase 5)
- Expanding test coverage (+50 tests, 76%→80% - Phase 6)
- Performance optimization (Numba JIT, caching - Phase 7)

### v2.0.0 Key Metrics

| Metric | Current | Target (v2.0) |
|--------|---------|---------------|
| Network flow tests skipped | 13 | 0 |
| Kalman filter duplicate code | ~300 lines | 0 |
| Spatial index implementations standardized | 0/7 | 7/7 |
| Module docstring quality | 85% | 95%+ |
| Jupyter tutorials | 0 | 8 |
| GPU speedup (Kalman batch) | N/A | 5-10x |
| GPU speedup (particle filters) | N/A | 8-15x |
| Unit tests | 2,057 | 2,100+ |
| Test coverage | 76% | 80%+ |
| Documentation quality | ~85% | 95%+ |

### Phase 1: Critical Fixes & Foundation (Months 1-3)

#### 1.1 Network Flow Performance [BLOCKER]

**Problem:** 13 skipped tests due to Bellman-Ford O(VE²) timeout limitation

**Solution:** Implement network simplex algorithm with O(VE log V) complexity
- Create `pytcl/assignment_algorithms/network_flow_simplex.py`
- Maintain 100% backward API compatibility
- Target: 50-100x faster for 100+ node problems
- Add 10+ new tests for simplex edge cases (degeneracy, cycling)

**Effort:** HIGH (3 weeks)

**Acceptance Criteria:**
- All 13 tests pass without skip markers
- Simplex benchmark ≥50x faster than Bellman-Ford on 20x20
- 100% backward compatibility

#### 1.2 Circular Imports Resolution

**Problem:** Late imports in `kalman/sr_ukf.py` use `# noqa: E402`

**Solution:** Refactor using TYPE_CHECKING blocks
- Use `from typing import TYPE_CHECKING` for forward references
- Move runtime imports to function-level scope
- Document dependency graph

**Effort:** MEDIUM (2 weeks)

#### 1.3 Empty Module Exports

**Problem:** Three modules have empty/incomplete `__all__` exports

**Solution:** Define clear public APIs or consolidate

**Effort:** LOW (1 week)

#### 1.4 Kalman Filter Code Consolidation

**Problem:** ~300 lines of duplicate matrix operations across 4 implementations

**Solution:** Extract to `kalman/matrix_utils.py`

**Effort:** MEDIUM (2 weeks)

### Phase 2: API Standardization & Infrastructure (Months 2-4)

#### 2.1 Spatial Index Interface Standardization

**Problem:** 7 spatial index implementations have inconsistent interfaces

**Solution:** Standardize to unified interface with `NeighborResult` NamedTuple

**Effort:** MEDIUM (3 weeks)

#### 2.2 Custom Exception Hierarchy

**Problem:** 50+ generic ValueError/ImportError across modules

**Solution:** Define custom exception hierarchy in `pytcl.core.exceptions`

**Effort:** MEDIUM (2 weeks)

#### 2.3 Optional Dependencies System

**Problem:** Plotly imports scattered without consistent error handling

**Solution:** Implement `pytcl.optdeps` with decorators and installation groups

**Effort:** MEDIUM (2 weeks)

### Phase 3: Documentation Expansion & Module Graduation (Months 3-6)

#### 3.1 Module Docstring Expansion

**Problem:** 4 modules have minimal 1-line docstrings

**Effort:** LOW (1 week)

#### 3.2 Function-Level Documentation

**Problem:** 20+ high-level functions lack docstring examples

**Effort:** LOW-MEDIUM (2 weeks)

#### 3.3 Module Graduation System

**Problem:** 25 modules mixed maturity levels; unclear production-readiness

**Effort:** MEDIUM (2 weeks)

### Phase 4: Jupyter Interactive Tutorials (Months 4-8)

#### 4.1 Notebook Creation (8 notebooks)

**Location:** `docs/notebooks/` with `.gitattributes` for `nbstripout`

Eight comprehensive notebooks covering:
1. Kalman Filters - Fundamentals to advanced filtering
2. Particle Filters - Resampling strategies and applications
3. Multi-Target Tracking - Data association, JPDA, MHT
4. Coordinate Systems - Transformations and projections
5. GPU Acceleration - CuPy integration and performance
6. Network Flow Solver - Simplex algorithm walkthrough
7. INS/GNSS Integration - Navigation system integration
8. Performance Optimization - Profiling and benchmarking

**Effort:** HIGH (6 weeks)

#### 4.2 Supporting Infrastructure

**Dataset Handling:** Create `examples/data/` directory  
**Binder Integration:** Set up cloud execution  
**CI Validation:** `pytest --nbval` integration

**Effort:** MEDIUM (2 weeks)

### Phase 5: GPU Acceleration Tier-1 (Months 6-10)

#### 5.1 CuPy-Based Kalman Filters

**Implementations:**
- `CuPyKalmanFilter` - Linear KF with batch processing
- `CuPyExtendedKalmanFilter` - EKF with nonlinear models
- `CuPyUnscentedKalmanFilter` - UKF with sigma points
- `CuPySRKalmanFilter` - Square-root for numerical stability

**Performance Target:** 5-10x speedup

**Effort:** HIGH (4 weeks)

#### 5.2 GPU Particle Filters

**Implementations:**
- GPU-accelerated resampling
- Importance function evaluation
- Particle weight evolution

**Performance Target:** 8-15x speedup

**Effort:** MEDIUM (3 weeks)

#### 5.3 Matrix Utilities

**Utilities:**
- CuPy Cholesky/QR factorization
- Memory pooling for repeated allocations
- Auto-offload on memory exhaustion

**Effort:** MEDIUM (2 weeks)

### Phase 6: Test Expansion & Coverage Improvement (Months 7-12)

**Current Status:** 2,057 tests, 76% line coverage  
**Target:** 2,100+ tests, 80%+ line coverage  
**Timeline:** Months 7-12 (15 weeks effort, concurrent with Phase 5 GPU work)

#### 6.1 Network Flow Tests Re-enablement

**Status:** 13 tests skipped (in `test_network_flow.py`, lines 85-215)

**Resolution:** Phase 1 network simplex implementation enables all tests

**Effort:** Included in Phase 1

#### 6.2 Kalman Filter Variant Tests (20+ new tests)

**Target Modules:**
- `kalman/sr_ukf.py` (6% → 50%+)
- `kalman/ud_filter.py` (11% → 60%+)
- `kalman/square_root.py` (19% → 70%+)

**Test Areas:**
- State prediction accuracy across dimensions (1D-100D)
- Covariance updates with positive definiteness maintenance
- Numerical stability edge cases
- Singular covariance recovery
- Large state dimension scenarios

**Effort:** MEDIUM (3 weeks)

#### 6.3 Advanced Filter Tests (15+ new tests)

**Target Module:** `dynamic_estimation/imm.py` (21% → 60%+)

**Test Areas:**
- Mode probability transitions
- Model likelihood evaluation
- State estimate merging consistency
- Mode-matched filtering accuracy
- Real-world scenarios (multimodal targets, mode switching)

**Effort:** MEDIUM (2 weeks)

#### 6.4 Signal Processing Detection Tests (20+ new tests)

**Target Module:** `signal_processing/detection.py` (34% → 65%+)

**Test Areas:**
- CFAR detector variants (CA, GO, SO, OS, 2D)
- Detection probability vs false alarm trade-off
- Receiver Operating Characteristic (ROC) curves
- Threshold selection algorithms
- Multi-hypothesis detection scenarios

**Effort:** MEDIUM (3 weeks)

#### 6.5 Terrain Loader Error Path Tests (15+ new tests)

**Target Module:** `terrain/loaders.py` (60% → 80%+)

**Test Areas:**
- Missing file handling
- Corrupted data detection
- Invalid format handling
- Coordinate system validation
- Out-of-bounds queries
- File permission errors

**Effort:** LOW-MEDIUM (2 weeks)

#### 6.6 Signal Processing Filter Tests (10+ new tests)

**Target Module:** `signal_processing/filters.py` (61% → 75%+)

**Test Areas:**
- Filter design edge cases
- Frequency response validation
- Phase distortion analysis
- Filter stability verification
- Low/high pass filter transitions

**Effort:** MEDIUM (2 weeks)

#### 6.7 Integration & Property-Based Tests (10+ new tests)

**New Test Categories:**
- Multi-module workflows (tracking + coordinate transforms)
- End-to-end tracking pipeline
- Kalman filter invariants (positive definite covariance)
- Coordinate transform round-trip properties
- Assignment optimality verification

**Implementation:**
- Hypothesis property-based tests for algorithm invariants
- Integration test suite for common workflows
- Real-world scenario tests

**Effort:** MEDIUM (2 weeks)

#### Phase 6 Summary Table

| Module | Current | Target | New Tests | Effort |
|--------|---------|--------|-----------|--------|
| Kalman filters (sr_ukf, ud, sr) | 6-19% | 50-70% | 20 | 3w |
| IMM filter | 21% | 60% | 15 | 2w |
| Signal detection | 34% | 65% | 20 | 3w |
| Signal filters | 61% | 75% | 10 | 2w |
| Terrain loaders | 60% | 80% | 15 | 2w |
| Network flow | N/A* | 100% | 13 | 1w |
| Integration/Property | - | - | 10 | 2w |
| **Total** | **76%** | **80%+** | **50+** | **15w** |

*Network flow: Re-enable 13 skipped tests after Phase 1 fixes

### Phase 7: Performance Optimization (Months 8-12)

#### 7.1 JIT Compilation with Numba

**Target Functions:**
- Kalman filter sigma point computation
- JPDA measurement likelihood
- Particle filter weight computation

**Expected Speedup:** 20% per optimized function

**Effort:** MEDIUM (2 weeks)

#### 7.2 Systematic Caching with lru_cache

**Target Functions:**
- Coordinate system Jacobians
- Legendre polynomials
- Transformation matrices

**Expected Speedup:** 25-40%

**Effort:** LOW-MEDIUM (2 weeks)

#### 7.3 Sparse Matrix Support

**New Functionality:**
- Optional scipy.sparse support in `assignment_algorithms/nd_assignment.py`
- Sparse cost matrix representation for large problems

**Performance Target:** 50% memory reduction

**Effort:** MEDIUM (3 weeks)

### Phase 8: Release Preparation (Months 13-18)

#### 8.1 v2.0-alpha (Month 12)

**Deliverables:**
- ✅ Network flow simplex algorithm implemented and tested
- ✅ Kalman filter code consolidated
- ✅ Module graduation completed
- ✅ API standardization complete
- ✅ GPU Tier-1 working and benchmarked
- ✅ Test coverage 80%+

#### 8.2 v2.0-beta (Month 14)

**Deliverables:**
- ✅ All 8 Jupyter notebooks complete
- ✅ Documentation expansion complete
- ✅ 50+ new tests integrated
- ✅ GPU performance benchmarked
- ✅ Integration tests complete

#### 8.3 v2.0-RC1 (Month 16)

**Deliverables:**
- ✅ Migration guide complete
- ✅ Deprecation warnings in place
- ✅ Performance benchmarks documented
- ✅ Installation instructions updated

#### 8.4 v2.0.0 (Month 18)

**Final Release** with all improvements integrated

### v2.0.0 Timeline

| Phase | Duration | Focus Area | Status |
|-------|----------|-----------|--------|
| **1** | Months 1-3 | Network flow, circular imports, consolidation | 🔄 Planned |
| **2** | Months 2-4 | API standardization, exceptions, optdeps | 🔄 Planned |
| **3** | Months 3-6 | Documentation, module graduation | 🔄 Planned |
| **4** | Months 4-8 | 8 Jupyter notebooks + CI integration | 🔄 Planned |
| **5** | Months 6-10 | GPU acceleration (CuPy Kalman, particles) | 🔄 Planned |
| **6** | Months 7-12 | +50 tests, 80%+ coverage, network flow re-enable | 🔄 Planned |
| **7** | Months 8-12 | Numba JIT, caching, sparse matrices | 🔄 Planned |
| **8** | Months 13-18 | Alpha → Beta → RC → Release | 🔄 Planned |

### v2.0.0 Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| Network simplex algorithm complexity | High | Low | Thorough testing, research phase |
| GPU memory constraints | Medium | Medium | Auto-offload strategy, documentation |
| Breaking API changes → user friction | High | Low | Deprecation path, migration guide |
| Skipped test complexity (13 tests) | High | Low | Phased implementation, benchmarking |
| Jupyter notebook maintenance | Medium | Medium | CI validation, doctest format |
| Test expansion timeline | Medium | Medium | Distribute across phases |

### v2.0.0 Dependencies & Resources

**Technical Skills Required**
- Numerical algorithms (network simplex, Kalman filters)
- GPU programming (CuPy, CUDA)
- Python profiling and optimization
- Documentation writing
- CI/CD infrastructure
- Test design and property-based testing

**External Dependencies**
- CuPy 12.0+ (GPU support)
- Plotly 5.0+ (visualization)
- Numba (JIT compilation)
- Hypothesis (property-based testing)
- Jupyter ecosystem
- RAPIDS (future, v2.1)

---

## Contributing

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (black formatting, NumPy docstrings)
4. Add tests for new functionality (aim for 80%+ coverage)
5. Submit a pull request

See the [original MATLAB library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary) for reference implementations.

---

**Last Updated:** January 4, 2026  
**Next Review:** Month 3 (after Phase 1-2 completion)  
**v2.0.0 Target Release:** Month 18 (Q4 2027)
