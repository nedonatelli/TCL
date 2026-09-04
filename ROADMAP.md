# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v2.8.0, released 3 September 2026
**Test Suite:** 8,200+ tests passing, ty-checked; every exported function is reached by a test with no standing exemptions (enforced by `tests/contract/test_public_api_coverage.py`, so the count tracks the surface automatically)
**Status:** On parity: the core tracking workflow is fully ported and
oracle-validated, and the full MATLAB surface is covered at roughly a
third by function count — see `docs/matlab_parity_inventory.rst`, which
supersedes any "feature-complete parity" phrasing. Per-release feature
summaries live in the CHANGELOG, not here.

This document covers **planned and future work only**. For what has already shipped, see
[CHANGELOG.md](CHANGELOG.md), the [GitHub releases](https://github.com/nedonatelli/TCL/releases),
and git history.

---

## Table of Contents

1. [Backlog](#backlog)
2. [Performance Targets](#performance-targets)
3. [Known Issues & Planned Fixes](#known-issues--planned-fixes)
4. [Long-Term Vision](#long-term-vision)
5. [Community Contribution Priorities](#community-contribution-priorities)
6. [Contributing](#contributing)

---

## Backlog

### Measured backlog (from the MATLAB parity inventory)

`docs/matlab_parity_inventory.rst` was produced by walking every directory of
the MATLAB TCL repository; its Absent/Weak/Divergent verdicts are the
evidence-based candidate list, as distinct from the aspirational items below.
No dates are attached because none have been decided:

- **Cubature point library remainder** — within the measured 79-file
  region-cubature subset (Cube_Space / Simplex / Sphere /
  Spherical_Surface), the 48 dimension-specialized subdirectory files
  (fixed 2D/3D formulas in Cube/, Square/, Tetrahedra/, Triangles/)
  deferred pending a consumer that needs their smaller point counts. The
  seven region directories outside that subset are now inventoried against
  the MATLAB tree: `Prism` (10 files), `Pyramid` (10), `Cross_Polytope`
  (3), `Exp_Weight` (3), `Weighted_Ellipse` (3), `Hexagon` (2) and
  `Spherical_Shell` (2) -- 33 files, none ported. (`Cubature_Points/` holds
  twelve subdirectories, not eleven; the twelfth, `Gaussian_Weight`, is not
  a gap -- all 10 of its files are already ported as `cubature_points.py`.)
  A further 12 files sit loose at the `Cubature_Points/` top level, of
  which two are ported (`calcCubPointMoments.m`, `thirdOrderStudentTCubPoints.m`);
  the rest are 1-D quadrature helpers plus single-region rules with no
  directory of their own, notably `thirdOrderTorusCubPoints.m` and
  `thirdOrderNDimShellCubPoints.m`. Three more `Cube_Space` files
  (`ClenshawCurtisPoints1D.m`, `FejerPoints1D.m`, `conformMapQuadPts1D.m`)
  are 1-D quadrature building blocks, not region-cubature rules themselves
  (no region-dimension argument) -- better scoped as a future
  `quadrature.py` extension than `region_cubature.py`, deferred pending a
  consumer.
- **Refraction suite remainder** — the suite itself shipped in v2.7.0
  (see CHANGELOG). Still open: the gas-table speed-of-sound algorithm
  (blocked on NRLMSISE-00); Jacchia 1971 (pure MATLAB and portable, but
  its validation oracle is the astrodynamics MEX chain plus JPL ephemeris
  data -- an astronomy-integration task); NRLMSISE-00 proper and its
  dependents as a standalone project (the MATLAB `.m` files are MEX
  stubs; the complete public-domain C implementation ships in the MATLAB
  tree and compiles/runs headlessly as a validation oracle).
  `Design_of_Lenses/` (3 optics files) stays excluded as out of scope
- **Localization-style static estimators, remainder** — nine of the 11
  shipped in v2.8.0 as `static_estimation.localization`. Still open:
  only the two `Uses_External_Solver` files (`DopplerOnlyInit6D`,
  `polyMeasConvert` — need the SCS solver or a scipy substitute)
- **Filter variants** — EnKF, ESRIF, QMC-Kalman, BLUE measurement updates,
  batch least squares, PCRLB/Riccati analysis tools
- **Direction-cosine UV measurement coordinates, remainder** — the core
  conversions shipped in v2.8.0 as `coordinate_systems.conversions.uv`
  (u-v <-> spherical, full bistatic r-u-v <-> Cartesian, ruv-to-ruv,
  r-u-v state conversion, camera-to-uv). Still open: the u-v measurement
  Jacobians/Hessians and the cubature/Taylor covariance conversions
  (`uv2SpherAngCubature`, `ruv2RuvCubature`, `monostatRuv2CartTaylor`,
  `cameraCoords2UVCoordsCubature`)
- **Time scales** — TDB/TCB/TCG, Besselian epochs, sidereal local time
- **Magnetic coordinate systems** — apex, quasi-dipole, centered-dipole
- **MOSPA/MMOSPA metrics**, interval scheduling, polynomials
- **NRLMSISE-00 proper** — load the NOAA coefficient tables and retire the
  barometric approximation's caveats (gh-79), plus HWM winds
- **HDF5 `states_only` covariance-transform mode** — a ~6.3x compression
  ceiling is reachable by reconstructing per-scan covariance from a
  steady-state Cholesky factor, but it touches every read path and breaks
  the bit-exact round-trip contract; deferred unless a real need appears
  (v2.2.0 measured and shipped byte-shuffle at 4.73x instead — see the
  CHANGELOG for the closed measurement).

### Modernization campaign — remaining

- **Unversioned, gated:** `[visualization-xy]` extra for large-dataset
  plotting, once xy has a stable release. It is the only item the campaign
  has left.

### Enhanced GPU Support

#### RAPIDS Integration for Distributed Computing

- **cuML integration**: GPU-accelerated clustering (k-means, DBSCAN) and statistical functions
- **cuDF support**: DataFrames for high-dimensional tracking data
- **Multi-GPU orchestration**: Distributed Kalman filtering across GPU clusters
- **Performance goal**: 50-100x speedup on 1000+ target scenarios
- Modules affected: `pytcl.clustering`, `pytcl.containers` (GPU k-d trees via cuML),
  `pytcl.dynamic_estimation` (multi-GPU particle filters)

#### Intel oneAPI Backend

- oneAPI support for Intel Arc and Data Center GPU Max
- Automatic backend selection (CuPy, MLX, oneAPI, NumPy)
- Unified performance profiling across all backends

#### Quantization & Model Compression

- Mixed-precision Kalman filtering (float32/float16)
- Covariance matrix compression for edge devices
- 50-70% memory reduction with <1% accuracy loss

### Advanced Tracking Capabilities

#### Multi-Hypothesis Tracking Enhancements

- Bernoulli filter for track initiation/termination
- PHD/CPHD filters for clutter-heavy scenarios
- Generalized labeled multi-Bernoulli (GLMB) tracker
- Intensity function visualization

#### Nonlinear Manifold Estimation

- Geodesic distance computation for rotation matrices (quaternion slerp is
  ported; a distance metric is not)
- Lie group integration for orientation tracking
- Applications: aircraft attitude, satellite orientation, marine headings
- Note: projection-based constrained filtering already exists
  (`ConstrainedEKF` in `pytcl.dynamic_estimation.kalman.constrained`)

### Data Management & Interoperability

#### Format Support Expansion

- Parquet format for cloud-native tracking data
- Apache Arrow integration for inter-process communication

#### ROS 2 Integration, Optional Plugin

- ROS 2 node wrappers for core tracking functions
- Real-time message handling for /tf transforms
- RADAR/LiDAR/Camera sensor plugins

### Performance & Analytics

#### Comprehensive Benchmarking Suite

- Extend the daily CPU benchmark CI to GPU and Apple Silicon runners
- Timeline profiling dashboard, memory allocation tracking
- Comparison with competing libraries (FilterPy, Stone Soup, etc.)

#### Advanced Diagnostic Tools

- Interactive dashboards on top of the existing static plotting
  (`pytcl.plotting` already provides covariance ellipses and NEES/NIS
  sequence plots; `pytcl.performance_evaluation` already computes purity,
  fragmentation and identity switches)
- Automatic anomaly detection in tracking results

### Documentation & Community

#### Case Study Library

- Air traffic control with 500+ simultaneous aircraft
- Autonomous vehicle fleet tracking and cooperative perception
- Maritime domain awareness with radar/AIS fusion
- Counter-UAS (C-UAS) multisensor tracking
- Space debris cataloging with optical/radar observations

Each case study: 200-400 lines of code, realistic datasets, benchmarked.

#### Extended API Documentation

- Video tutorials for major modules
- Interactive filter playground (web-based parameter tuning)
- Algorithm comparison visualizations
- Community-contributed examples (with vetting process)

---

## Performance Targets

No performance target is open. The two previously tracked targets (JPDA at
100 targets / 50 measurements, Hungarian assignment at 500x500) are met, as
is the `assign2d` augmented path the v2.5.0 campaign parked; all three are
now guarded by CI-calibrated SLOs in `.benchmarks/slos.json`, whose
`_derivation` fields carry each threshold and how it was computed. The
measurements that closed them, and the reasoning behind each, are in
[CHANGELOG.md](CHANGELOG.md) -- forward-only: a closed target gets a
pointer to what enforces it, not a stale number kept here.

---

## Known Issues & Planned Fixes

Resolved issues live in [CHANGELOG.md](CHANGELOG.md); this section lists only
what is still open.

### Open

| Issue | Status |
|-------|--------|
| Sphinx prose code blocks are not all executed | Every `pytcl` import in `docs/` is checked (244/244 resolve) and the architecture and data-structures pages run under tests, but the remaining prose blocks are still not run |
| CuPy tests skip on machines without an NVIDIA GPU | By design; exercised manually on real NVIDIA hardware for 2.0.0 and 2.1.0 (see the CHANGELOG). The manual `GPU` workflow was dispatched for the first time during v2.8.0 release prep and sat queued: its default `gpu-t4-4-core` runner label is not provisioned for this account, so the workflow cannot run until a GitHub GPU larger runner is added or it is dispatched with the `runner` input pointing at a self-hosted GPU machine. Every hardware run to date remains ad hoc |

### Medium Priority

| Issue | Effect | Mitigation |
|-------|--------|-----------|
| EGM2008 data file (1.3GB) | Geoid height tests skipped without download | Optional download, fallback to EGM96 |
| pywavelets optional dep | Wavelet tests conditional | Falls back to signal processing module |
| Global Legendre cache | Thread safety for concurrent use | Use `clear_legendre_cache()` if needed |

### Limitations vs MATLAB TCL Original

| Feature | Gap | Plan |
|---------|-----|------|
| Distributed tracking | Not implemented | Planned (RAPIDS; unscheduled) |
| Real-time C++ bindings | Not implemented | pybind11 if demand materializes |

### Performance Limitations

- **Very large arrays (>10GB)**: Out-of-core support not implemented (could use HDF5 chunking)
- **Thread safety**: Most functions thread-safe, but some caches are global (mitigated by clearing)
- **Real-time hard deadlines**: Python GC unpredictability (mitigated by profiling tools)

---

## Long-Term Vision

### v2.x Series Direction

#### Core Tracking Evolution

1. **Real-time Particle Swarm Optimization tracking**
   - Swarm intelligence for clutter-heavy tracking
   - Cooperative multi-target estimation
2. **Quantum-inspired algorithms**
   - Quantum annealing backend for assignment (via D-Wave)
   - Quantum simulation of filter dynamics
3. **Adaptive learning tracking**
   - Neural network-augmented Kalman filters
   - Transfer learning for new sensor types

#### Infrastructure Maturation

Unscheduled, in rough order:

- RAPIDS, distributed tracking, advanced diagnostics
- Extended ecosystem (ROS 2, autonomous systems)
- Emerging tech (quantum, federated learning)

#### Community & Ecosystem

- 500+ GitHub stars, 50+ external contributions
- Integration with major frameworks (CARLA, AirSim for autonomous systems)
- Industrial adoption (defense contractors, automotive OEMs)

### v3.0.0 Vision

- Full async/await support for real-time systems
- Native WebAssembly compilation for browser-based tracking
- Federated learning for multi-agent scenarios
- Quantum computing backend exploration

---

## Community Contribution Priorities

### High-Impact Opportunities (Seeking Contributors)

#### 1. Documentation & Tutorials ⭐⭐⭐ (Low barrier to entry)

- Video tutorials for each major module (10-20 min each)
- Blog posts on advanced techniques (JPDA, MHT, particle filters)
- Real-world case studies with public datasets
- Glossary of tracking terminology

**Effort:** 20-40 hours per item | **Skills:** technical writing | **Impact:** high

#### 2. ROS 2 Integration Layer ⭐⭐⭐ (Medium barrier)

- ROS 2 node wrappers for core modules
- Sensor message adapters (RADAR, LiDAR, camera)
- Autonomous vehicle examples

**Effort:** 60-100 hours | **Skills:** ROS 2 experience | **Impact:** very high

#### 3. Additional GPU Backends ⭐⭐ (Medium-high barrier)

- Intel oneAPI backend (similar to CuPy/MLX structure)
- AMD ROCm support
- Vulkan compute for cross-platform GPU access

**Effort:** 80-150 hours per backend | **Skills:** GPU programming | **Impact:** high

#### 4. Performance Optimization ⭐⭐⭐ (High barrier)

- Profile existing algorithms for bottlenecks
- Implement missing Numba JIT targets
- SIMD optimization
- Caching opportunities analysis

**Effort:** 40-80 hours per element | **Skills:** profiling, numerical computing | **Impact:** medium-high

#### 5. Additional Assignment Algorithms ⭐⭐ (Medium-high barrier)

- Genetic algorithm-based assignment
- Ant colony optimization for dynamic assignment
- Deep learning-based cost matrix prediction

**Effort:** 50-100 hours per algorithm | **Skills:** algorithm implementation | **Impact:** medium

### Getting Started for Potential Contributors

**Repositories:**
- Python port (this repo): https://github.com/nedonatelli/TCL — published on PyPI as `nrl-tracker`
- MATLAB reference: https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary

**Development setup:**
```bash
git clone https://github.com/nedonatelli/TCL.git
cd TCL
uv sync
uv run pytest
```

---

## Contributing

Contributions are welcome! If you'd like to work on any of these features:

1. Open an issue to discuss your planned implementation
2. Fork the repository and create a feature branch
3. Follow the existing code style (ruff formatting, NumPy docstrings)
4. Add tests for new functionality (CI enforces >=90% patch coverage)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and the
[original MATLAB library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary)
for reference implementations.

---

**Next:** the measured backlog above; the `[visualization-xy]` extra stays
gated on an upstream xy stable release.
No dates are attached to anything in this document: none have been decided.
