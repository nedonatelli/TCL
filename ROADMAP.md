# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v2.4.0, released 17 August 2026
**Test Suite:** 7,000+ tests passing, ty-checked; every exported function is reached by a test with no standing exemptions (enforced by `tests/contract/test_public_api_coverage.py`, so the count tracks the surface automatically)
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

1. [Post-v2.0.0 Roadmap](#post-v200-roadmap)
2. [Performance Targets](#performance-targets)
3. [Known Issues & Planned Fixes](#known-issues--planned-fixes)
4. [Breaking Changes for v2.0.0](#breaking-changes-for-v200)
5. [Long-Term Vision](#long-term-vision)
6. [Community Contribution Priorities](#community-contribution-priorities)
7. [Contributing](#contributing)

---

## Post-v2.0.0 Roadmap

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
  remaining seven region-cubature directories (torus, cone, wedge, etc.)
  are out of this subset entirely and have not been inventoried. Three
  more `Cube_Space` files (`ClenshawCurtisPoints1D.m`, `FejerPoints1D.m`,
  `conformMapQuadPts1D.m`) are 1-D quadrature building blocks, not
  region-cubature rules themselves (no region-dimension argument) --
  better scoped as a future `quadrature.py` extension than
  `region_cubature.py`, deferred pending a consumer.
- **Refraction suite** — entirely unported (astronomical refraction,
  standard-refraction ray tracing, refractivity models, humidity conversions)
- **Localization-style static estimators** — Cartesian TDOA, Doppler-only
  init, direction-only; MATLAB's Static_Estimation content is almost all
  absent (great-circle TDOA is the exception, ported as
  `great_circle_tdoa_loc`)
- **Filter variants** — EnKF, ESRIF, QMC-Kalman, BLUE measurement updates,
  batch least squares, PCRLB/Riccati analysis tools
- **Direction-cosine UV measurement coordinates** — the angle-only u-v
  (and u-v-w) system of planar phased arrays, with conversions and
  Jacobians; distinct from the ported range+direction-cosine `cart2ruv`
- **Time scales** — TDB/TCB/TCG, Besselian epochs, sidereal local time
- **Magnetic coordinate systems** — apex, quasi-dipole, centered-dipole
- **MOSPA/MMOSPA metrics**, interval scheduling, polynomials
- **NRLMSISE-00 proper** — load the NOAA coefficient tables and retire the
  barometric approximation's caveats (gh-79), plus HWM winds
- **IGRF-14 coefficients** (IGRF-13's validity window ended 2025.0) — the
  embedded `IGRF13` table is now out of its official range; `igrf()` and
  `create_igrf13_coefficients()` document this and extrapolate the
  2020-2025 secular variation past 2025.0, but a real IGRF-14 table
  (expected from IAGA) should replace it

- **HDF5 `states_only` covariance-transform mode** — a ~6.3x compression
  ceiling is reachable by reconstructing per-scan covariance from a
  steady-state Cholesky factor, but it touches every read path and breaks
  the bit-exact round-trip contract; deferred unless a real need appears
  (v2.2.0 measured and shipped byte-shuffle at 4.73x instead — see the
  CHANGELOG for the closed measurement).

### Modernization campaign — remaining

- **Unversioned, gated:** `[visualization-xy]` extra for large-dataset
  plotting, once xy has a stable release. Everything else in the campaign
  (uv/ty tooling, v2.1.0 Diagnostics, v2.2.0 Results I/O, v2.3.0 typed
  configs + sessions) has shipped — see the CHANGELOG.

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

Both previously tracked targets are met and now guarded by CI SLOs
(`.benchmarks/slos.json`) rather than kept here as an open-work table --
forward-only: a closed target gets a one-line pointer to what enforces it,
not a stale number.

- **JPDA (100 targets, 50 meas):** 45.31 ms (task C1's `time.perf_counter`
  profiling-run baseline, Apple M3 Max, 2026-08-18) -> 33.55 ms (Apple M3
  Max, 2026-08-18) after wiring the previously-dead Mahalanobis njit
  fast-path kernels (`pytcl/assignment_algorithms/gating.py`) into
  `mahalanobis_distance()`. CHANGELOG.md's `[Unreleased]` entry for this
  change instead cites 43.89 ms -- task C2's own `pytest-benchmark`
  pre-optimization re-measurement, taken fresh at the start of that task on
  the same machine and date; the two "before" numbers come from different
  measurement runs/methodologies, not a discrepancy. Both figures already
  beat the old "89 ms -> <75 ms" row, which had no in-repo provenance (no
  commit, fixture, or script reproduced either number). Guarded by
  `test_jpda_update_100_targets_50_meas`'s SLO (111.18 ms mean).
- **Hungarian Assignment (500x500 dense):** 5.00 ms median (Apple M3 Max,
  2026-08-18) -- `hungarian()` is a thin wrapper around scipy's
  `linear_sum_assignment`, so this is scipy's own number, not a pytcl
  optimization target; the old "45 ms -> <30 ms" row also had no in-repo
  provenance. Guarded by `test_hungarian_dense_500x500`'s SLO
  (16.57 ms mean).
- **assign2d augmented path (500x500, finite `cost_of_non_assignment`):**
  the v2.5.0 campaign parked whether this path -- which builds an
  (n+m)x(n+m) = 1000x1000 augmented matrix before delegating to scipy,
  double the plain Hungarian case above -- deserved its own perf target.
  Answered by measurement (perf-levers task 2, Apple M3 Max, 2026-08-18):
  9.0375 ms median; cProfile of the same scenario attributes 96.7% of
  cumulative time to `scipy.optimize.linear_sum_assignment` and 1.1% to
  the `np.full` augmented-matrix construction, well under the 10%
  trivial-fix threshold, so no code change was made -- scipy's 1000x1000
  solve is the floor, same as the plain case. Guarded by
  `test_assign2d_augmented_500x500`'s SLO (29.946 ms mean).

Both SLO thresholds are CI-calibrated (local measurement x measured
CI/local hardware ratio x 1.5 headroom), not bare M3 Max numbers; full
derivation in task C2's performance report (local-only, untracked
artifact of the v2.5.0 region-lcd-perf campaign).


---

## Known Issues & Planned Fixes

Resolved issues live in [CHANGELOG.md](CHANGELOG.md); this section lists only
what is still open.

### Open

| Issue | Status |
|-------|--------|
| Sphinx prose code blocks are not all executed | Every `pytcl` import in `docs/` is checked (244/244 resolve) and the architecture and data-structures pages run under tests, but the remaining prose blocks are still not run |
| CuPy tests skip on machines without an NVIDIA GPU | By design. Exercised on real hardware twice: pre-2.0.0 on an RTX 5080 (found the doctest interop bugs now pinned by `test_gpu_doctest_hygiene.py`; no run artifact was kept) and for 2.1.0 on an RTX 5090/CUDA 13 (recorded; caught and fixed the `[gpu]` extra's missing CUDA 12 wheels). The manual `GPU` workflow exists for on-demand re-runs but has never been dispatched -- hardware runs to date were manual |

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

## Breaking Changes for v2.0.0

### API Changes Summary

The breaking changes, each documented with before/after in
``docs/migration_v1_to_v2.rst``:

- ``query(k)`` rejects ``k > n_samples`` instead of padding with index 0
- INS/GNSS ``position_cov`` is in ``[rad, rad, m]``;
  ``position_std_to_error_state_units`` converts
- ``compute_dop`` takes ``user_lla`` for meaningful HDOP/VDOP
- ``detection_probability`` drops the inert ``swerling_case`` argument
- ``snr_loss`` requires ``pfa`` and covers CA-CFAR only
- ``SQLStorage()`` takes no ``db_type``; read-mode ``open`` no longer creates
  files; ``store_array`` replaces on both backends
- GPU filter callbacks take the whole ``(N, dim)`` batch on the active backend
- ``NRLMSISE00`` family renamed ``SimplifiedThermosphere`` (gh-79)
- ``nuttall_q`` renamed ``rician_cdf`` (warning alias retained)
- ``pytcl.logging_config`` and
  ``pytcl.assignment_algorithms.network_simplex`` removed outright
- Several functions return different (correct) numbers with unchanged
  signatures — see the migration guide's dedicated section before comparing
  against recorded v1.x baselines

**Exception handling** (unchanged in 2.0.0, listed for migrating v1.x users):
all library exceptions inherit from ``pytcl.TCLError``, with specific types
like ``pytcl.DimensionError`` available for narrow catches.

### Backward Compatibility Layer

There is none, deliberately. The 2.0.0 breaks are hard breaks — `nuttall_q`
(a rename, kept as a warning alias) is the sole exception — and
`docs/migration_v1_to_v2.rst` is the migration path.

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
4. Add tests for new functionality (aim for 80%+ coverage)
5. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines and the
[original MATLAB library](https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary)
for reference implementations.

---

**Next:** the measured backlog above; the `[visualization-xy]` extra stays
gated on an upstream xy stable release.
No dates are attached to anything in this document: none have been decided.
