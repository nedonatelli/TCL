# TCL (Tracker Component Library) - Development Roadmap

**Current Version:** v2.12.0, released 19 November 2026
**Test Suite:** 11,000+ tests passing, ty-checked; every exported function is reached by a test with no standing exemptions (enforced by `tests/contract/test_public_api_coverage.py`, so the count tracks the surface automatically)
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

### v2.13.0 -- planned

<!-- docs/roadmap.rst includes everything from here to the end marker. -->
<!-- roadmap-next:start -->

The audit's Tier 4 ports, moved out of v2.12 so that release stays
correctness-only, plus follow-ups from v2.12. Names are the pytcl
names each would take:

- **EGM gravity disturbance / anomaly / deflection reference values.**
  `gravity_disturbance`, `gravity_anomaly` and `deflection_of_vertical` have no
  value tests: they need authoritative EGM2008 reference values (NGA).
  `tests/unit/test_egm_comprehensive.py` has never run anywhere (it needs
  `EGM2008.cof`); someone with that file must run it once before its oracle
  claims count.
- **`simple_iri` checked against IRI-2020** (pending IRI permission).
- **Near-antipodal and long-way near-parallel `lambert_universal`.**
  v2.12.0 rejects `|sin(dnu)| < 1e-7` near `dnu = pi` and, on the long
  way, near `2 pi`. Above that the `arccos` transfer angle still loses
  accuracy near pi (relative miss 6.8e-3 at `sin(dnu) = 1e-7` and 1.1e-3
  at 1e-6 for `|r2| = 1.3 |r1|`, `tof = 5000` s; up to 0.28 relative at `sin(dnu)` = 1.1e-7 over `|r2| / |r1|` 0.8 to 2 and `tof` 1500 to 8000 s). The long way near
  `2 pi` is worse: for `sin(dnu)` from 1e-7 to 1e-2 it either raises "did
  not converge" (as throughout the sampled 1e-5 to 1.2e-2 for that
  geometry) or returns a nearly radial `v1` missing `r2` by order 1, at up
  to `sin(dnu)` of about 8e-5 over `|r2| / |r1|` of 0.8 to 2 and `tof` of 1500
  to 8000 s.
  Compute `dnu = atan2(|r1 x r2|, r1 . r2)` and make the
  `abs(tof_calc - tof) < tol` stopping test relative to `tof`: the atan2
  change alone moved ordinary results by up to 3.3e-13 relative and flipped
  14 of 3000 random geometries to "did not converge" in one branch, because
  the absolute 1e-10 s test sits at the floating-point floor.
- **`hyp1f1` overflow proofs: speed and coverage.** v2.12.0 returns `+-inf`
  only for a proven overflow. The proof is a Python-level series sum, so an
  array of overflowing elements runs 17 to 28 times slower than 2.11.1's
  scipy pass-through at `abs(a)`, `abs(b)` up to 1000, and with
  cancellation in the series (`abs(a)`, `abs(b)` above about 200) up to 11%
  of overflows stay unproven and return NaN with a `RuntimeWarning`. Move
  the term recurrence into a numba kernel, and prove overflows past the
  cancellation (for example from the tail beyond `k = ceil(-a)`, or in
  higher precision).
- **Pseudo-inverse sites to the unidentifiable-covariance convention.**
  `least_squares.weighted_least_squares` (`pinv(AtWA)`),
  `information_filter.py` and `smoothers.py` still return finite
  pseudo-inverse values where CONTRIBUTING.md's convention returns `inf`
  variance with a `RuntimeWarning`; migrate them.
- **Matrix permanent.** MATLAB `Mathematical_Functions/Combinatorics/perm.m`
  (Ryser's formula); pytcl `matrix_permanent`.
- **`stateCart2Ruv`.** MATLAB
  `Coordinate_Systems/State_Coordinate_System_Conversion/stateCart2Ruv.m`;
  pytcl `state_cart2ruv`, the inverse of the shipped `state_ruv2cart`.
- **Spherical-harmonic dV/dlon and Hessian.** MATLAB
  `Mathematical_Functions/Spherical_Harmonics/spherHarmonicEval.m`; pytcl
  `spherical_harmonic_sum` gains the longitude derivative and Hessian.
  `gravity_acceleration`'s east component is hard-coded to 0 until then.
- **Runnalls `gammaBound` / `KMax`.** MATLAB
  `Clustering_and_Mixture_Reduction/RunnalsGaussMixRed.m`; pytcl
  `reduce_mixture_runnalls` gains `gamma` and `k_max`. MATLAB's `BDist`
  omits the 1/2 of Runnalls' Equation 21, so its `gammaBound` is 2x
  pytcl's cost.
- **IMM GPB1 / GPB2 / AMM family.** MATLAB
  `Dynamic_Estimation/State_Propagation/multipleModelPred.m` and
  `Dynamic_Estimation/Measurement_Update/Complete_Measurement_Updates/multipleModelUpdate.m`;
  pytcl `pytcl.dynamic_estimation.imm` gains the `GPB1`, `GPB2` and `AMM`
  algorithms.
- **Robust interior `ecef2geodetic` solver.** A robust solver for points
  near the Earth's center. Measured on WGS84, `'direct'` returns NaN or
  values wrong by up to 3.5e159 m within 4.28e4 m, and `'iterative'` is
  off by up to 4.25e4 m and by more than 1 mm out to 3.14e5 m; v2.12.0
  warns within 4.27e5 m (`'direct'`) and 1.28e6 m (`'iterative'`).
- **Geomagnetic latitude quantizations near the poles.** The WMM/IGRF
  field cache rounds latitude to 6 decimals by default
  (`configure_magnetic_cache`), so public `wmm`, `igrf` and the heading
  conversions never evaluate closer than 3.3e-7 rad to a pole; and
  `emm`/`wmm` convert geodetic to geocentric latitude with
  `arcsin(z / r)`, which maps geodetic colatitudes below about 1e-8 rad
  onto the pole. `arctan2(z, p)` removes the second; the first needs a
  cache key that keeps the polar colatitude.
- **Vincenty cache input rounding.** `navigation.geodesy`'s inverse/direct
  caches round their inputs to 1e-10 rad (about 0.64 mm) before computing,
  the same rounding the great-circle caches no longer do; keying on the
  exact inputs removes it.
- **Remaining inline squared-Mahalanobis copies.** `trackers/multi_target.py`,
  `trackers/single_target.py`, `trackers/mht.py` and
  `clustering/gaussian_mixture.py` each compute the squared distance inline
  instead of through the shared Cholesky body in
  `kalman.matrix_utils.compute_mahalanobis_distance`.
- **Coplanar-sensor range-only localization (beyond MATLAB, pending owner
  sign-off).** MATLAB `rangeOnlyStaticLocEstNP.m` excludes coplanar
  sensors, and since v2.12.0 `range_only_static_loc_est_np` raises
  `LinAlgError` for them. The geometry still determines the target up to
  its mirror image in the sensors' plane: a least-squares solve on
  `[S_plane, -r]` recovers x, y and `abs(z)` exactly, so the extension
  would return both mirror solutions.

<!-- roadmap-next:end -->

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
  `thirdOrderNDimShellCubPoints.m`.
- **Refraction suite remainder** —
  `Design_of_Lenses/` (3 optics files) stays excluded as out of scope
- **Localization-style static estimators, remainder** — nine of the 11
  shipped in v2.8.0 as `static_estimation.localization`. Still open:
  only the two `Uses_External_Solver` files (`DopplerOnlyInit6D`,
  `polyMeasConvert`). Investigated 2026-09-04: contrary to earlier
  notes here, they do not use SCS — both build 6-7-variable
  multivariate polynomial systems and shell out to an external
  homotopy-continuation solver (Bertini/PHCpack/Macaulay2) through
  `solvePolySysWithExtProg` (734 lines), beyond
  `poly_roots_multi_dim`'s <=3-variable envelope. A port means the
  system construction plus that bridge, with the solver executables as
  optional external tools (tests skipping when absent, like the
  terrain data files) — a full campaign tier of work
- **Direction-cosine UV measurement coordinates, remainder** — the core
  conversions shipped in v2.8.0 (`coordinate_systems.conversions.uv`) and
  the cubature/Taylor covariance conversions in v2.9.0
  (`covariance_conversions`); v2.10.0 completed the family with the
  measurement Jacobians/component gradients
  (`jacobians.measurement_jacobians`) and the Hessians and cross
  gradients/Hessians (`coordinate_systems.hessians`)
- Multivariate **polynomials** remainder
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
| CuPy tests skip on machines without an NVIDIA GPU | By design; exercised manually on real NVIDIA hardware for 2.0.0 and 2.1.0 (see the CHANGELOG). The manual `GPU` workflow (`.github/workflows/gpu.yml`) has no default runner: `workflow_dispatch` requires a `runner` input, the label of a registered self-hosted runner on an NVIDIA machine, and the job runs on `runs-on: ${{ inputs.runner }}`. No GitHub-hosted GPU runner is provisioned for this repository and no self-hosted runner is registered, so the workflow cannot run yet; the one run against the old `gpu-t4-4-core` default queued and was canceled. Every hardware run to date remains ad hoc |
| Debye functions are 20-30x slower on large arrays at large x | Open; not claimed by a release |
| `lambert_w_real` is 1 ulp inside -1/e (2.9e-9 / 1.5e-8 relative) | Inherited from scipy; open |
| pFq max-term convergence is fooled by series whose early terms are tiny | Open |
| `hyp1f1_regularized` prefactor overflows at large `m` or `abs(z)` | At a non-positive integer `b = -m` the prefactor `(a)_{m+1} z^{m+1} / (m+1)!` is evaluated in double precision and returns NaN or `+-inf` for a finite value (`hyp1f1_regularized(0.5, -150, 3)` is `inf`, the value 1.01e72); open, unchanged in v2.12.0 |
| `us_standard_atmosphere_1976` clamps above 86 km | Open |
| NRLMSISE warnings are still `UserWarning` | Open |
| AIS message type 27 | Open; carried from the audit, not claimed by a release |
| State-ordering converter | Open; carried from the audit, not claimed by a release |
| The INS 15x15 error-state model rebuild | Open; the strict-xfail inventory in `tests/validation/test_ins_error_matrix.py` is the acceptance list |
| Magnetism validity windows on `MagneticCoefficients` | Open; carried from the audit, not claimed by a release |
| The GEBCO mask channel | Open; carried from the audit, not claimed by a release |
| `mercator` with multi-element arrays | Open; carried from the audit, not claimed by a release |
| `ecef2geodetic` is unreliable near the Earth's center | `'direct'` returns NaN or wildly wrong values within 4.28e4 m, and `'iterative'` errs by more than 1 mm within 3.14e5 m (up to 4.25e4 m), measured on WGS84. v2.12.0 returns the values unchanged and emits a `RuntimeWarning` within 4.27e5 m (`'direct'`) or 1.28e6 m (`'iterative'`); a robust interior solver is planned for v2.13 |

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
- `geodetic2ecef` returns `(3,) + broadcast_shape` for array input, so
  size-1 array input gives `(3, 1)`; scalars stay `(3,)`. Breaking for a
  STABLE module; also requires the four `float(ecef[2])` callers in
  `gravity/models.py:293`, `gravity/egm.py:494,632`,
  `gravity/clenshaw.py:499` to take `.item()`.

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

**Next:** v2.13.0 above, then the measured backlog; the `[visualization-xy]` extra stays
gated on an upstream xy stable release.
No dates are attached to anything in this document: none have been decided.
