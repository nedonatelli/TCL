# Correctness Audit Ledger

Tracks the validation status of every public function, method, and class in
`pytcl/` ahead of v2.0.0. Definitions of the validation classes are in
[CONTRIBUTING.md](CONTRIBUTING.md#test-validation-classes).

**Goal:** zero UNTESTED, and REFERENCE or PROPERTY class for all numerical
code, before v2.0.0-alpha.

Inventory (2026-07-26): **1,351 public functions/methods, 226 classes** across
21 packages.

## Status by package

| Package | Public API | Status | Notes |
|---------|-----------:|--------|-------|
| magnetism | 25 | ✅ REFERENCE | WMM2020/WMM2025/IGRF-13/WMMHR2025 vs pygeomag + NOAA test values (<0.1 nT), poles vs published locations |
| atmosphere (models) | 12 | ✅ REFERENCE | US76 vs published tables; NRLMSISE-00 range-checked; ionosphere property-tested |
| astronomical (relativity) | — | ✅ REFERENCE | de Sitter, Lense–Thirring, 1PN, Shapiro vs literature values |
| coordinate_systems (UTM, SEZ) | — | ✅ REFERENCE | UTM vs pyproj (sub-mm); SEZ convention + round trips |
| assignment_algorithms (flow) | — | ✅ REFERENCE | min-cost flow vs scipy linear_sum_assignment |
| gravity (Legendre, synthesis) | — | ✅ PROPERTY | derivative vs finite differences (4e-9); dual-implementation agreement |
| coordinate_systems | 70 | ✅ Wave 1 done | 37 REFERENCE + 32 PROPERTY + 1 structural; 5 bugs fixed (ecef2geodetic direct, polar stereographic S, rotmat2euler XYZ, azimuthal radius, axis-angle near pi) |
| mathematical_functions (special/quad/interp) | 103 | ✅ Wave 1 done | 76 REFERENCE + 27 PROPERTY, 0 untested; 8 bugs fixed (Debye family ×4, Swerling 1-4, wright_omega, marcum broadcasting ×2) |
| mathematical_functions (signal/transforms/stats/matrix) | 134 | ✅ Wave 1 done | 107 REFERENCE + 22 PROPERTY; 5 bugs fixed (CWT dilation, OS/GO/SO-CFAR thresholds, wavelet scale map, gaussian wavelet) |
| navigation | 69 | ✅ Wave 1 done | 42 REFERENCE + 20 PROPERTY + 7 structural; gyrocompass_alignment fixed (45 deg heading errors) |
| containers + clustering | 118 | ✅ Wave 1 done | 96 REFERENCE + 5 PROPERTY, 0 untested; CoverTree search rewritten (669 brute-force failures → 0) |
| dynamic_estimation | 101 | ✅ Wave 2 done | 90 REFERENCE + 8 PROPERTY; 7 bugs fixed (UD covariance, SRIF predict, two-filter smoother, PF likelihood, IF diffuse start, SR-UKF downdate, GSF prune) |
| astronomical | 118 | ✅ Wave 2 done | 62 REFERENCE + 35 PROPERTY; 9 bugs fixed (GMST 0.74 deg, nutation args, TEME sign, SGP4 x4 → <1 mm vs Vallado, lambert_izzo rewrite, deflection angle); SDP4 deep-space port flagged |
| assignment_algorithms + static_estimation | 91 | ✅ Wave 2 done | 62 REFERENCE + 10 PROPERTY; 11 bugs fixed (Murty x4, default flow path suboptimal 64%, JPDA Pd^2 + covariance, 3D auction infeasible, TLS sign, more); 7 design issues reported |
| gravity + tides | 60 | ✅ Wave 2 done | 36 REFERENCE + 4 PROPERTY; 12 bugs fixed (Legendre sqrt2 at source, solid tide frame/amplitude/sign, pole tide, atm loading 100x, geoid reference field, EGM parser D-exponents); degree>500 scaling limit flagged |
| dynamic_models + performance_evaluation + trackers | 72 | ✅ Wave 2 done | 55 REFERENCE + 4 PROPERTY; 2 bugs fixed (q_singer 4 orders off/non-PSD, q_continuous_white_noise); CT Jacobian + MHT score + dead frag counter reported |
| io | 113 | ⬜ Wave 3 | round-trip/property (HDF5/SQL) |
| core + terrain + plotting + misc | ~100 | ⬜ Wave 3 | behavioral contracts |
| gpu | 56 | ⬜ Wave 3 | MLX backend on Apple Silicon; CuPy needs NVIDIA hardware |

## Findings log

Confirmed bugs found and fixed during the audit are recorded in CHANGELOG.md;
suspected-but-unconfirmed issues get GitHub issues and are linked here.

| Date | Package | Finding | Resolution |
|------|---------|---------|------------|
| 2026-07-25 | multiple | 9 library bugs (network flow, UTM, atmosphere ×2, Legendre derivative, gravity signs, chol_semi_def, Swerling, lambert_w) | Fixed in v1.15.1 |
| 2026-07-25/26 | magnetism, relativity, SEZ | Issue #3 (synthesis normalization, coefficient corruption, formula errors, convention) | Fixed in v1.16.0 |
| 2026-07-26 | Wave 1 (5 packages) | 21 bugs fixed: CWT never dilated, OS-CFAR 14x design pfa, CoverTree invalid search, gyrocompass 45-deg errors, ecef2geodetic 37 km, polar stereographic S hemisphere, rotmat2euler XYZ negated, Debye family, Swerling 1-4, wright_omega overflow, more | Fixed on audit branch; ~15 ambiguous items for triage |
| 2026-07-26 | Wave 2 (5 groups) | 41 bugs fixed: SGP4 x4 (728 km/day → <1 mm), GMST double-count, lambert_izzo rewrite, solid tides had no semidiurnal component, atm loading 100x, Legendre sqrt2 root cause, UD/SRIF/smoother/PF/IF/SR-UKF filter-core errors, Murty x4, JPDA Pd^2, default assignment path suboptimal, q_singer, more | Fixed on audit branch; ~12 design-level items for triage |
