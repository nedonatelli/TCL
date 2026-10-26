Development Roadmap
===================

This page summarizes where the library is headed. The authoritative,
regularly updated plan lives in `ROADMAP.md
<https://github.com/nedonatelli/TCL/blob/main/ROADMAP.md>`_; release
history lives in `CHANGELOG.md
<https://github.com/nedonatelli/TCL/blob/main/CHANGELOG.md>`_.

Current State (v2.11.1)
-----------------------

* **8,500+ tests** passing; coverage measured honestly (94.6% as CI's
  branch-coverage gate measures it, ubuntu-3.11 cell), docstring examples
  run in CI; v2.11.1 is a correctness patch (see the CHANGELOG for its
  behaviour notes)
* **The audit release** (v2.11.0): every silent-wrong-answer path from
  the full-codebase audit closed (Vincenty antipodal, gating without
  R, hypergeometric divergence, silent numerical fallbacks); the MHT
  tracker rebuilt so hypotheses actually branch (plus a Bernoulli
  filter); duplicate implementations removed, delegated, or factored;
  EPSG-9809 oblique stereographic and exact geodesic azimuthal
  equidistant projections matching PROJ to nanometers; the gravity
  ellipsoidal-parameter conversions ported against fresh MATLAB fixtures; abi3 wheels and
  PEP 639 metadata; 23 evidence-backed maturity promotions
* **NRLMSISE-00 proper and the long-tail port** (v2.10.0): the
  vendored reference C compiled as pytcl's first extension (with
  platform wheels and a machine-precision pure-Python fallback,
  cross-validated against the official Fortran lineage), the
  SOFA/ERFA time scales, the full bistatic measurement
  Jacobian/Hessian/cross-derivative suite, MOSPA/MMOSPA metrics,
  interval scheduling, and magnetic apex/quasi-dipole/centered-dipole
  coordinates by field-line tracing -- all fixture-validated with
  independent second oracles, with further upstream MATLAB defects
  documented
* **Estimation expansion** (v2.9.0): the ensemble Kalman filter and
  extended SRIF, the Monte-Carlo (QMC) Kalman family, BLUE polar and
  spherical measurement updates, batch least-squares estimation, the
  nine MATLAB batch/interval smoothers with square-root cubature
  predict/update, the complete Performance_Prediction directory
  (modified Riccati, asymptotic FIM, PCRLB recursions, track purity
  and untrackability), and the u-v/r-u-v/camera covariance measurement
  conversions -- all fixture-validated, with thirteen further upstream
  MATLAB defects found, fixed and documented
* **Static estimation suite** (v2.8.0): nine of MATLAB's 11
  Static_Estimation localization estimators (TDOA, bistatic range-only,
  range-rate-only, Doppler-ratio, bearings-only, and the
  polynomial-measurement Fisher information matrix), built on a Macaulay
  null-space multivariate polynomial solver; plus the direction-cosine
  u-v measurement coordinate system and IGRF-14 — all validated against
  MATLAB fixtures, with four upstream MATLAB defects found, fixed and
  documented
* **MATLAB refraction suite** (v2.7.0): humidity and dew point,
  astronomical refraction, standard-exponential-model radar refraction
  (bistatic r-u-v ray tracing, bias approximation, cubature conversions)
  and speed of sound, every function validated against fixtures captured
  from the MATLAB source tree
* **Audited**: the post-v2.5.0 audit verified the MATLAB parity inventory
  against the source tree, fixed eleven wrong-answer code defects and ~60
  incorrect docstrings, and left permanent gates behind -- dead-parameter
  detection, patch coverage on every PR, executed markdown/rst examples,
  a complete module-maturity registry checked both ways, and weekly
  mutation testing over the STABLE modules
* **Typed configs and sessions** (:doc:`sessions`): ``msgspec.Struct``
  configs (``IMMConfig``, ``GaussianSumConfig``, ``RBPFConfig``,
  ``SingleTargetConfig``, ``MultiTargetConfig``) accepted via a
  keyword-only ``config=``, and full state snapshot/resume
  (``pytcl.io.save_session``/``load_session``) for six tracker and filter
  classes, bit-exact resume for the four deterministic ones
* **Validated geophysics**: WMM2025 (default), WMM2020, IGRF-14 (with IGRF-13 retained), and
  WMMHR2025 magnetic models verified against independent references to
  sub-nT accuracy; UTM projections verified against EPSG to sub-millimeter
* **Real-data validation**: recorded ADS-B air traffic scores the tracking
  chain; vendored Space-Track TLE history scores SGP4/SDP4 self-prediction
  at 1-28 day horizons
* **GPU acceleration**: dual-backend (CuPy for NVIDIA CUDA, MLX for Apple
  Silicon) batch Kalman and particle filters, both verified on real
  hardware
* **Diagnostics**: opt-in observability (:doc:`diagnostics`) — silent by
  default, with gating/association/filter-health/data-file instrumentation
* **Results I/O** (:doc:`results_io`): CSV/Parquet measurement ingest,
  ``to_polars()`` accessors, msgspec JSON/MessagePack and ASDF
  serialization, and AIS NMEA decoding
* **Real ship traffic**: a recorded AIS capture (299 vessels) scores the
  tracking chain against self-broadcast speed, alongside the ADS-B and TLE
  fixtures
* **Track management**: SQL and HDF5 persistence with lifecycle management
  and v1.x migration tools
* **Tooling**: uv-managed workflow, ruff (lint + format), ty as the type
  gate, prek hooks, pinned CI toolchain

Next
----

* **v2.12.0 planned**: the audit carry-forward. Vacuous CI gates fixed,
  dependency floors (numpy 2.3.2, scipy 1.16.1, numba 0.63.0) enforced by
  a ``lowest-direct`` CI cell, ``'w'`` mode truncating for
  ``TrackDatabaseManager`` and ``SQLStorage``, unknown schema versions
  rejected, ``LSResult.residuals`` as the residual vector, exact
  ``factorial``/``factorial2``, ``ecef2geodetic`` rejecting unknown
  methods, the information-filter singular-``Y`` prediction re-derived,
  ``marcum_q`` at large arguments, CFAR empty windows as NaN, ``tau_scale``
  as the Maronna-Zamar scale, ``q_coord_turn_2d`` matching ``QCoordTurn``,
  ``simple_iri`` checked against IRI-2020, and the BLE001/B905 lint rules,
  the duplicate-implementation consolidation and the vacuous-test
  paydown.

* **v2.13.0 planned**: the audit's Tier 4 ports -- matrix permanent,
  ``stateCart2Ruv``, spherical-harmonic dV/dlon and Hessian, Runnalls
  ``gammaBound``/``KMax``, and the IMM GPB1/GPB2/AMM family.

* **v3.0.0**: ``geodetic2ecef`` returning ``(3, 1)`` for size-1 array
  input (a breaking change for a STABLE module).

* **v2.13.0 planned changes**: a robust interior ``ecef2geodetic``
  solver for points near the Earth's centre, where on WGS84 ``'direct'``
  fails within 4.28e4 m and ``'iterative'`` errs by more than 1 mm within
  3.14e5 m (v2.12.0 warns within 4.27e5 m and 1.28e6 m respectively).

* **Measured backlog**: the remaining MATLAB parity gaps
  (:doc:`matlab_parity_inventory`) — the cubature-point library's
  dimension-specialized subdirectory files and out-of-scope region
  types (LCD samples and the general-dimension region-cubature rules
  have shipped), the multivariate-polynomials remainder,
  calendar/timezone utilities, and filter variants.

Long term (unscheduled)
-----------------------

RAPIDS and distributed tracking, ROS 2 integration, PHD/CPHD and labeled
multi-Bernoulli trackers, and the exploratory items listed under
"Long-Term Vision" in ROADMAP.md.

Contributing
------------

See `CONTRIBUTING.md
<https://github.com/nedonatelli/TCL/blob/main/CONTRIBUTING.md>`_ for
guidelines and the high-impact contribution areas listed in ROADMAP.md.
