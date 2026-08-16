Development Roadmap
===================

This page summarizes where the library is headed. The authoritative,
regularly updated plan lives in `ROADMAP.md
<https://github.com/nedonatelli/TCL/blob/main/ROADMAP.md>`_; release
history lives in `CHANGELOG.md
<https://github.com/nedonatelli/TCL/blob/main/CHANGELOG.md>`_.

Current State (v2.3.0)
----------------------

* **6,700+ tests** passing, 90% coverage, docstring examples run in CI
* **Typed configs and sessions** (:doc:`sessions`): ``msgspec.Struct``
  configs (``IMMConfig``, ``GaussianSumConfig``, ``RBPFConfig``,
  ``SingleTargetConfig``, ``MultiTargetConfig``) accepted via a
  keyword-only ``config=``, and full state snapshot/resume
  (``pytcl.io.save_session``/``load_session``) for six tracker and filter
  classes, bit-exact resume for the four deterministic ones
* **Validated geophysics**: WMM2025 (default), WMM2020, IGRF-13, and
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

Released
--------

* **v2.0.0** (2026-08-06) — the correctness release: the pre-2.0 audit
  closed, hard API breaks documented in the
  :doc:`migration guide <migration_v1_to_v2>`.
* **v2.1.0** (2026-08-10) — the Diagnostics release: ``pytcl.diagnostics``,
  the Gaussian cubature-point library with CKF hookup, real-data
  validations, and the completed uv/ty toolchain migration.
* **v2.2.0** (2026-08-12) — the Results I/O release: polars ingest
  (CSV/Parquet) and ``to_polars()`` accessors, msgspec JSON/MessagePack
  serialization, ASDF export, AIS NMEA decoding validated against a
  recorded capture of 299 real ships, and HDF5 track-storage compression
  measured at 4.73x.
* **v2.3.0** (2026-08-16) — the typed configs + save/restore release:
  filter/tracker configs as ``msgspec.Struct`` accepted via a
  keyword-only ``config=``, and full tracker/filter state snapshot and
  resume, bit-exact for the four deterministic classes.

Next
----

* **Measured backlog**: the remaining MATLAB parity gaps
  (:doc:`matlab_parity_inventory`) — Genz-Keister and LCD cubature points,
  the refraction suite, localization estimators, and filter variants.

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
