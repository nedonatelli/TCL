Development Roadmap
===================

This page summarizes where the library is headed. The authoritative,
regularly updated plan lives in `ROADMAP.md
<https://github.com/nedonatelli/TCL/blob/main/ROADMAP.md>`_; release
history lives in `CHANGELOG.md
<https://github.com/nedonatelli/TCL/blob/main/CHANGELOG.md>`_.

Current State (v2.1.0)
----------------------

* **6,200+ tests** passing, 90% coverage, docstring examples run in CI
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

Next
----

* **v2.2.0 — Results I/O**: polars ingest (CSV/Parquet) and ``to_polars()``
  accessors (the ``dataframe`` extra), msgspec export of track histories
  and states to JSON/MessagePack (msgspec now a core dependency), ASDF
  export/import of track histories and states (the ``asdf`` extra), AIS
  NMEA decoding and position-report extraction (the ``ais`` extra, pyais),
  and measured HDF5 byte-shuffle compression (4.73x, on by default).
* **v2.3.0 — Typed configs + save/restore**: filter/tracker configs as
  ``msgspec.Struct``, full tracker state snapshot/resume.
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
