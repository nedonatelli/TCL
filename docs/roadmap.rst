Development Roadmap
===================

This page summarizes where the library is headed. The authoritative,
regularly updated plan lives in `ROADMAP.md
<https://github.com/nedonatelli/TCL/blob/main/ROADMAP.md>`_; release
history lives in `CHANGELOG.md
<https://github.com/nedonatelli/TCL/blob/main/CHANGELOG.md>`_.

Current State (v1.16.0)
-----------------------

* **100% MATLAB TCL parity** across all tier 1 and tier 2 components
* **3,322 tests** passing, 80% line coverage, docstring examples run in CI
* **Validated geophysics**: WMM2025 (default), WMM2020, IGRF-13, and
  WMMHR2025 magnetic models verified against independent references to
  sub-nT accuracy; UTM projections verified against EPSG to sub-millimeter
* **GPU acceleration**: dual-backend (CuPy for NVIDIA CUDA, MLX for Apple
  Silicon) batch Kalman and particle filters
* **Track management**: SQL and HDF5 persistence with lifecycle management
  and v1.x migration tools
* **Tooling**: ruff (lint + format), mypy --strict, pinned CI toolchain

v2.0.0 (Target: Q4 2026)
------------------------

All development phases are complete; remaining work is release
preparation (Phase 9):

* Final integration testing across all subsystems
* Track-management quality gates (throughput, latency, compression)
* Alpha → beta → release-candidate cycle with community feedback
* Migration guide and deprecation path for v1.x users
* Removal of dead modules (``network_simplex``, ``logging_config``) -- **done**

v2.1 (2027)
-----------

* RAPIDS integration for distributed, multi-GPU tracking
* Intel oneAPI backend
* PHD/CPHD and labeled multi-Bernoulli trackers
* Terrain loader signature fixes
* Expanded format support (Parquet, Arrow, ASDF) and optional ROS 2 plugin

Contributing
------------

See `CONTRIBUTING.md
<https://github.com/nedonatelli/TCL/blob/main/CONTRIBUTING.md>`_ for
guidelines and the high-impact contribution areas listed in ROADMAP.md.
