Development Roadmap
===================

This page summarizes where the library is headed. The authoritative,
regularly updated plan lives in `ROADMAP.md
<https://github.com/nedonatelli/TCL/blob/main/ROADMAP.md>`_; release
history lives in `CHANGELOG.md
<https://github.com/nedonatelli/TCL/blob/main/CHANGELOG.md>`_.

Current State (v2.0.0)
----------------------

* **6,000+ tests** passing, 90% line coverage, docstring examples run in CI
* **Validated geophysics**: WMM2025 (default), WMM2020, IGRF-13, and
  WMMHR2025 magnetic models verified against independent references to
  sub-nT accuracy; UTM projections verified against EPSG to sub-millimeter
* **GPU acceleration**: dual-backend (CuPy for NVIDIA CUDA, MLX for Apple
  Silicon) batch Kalman and particle filters
* **Track management**: SQL and HDF5 persistence with lifecycle management
  and v1.x migration tools
* **Tooling**: ruff (lint + format), mypy --strict, pinned CI toolchain

v2.0.0 Release
--------------

Complete and verified on ``main``; the remaining step is the tag, which
publishes to PyPI. The release is direct -- no alpha/beta/RC cycle -- and
the breaks are hard breaks with no deprecation path:
:doc:`migration guide <migration_v1_to_v2>` covers every change with
before/after snippets.

v2.1
----

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
