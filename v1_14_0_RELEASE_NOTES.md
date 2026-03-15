v1.14.0 Release Notes
=====================

Release Date: March 15, 2026

Overview
--------

**v1.14.0 upgrades geophysical data support and delivers significant performance
improvements to the Enhanced Magnetic Model.** This release adds GEBCO 2025 terrain
data, WMMHR2025 magnetic model support, EMM array input handling, and a 6-12x
speedup in spherical harmonic evaluation.

Key Features
~~~~~~~~~~~~

**GEBCO 2025 Default**
   The default terrain dataset has been upgraded from GEBCO 2024 to GEBCO 2025.
   All ``load_gebco()`` and ``get_gebco_metadata()`` calls now default to
   ``version="GEBCO2025"``. Previous versions (2024, 2023, 2022) remain available.

**WMMHR2025 Support**
   World Magnetic Model High Resolution 2025 (degree 133, ~300 km spatial
   resolution) is now available via ``wmmhr()`` or ``emm(model="WMMHR2025")``.
   Requires the WMMHR2025.COF coefficient file in ``~/.pytcl/data/``.

**EMM Array Inputs**
   ``emm()``, ``emm_declination()``, ``emm_inclination()``, and ``emm_intensity()``
   now accept NumPy array inputs for lat, lon, and height. Scalar and array
   inputs are both supported with proper broadcasting.

**EMM Performance Optimization**
   Vectorized the spherical harmonic summation in ``_high_res_field_spherical()``,
   replacing nested Python loops with NumPy array operations:

   +--------------------------+----------+---------+---------+
   | Benchmark                | Before   | After   | Speedup |
   +==========================+==========+=========+=========+
   | Single point (n_max=36)  | 1.29 ms  | 0.20 ms | 6.5x    |
   +--------------------------+----------+---------+---------+
   | Single point (n_max=133) | 16.9 ms  | 1.45 ms | 11.7x   |
   +--------------------------+----------+---------+---------+
   | 10 points (n_max=133)    | 204 ms   | 60 ms   | 3.4x    |
   +--------------------------+----------+---------+---------+

New Optional Extras
~~~~~~~~~~~~~~~~~~~

- ``pip install nrl-tracker[terrain]`` — installs netCDF4 for GEBCO/Earth2014 loading
- ``pip install nrl-tracker[storage]`` — installs h5py for HDF5 track storage

Both are included in ``pip install nrl-tracker[all]``.

Refactoring
~~~~~~~~~~~

- **Centralized ``get_data_dir()``**: Moved from 3 duplicate copies (terrain,
  magnetism, gravity) into ``pytcl.core.paths``. All modules now import from there.
- **Test quality**: Fixed ~30 broken/dead test assertions across terrain loaders,
  EMM, and lambert transfer tests. Tightened skip guards to only catch
  ``FileNotFoundError``/``DependencyError`` instead of bare ``Exception``.
- **CLAUDE.md**: Added project conventions and setup guide.

Bug Fixes
~~~~~~~~~

- **EMM lon broadcast**: Passing scalar ``lon`` with array ``lat`` would silently
  truncate results to one element. Now properly broadcasts all inputs.
- **Terrain test units**: Tests were passing degrees where radians were expected,
  causing all real-data tests to silently skip.
- **Lambert test assertions**: ``isinstance(result, dict)`` was always ``False``
  on tuple returns from ``hohmann_transfer``/``bi_elliptic_transfer``, so the
  assertions inside never executed.
- **Dead assertion**: ``abs(result_2020.X - result_2025.X) >= 0`` (always true)
  replaced with ``!= 0`` to actually test secular variation.

Quality Metrics
~~~~~~~~~~~~~~~

- **1,048 functions** across **133 modules**
- **3,306 tests** — 0 skipped when all data files are installed
- **80% code coverage**
- **100% mypy --strict compliance**

Upgrade Guide
~~~~~~~~~~~~~

**From v1.13.2:**

1. The default GEBCO version is now ``"GEBCO2025"``. If you explicitly pass
   ``version="GEBCO2024"``, no change is needed. If you rely on the default,
   download the GEBCO 2025 data from https://www.gebco.net/ and place
   ``GEBCO_2025.nc`` in ``~/.pytcl/data/``.

2. ``emm_declination()``, ``emm_inclination()``, and ``emm_intensity()`` now
   return ``ndarray`` when given array inputs (previously raised an error).
   Scalar inputs still return ``float``.

3. ``get_data_dir()`` is now canonical in ``pytcl.core.paths``. The re-exports
   from ``pytcl.terrain.loaders`` and ``pytcl.magnetism.emm`` still work.

4. Install new extras if needed:

   .. code-block:: bash

      pip install nrl-tracker[terrain]   # for GEBCO/Earth2014
      pip install nrl-tracker[storage]   # for HDF5 track storage
