Diagnostics Guide
==================

Overview
--------

:mod:`pytcl.diagnostics` is pytcl's opt-in logging, instrumentation, and
progress-reporting layer. It is the redesigned successor to the
``pytcl.logging_config`` module removed in v2.0.0 -- there is no
compatibility shim, and no code should import the old module.

Two rules govern the whole module:

- **Silent by default.** Importing ``pytcl`` disables the ``pytcl`` loguru
  namespace and installs no handlers. A fresh interpreter that imports
  ``pytcl`` and runs a filter step prints nothing, to either stream.
- **Behaviorally neutral.** Enabling diagnostics never changes a numerical
  result. Every instrumentation call sits behind a guard and reads values
  already computed for the algorithm itself -- it never causes an extra
  matrix inversion or perturbs a cache.

Enable and Disable
-------------------

.. code-block:: python

   import pytcl

   pytcl.enable_debug_logging()   # DEBUG-level output to stderr, from here on
   ...
   pytcl.disable_debug_logging()  # back to complete silence

Both functions are re-exported at the top level and are idempotent:
calling ``enable_debug_logging()`` twice replaces the handler rather than
stacking a second one, and ``disable_debug_logging()`` is a no-op if
diagnostics are already off.

.. code-block:: python

   from pytcl.diagnostics import diagnostics_enabled

   pytcl.enable_debug_logging(level="INFO")  # raise the floor above DEBUG
   assert diagnostics_enabled() is True

``diagnostics_enabled()`` is the guard every instrumentation site checks
before doing any work, so the disabled path costs one boolean check and
nothing else -- no string formatting, no payload construction.

The Silence Guarantee
----------------------

The guarantee is tested at the process level, not just at the API surface:
a subprocess that imports ``pytcl`` and calls ``kf_predict`` with no prior
``enable_debug_logging()`` call produces zero bytes on stdout and stderr,
and the loguru handler count before and after import is identical. Nothing
pytcl does at import time or during normal operation writes to either
stream unless diagnostics have been explicitly enabled.

What Each Family Logs
----------------------

Four instrumentation families exist, all bound with a ``site`` tag so a
handler can filter or route by category. Enable logging first:

.. code-block:: python

   import numpy as np
   import pytcl
   from pytcl.trackers import MultiTargetTracker

   pytcl.enable_debug_logging()

   tracker = MultiTargetTracker(
       state_dim=4, meas_dim=2,
       F=np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
       H=np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0]]),
       Q=np.eye(4) * 0.01, R=np.eye(2) * 1.0,
   )
   tracker.process([np.array([0.0, 0.0])], dt=1.0)
   tracker.process([np.array([0.1, 0.1]), np.array([500.0, 500.0])], dt=1.0)

Sample output (timestamps trimmed):

.. code-block:: text

   DEBUG    | pytcl - track 0: gated out 1 of 2 measurements: m1 d=482.13>thr=9.21
   DEBUG    | pytcl - GNN assignment: 1 pair(s) [(0, 0)], total_cost=0.0142
   DEBUG    | pytcl - track 0: nis=0.3821 (window_mean=0.3821, n=1) cov_condition=1.2400e+01

**Gating** (``site="gating"``) -- ``MultiTargetTracker`` logs which
measurements were rejected by the gate for each track before association
runs, with the Mahalanobis distance and threshold that rejected them.

**Association** (``site="association"``) -- the resulting GNN
track-to-measurement pairing and its total cost; ``JPDATracker`` logs the
top marginal probability per track (``site="jpda"``), and ``MHTTracker``
logs per-scan hypothesis counts, how many were pruned, and the best
surviving hypothesis score (``site="mht"``).

**Filter health** (``site="filter_health"``) -- :func:`~pytcl.diagnostics.log_filter_health`
logs a per-update NIS (normalized innovation squared) and covariance
condition number snapshot for every track, at DEBUG. It escalates to
WARNING when either symptom of a diverging filter appears: the current
NIS exceeds three times the mean of its recent window, or the covariance
condition number exceeds ``1e12``. ``MultiTargetTracker`` calls it
automatically after every track update, reusing the innovation covariance
inverse already computed for the Kalman gain -- no extra work is done to
produce the health snapshot.

**Data-file resolution** (``site="data-files"``) -- :func:`~pytcl.core.paths.get_data_dir`
and the terrain/magnetism/gravity coefficient loaders log every candidate
path they try, whether ``PYTCL_DATA_DIR`` is overriding the default, and
which candidate (if any) was found, at DEBUG. This is the fastest way to
see why a loader raised ``FileNotFoundError``: enable diagnostics and the
log shows exactly which directories and filename patterns were tried.

Progress Bars and Track Tables
--------------------------------

:func:`~pytcl.diagnostics.progress_bar` wraps an iterable in an ASCII
progress display on stderr, independent of ``enable_debug_logging()`` --
it is a UI element, not a log record:

.. code-block:: python

   from pytcl.diagnostics import progress_bar

   for item in progress_bar(range(1000), description="processing"):
       ...

The terrain loaders take a ``progress`` flag that wires this in directly:

.. code-block:: python

   import math

   from pytcl.terrain.loaders import load_earth2014, load_gebco

   lat_min, lat_max = math.radians(34.0), math.radians(35.0)
   lon_min, lon_max = math.radians(-119.0), math.radians(-118.0)

   grid = load_earth2014(lat_min, lat_max, lon_min, lon_max, progress=True)
   dem = load_gebco(lat_min, lat_max, lon_min, lon_max, progress=True)

``load_earth2014`` shows a genuine row-by-row bar; ``load_gebco`` reads
its region in one NetCDF slice with no natural loop to attach a bar to,
so ``progress=True`` instead logs DEBUG start/finish markers around the
read (visible only with diagnostics enabled). Either way, passing
``progress=True`` routes around the ``lru_cache``-backed default load
path -- it neither reads from the cache nor populates it, so toggling the
flag never forces a redundant re-parse of the underlying file (up to
~7.5 GB for GEBCO, ~455 MB per Earth2014 layer) on the default,
non-progress path.

:func:`~pytcl.diagnostics.track_table` renders a summary table of a
tracker's current tracks (id, status, position, speed) to the console:

.. code-block:: python

   from pytcl.diagnostics import track_table

   tracks = tracker.process([np.array([0.15, 0.15])], dt=1.0)
   track_table(tracks)

.. code-block:: text

   +----+-----------+---------------+-------+
   | id | status    | position      | speed |
   |----+-----------+---------------+-------|
   |  0 | CONFIRMED | (0.1, 0.1)    |  0.14 |
   +----+-----------+---------------+-------+

ASCII-Only Output
------------------

Every character :mod:`pytcl.diagnostics` writes -- log lines, progress
bars, and track tables -- is restricted to what encodes cleanly under
``cp1252``. This is not a style preference: Windows crashes on the
default console encoding when a redirected stdout/stderr receives a
character outside that codec, so a library that wants to be usable from a
Windows batch file or CI runner cannot emit box-drawing glyphs or Unicode
block characters. Concretely:

- ``track_table`` renders with ``rich.box.ASCII`` and ``safe_box=True``
  instead of rich's default Unicode box-drawing characters.
- ``progress_bar`` uses plain text columns (``completed/total``,
  percentage) rather than rich's default ``BarColumn``, which renders
  with Unicode block characters.
- The log format string uses only ASCII punctuation.

``tests/contract/test_console_encoding.py`` asserts this for the whole
package, not just this module.

See Also
--------

- :doc:`troubleshooting` - general debugging guidance
- :doc:`gpu_acceleration` - GPU backend selection and diagnostics
- :doc:`kalman_filter_tuning` - interpreting NIS and filter-health symptoms
