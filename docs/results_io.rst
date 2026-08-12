Results I/O
============

:mod:`pytcl.io` and :mod:`pytcl.transponders` cover the two halves of a
tracking pipeline's I/O: reading measurements in (CSV/Parquet), and getting
tracks and filter states back out (polars DataFrames, msgspec bytes, ASDF,
HDF5). :mod:`pytcl.transponders.ais` decodes real-world AIS traffic into the
same measurement shapes.

Every function below that touches an optional dependency (``polars``,
``pyais``, ``asdf``) raises :class:`~pytcl.core.exceptions.DependencyError`
naming the extra to install if it is missing -- ``h5py`` and ``msgspec`` are
core dependencies and need nothing extra.

Reading Measurements (CSV/Parquet)
------------------------------------

``read_measurements_csv`` and ``read_measurements_parquet``
(:mod:`pytcl.io.readers`, the ``dataframe`` extra) turn a flat table -- one
row per measurement report -- into a `MeasurementSet`: measurements grouped
into scans by the exact value of a timestamp column, ascending. Three
keyword-only arguments do the column mapping:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Parameter
     - Meaning
   * - ``time_column``
     - Column whose exact value groups rows into the same scan
   * - ``measurement_columns``
     - Columns stacked, in order, into each scan's ``(n_k, n_cols)`` matrix
   * - ``id_column``
     - Optional column threaded through as ``MeasurementSet.ids``

A missing column raises ``ValueError`` listing every available column, so a
typo in the mapping fails immediately rather than silently dropping data.

.. code-block:: python

   import tempfile
   from pathlib import Path

   from pytcl.io import read_measurements_csv

   csv_text = (
       "t,x,y,sensor_id\n"
       "0.0,10.0,20.0,S1\n"
       "0.0,10.3,19.8,S2\n"
       "1.0,11.1,21.4,S1\n"
   )
   tmpdir = tempfile.TemporaryDirectory()
   csv_path = Path(tmpdir.name) / "detections.csv"
   csv_path.write_text(csv_text)

   ms = read_measurements_csv(
       csv_path,
       time_column="t",
       measurement_columns=["x", "y"],
       id_column="sensor_id",
   )
   print(ms.times.tolist())          # [0.0, 1.0]
   print(ms.scans[0].tolist())       # 2 rows at t=0.0
   print(ms.ids[0].tolist())         # ['S1', 'S2']

``read_measurements_parquet`` has the identical signature and grouping
contract, parsed with ``polars.read_parquet`` instead of ``polars.read_csv``
-- a CSV and a Parquet file holding the same rows produce bitwise-identical
`MeasurementSet` values.

Track Histories as DataFrames
--------------------------------

``tracks_to_polars`` (:mod:`pytcl.io.dataframes`, the ``dataframe`` extra)
flattens a per-scan track history -- the same ``list[list[Track]]`` shape
`MultiTargetTracker.process` accumulates -- into a **long** table: one row
per ``(scan, track)`` pair.

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Column
     - Type
     - Contents
   * - ``track_id``
     - Int64
     - Track identifier
   * - ``t``
     - Float64
     - Scan timestamp
   * - ``status``
     - String
     - Track status (``TrackStatus.value``)
   * - ``state``
     - List[Float64]
     - State estimate vector
   * - ``covariance``
     - List[Float64]
     - Row-major flattened covariance, length ``len(state) ** 2``

``explode_state_columns`` widens the ``state`` list column into one named
Float64 column per component, for a layout you know ahead of time; it raises
``ValueError`` if ``layout``'s length does not match the state dimension --
except for a zero-row ``df``, where the dimension can't be read from empty
data, so the check is skipped and the ``layout``-named columns are added
empty instead.

.. code-block:: python

   import numpy as np
   from pytcl.dynamic_models import f_constant_velocity
   from pytcl.io import explode_state_columns, tracks_to_polars
   from pytcl.trackers import MultiTargetTracker

   F = f_constant_velocity(1.0, num_dims=2)  # [x, vx, y, vy] layout
   H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
   tracker = MultiTargetTracker(
       state_dim=4, meas_dim=2, F=F, H=H,
       Q=np.eye(4) * 0.01, R=np.eye(2) * 4.0, confirm_hits=1,
   )

   history, times = [], []
   for k in range(3):
       history.append(tracker.process([np.array([float(k), 2.0 * k])], dt=1.0))
       times.append(float(k))

   df = tracks_to_polars(history, times)
   print(df.columns)            # ['track_id', 't', 'status', 'state', 'covariance']
   print(df.height)              # one row per (scan, track)

   wide = explode_state_columns(df, ["x", "vx", "y", "vy"])
   print(wide.select(["track_id", "t", "x", "y"]).height)

``metrics_to_polars(times, **series)`` builds an unrelated, simpler table --
one ``t`` column plus one Float64 column per named 1-D series (e.g.
``metrics_to_polars(t, ospa=ospa_values)``) -- for scalar-per-scan
evaluation metrics rather than per-track state. Both DataFrame kinds write
to Parquet with the ordinary ``df.write_parquet(path)`` and read back with
``pl.read_parquet`` or, for a metrics-style table, `read_measurements_csv`
/ `read_measurements_parquet` on the far side of a pipeline (the export and
ingest sides share a column-oriented convention by design).

Serialization Fidelity: MessagePack vs JSON
-----------------------------------------------

``encode_tracks`` / ``decode_tracks`` and ``encode_states`` / ``decode_states``
(:mod:`pytcl.io.serialize`, msgspec -- a core dependency) serialize the same
track-history and single-state shapes to bytes, in one of two wire formats
selected by ``fmt``:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - Property
     - ``fmt="msgpack"`` (default)
     - ``fmt="json"``
   * - Encoding
     - Compact binary
     - Human-readable text
   * - NaN / Inf
     - Preserved exactly (bit-identical ``float64``)
     - Not representable -- ``encode_*`` raises ``ValueError`` *before*
       writing anything
   * - Finite-value round-trip
     - Bit-exact (``tobytes()`` equal)
     - Bit-exact (``tobytes()`` equal)
   * - Best for
     - Archival, wire transport, anything that might carry a coasting
       track's inflated (but finite) or degenerate covariance
     - Interop with non-Python tooling, human inspection, diffing

Decoding is strict either way: msgspec validates the bytes against the
target ``Struct`` and raises rather than returning a partially-populated
result.

.. code-block:: python

   import numpy as np

   from pytcl.io import decode_states, encode_states

   x = np.array([100.0, 5.0, np.nan])   # a NaN slipped into a coasting track
   P = np.eye(3)

   blob = encode_states(x, P, fmt="msgpack")
   x2, P2 = decode_states(blob, fmt="msgpack")
   print(x2.tobytes() == x.tobytes())    # True -- bit-exact, NaN included

   try:
       encode_states(x, P, fmt="json")
   except ValueError as exc:
       print("json rejected it:", "non-finite" in str(exc))

AIS Decoding and Position Reports
-------------------------------------

:mod:`pytcl.transponders.ais` (the ``ais`` extra, via `pyais
<https://pypi.org/project/pyais/>`_) is this port's counterpart to the
MATLAB TCL's ``Transponders/decodeAISString``, which wraps libais.
``decode_ais`` reassembles and decodes ``!AIVDM``/``!AIVDO`` NMEA sentences,
skipping lines it cannot decode rather than raising (a batch decode over
potentially noisy logs). ``ais_position_reports`` pulls out message types 1,
2, 3, 18, 19 (Class A/B position reports) as parallel arrays, converting
ITU-R M.1371's "not available" sentinels (91 deg lat, 181 deg lon, 102.3 kn
SOG, 360 deg COG, heading 511) to NaN.

**Units matter here**: ``lat``, ``lon``, ``cog`` and ``heading`` come back in
**radians** (this library's angle convention at every API boundary, degrees
nowhere), and ``sog`` in **m/s** -- pyais itself reports degrees and knots,
so both conversions happen inside `ais_position_reports`.

.. code-block:: python

   import numpy as np

   from pytcl.transponders.ais import ais_position_reports, decode_ais

   # A widely published type-1 (Class A position report) test sentence.
   vdm = "!AIVDM,1,1,,A,15M67FC000G?ufbE`FepT@3n00Sa,0*5C"

   messages = decode_ais(vdm)
   print(messages[0].msg_type, messages[0].mmsi)   # 1 366053209

   rep = ais_position_reports(vdm)
   print(rep.lat.dtype)                             # float64
   print(round(float(np.degrees(rep.lat[0])), 4))    # degrees, for a human to read
   print(round(float(rep.sog[0]), 3))                # m/s, already converted

Feeding ``times`` (one receiver timestamp per decoded message, before the
position-report filter) threads a ``t`` column through `PositionReports`,
ready for `tracks_to_polars`-style downstream handling or a per-ship
constant-velocity filter keyed on ``mmsi``.

**Validated against real traffic**: ``tests/validation/test_ais_tracking.py``
runs a per-ship constant-velocity Kalman filter, positions only, over 6,808
position reports from 299 real ships -- captured from Kystverket's open AIS
feed off the Norwegian coast -- and scores the recovered speed against each
ship's self-broadcast SOG, a quantity the filter is never given: median
error 0.0134 m/s against a calibrated 0.03 m/s envelope. Reproduce:

.. code-block:: bash

   uv run pytest tests/validation/test_ais_tracking.py -q

The capture provenance, message-type histogram, independence argument and
full calibration record (including the ``PROCESS_VAR`` sweep the 0.03 m/s
envelope was derived from) are in ``tests/fixtures/ais/SOURCES.md``.

ASDF
-----

``save_tracks_asdf`` / `load_tracks_asdf` and ``save_states_asdf`` /
`load_states_asdf` (:mod:`pytcl.io.asdf_io`, the ``asdf`` extra) write the
same track-history and single-state shapes as `pytcl.io.serialize` to an
`ASDF <https://asdf.readthedocs.io/>`_ file: a self-describing,
schema-versioned ndarray tree, useful for archiving tracking results
alongside other ASDF-native metadata (WCS, provenance). States must be a
uniform dimension within one history; `save_tracks_asdf` raises
``ValueError`` naming the offending scan and track otherwise.

.. code-block:: python

   import tempfile
   from pathlib import Path

   import numpy as np
   from pytcl.io import load_tracks_asdf, save_tracks_asdf
   from pytcl.trackers import Track, TrackStatus

   track = Track(
       id=1, state=np.array([1.0, 2.0]), covariance=np.eye(2),
       status=TrackStatus.CONFIRMED, hits=1, misses=0, time=0.0,
   )
   tmpdir = tempfile.TemporaryDirectory()
   path = Path(tmpdir.name) / "tracks.asdf"

   save_tracks_asdf(path, [[track]], [0.0])
   times, history = load_tracks_asdf(path)
   print(times)                       # [0.0]
   print(history[0][0].state.tolist())  # [1.0, 2.0]

HDF5 Compression
-------------------

``TrackHDF5Storage`` (:mod:`pytcl.io.hdf5_track_storage`, ``h5py`` -- a core
dependency) is the archival backend for large tracking scenarios. Its
default configuration (``chunk_size=1000``, ``compression="gzip"``,
``compression_level=4``, ``shuffle=True``) was measured, not assumed: on a
100-track x 500-scan, 6-D benchmark whose covariances come from a *real*
converged constant-velocity Kalman filter (not random noise -- a converged
filter's covariance settles to a near-constant matrix, which is what makes
it compressible), the shipped configuration reaches **4.73x** compression,
up from a 4.42x baseline with the byte-shuffle filter off (+7.1%).
Time-aligned chunk shapes were evaluated too and found to already be the
existing behavior (measured 0.0% additional gain). Reproduce the full
measurement:

.. code-block:: bash

   uv run pytest tests/unit/test_hdf5_compression.py -q

.. code-block:: python

   import tempfile
   from pathlib import Path

   import numpy as np
   from pytcl.io import TrackHDF5Storage

   rng = np.random.default_rng(0)
   n_scans = 50
   states = np.cumsum(rng.normal(0, 0.1, size=(n_scans, 4)), axis=0)
   covariances = np.tile(np.eye(4) * 2.0, (n_scans, 1, 1))
   timestamps = np.arange(n_scans, dtype=np.float64)

   tmpdir = tempfile.TemporaryDirectory()
   path = Path(tmpdir.name) / "tracks.h5"

   with TrackHDF5Storage(str(path)) as store:  # shuffle=True by default
       store.open(mode="w")
       store.store_track("trk_001", states, covariances, timestamps)

   with TrackHDF5Storage(str(path)) as store:
       store.open(mode="r")
       traj = store.retrieve_track("trk_001")
   print(traj["states"].shape)   # (50, 4)

   raw_bytes = 8 * (states.size + covariances.size + timestamps.size)
   print(f"on-disk vs raw: {path.stat().st_size} / {raw_bytes} bytes")
