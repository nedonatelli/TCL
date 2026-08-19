I/O and Storage
===============

Persistent storage backends, tabular measurement ingest, serialization, and
session snapshot/resume. See :doc:`../results_io` and :doc:`../sessions` for
the narrative guides.

.. automodule:: pytcl.io
   :no-members:
   :no-undoc-members:

Storage Interface
-----------------

Abstract storage interface shared by the SQL and HDF5 backends.

.. automodule:: pytcl.io.storage
   :members:
   :undoc-members:
   :show-inheritance:

SQL Storage
-----------

SQLite-backed structured storage for metadata and query-driven access.

.. automodule:: pytcl.io.sql_storage
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Storage
------------

HDF5 storage for large numerical arrays.

.. automodule:: pytcl.io.hdf5_storage
   :members:
   :undoc-members:
   :show-inheritance:

HDF5 Track Storage
------------------

HDF5-backed archival storage for large-scale tracking datasets.

.. automodule:: pytcl.io.hdf5_track_storage
   :members:
   :undoc-members:
   :show-inheritance:

Track Database
--------------

SQL-backed track lifecycle management: detections, initiation, maintenance.

.. automodule:: pytcl.io.track_database
   :members:
   :undoc-members:
   :show-inheritance:

Measurement Readers
-------------------

CSV and Parquet readers for tabular measurement data.

.. automodule:: pytcl.io.readers
   :members:
   :undoc-members:
   :show-inheritance:

DataFrame Accessors
-------------------

polars accessors for track histories and scalar metrics (``dataframe`` extra).

.. automodule:: pytcl.io.dataframes
   :members:
   :undoc-members:
   :show-inheritance:

Serialization
-------------

msgspec JSON and MessagePack serialization for filter states and tracks.

.. automodule:: pytcl.io.serialize
   :members:
   :undoc-members:
   :show-inheritance:

ASDF Export
-----------

ASDF archival export/import (``asdf`` extra).

.. automodule:: pytcl.io.asdf_io
   :members:
   :undoc-members:
   :show-inheritance:

Sessions
--------

Full tracker/filter state snapshot and resume.

.. automodule:: pytcl.io.session
   :members:
   :undoc-members:
   :show-inheritance:

Migration Tools
---------------

Utilities for moving v1.x tracking pipelines and stored data to v2.x.

.. automodule:: pytcl.io.migration
   :members:
   :undoc-members:
   :show-inheritance:

v1.x Compatibility Adapters
---------------------------

Adapters connecting v1.x filter outputs to the v2.x storage layer.

.. automodule:: pytcl.io.compat
   :members:
   :undoc-members:
   :show-inheritance:
