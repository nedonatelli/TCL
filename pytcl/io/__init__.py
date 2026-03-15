"""Data I/O and storage module for pytcl.

Provides persistent storage backends for pytcl data including:
- HDF5: Efficient storage of large numerical arrays
- SQL: Structured data storage and metadata management

Examples
--------
Store tracking data in HDF5:

>>> from pytcl.io import HDF5Storage
>>> with HDF5Storage() as store:
...     store.open("tracking.h5", mode="w")
...     store.store_array("states", track_states)
...     store.store_scalar("num_tracks", 42)

Query structured data in SQL:

>>> from pytcl.io import SQLStorage
>>> with SQLStorage() as store:
...     store.open("tracking.db", mode="a")
...     store.store_group("mission")
...     store.store_scalar("mission/start_time", 1234567890)
...     keys = store.list_keys("mission")
"""

from pytcl.io.compat import (
    EKFTrackAdapter,
    IMMTrackAdapter,
    KalmanTrackAdapter,
    ParticleFilterTrackAdapter,
    TrackerDatabaseAdapter,
    UKFTrackAdapter,
    store_filter_result,
)
from pytcl.io.hdf5_storage import HDF5Storage
from pytcl.io.hdf5_track_storage import TrackHDF5Storage
from pytcl.io.migration import MigrationHelper
from pytcl.io.sql_storage import SQLStorage
from pytcl.io.storage import StorageBackend
from pytcl.io.track_database import TrackDatabaseManager, TrackDatabaseStatus

__all__ = [
    "StorageBackend",
    "HDF5Storage",
    "SQLStorage",
    "TrackDatabaseManager",
    "TrackDatabaseStatus",
    "TrackHDF5Storage",
    "KalmanTrackAdapter",
    "EKFTrackAdapter",
    "UKFTrackAdapter",
    "IMMTrackAdapter",
    "ParticleFilterTrackAdapter",
    "TrackerDatabaseAdapter",
    "store_filter_result",
    "MigrationHelper",
]
