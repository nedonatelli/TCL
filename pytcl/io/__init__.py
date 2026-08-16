"""Data I/O and storage module for pytcl.

Provides persistent storage backends for pytcl data including:
- HDF5: Efficient storage of large numerical arrays
- SQL: Structured data storage and metadata management

Examples
--------
Store tracking data in HDF5:

>>> from pytcl.io import HDF5Storage
>>> with HDF5Storage() as store:  # doctest: +SKIP
...     store.open("tracking.h5", mode="w")
...     store.store_array("states", track_states)
...     store.store_scalar("num_tracks", 42)

Query structured data in SQL:

>>> from pytcl.io import SQLStorage
>>> with SQLStorage() as store:  # doctest: +SKIP
...     store.open("tracking.db", mode="a")
...     store.store_group("mission")
...     store.store_scalar("mission/start_time", 1234567890)
...     keys = store.list_keys("mission")
"""

from pytcl.io.asdf_io import (
    load_states_asdf,
    load_tracks_asdf,
    save_states_asdf,
    save_tracks_asdf,
)
from pytcl.io.compat import (
    EKFTrackAdapter,
    IMMTrackAdapter,
    KalmanTrackAdapter,
    ParticleFilterTrackAdapter,
    TrackerDatabaseAdapter,
    UKFTrackAdapter,
    store_filter_result,
)
from pytcl.io.dataframes import (
    explode_state_columns,
    metrics_to_polars,
    tracks_to_polars,
)
from pytcl.io.hdf5_storage import HDF5Storage
from pytcl.io.hdf5_track_storage import TrackHDF5Storage
from pytcl.io.migration import MigrationHelper
from pytcl.io.readers import (
    MeasurementSet,
    read_measurements_csv,
    read_measurements_parquet,
)
from pytcl.io.serialize import (
    SimpleTrack,
    StateRecord,
    TrackRecord,
    TrackSet,
    decode_states,
    decode_tracks,
    encode_states,
    encode_tracks,
)
from pytcl.io.session import (
    SESSION_SCHEMA_VERSION,
    load_session,
    load_session_file,
    save_session,
    save_session_file,
)
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
    "TrackRecord",
    "TrackSet",
    "StateRecord",
    "SimpleTrack",
    "encode_tracks",
    "decode_tracks",
    "encode_states",
    "decode_states",
    "tracks_to_polars",
    "explode_state_columns",
    "metrics_to_polars",
    "MeasurementSet",
    "read_measurements_csv",
    "read_measurements_parquet",
    "save_tracks_asdf",
    "load_tracks_asdf",
    "save_states_asdf",
    "load_states_asdf",
    "SESSION_SCHEMA_VERSION",
    "save_session",
    "save_session_file",
    "load_session",
    "load_session_file",
]
