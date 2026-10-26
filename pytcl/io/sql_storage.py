"""SQL storage backend for pytcl data persistence.

Provides structured data storage using SQLite. Good for metadata, tracks,
measurements, and searchable structured data.
"""

import json
import os
import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.io.storage import StorageBackend


def _read_only_uri(path: str, pathmod: Any = os.path) -> str:
    """Return a SQLite ``mode=ro`` URI for ``path``.

    ``Path.resolve().as_uri()`` turned a UNC path, or a mapped drive that
    resolves to one, into ``file://server/share/...``; SQLite accepts only
    an empty or ``localhost`` authority there and refused to open it.
    ``pathmod`` is ``os.path``; tests pass ``ntpath`` to check Windows paths.
    """
    absolute = pathmod.abspath(path).replace(pathmod.sep, "/")
    if not absolute.startswith("/"):
        absolute = "/" + absolute
    return "file://" + quote(absolute, safe="/:") + "?mode=ro"


def _connect(path: str, mode: str) -> tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Open ``path`` under ``mode``.

    'r' uses SQLite's own read-only open as a backstop to the explicit
    ``_check_writable`` guards on the public write methods.
    """
    if mode == "r":
        connection = sqlite3.connect(_read_only_uri(path), uri=True)
    else:
        connection = sqlite3.connect(path)
    return connection, connection.cursor()


_SQLITE_SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")


def _truncate_sqlite(path: Path) -> None:
    """Remove ``path`` and its SQLite journal files, if present.

    A ``-journal`` or ``-wal`` left behind by a crashed writer would
    otherwise be replayed into the new database.
    """
    sidecars = (Path(f"{path}{suffix}") for suffix in _SQLITE_SIDECAR_SUFFIXES)
    for target in (path, *sidecars):
        target.unlink(missing_ok=True)


class SQLStorage(StorageBackend):
    """SQL-based storage backend.

    Stores structured data, metadata, and searchable information using SQL.
    SQLite only: ``open()`` passes its ``path`` straight to ``sqlite3.connect``,
    so a driver connection string is not accepted. The ``db_type`` argument
    that once selected a backend was removed in v2.0.0 (see the ``__init__``
    Notes).

    Ideal for:
    - Track metadata and state
    - Measurements and detections
    - Sensor configurations
    - Any structured, queryable data

    Examples
    --------
    >>> from pytcl.io import SQLStorage
    >>> with SQLStorage() as store:  # doctest: +SKIP
    ...     store.open("tracking.db", mode="w")
    ...     store.store_array("tracks", track_states)
    ...     store.store_scalar("mission/start_time", 1234567890)
    ...     results = store.retrieve_array("tracks")
    """

    def __init__(self) -> None:
        """Initialize SQLite storage.

        Notes
        -----
        This used to take a ``db_type`` argument documented as accepting
        "'sqlite' (default) or connection string for other databases". Any
        value other than ``'sqlite'`` made ``open()`` do nothing at all -- no
        connection was established -- after which every method raised
        ``RuntimeError("Storage not open")``. The argument advertised a
        capability that did not exist, so it has been removed (gh-21). Adding
        real support for another backend is a feature, not a parameter.
        """
        self._connection: Optional[sqlite3.Connection] = None
        self._cursor: Optional[sqlite3.Cursor] = None
        self._path: Optional[Path] = None
        self._mode: Optional[str] = None
        self._metadata_table = "_pytcl_metadata"
        self._arrays_table = "_pytcl_arrays"

    def open(self, path: str, mode: str = "r") -> None:
        """Open a SQLite database connection.

        Parameters
        ----------
        path : str
            Path to the SQLite database file.
        mode : str, optional
            'r' (read-only), 'w' (create), 'a' (create if absent, else
            append). Default is 'r'. 'w' creates the file, discarding any
            existing file and its SQLite journal files (matching
            ``HDF5Storage``).

        Raises
        ------
        FileNotFoundError
            If ``mode='r'`` and the file does not exist.
        ValueError
            If ``mode`` is not one of 'r', 'w', 'a'.

        Notes
        -----
        Read mode is enforced: writes through a ``'r'`` handle raise
        :class:`PermissionError` (they used to persist).

        ``'w'`` unlinks the path before connecting. On Windows that raises
        :class:`PermissionError` while another process holds the file open.
        A symlinked path is replaced by a regular file; the symlink's target
        is left intact. Calling ``open('w')`` again without :meth:`close`
        leaks the old connection, and on Windows the unlink then raises
        :class:`PermissionError` because that connection still holds the
        file.

        Opening a nonexistent path for reading used to create an empty
        database, because ``sqlite3.connect`` creates the file whatever the
        caller intended. Reads against it then failed with
        ``sqlite3.OperationalError`` about a missing table rather than the
        documented ``KeyError``, and the caller was left holding a stray file
        they never asked for (gh-21).
        """
        if mode not in ("r", "w", "a"):
            raise ValueError(f"mode must be 'r', 'w' or 'a', got {mode!r}")

        self._path = Path(path)
        self._mode = mode

        if mode == "w":
            _truncate_sqlite(self._path)

        if mode == "r" and not self._path.exists():
            raise FileNotFoundError(
                f"No such database: {path}. Opening for reading does not "
                f"create one; use mode='w' or mode='a' to create it."
            )

        self._connection, self._cursor = _connect(str(self._path), mode)

        if mode in ("w", "a"):
            self._initialize_tables()

    def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            self._connection.commit()
            self._connection.close()
            self._connection = None
            self._cursor = None

    def __enter__(self) -> "SQLStorage":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def store_array(
        self,
        name: str,
        data: ArrayLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store array as blob in SQL table.

        Parameters
        ----------
        name : str
            Array name/key
        data : ArrayLike
            Array to store
        metadata : dict, optional
            Associated metadata
        """
        self._check_writable()

        arr = np.asarray(data)

        # Serialize array to bytes
        binary_data = arr.tobytes()
        shape_json = json.dumps(list(arr.shape))
        dtype_str = str(arr.dtype)

        # Serialize metadata
        meta_json = json.dumps(metadata or {})

        # Store in arrays table
        try:
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._arrays_table}
                   (name, dtype, shape, data, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, dtype_str, shape_json, binary_data, meta_json),
            )
        except sqlite3.OperationalError:
            self._initialize_tables()
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._arrays_table}
                   (name, dtype, shape, data, metadata)
                   VALUES (?, ?, ?, ?, ?)""",
                (name, dtype_str, shape_json, binary_data, meta_json),
            )

        self._connection.commit()

    def retrieve_array(self, name: str) -> NDArray[Any]:
        """Retrieve a stored array.

        Parameters
        ----------
        name : str
            Array name/key

        Returns
        -------
        ndarray
            The reconstructed array
        """
        if self._cursor is None:
            raise RuntimeError("Storage not open. Call open() first.")

        self._cursor.execute(
            f"""SELECT dtype, shape, data FROM {self._arrays_table}
               WHERE name = ? LIMIT 1""",
            (name,),
        )

        row = self._cursor.fetchone()
        if row is None:
            raise KeyError(f"Array '{name}' not found in SQL database")

        dtype_str, shape_json, binary_data = row

        # Reconstruct array
        shape = tuple(json.loads(shape_json))
        dtype = np.dtype(dtype_str)
        arr = np.frombuffer(binary_data, dtype=dtype).reshape(shape)

        return arr.copy()

    def store_scalar(
        self,
        name: str,
        value: Union[int, float, str, bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a scalar value as metadata.

        Parameters
        ----------
        name : str
            Scalar name/key
        value : scalar
            Value to store
        metadata : dict, optional
            Associated metadata
        """
        self._check_writable()

        value_json = json.dumps({"value": value, "type": type(value).__name__})
        meta_json = json.dumps(metadata or {})

        try:
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._metadata_table}
                   (key, value, metadata)
                   VALUES (?, ?, ?)""",
                (name, value_json, meta_json),
            )
        except sqlite3.OperationalError:
            self._initialize_tables()
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._metadata_table}
                   (key, value, metadata)
                   VALUES (?, ?, ?)""",
                (name, value_json, meta_json),
            )

        self._connection.commit()

    def retrieve_scalar(self, name: str) -> Union[int, float, str, bool]:
        """Retrieve a scalar value.

        Parameters
        ----------
        name : str
            Scalar name/key

        Returns
        -------
        Scalar value
        """
        if self._cursor is None:
            raise RuntimeError("Storage not open. Call open() first.")

        self._cursor.execute(
            f"""SELECT value FROM {self._metadata_table}
               WHERE key = ? LIMIT 1""",
            (name,),
        )

        row = self._cursor.fetchone()
        if row is None:
            raise KeyError(f"Scalar '{name}' not found in SQL database")

        value_json = json.loads(row[0])
        return value_json.get("value")

    def store_group(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Mark a group/namespace for organization.

        Parameters
        ----------
        name : str
            Group name
        metadata : dict, optional
            Group-level metadata
        """
        self._check_writable()

        # Store as special metadata entry
        group_marker = json.dumps({"_is_group": True, **(metadata or {})})

        try:
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._metadata_table}
                   (key, value, metadata)
                   VALUES (?, ?, ?)""",
                (f"_group:{name}", group_marker, "{}"),
            )
        except sqlite3.OperationalError:
            self._initialize_tables()
            self._cursor.execute(
                f"""INSERT OR REPLACE INTO {self._metadata_table}
                   (key, value, metadata)
                   VALUES (?, ?, ?)""",
                (f"_group:{name}", group_marker, "{}"),
            )

        self._connection.commit()

    def list_keys(self, group: str = "/") -> List[str]:
        """List all keys in storage or a group.

        Parameters
        ----------
        group : str, optional
            Group path. Lists the keys under ``group + "/"``, matched
            literally and case-sensitively (``_`` and ``%`` are not
            wildcards; ``"a"`` lists neither ``"ab/x"``, ``"A/x"`` nor the
            bare key ``"a"``). A trailing ``/`` is optional. Default is "/"
            (all keys); ``""`` also lists all keys.

        Returns
        -------
        list of str
            Keys in the group
        """
        if self._cursor is None:
            raise RuntimeError("Storage not open. Call open() first.")

        if group.strip("/"):
            # Compare "group/" as an exact, case-sensitive prefix. LIKE treated
            # "_" as a wildcard, folded ASCII case, and had no "/" boundary,
            # so "a_b" also listed "aXb/y", "A_B/y" and "a_bc/z".
            prefix = group.rstrip("/") + "/"
            self._cursor.execute(
                f"""SELECT name FROM {self._arrays_table}
                   WHERE substr(name, 1, ?) = ?
                   UNION
                   SELECT key FROM {self._metadata_table}
                   WHERE substr(key, 1, ?) = ?""",
                (len(prefix), prefix, len(prefix), prefix),
            )
        else:
            # List all keys
            self._cursor.execute(f"""SELECT name FROM {self._arrays_table}
                   UNION
                   SELECT key FROM {self._metadata_table}""")

        return [row[0] for row in self._cursor.fetchall()]

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for an entry.

        Parameters
        ----------
        name : str
            Entry name

        Returns
        -------
        dict
            Metadata
        """
        if self._cursor is None:
            raise RuntimeError("Storage not open. Call open() first.")

        # Try arrays table first
        self._cursor.execute(
            f"""SELECT metadata FROM {self._arrays_table}
               WHERE name = ? LIMIT 1""",
            (name,),
        )
        row = self._cursor.fetchone()

        if row:
            return json.loads(row[0])

        # Try metadata table
        self._cursor.execute(
            f"""SELECT metadata FROM {self._metadata_table}
               WHERE key = ? LIMIT 1""",
            (name,),
        )
        row = self._cursor.fetchone()

        if row:
            return json.loads(row[0])

        raise KeyError(f"Entry '{name}' not found")

    def delete(self, name: str) -> None:
        """Delete an entry.

        Parameters
        ----------
        name : str
            Entry name
        """
        self._check_writable()

        self._cursor.execute(
            f"""DELETE FROM {self._arrays_table} WHERE name = ?""", (name,)
        )
        self._cursor.execute(
            f"""DELETE FROM {self._metadata_table} WHERE key = ?""", (name,)
        )
        self._connection.commit()

    def _check_writable(self) -> None:
        if self._cursor is None:
            raise RuntimeError("Storage not open. Call open() first.")
        if self._mode == "r":
            raise PermissionError(
                "Storage was opened read-only (mode='r'); reopen with "
                "mode='a' or mode='w' to write."
            )

    def flush(self) -> None:
        """Commit any pending transactions."""
        if self._connection is not None:
            self._connection.commit()

    def _initialize_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        if self._cursor is None:
            return

        # Metadata table
        self._cursor.execute(f"""CREATE TABLE IF NOT EXISTS {self._metadata_table} (
               key TEXT PRIMARY KEY,
               value TEXT,
               metadata TEXT DEFAULT '{{}}'
           )""")

        # Arrays table
        self._cursor.execute(f"""CREATE TABLE IF NOT EXISTS {self._arrays_table} (
               name TEXT PRIMARY KEY,
               dtype TEXT,
               shape TEXT,
               data BLOB,
               metadata TEXT DEFAULT '{{}}'
           )""")

        self._connection.commit()
