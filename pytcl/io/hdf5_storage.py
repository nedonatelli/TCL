"""HDF5 storage backend for pytcl data persistence.

Provides efficient storage of large numerical arrays using HDF5.
Requires h5py package.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.io.storage import StorageBackend

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class HDF5Storage(StorageBackend):
    """HDF5-based storage backend.

    Efficiently stores large arrays and structured data using HDF5 format.
    Ideal for numerical data, model coefficients, and time series.

    Examples
    --------
    >>> from pytcl.io import HDF5Storage
    >>> with HDF5Storage() as store:  # doctest: +SKIP
    ...     store.open("data.h5", mode="w")
    ...     store.store_array("gravity/egm96", coefficients)
    ...     store.store_scalar("header/version", 1)
    ...     result = store.retrieve_array("gravity/egm96")
    """

    def __init__(self):
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required for HDF5Storage. Install with: pip install h5py"
            )
        self._file = None
        self._path = None
        self._mode = None

    def open(self, path: str, mode: str = "r") -> None:
        """Open an HDF5 file.

        Parameters
        ----------
        path : str
            Path to HDF5 file
        mode : str, optional
            'r' (read), 'w' (write), 'a' (append). Default is 'r'.
        """
        self._path = Path(path)
        self._mode = mode
        self._file = h5py.File(str(self._path), mode=mode)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def store_array(
        self,
        name: str,
        data: ArrayLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a numpy array as an HDF5 dataset.

        Parameters
        ----------
        name : str
            Dataset path (e.g., "groups/subgroup/data")
        data : ArrayLike
            Array to store
        metadata : dict, optional
            Metadata stored as HDF5 attributes
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        # Ensure parent groups exist
        self._ensure_groups(name)

        # Store array
        arr = np.asarray(data)
        dataset = self._file.create_dataset(name, data=arr)

        # Store metadata as attributes
        if metadata:
            for key, value in metadata.items():
                try:
                    dataset.attrs[key] = value
                except (TypeError, ValueError):
                    # For non-serializable objects, store as JSON string
                    dataset.attrs[key] = json.dumps(str(value))

    def retrieve_array(self, name: str) -> NDArray:
        """Retrieve a stored array.

        Parameters
        ----------
        name : str
            Dataset path

        Returns
        -------
        ndarray
            The stored array
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        if name not in self._file:
            raise KeyError(f"Dataset '{name}' not found in HDF5 file")

        return np.array(self._file[name])

    def store_scalar(
        self,
        name: str,
        value: Union[int, float, str, bool],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a scalar value.

        Parameters
        ----------
        name : str
            Scalar name/path
        value : scalar
            Value to store
        metadata : dict, optional
            Associated metadata
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        # Ensure parent groups exist
        self._ensure_groups(name)

        # Store as dataset with shape ()
        dataset = self._file.create_dataset(name, data=value)

        if metadata:
            for key, val in metadata.items():
                try:
                    dataset.attrs[key] = val
                except (TypeError, ValueError):
                    dataset.attrs[key] = json.dumps(str(val))

    def retrieve_scalar(self, name: str) -> Union[int, float, str, bool]:
        """Retrieve a scalar value.

        Parameters
        ----------
        name : str
            Scalar name/path

        Returns
        -------
        Scalar value
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        if name not in self._file:
            raise KeyError(f"Scalar '{name}' not found in HDF5 file")

        value = self._file[name][()]
        # Handle bytes (HDF5 stores strings as bytes)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value.item() if hasattr(value, "item") else value

    def store_group(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a group for organizing related datasets.

        Parameters
        ----------
        name : str
            Group path
        metadata : dict, optional
            Group-level metadata
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        # Create groups recursively if they don't already exist
        if name not in self._file:
            self._file.create_group(name)

        if metadata:
            group = self._file[name]
            for key, val in metadata.items():
                try:
                    group.attrs[key] = val
                except (TypeError, ValueError):
                    group.attrs[key] = json.dumps(str(val))

    def list_keys(self, group: str = "/") -> List[str]:
        """List datasets and groups in a location.

        Parameters
        ----------
        group : str, optional
            Group path. Default is root.

        Returns
        -------
        list of str
            Keys in the group
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        try:
            return list(self._file[group].keys())
        except KeyError:
            raise KeyError(f"Group '{group}' not found in HDF5 file")

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a dataset or group.

        Parameters
        ----------
        name : str
            Dataset/group name

        Returns
        -------
        dict
            Metadata attributes
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        if name not in self._file:
            raise KeyError(f"'{name}' not found in HDF5 file")

        obj = self._file[name]
        metadata = {}

        for key, value in obj.attrs.items():
            if isinstance(value, str):
                # store_* serializes non-native values as json.dumps(str(v)),
                # which always decodes back to a str. Only reverse that case;
                # leave user strings that happen to parse as JSON (e.g. "123",
                # "true") untouched so string metadata round-trips exactly.
                try:
                    parsed = json.loads(value)
                    metadata[key] = parsed if isinstance(parsed, str) else value
                except (json.JSONDecodeError, TypeError):
                    metadata[key] = value
            else:
                metadata[key] = value

        return metadata

    def delete(self, name: str) -> None:
        """Delete a dataset or group.

        Parameters
        ----------
        name : str
            Dataset/group path
        """
        if self._file is None:
            raise RuntimeError("Storage file not open. Call open() first.")

        if name in self._file:
            del self._file[name]

    def flush(self) -> None:
        """Ensure all data is written to disk."""
        if self._file is not None:
            self._file.flush()

    def _ensure_groups(self, path: str, is_group: bool = False) -> None:
        """Ensure parent groups exist, creating them if needed.

        Parameters
        ----------
        path : str
            Full path to dataset or group
        is_group : bool, optional
            If True, treat path as group. Default treats as dataset path.
        """
        if "/" not in path:
            return

        # Get parent path
        if is_group:
            parent = path
        else:
            parent = path.rsplit("/", 1)[0]

        if parent and parent != "/" and parent not in self._file:
            self._file.create_group(parent)
