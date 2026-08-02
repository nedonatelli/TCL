"""Abstract storage interface for pytcl data persistence.

This module provides interfaces for storing and retrieving pytcl data
in different formats (HDF5, SQL, etc.).
"""

from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, Dict, List, Optional, Union

from numpy.typing import ArrayLike, NDArray


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Provides a unified interface for storing and retrieving arrays,
    metadata, and structured data in various formats.
    """

    @abstractmethod
    def open(self, path: str, mode: str = "r") -> None:
        """Open a storage file or database connection.

        Parameters
        ----------
        path : str
            Path to storage file or database URI
        mode : str, optional
            Open mode: 'r' (read), 'w' (write), 'a' (append). Default is 'r'.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""
        pass

    @abstractmethod
    def __enter__(self) -> "StorageBackend":
        """Context manager entry."""
        return self

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit."""
        pass

    @abstractmethod
    def store_array(
        self,
        name: str,
        data: ArrayLike,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a numpy array, replacing any array already under that name.

        Parameters
        ----------
        name : str
            Dataset name/key for storage
        data : ArrayLike
            Numpy array to store
        metadata : dict, optional
            Associated metadata (e.g., units, description). Replaced wholesale
            along with the array; metadata from a previous store under the same
            name does not survive.

        Notes
        -----
        The replace-on-collision rule is stated here because the backends used
        to disagree and neither said so: ``SQLStorage`` replaced, while
        ``HDF5Storage`` let h5py raise ``ValueError`` on an existing name
        (gh-21). Code written against one backend broke on the other, and the
        base class -- the only place the contract could live -- was silent.
        """
        pass

    @abstractmethod
    def retrieve_array(self, name: str) -> NDArray[Any]:
        """Retrieve a stored numpy array.

        Parameters
        ----------
        name : str
            Dataset name/key

        Returns
        -------
        ndarray
            The stored array
        """
        pass

    @abstractmethod
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
            Scalar name/key
        value : int, float, str, bool
            Scalar value to store
        metadata : dict, optional
            Associated metadata
        """
        pass

    @abstractmethod
    def retrieve_scalar(self, name: str) -> Union[int, float, str, bool]:
        """Retrieve a stored scalar value.

        Parameters
        ----------
        name : str
            Scalar name/key

        Returns
        -------
        Scalar value
        """
        pass

    @abstractmethod
    def store_group(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Create a logical group/container for related data.

        Parameters
        ----------
        name : str
            Group name
        metadata : dict, optional
            Group-level metadata
        """
        pass

    @abstractmethod
    def list_keys(self, group: str = "/") -> List[str]:
        """List all stored keys/datasets in a group.

        Parameters
        ----------
        group : str, optional
            Group path. Default is root ("/")

        Returns
        -------
        list of str
            Keys in the group
        """
        pass

    @abstractmethod
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata associated with a dataset.

        Parameters
        ----------
        name : str
            Dataset name

        Returns
        -------
        dict
            Metadata dictionary
        """
        pass

    @abstractmethod
    def delete(self, name: str) -> None:
        """Delete a dataset or group.

        Parameters
        ----------
        name : str
            Dataset/group name to delete
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Ensure all data is written to disk."""
        pass
