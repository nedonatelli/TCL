"""Tests for pytcl.io storage backends."""

import tempfile

import numpy as np
import pytest

from pytcl.io import HDF5Storage, SQLStorage, StorageBackend


class TestHDF5Storage:
    """Tests for HDF5 storage backend."""

    @pytest.fixture
    def temp_file(self):
        """Create temporary HDF5 file."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            yield f.name

    def test_context_manager(self, temp_file):
        """Test HDF5Storage context manager."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_scalar("test", 42)

        # Verify data is closed properly
        assert store._file is None

    def test_store_retrieve_array(self, temp_file):
        """Test storing and retrieving arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)

        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array("test_array", data, metadata={"units": "meters"})
            store.flush()

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            retrieved = store.retrieve_array("test_array")

        np.testing.assert_array_equal(retrieved, data)
        assert retrieved.dtype == np.float32

    def test_store_retrieve_scalar(self, temp_file):
        """Test storing and retrieving scalars."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_scalar("test_int", 42)
            store.store_scalar("test_float", 3.14)
            store.store_scalar("test_str", "hello")
            store.flush()

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            assert store.retrieve_scalar("test_int") == 42
            assert store.retrieve_scalar("test_float") == pytest.approx(3.14)
            assert store.retrieve_scalar("test_str") == "hello"

    def test_store_group(self, temp_file):
        """Test group creation."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_group("gravity", metadata={"model": "EGM96"})
            store.store_array("gravity/coefficients", np.array([1, 2, 3]))

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            keys = store.list_keys("gravity")
            assert "coefficients" in keys

    def test_list_keys(self, temp_file):
        """Test listing keys."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array("a", np.array([1]))
            store.store_array("b", np.array([2]))
            store.store_scalar("c", 3)

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            keys = store.list_keys()
            assert "a" in keys
            assert "b" in keys
            assert "c" in keys

    def test_get_metadata(self, temp_file):
        """Test metadata retrieval."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array(
                "data", np.array([1, 2, 3]), metadata={"units": "m/s", "source": "GPS"}
            )

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            meta = store.get_metadata("data")
            assert meta["units"] == "m/s"
            assert meta["source"] == "GPS"

    def test_delete(self, temp_file):
        """Test deleting datasets."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array("data", np.array([1, 2, 3]))
            store.delete("data")

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            with pytest.raises(KeyError):
                store.retrieve_array("data")

    def test_nested_groups(self, temp_file):
        """Test nested group creation."""
        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array("gravity/models/egm96/coefficients", np.array([1, 2, 3]))
            store.flush()

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            data = store.retrieve_array("gravity/models/egm96/coefficients")
            np.testing.assert_array_equal(data, np.array([1, 2, 3]))

    def test_large_array(self, temp_file):
        """Test storing large arrays."""
        large_data = np.random.randn(1000, 1000)

        with HDF5Storage() as store:
            store.open(temp_file, mode="w")
            store.store_array("large", large_data)
            store.flush()

        with HDF5Storage() as store:
            store.open(temp_file, mode="r")
            retrieved = store.retrieve_array("large")
            np.testing.assert_array_equal(retrieved, large_data)


class TestSQLStorage:
    """Tests for SQL storage backend."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary SQLite database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name

    def test_context_manager(self, temp_db):
        """Test SQLStorage context manager."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_scalar("test", 42)

        assert store._connection is None

    def test_store_retrieve_array(self, temp_db):
        """Test storing and retrieving arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)

        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_array("test_array", data, metadata={"units": "meters"})

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            retrieved = store.retrieve_array("test_array")

        np.testing.assert_array_equal(retrieved, data)
        assert retrieved.dtype == np.int32

    def test_store_retrieve_scalar(self, temp_db):
        """Test storing and retrieving scalars."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_scalar("test_int", 42)
            store.store_scalar("test_float", 3.14)
            store.store_scalar("test_str", "hello")

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            assert store.retrieve_scalar("test_int") == 42
            assert store.retrieve_scalar("test_float") == pytest.approx(3.14)
            assert store.retrieve_scalar("test_str") == "hello"

    def test_store_group(self, temp_db):
        """Test group creation."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_group("mission", metadata={"start": "2025-01-01"})
            store.store_scalar("mission/duration", 3600)

    def test_list_keys(self, temp_db):
        """Test listing keys."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_array("data1", np.array([1]))
            store.store_array("data2", np.array([2]))
            store.store_scalar("scalar1", 3)

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            keys = store.list_keys()
            assert "data1" in keys
            assert "data2" in keys
            assert "scalar1" in keys

    def test_get_metadata(self, temp_db):
        """Test metadata retrieval."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_array(
                "data", np.array([1, 2, 3]), metadata={"units": "m/s", "sensor": "IMU"}
            )

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            meta = store.get_metadata("data")
            assert meta["units"] == "m/s"
            assert meta["sensor"] == "IMU"

    def test_delete(self, temp_db):
        """Test deleting entries."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_scalar("test", 42)
            store.delete("test")

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            with pytest.raises(KeyError):
                store.retrieve_scalar("test")

    def test_float64_array(self, temp_db):
        """Test storing float64 arrays."""
        data = np.array([[1.5, 2.7], [3.2, 4.9]], dtype=np.float64)

        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_array("floats", data)

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            retrieved = store.retrieve_array("floats")
            np.testing.assert_array_almost_equal(retrieved, data)

    def test_replace_existing(self, temp_db):
        """Test replacing existing data."""
        with SQLStorage() as store:
            store.open(temp_db, mode="w")
            store.store_scalar("value", 10)
            store.store_scalar("value", 20)

        with SQLStorage() as store:
            store.open(temp_db, mode="r")
            assert store.retrieve_scalar("value") == 20


class TestStorageBackend:
    """Tests for StorageBackend abstract class."""

    def test_is_abstract(self):
        """Test that StorageBackend is abstract."""
        with pytest.raises(TypeError):
            StorageBackend()


class TestStorageInteroperability:
    """Test compatibility between storage backends."""

    def test_array_compatibility(self):
        """Test that arrays stored in one format can be understood by others."""
        data_1d = np.array([1, 2, 3, 4, 5])
        data_2d = np.array([[1, 2], [3, 4], [5, 6]])

        with (
            tempfile.NamedTemporaryFile(suffix=".h5") as hf,
            tempfile.NamedTemporaryFile(suffix=".db") as sf,
        ):

            # Store in HDF5
            with HDF5Storage() as store:
                store.open(hf.name, mode="w")
                store.store_array("data1d", data_1d)
                store.store_array("data2d", data_2d)

            # Store in SQL
            with SQLStorage() as store:
                store.open(sf.name, mode="w")
                store.store_array("data1d", data_1d)
                store.store_array("data2d", data_2d)

            # Verify matching results
            with HDF5Storage() as h_store:
                h_store.open(hf.name, mode="r")
                h_data1d = h_store.retrieve_array("data1d")
                h_data2d = h_store.retrieve_array("data2d")

            with SQLStorage() as s_store:
                s_store.open(sf.name, mode="r")
                s_data1d = s_store.retrieve_array("data1d")
                s_data2d = s_store.retrieve_array("data2d")

            np.testing.assert_array_equal(h_data1d, s_data1d)
            np.testing.assert_array_equal(h_data2d, s_data2d)
