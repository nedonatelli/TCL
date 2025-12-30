"""
Pytest configuration and fixtures for Tracker Component Library tests.
"""

import numpy as np
import pytest


@pytest.fixture
def random_seed():
    """Set a fixed random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def tolerance():
    """Standard numerical tolerance for comparisons."""
    return {"rtol": 1e-10, "atol": 1e-14}


@pytest.fixture
def loose_tolerance():
    """Looser tolerance for iterative algorithms."""
    return {"rtol": 1e-6, "atol": 1e-10}


@pytest.fixture
def random_vector_3d(random_seed):
    """Generate a random 3D vector."""
    return np.random.randn(3)


@pytest.fixture
def random_unit_vector_3d(random_seed):
    """Generate a random unit 3D vector."""
    v = np.random.randn(3)
    return v / np.linalg.norm(v)


@pytest.fixture
def random_rotation_matrix(random_seed):
    """Generate a random 3x3 rotation matrix."""
    # Use QR decomposition of random matrix
    A = np.random.randn(3, 3)
    Q, R = np.linalg.qr(A)
    # Ensure proper rotation (det = +1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


@pytest.fixture
def random_covariance_matrix(random_seed):
    """Generate a random positive definite covariance matrix."""
    A = np.random.randn(4, 4)
    return A @ A.T + 0.1 * np.eye(4)


@pytest.fixture
def sample_track_state():
    """Sample state vector for tracking [x, vx, y, vy]."""
    return np.array([1000.0, 10.0, 2000.0, -5.0])


@pytest.fixture
def sample_covariance():
    """Sample covariance matrix for a 4-state tracker."""
    return np.diag([100.0, 1.0, 100.0, 1.0])


class NumpyTestCase:
    """Base class with numpy assertion helpers."""

    def assert_allclose(self, actual, expected, rtol=1e-10, atol=1e-14, **kwargs):
        """Assert arrays are close within tolerance."""
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)

    def assert_array_equal(self, actual, expected):
        """Assert arrays are exactly equal."""
        np.testing.assert_array_equal(actual, expected)

    def assert_shape(self, arr, expected_shape):
        """Assert array has expected shape."""
        assert (
            arr.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {arr.shape}"

    def assert_symmetric(self, arr, rtol=1e-10):
        """Assert matrix is symmetric."""
        np.testing.assert_allclose(arr, arr.T, rtol=rtol)

    def assert_positive_definite(self, arr, tol=1e-10):
        """Assert matrix is positive definite."""
        eigenvalues = np.linalg.eigvalsh(arr)
        assert np.all(
            eigenvalues > -tol
        ), f"Matrix not positive definite: min eigenvalue = {np.min(eigenvalues)}"

    def assert_orthogonal(self, arr, rtol=1e-10):
        """Assert matrix is orthogonal."""
        n = arr.shape[0]
        np.testing.assert_allclose(arr @ arr.T, np.eye(n), rtol=rtol)
        np.testing.assert_allclose(arr.T @ arr, np.eye(n), rtol=rtol)

    def assert_rotation_matrix(self, arr, rtol=1e-10):
        """Assert matrix is a proper rotation matrix."""
        self.assert_orthogonal(arr, rtol=rtol)
        det = np.linalg.det(arr)
        assert abs(det - 1.0) < rtol, f"Determinant should be +1, got {det}"


@pytest.fixture
def numpy_test():
    """Provide NumpyTestCase helper methods."""
    return NumpyTestCase()
