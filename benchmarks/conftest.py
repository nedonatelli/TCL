"""
Benchmark fixtures with session-scoped expensive setup.

These fixtures pre-compute test data to avoid expensive setup during
benchmark iterations. Session scope ensures data is computed once per
test session rather than per test function.
"""

import numpy as np
import pytest

# =============================================================================
# Kalman Filter Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def kalman_test_matrices():
    """
    Pre-computed Kalman filter test matrices of various sizes.

    Returns dict keyed by state dimension with F, Q, P, x matrices.
    """
    np.random.seed(42)
    sizes = [4, 6, 9, 12]
    matrices = {}
    for n in sizes:
        # Create stable transition matrix (eigenvalues < 1)
        F = np.eye(n) + 0.05 * np.random.randn(n, n)
        Q = np.eye(n) * 0.01
        P = np.eye(n) * 0.1
        x = np.random.randn(n)
        matrices[n] = {"F": F, "Q": Q, "P": P, "x": x}
    return matrices


@pytest.fixture(scope="session")
def measurement_matrices():
    """
    Pre-computed measurement matrices for update benchmarks.

    Returns dict keyed by (state_dim, meas_dim) tuple with H, R matrices.
    """
    np.random.seed(42)
    configs = [(4, 2), (6, 3), (9, 3), (12, 4)]
    matrices = {}
    for n, m in configs:
        # Create measurement matrix that observes subset of states
        H = np.zeros((m, n))
        for i in range(m):
            H[i, i * (n // m)] = 1.0
        R = np.eye(m) * 0.1
        z = np.random.randn(m)
        matrices[(n, m)] = {"H": H, "R": R, "z": z}
    return matrices


# =============================================================================
# Gating / Distance Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def random_point_clouds():
    """
    Pre-generated point clouds for clustering/gating benchmarks.

    Returns dict keyed by n_points with sub-dict keyed by dimension.
    """
    np.random.seed(42)
    clouds = {}
    for n in [100, 500, 1000, 5000]:
        clouds[n] = {
            2: np.random.randn(n, 2),
            3: np.random.randn(n, 3),
            6: np.random.randn(n, 6),
        }
    return clouds


@pytest.fixture(scope="session")
def covariance_matrices():
    """
    Pre-computed positive definite covariance matrices.

    Returns dict keyed by dimension.
    """
    np.random.seed(42)
    matrices = {}
    for d in [2, 3, 4, 6, 9]:
        # Generate random positive definite matrix
        A = np.random.randn(d, d)
        matrices[d] = A @ A.T + np.eye(d) * 0.1
    return matrices


@pytest.fixture(scope="session")
def gating_test_data():
    """
    Pre-computed data for gating benchmarks.

    Returns dict with tracks, measurements, and covariances.
    """
    np.random.seed(42)
    n_tracks = 20
    n_meas = 50
    dim = 3

    return {
        "track_states": np.random.randn(n_tracks, dim),
        "track_covs": np.array([np.eye(dim) * 0.1 for _ in range(n_tracks)]),
        "measurements": np.random.randn(n_meas, dim),
        "meas_covs": np.array([np.eye(dim) * 0.05 for _ in range(n_meas)]),
    }


# =============================================================================
# Rotation Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def rotation_test_data():
    """
    Pre-computed rotation test inputs.

    Returns dict with euler_angles, quaternions, rotation_matrices.
    """
    np.random.seed(42)
    n = 1000

    # Euler angles in valid range
    euler_angles = np.random.uniform(-np.pi, np.pi, (n, 3))

    # Unit quaternions
    quats = np.random.randn(n, 4)
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)

    # Random rotation matrices (via QR decomposition)
    rotation_matrices = []
    for _ in range(n):
        Q, _ = np.linalg.qr(np.random.randn(3, 3))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        rotation_matrices.append(Q)
    rotation_matrices = np.array(rotation_matrices)

    return {
        "euler_angles": euler_angles,
        "quaternions": quats,
        "rotation_matrices": rotation_matrices,
        "single_euler": euler_angles[0],
        "single_quat": quats[0],
        "single_rotmat": rotation_matrices[0],
    }


# =============================================================================
# Signal Processing Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def cfar_test_signals():
    """
    Pre-generated CFAR test signals with embedded targets.

    Returns dict keyed by signal length.
    """
    np.random.seed(42)
    signals = {}
    for n in [1000, 5000, 10000]:
        # Exponential noise (Rayleigh power)
        sig = np.random.exponential(1.0, n)
        # Add targets at known locations
        sig[n // 4] = 50.0
        sig[n // 2] = 100.0
        sig[3 * n // 4] = 30.0
        signals[n] = sig
    return signals


@pytest.fixture(scope="session")
def cfar_2d_test_data():
    """
    Pre-generated 2D CFAR test data (range-Doppler maps).

    Returns dict keyed by shape tuple.
    """
    np.random.seed(42)
    data = {}
    for shape in [(64, 64), (128, 128), (256, 256)]:
        # Exponential noise
        rd_map = np.random.exponential(1.0, shape)
        # Add targets
        rd_map[shape[0] // 4, shape[1] // 4] = 50.0
        rd_map[shape[0] // 2, shape[1] // 2] = 100.0
        data[shape] = rd_map
    return data


# =============================================================================
# Clustering Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def clustering_test_data():
    """
    Pre-generated clustering test data with known cluster structure.

    Returns dict keyed by n_points.
    """
    np.random.seed(42)
    data = {}
    for n in [100, 500, 1000]:
        # Generate 5 clusters
        centers = np.random.randn(5, 3) * 5
        points = []
        for center in centers:
            cluster_points = center + np.random.randn(n // 5, 3) * 0.5
            points.append(cluster_points)
        data[n] = np.vstack(points)
    return data


# =============================================================================
# JPDA Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def jpda_test_data():
    """
    Pre-computed JPDA test scenarios.

    Returns dict with different track/measurement configurations.
    """
    np.random.seed(42)

    scenarios = {}

    # Small scenario: 5 tracks, 10 measurements
    scenarios["small"] = {
        "n_tracks": 5,
        "n_meas": 10,
        "likelihood_matrix": np.random.rand(5, 10) * 0.1,
        "detection_prob": 0.9,
        "clutter_density": 1e-6,
    }

    # Medium scenario: 10 tracks, 20 measurements
    scenarios["medium"] = {
        "n_tracks": 10,
        "n_meas": 20,
        "likelihood_matrix": np.random.rand(10, 20) * 0.1,
        "detection_prob": 0.9,
        "clutter_density": 1e-6,
    }

    # Large scenario: 20 tracks, 50 measurements
    scenarios["large"] = {
        "n_tracks": 20,
        "n_meas": 50,
        "likelihood_matrix": np.random.rand(20, 50) * 0.1,
        "detection_prob": 0.9,
        "clutter_density": 1e-6,
    }

    return scenarios


# =============================================================================
# Benchmark Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark")
    config.addinivalue_line(
        "markers", "light: mark benchmark as part of light (PR) suite"
    )
    config.addinivalue_line(
        "markers", "full: mark benchmark as part of full (main) suite"
    )
