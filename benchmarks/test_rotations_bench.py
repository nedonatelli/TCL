"""
Benchmarks for rotation operations.

These are light benchmarks that run on every PR to catch performance regressions
in the Numba-optimized rotation computations.
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.rotations import (
    euler2rotmat,
    quat2rotmat,
    quat_multiply,
    quat_rotate,
    rotmat2euler,
    rotmat2quat,
    rotx,
    roty,
    rotz,
)


class TestBasicRotationBenchmarks:
    """Benchmark basic rotation matrix construction."""

    @pytest.mark.light
    def test_rotx(self, benchmark):
        """Benchmark X-axis rotation matrix."""
        angle = 0.5

        # Warm up Numba JIT
        _ = rotx(angle)

        result = benchmark(rotx, angle)

        assert result.shape == (3, 3)

    @pytest.mark.light
    def test_roty(self, benchmark):
        """Benchmark Y-axis rotation matrix."""
        angle = 0.5

        # Warm up Numba JIT
        _ = roty(angle)

        result = benchmark(roty, angle)

        assert result.shape == (3, 3)

    @pytest.mark.light
    def test_rotz(self, benchmark):
        """Benchmark Z-axis rotation matrix."""
        angle = 0.5

        # Warm up Numba JIT
        _ = rotz(angle)

        result = benchmark(rotz, angle)

        assert result.shape == (3, 3)


class TestEulerRotationBenchmarks:
    """Benchmark Euler angle conversions."""

    @pytest.mark.light
    def test_euler2rotmat_single(self, benchmark, rotation_test_data):
        """Benchmark single Euler to rotation matrix conversion."""
        euler = rotation_test_data["single_euler"]

        # Warm up
        _ = euler2rotmat(euler[0], euler[1], euler[2])

        result = benchmark(euler2rotmat, euler[0], euler[1], euler[2])

        assert result.shape == (3, 3)

    @pytest.mark.light
    def test_rotmat2euler_single(self, benchmark, rotation_test_data):
        """Benchmark single rotation matrix to Euler conversion."""
        rotmat = rotation_test_data["single_rotmat"]

        result = benchmark(rotmat2euler, rotmat)

        assert len(result) == 3

    @pytest.mark.full
    def test_euler2rotmat_batch_1000(self, benchmark, rotation_test_data):
        """Benchmark 1000 Euler to rotation matrix conversions."""
        eulers = rotation_test_data["euler_angles"]

        def convert_all():
            results = []
            for e in eulers:
                results.append(euler2rotmat(e[0], e[1], e[2]))
            return results

        # Warm up
        _ = euler2rotmat(eulers[0, 0], eulers[0, 1], eulers[0, 2])

        result = benchmark(convert_all)

        assert len(result) == 1000


class TestQuaternionBenchmarks:
    """Benchmark quaternion operations."""

    @pytest.mark.light
    def test_quat2rotmat_single(self, benchmark, rotation_test_data):
        """Benchmark single quaternion to rotation matrix."""
        quat = rotation_test_data["single_quat"]

        result = benchmark(quat2rotmat, quat)

        assert result.shape == (3, 3)

    @pytest.mark.light
    def test_rotmat2quat_single(self, benchmark, rotation_test_data):
        """Benchmark single rotation matrix to quaternion."""
        rotmat = rotation_test_data["single_rotmat"]

        result = benchmark(rotmat2quat, rotmat)

        assert result.shape == (4,)

    @pytest.mark.light
    def test_quat_multiply(self, benchmark, rotation_test_data):
        """Benchmark quaternion multiplication."""
        quats = rotation_test_data["quaternions"]
        q1 = quats[0]
        q2 = quats[1]

        result = benchmark(quat_multiply, q1, q2)

        assert result.shape == (4,)

    @pytest.mark.light
    def test_quat_rotate(self, benchmark, rotation_test_data):
        """Benchmark quaternion vector rotation."""
        quat = rotation_test_data["single_quat"]
        vector = np.array([1.0, 0.0, 0.0])

        result = benchmark(quat_rotate, quat, vector)

        assert result.shape == (3,)


class TestBatchRotationBenchmarks:
    """Benchmark batch rotation operations."""

    @pytest.mark.full
    def test_quat_multiply_batch_1000(self, benchmark, rotation_test_data):
        """Benchmark 1000 quaternion multiplications."""
        quats = rotation_test_data["quaternions"]

        def multiply_chain():
            result = quats[0].copy()
            for q in quats[1:]:
                result = quat_multiply(result, q)
            return result

        result = benchmark(multiply_chain)

        assert result.shape == (4,)

    @pytest.mark.full
    def test_quat_rotate_batch_1000(self, benchmark, rotation_test_data):
        """Benchmark 1000 quaternion vector rotations."""
        quat = rotation_test_data["single_quat"]
        np.random.seed(42)
        vectors = np.random.randn(1000, 3)

        def rotate_all():
            results = []
            for v in vectors:
                results.append(quat_rotate(quat, v))
            return results

        result = benchmark(rotate_all)

        assert len(result) == 1000
