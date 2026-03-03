"""Comprehensive tests for coordinate rotations to improve coverage.

This module provides additional tests for Tier 3 coverage improvement of
coordinate_systems/rotations (67.8% -> ~75% target).
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.rotations import (
    rotx,
    roty,
    rotz,
    euler2rotmat,
    rotmat2euler,
    axisangle2rotmat,
    rotmat2axisangle,
    quat2rotmat,
    rotmat2quat,
    euler2quat,
    quat2euler,
    quat_multiply,
    quat_conjugate,
    quat_inverse,
    quat_rotate,
    slerp,
)


class TestBasicRotationMatrices:
    """Tests for basic rotation matrix generation."""
    
    def test_rotx_identity(self):
        """Test rotation about X axis with zero angle."""
        R = rotx(0.0)
        assert R.shape == (3, 3)
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_rotx_90_degrees(self):
        """Test 90 degree rotation about X axis."""
        R = rotx(np.pi / 2)
        assert R.shape == (3, 3)
        
        # X axis should be unchanged
        v = np.array([1, 0, 0])
        result = R @ v
        np.testing.assert_almost_equal(result, v)
    
    def test_roty_identity(self):
        """Test rotation about Y axis with zero angle."""
        R = roty(0.0)
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_roty_90_degrees(self):
        """Test 90 degree rotation about Y axis."""
        R = roty(np.pi / 2)
        
        # Y axis should be unchanged
        v = np.array([0, 1, 0])
        result = R @ v
        np.testing.assert_almost_equal(result, v)
    
    def test_rotz_identity(self):
        """Test rotation about Z axis with zero angle."""
        R = rotz(0.0)
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_rotz_90_degrees(self):
        """Test 90 degree rotation about Z axis."""
        R = rotz(np.pi / 2)
        
        # Z axis should be unchanged
        v = np.array([0, 0, 1])
        result = R @ v
        np.testing.assert_almost_equal(result, v)
    
    def test_rotation_matrices_orthogonality(self):
        """Test that rotation matrices are orthogonal."""
        angles = [0, np.pi/6, np.pi/4, np.pi/2, np.pi]
        
        for angle in angles:
            R = rotx(angle)
            # R @ R.T should equal identity
            np.testing.assert_almost_equal(R @ R.T, np.eye(3))
            
            R = roty(angle)
            np.testing.assert_almost_equal(R @ R.T, np.eye(3))
            
            R = rotz(angle)
            np.testing.assert_almost_equal(R @ R.T, np.eye(3))
    
    def test_rotation_determinant(self):
        """Test that rotation matrices have determinant 1."""
        angles = [0, np.pi/6, np.pi/4, np.pi/2]
        
        for angle in angles:
            for rot_func in [rotx, roty, rotz]:
                R = rot_func(angle)
                det = np.linalg.det(R)
                np.testing.assert_almost_equal(det, 1.0)


class TestEulerAngleConversions:
    """Tests for Euler angle conversions."""
    
    def test_euler2rotmat_identity(self):
        """Test Euler angles (0,0,0) produce identity."""
        R = euler2rotmat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_euler2rotmat_orthogonality(self):
        """Test Euler-generated matrices are orthogonal."""
        angle_sets = [(0.1, 0.2, 0.3), (np.pi/6, np.pi/4, np.pi/3)]
        
        for roll, pitch, yaw in angle_sets:
            R = euler2rotmat(np.array([yaw, pitch, roll]))
            np.testing.assert_almost_equal(R @ R.T, np.eye(3))
            det = np.linalg.det(R)
            np.testing.assert_almost_equal(det, 1.0)
    
    def test_rotmat2euler_identity(self):
        """Test identity matrix produces zero Euler angles."""
        roll, pitch, yaw = rotmat2euler(np.eye(3))
        np.testing.assert_almost_equal([roll, pitch, yaw], [0, 0, 0])
    
    def test_euler_roundtrip(self):
        """Test euler -> rotmat -> euler conversion."""
        test_angles = [
            (0.1, 0.2, 0.3),
            (np.pi/4, np.pi/6, np.pi/3),
            (0.5, 0.5, 0.5),
        ]
        
        for roll, pitch, yaw in test_angles:
            R = euler2rotmat(np.array([yaw, pitch, roll]))
            roll2, pitch2, yaw2 = rotmat2euler(R)
            np.testing.assert_almost_equal([yaw, pitch, roll], [roll2, pitch2, yaw2], decimal=5)


class TestAxisAngleConversions:
    """Tests for axis-angle representations."""
    
    def test_axisangle2rotmat_identity(self):
        """Test zero angle produces identity."""
        axis = np.array([1, 0, 0])
        R = axisangle2rotmat(axis, 0.0)
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_axisangle_orthogonality(self):
        """Test axis-angle matrices are orthogonal."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 4
        
        R = axisangle2rotmat(axis, angle)
        np.testing.assert_almost_equal(R @ R.T, np.eye(3))
        np.testing.assert_almost_equal(np.linalg.det(R), 1.0)
    
    def test_rotmat2axisangle_identity(self):
        """Test identity recovers zero angle."""
        axis, angle = rotmat2axisangle(np.eye(3))
        np.testing.assert_almost_equal(angle, 0.0)
    
    def test_axisangle_roundtrip(self):
        """Test axis-angle roundtrip conversion."""
        test_cases = [
            (np.array([1, 0, 0]), np.pi / 4),
            (np.array([0, 1, 0]), np.pi / 6),
            (np.array([1, 1, 1]) / np.sqrt(3), np.pi / 3),
        ]
        
        for axis, angle in test_cases:
            R = axisangle2rotmat(axis, angle)
            axis2, angle2 = rotmat2axisangle(R)
            
            np.testing.assert_almost_equal(angle, angle2, decimal=5)


class TestQuaternionConversions:
    """Tests for quaternion conversions."""
    
    def test_quat2rotmat_identity(self):
        """Test identity quaternion produces identity matrix."""
        q = np.array([1, 0, 0, 0])
        R = quat2rotmat(q)
        np.testing.assert_almost_equal(R, np.eye(3))
    
    def test_quat2rotmat_orthogonality(self):
        """Test quaternion-generated matrices are orthogonal."""
        q = np.array([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0])  # Normalized
        R = quat2rotmat(q)
        np.testing.assert_almost_equal(R @ R.T, np.eye(3))
    
    def test_rotmat2quat_identity(self):
        """Test identity matrix produces identity quaternion."""
        q = rotmat2quat(np.eye(3))
        # Should be [±1, 0, 0, 0]
        assert abs(q[0]) == pytest.approx(1.0)
        np.testing.assert_almost_equal(np.abs(q[1:]), [0, 0, 0])
    
    def test_quat_roundtrip(self):
        """Test quaternion roundtrip conversion."""
        q_original = np.array([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0])
        R = quat2rotmat(q_original)
        q_recovered = rotmat2quat(R)
        
        # Quaternions q and -q represent same rotation
        if np.dot(q_original, q_recovered) < 0:
            q_recovered = -q_recovered
        
        np.testing.assert_almost_equal(q_original, q_recovered, decimal=5)


class TestEulerQuaternionConversions:
    """Tests for conversions between Euler angles and quaternions."""
    
    def test_euler2quat_identity(self):
        """Test zero Euler angles produce identity quaternion."""
        q = euler2quat(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_almost_equal(np.abs(q[0]), 1.0)
        np.testing.assert_almost_equal(np.abs(q[1:]), [0, 0, 0])
    
    def test_quat2euler_identity(self):
        """Test identity quaternion produces zero Euler angles."""
        roll, pitch, yaw = quat2euler(np.array([1, 0, 0, 0]))
        np.testing.assert_almost_equal([roll, pitch, yaw], [0, 0, 0])
    
    def test_euler_quat_roundtrip(self):
        """Test Euler -> quat -> Euler conversion."""
        test_angles = [
            (0.1, 0.2, 0.3),
            (np.pi / 4, np.pi / 6, 0.5),
        ]
        
        for roll, pitch, yaw in test_angles:
            q = euler2quat(np.array([roll, pitch, yaw]))
            roll2, pitch2, yaw2 = quat2euler(q)
            np.testing.assert_almost_equal([roll, pitch, yaw], [roll2, pitch2, yaw2], decimal=5)


class TestQuaternionOperations:
    """Tests for quaternion arithmetic."""
    
    def test_quat_multiply_identity(self):
        """Test multiplication by identity quaternion."""
        q1 = np.array([1, 0, 0, 0])  # Identity
        q2 = np.array([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0])
        
        result = quat_multiply(q1, q2)
        
        # Normalize for comparison
        result = result / np.linalg.norm(result)
        q2_norm = q2 / np.linalg.norm(q2)
        
        np.testing.assert_almost_equal(result, q2_norm)
    
    def test_quat_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([1, 2, 3, 4])
        q_conj = quat_conjugate(q)
        
        assert q_conj[0] == q[0]
        np.testing.assert_almost_equal(q_conj[1:], -q[1:])
    
    def test_quat_inverse(self):
        """Test quaternion inverse."""
        q = np.array([1, 0.5, 0.3, 0.2])
        q = q / np.linalg.norm(q)  # Normalize
        
        q_inv = quat_inverse(q)
        
        # q * q_inv should give identity (up to scaling)
        result = quat_multiply(q, q_inv)
        result = result / np.linalg.norm(result)
        
        np.testing.assert_almost_equal(np.abs(result[0]), 1.0)
        np.testing.assert_almost_equal(np.abs(result[1:]), [0, 0, 0])
    
    def test_quat_rotate_identity(self):
        """Test rotating vector with identity quaternion."""
        q = np.array([1, 0, 0, 0])
        v = np.array([1, 2, 3])
        
        v_rotated = quat_rotate(q, v)
        np.testing.assert_almost_equal(v_rotated, v)
    
    def test_quat_rotate_vector_magnitude(self):
        """Test that quat_rotate preserves vector magnitude."""
        q = np.array([np.cos(np.pi/8), np.sin(np.pi/8), 0, 0])
        v = np.array([1, 2, 3])
        
        v_rotated = quat_rotate(q, v)
        
        mag_original = np.linalg.norm(v)
        mag_rotated = np.linalg.norm(v_rotated)
        
        np.testing.assert_almost_equal(mag_original, mag_rotated)


class TestSphericalLinearInterpolation:
    """Tests for SLERP (spherical linear interpolation)."""
    
    def test_slerp_endpoints(self):
        """Test SLERP at t=0 and t=1."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        
        # At t=0
        result = slerp(q1, q2, 0.0)
        assert np.linalg.norm(result - q1) < 0.1
        
        # At t=1
        result = slerp(q1, q2, 1.0)
        assert np.linalg.norm(result - q2) < 0.1
    
    def test_slerp_midpoint(self):
        """Test SLERP at midpoint."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        
        result = slerp(q1, q2, 0.5)
        
        # Result should be normalized
        norm = np.linalg.norm(result)
        np.testing.assert_almost_equal(norm, 1.0)


class TestRotationCompositions:
    """Tests for composing rotations."""
    
    def test_sequential_rotations(self):
        """Test composing sequential rotations."""
        # Rotate 90° about X, then 90° about Y
        Rx = rotx(np.pi / 2)
        Ry = roty(np.pi / 2)
        
        # Composition: Ry @ Rx
        R_composed = Ry @ Rx
        
        # Test on a vector
        v = np.array([1, 0, 0])
        
        # Apply rotations sequentially
        v1 = Rx @ v
        v2 = Ry @ v1
        
        # Apply composed rotation
        v_composed = R_composed @ v
        
        np.testing.assert_almost_equal(v2, v_composed)
    
    def test_euler_composition_vs_individual(self):
        """Test Euler angles composed vs individual rotations."""
        roll, pitch, yaw = 0.1, 0.2, 0.3
        
        R_euler = euler2rotmat(np.array([yaw, pitch, roll]))
        
        # Individual rotations (Z-Y-X order)
        Rz = rotz(yaw)
        Ry = roty(pitch)
        Rx = rotx(roll)
        
        R_individual = Rz @ Ry @ Rx
        
        np.testing.assert_almost_equal(R_euler, R_individual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
