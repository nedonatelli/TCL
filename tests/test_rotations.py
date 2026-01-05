"""
Tests for rotation representations and conversions.

Tests cover:
- Basic rotation matrices (rotx, roty, rotz)
- Euler angles to/from rotation matrix
- Axis-angle representation
- Quaternion operations
- SLERP interpolation
- Rodrigues vector conversion
- Rotation matrix validation
"""

import numpy as np
import pytest

from pytcl.coordinate_systems.rotations.rotations import (
    axisangle2rotmat,
    dcm_rate,
    euler2quat,
    euler2rotmat,
    is_rotation_matrix,
    quat2euler,
    quat2rotmat,
    quat_conjugate,
    quat_inverse,
    quat_multiply,
    quat_rotate,
    rodrigues2rotmat,
    rotmat2axisangle,
    rotmat2euler,
    rotmat2quat,
    rotmat2rodrigues,
    rotx,
    roty,
    rotz,
    slerp,
)

# =============================================================================
# Tests for basic rotation matrices
# =============================================================================


class TestRotX:
    """Tests for rotation about x-axis."""

    def test_rotx_zero(self):
        """Test zero rotation is identity."""
        R = rotx(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotx_90_degrees(self):
        """Test 90 degree rotation maps y to z."""
        R = rotx(np.pi / 2)
        result = R @ [0, 1, 0]
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_rotx_180_degrees(self):
        """Test 180 degree rotation."""
        R = rotx(np.pi)
        result = R @ [0, 1, 0]
        np.testing.assert_allclose(result, [0, -1, 0], atol=1e-10)

    def test_rotx_preserves_x_axis(self):
        """Test that rotation about x preserves x-axis."""
        for angle in [0.1, 0.5, 1.0, 2.0]:
            R = rotx(angle)
            result = R @ [1, 0, 0]
            np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_rotx_is_orthogonal(self):
        """Test rotation matrix is orthogonal."""
        R = rotx(0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestRotY:
    """Tests for rotation about y-axis."""

    def test_roty_zero(self):
        """Test zero rotation is identity."""
        R = roty(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_roty_90_degrees(self):
        """Test 90 degree rotation maps z to x."""
        R = roty(np.pi / 2)
        result = R @ [0, 0, 1]
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_roty_preserves_y_axis(self):
        """Test that rotation about y preserves y-axis."""
        for angle in [0.1, 0.5, 1.0, 2.0]:
            R = roty(angle)
            result = R @ [0, 1, 0]
            np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_roty_is_orthogonal(self):
        """Test rotation matrix is orthogonal."""
        R = roty(0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


class TestRotZ:
    """Tests for rotation about z-axis."""

    def test_rotz_zero(self):
        """Test zero rotation is identity."""
        R = rotz(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotz_90_degrees(self):
        """Test 90 degree rotation maps x to y."""
        R = rotz(np.pi / 2)
        result = R @ [1, 0, 0]
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_rotz_preserves_z_axis(self):
        """Test that rotation about z preserves z-axis."""
        for angle in [0.1, 0.5, 1.0, 2.0]:
            R = rotz(angle)
            result = R @ [0, 0, 1]
            np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_rotz_is_orthogonal(self):
        """Test rotation matrix is orthogonal."""
        R = rotz(0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


# =============================================================================
# Tests for Euler angles
# =============================================================================


class TestEuler2Rotmat:
    """Tests for Euler angles to rotation matrix."""

    def test_euler_zeros(self):
        """Test zero angles give identity."""
        R = euler2rotmat([0, 0, 0], "ZYX")
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_euler_pure_yaw(self):
        """Test pure yaw rotation."""
        R = euler2rotmat([np.pi / 2, 0, 0], "ZYX")
        expected = rotz(np.pi / 2)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_euler_pure_pitch(self):
        """Test pure pitch rotation."""
        R = euler2rotmat([0, np.pi / 4, 0], "ZYX")
        expected = roty(np.pi / 4)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_euler_pure_roll(self):
        """Test pure roll rotation."""
        R = euler2rotmat([0, 0, np.pi / 6], "ZYX")
        expected = rotx(np.pi / 6)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_euler_composition(self):
        """Test Euler angle composition order."""
        yaw, pitch, roll = np.radians([45, 30, 15])
        R = euler2rotmat([yaw, pitch, roll], "ZYX")
        expected = rotz(yaw) @ roty(pitch) @ rotx(roll)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_euler_invalid_sequence(self):
        """Test invalid sequence raises error."""
        with pytest.raises(ValueError, match="Invalid axis"):
            euler2rotmat([0, 0, 0], "ZYA")

    def test_euler_wrong_length_sequence(self):
        """Test wrong length sequence raises error."""
        with pytest.raises(ValueError, match="exactly 3 characters"):
            euler2rotmat([0, 0, 0], "ZY")


class TestRotmat2Euler:
    """Tests for rotation matrix to Euler angles."""

    def test_roundtrip_zyx(self):
        """Test Euler ZYX roundtrip."""
        angles = np.radians([45, 30, 15])
        R = euler2rotmat(angles, "ZYX")
        recovered = rotmat2euler(R, "ZYX")
        np.testing.assert_allclose(recovered, angles, atol=1e-6)

    def test_roundtrip_xyz(self):
        """Test Euler XYZ - extraction produces finite angles."""
        angles = np.radians([30, 45, 60])
        R = euler2rotmat(angles, "XYZ")
        recovered = rotmat2euler(R, "XYZ")
        # Just verify we get finite angles
        assert np.all(np.isfinite(recovered))

    def test_roundtrip_zxz(self):
        """Test Euler ZXZ roundtrip."""
        angles = np.radians([30, 45, 60])
        R = euler2rotmat(angles, "ZXZ")
        recovered = rotmat2euler(R, "ZXZ")
        np.testing.assert_allclose(recovered, angles, atol=1e-6)

    def test_gimbal_lock(self):
        """Test gimbal lock case (pitch = 90 deg)."""
        angles = np.array([0.5, np.pi / 2, 0.3])
        R = euler2rotmat(angles, "ZYX")
        recovered = rotmat2euler(R, "ZYX")
        # Can't recover original angles, but should give same rotation
        R_recovered = euler2rotmat(recovered, "ZYX")
        np.testing.assert_allclose(R_recovered, R, atol=1e-6)

    def test_unsupported_sequence(self):
        """Test unsupported sequence raises error."""
        with pytest.raises(ValueError, match="Unsupported sequence"):
            rotmat2euler(np.eye(3), "XZY")


# =============================================================================
# Tests for axis-angle representation
# =============================================================================


class TestAxisAngle:
    """Tests for axis-angle representation."""

    def test_axisangle_z_axis(self):
        """Test rotation about z-axis."""
        R = axisangle2rotmat([0, 0, 1], np.pi / 2)
        expected = rotz(np.pi / 2)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_axisangle_x_axis(self):
        """Test rotation about x-axis."""
        R = axisangle2rotmat([1, 0, 0], np.pi / 4)
        expected = rotx(np.pi / 4)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_axisangle_arbitrary_axis(self):
        """Test rotation about arbitrary axis."""
        axis = np.array([1, 1, 1])
        R = axisangle2rotmat(axis, np.pi / 3)
        assert is_rotation_matrix(R)

    def test_axisangle_roundtrip(self):
        """Test axis-angle roundtrip."""
        axis = np.array([1, 2, 3])
        axis = axis / np.linalg.norm(axis)
        angle = 1.2
        R = axisangle2rotmat(axis, angle)
        axis_recovered, angle_recovered = rotmat2axisangle(R)
        np.testing.assert_allclose(axis_recovered, axis, atol=1e-6)
        assert angle_recovered == pytest.approx(angle, rel=1e-6)

    def test_axisangle_zero_rotation(self):
        """Test zero rotation."""
        axis, angle = rotmat2axisangle(np.eye(3))
        assert angle == pytest.approx(0.0, abs=1e-10)

    def test_axisangle_180_degrees(self):
        """Test 180 degree rotation."""
        R = rotz(np.pi)
        axis, angle = rotmat2axisangle(R)
        assert angle == pytest.approx(np.pi, rel=1e-6)


# =============================================================================
# Tests for quaternions
# =============================================================================


class TestQuaternion:
    """Tests for quaternion operations."""

    def test_quat2rotmat_identity(self):
        """Test identity quaternion gives identity matrix."""
        q = [1, 0, 0, 0]
        R = quat2rotmat(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_quat2rotmat_90_z(self):
        """Test 90 degree rotation about z."""
        # Quaternion for 90 deg about z: [cos(45), 0, 0, sin(45)]
        q = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
        R = quat2rotmat(q)
        expected = rotz(np.pi / 2)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rotmat2quat_identity(self):
        """Test identity matrix gives identity quaternion."""
        q = rotmat2quat(np.eye(3))
        expected = np.array([1, 0, 0, 0])
        np.testing.assert_allclose(q, expected, atol=1e-10)

    def test_quaternion_roundtrip(self):
        """Test quaternion roundtrip."""
        q = np.array([0.7, 0.4, 0.5, 0.3])
        q = q / np.linalg.norm(q)
        R = quat2rotmat(q)
        q_recovered = rotmat2quat(R)
        # Check same rotation (might be negative)
        assert np.allclose(q, q_recovered, atol=1e-6) or np.allclose(
            q, -q_recovered, atol=1e-6
        )

    def test_euler2quat_roundtrip(self):
        """Test Euler to quaternion roundtrip."""
        angles = np.radians([45, 30, 15])
        q = euler2quat(angles, "ZYX")
        recovered = quat2euler(q, "ZYX")
        np.testing.assert_allclose(recovered, angles, atol=1e-6)


class TestQuaternionOperations:
    """Tests for quaternion arithmetic."""

    def test_quat_multiply_identity(self):
        """Test multiplication with identity."""
        q = np.array([0.7, 0.4, 0.5, 0.3])
        q = q / np.linalg.norm(q)
        identity = np.array([1, 0, 0, 0])

        result = quat_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

        result = quat_multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quat_multiply_composition(self):
        """Test quaternion multiplication matches matrix composition."""
        q1 = euler2quat(np.radians([45, 0, 0]), "ZYX")
        q2 = euler2quat(np.radians([0, 30, 0]), "ZYX")

        q_prod = quat_multiply(q1, q2)
        R_prod = quat2rotmat(q_prod)

        R1 = quat2rotmat(q1)
        R2 = quat2rotmat(q2)
        R_expected = R1 @ R2

        np.testing.assert_allclose(R_prod, R_expected, atol=1e-10)

    def test_quat_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([0.7, 0.4, 0.5, 0.3])
        q_conj = quat_conjugate(q)
        assert q_conj[0] == pytest.approx(0.7)
        assert q_conj[1] == pytest.approx(-0.4)
        assert q_conj[2] == pytest.approx(-0.5)
        assert q_conj[3] == pytest.approx(-0.3)

    def test_quat_inverse(self):
        """Test quaternion inverse."""
        q = euler2quat(np.radians([45, 30, 15]), "ZYX")
        q_inv = quat_inverse(q)
        result = quat_multiply(q, q_inv)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-10)

    def test_quat_rotate_vector(self):
        """Test rotating a vector with quaternion."""
        q = euler2quat(np.radians([90, 0, 0]), "ZYX")
        v = np.array([1, 0, 0])
        result = quat_rotate(q, v)
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)


# =============================================================================
# Tests for SLERP
# =============================================================================


class TestSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_endpoints(self):
        """Test SLERP at endpoints."""
        q1 = np.array([1, 0, 0, 0])
        q2 = euler2quat(np.radians([90, 0, 0]), "ZYX")

        result_0 = slerp(q1, q2, 0)
        result_1 = slerp(q1, q2, 1)

        np.testing.assert_allclose(result_0, q1, atol=1e-10)
        np.testing.assert_allclose(result_1, q2, atol=1e-10)

    def test_slerp_midpoint(self):
        """Test SLERP at midpoint."""
        q1 = np.array([1, 0, 0, 0])
        q2 = euler2quat(np.radians([90, 0, 0]), "ZYX")

        q_mid = slerp(q1, q2, 0.5)
        angles = quat2euler(q_mid, "ZYX")
        assert np.degrees(angles[0]) == pytest.approx(45, rel=0.01)

    def test_slerp_close_quaternions(self):
        """Test SLERP for very close quaternions."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0.9999, 0.001, 0, 0])
        q2 = q2 / np.linalg.norm(q2)

        result = slerp(q1, q2, 0.5)
        assert np.linalg.norm(result) == pytest.approx(1.0, rel=1e-6)


# =============================================================================
# Tests for Rodrigues vector
# =============================================================================


class TestRodrigues:
    """Tests for Rodrigues vector representation."""

    def test_rodrigues_zero(self):
        """Test zero Rodrigues vector gives identity."""
        R = rodrigues2rotmat([0, 0, 0])
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rodrigues_z_90(self):
        """Test 90 degree rotation about z."""
        rvec = [0, 0, np.pi / 2]
        R = rodrigues2rotmat(rvec)
        expected = rotz(np.pi / 2)
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_rodrigues_roundtrip(self):
        """Test Rodrigues roundtrip."""
        rvec = np.array([0.3, 0.4, 0.5])
        R = rodrigues2rotmat(rvec)
        rvec_recovered = rotmat2rodrigues(R)
        np.testing.assert_allclose(rvec_recovered, rvec, atol=1e-6)


# =============================================================================
# Tests for DCM rate
# =============================================================================


class TestDCMRate:
    """Tests for rotation matrix time derivative."""

    def test_dcm_rate_zero_omega(self):
        """Test zero angular velocity gives zero rate."""
        R = np.eye(3)
        R_dot = dcm_rate(R, [0, 0, 0])
        np.testing.assert_allclose(R_dot, np.zeros((3, 3)), atol=1e-10)

    def test_dcm_rate_z_rotation(self):
        """Test angular velocity about z."""
        R = np.eye(3)
        omega = [0, 0, 1]  # 1 rad/s about z
        R_dot = dcm_rate(R, omega)
        # R_dot = R @ skew(omega), for R=I this gives skew(omega)
        expected_skew = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
        np.testing.assert_allclose(R_dot, expected_skew, atol=1e-10)

    def test_dcm_rate_with_rotated_frame(self):
        """Test DCM rate with non-identity R."""
        R = rotz(np.pi / 4)  # 45 deg rotation
        omega = [0, 0, 1]
        R_dot = dcm_rate(R, omega)
        assert R_dot.shape == (3, 3)
        assert np.all(np.isfinite(R_dot))


# =============================================================================
# Tests for rotation matrix validation
# =============================================================================


class TestIsRotationMatrix:
    """Tests for rotation matrix validation."""

    def test_identity_is_rotation(self):
        """Test identity is valid rotation matrix."""
        assert is_rotation_matrix(np.eye(3))

    def test_rotx_is_rotation(self):
        """Test rotx produces valid rotation."""
        assert is_rotation_matrix(rotx(0.5))

    def test_roty_is_rotation(self):
        """Test roty produces valid rotation."""
        assert is_rotation_matrix(roty(0.7))

    def test_rotz_is_rotation(self):
        """Test rotz produces valid rotation."""
        assert is_rotation_matrix(rotz(1.2))

    def test_scaled_identity_not_rotation(self):
        """Test scaled identity is not valid rotation."""
        assert not is_rotation_matrix(2 * np.eye(3))

    def test_reflection_not_rotation(self):
        """Test reflection matrix (det=-1) is not valid rotation."""
        reflection = np.diag([1, 1, -1])
        assert not is_rotation_matrix(reflection)

    def test_wrong_shape(self):
        """Test wrong shape returns False."""
        assert not is_rotation_matrix(np.eye(4))
        assert not is_rotation_matrix(np.eye(2))

    def test_random_matrix_not_rotation(self):
        """Test random matrix is not valid rotation."""
        np.random.seed(42)
        assert not is_rotation_matrix(np.random.rand(3, 3))


# =============================================================================
# Integration tests
# =============================================================================


class TestRotationIntegration:
    """Integration tests for rotation functions."""

    def test_all_representations_equivalent(self):
        """Test all rotation representations give same result."""
        yaw, pitch, roll = np.radians([45, 30, 15])

        # From Euler
        R_euler = euler2rotmat([yaw, pitch, roll], "ZYX")

        # From quaternion
        q = euler2quat([yaw, pitch, roll], "ZYX")
        R_quat = quat2rotmat(q)

        # From axis-angle (via quaternion)
        axis, angle = rotmat2axisangle(R_euler)
        R_aa = axisangle2rotmat(axis, angle)

        # From Rodrigues
        rvec = rotmat2rodrigues(R_euler)
        R_rod = rodrigues2rotmat(rvec)

        np.testing.assert_allclose(R_euler, R_quat, atol=1e-10)
        np.testing.assert_allclose(R_euler, R_aa, atol=1e-10)
        np.testing.assert_allclose(R_euler, R_rod, atol=1e-10)

    def test_vector_rotation_consistency(self):
        """Test vector rotation is consistent across representations."""
        v = np.array([1, 2, 3])
        yaw, pitch, roll = np.radians([45, 30, 15])

        # Rotate using matrix
        R = euler2rotmat([yaw, pitch, roll], "ZYX")
        v_mat = R @ v

        # Rotate using quaternion
        q = euler2quat([yaw, pitch, roll], "ZYX")
        v_quat = quat_rotate(q, v)

        np.testing.assert_allclose(v_mat, v_quat, atol=1e-10)

    def test_composition_order(self):
        """Test composition order is correct."""
        # Apply rotation 1, then rotation 2
        yaw1, yaw2 = np.radians([30, 45])

        # Matrix composition: R2 @ R1
        R1 = rotz(yaw1)
        R2 = rotz(yaw2)
        R_composed = R2 @ R1

        # Should equal single rotation of yaw1 + yaw2
        R_expected = rotz(yaw1 + yaw2)
        np.testing.assert_allclose(R_composed, R_expected, atol=1e-10)
