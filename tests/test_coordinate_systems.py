"""
Tests for coordinate_systems module.

Tests cover:
- Spherical/polar/cylindrical coordinate conversions
- r-u-v (direction cosine) conversions
- Rotation matrices (rotx, roty, rotz)
- Euler angles <-> rotation matrix conversions
- Axis-angle <-> rotation matrix conversions
- Quaternion operations and conversions
- SLERP interpolation
- Rodrigues vector conversions
"""

import numpy as np
import pytest

from pytcl.coordinate_systems import (  # Spherical/polar conversions; Rotation operations
    axisangle2rotmat,
    cart2cyl,
    cart2pol,
    cart2ruv,
    cart2sphere,
    cyl2cart,
    dcm_rate,
    euler2quat,
    euler2rotmat,
    is_rotation_matrix,
    pol2cart,
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
    ruv2cart,
    slerp,
    sphere2cart,
)


class TestSphericalConversions:
    """Tests for Cartesian <-> spherical coordinate conversions."""

    def test_cart2sphere_basic(self):
        """Test basic Cartesian to spherical conversion."""
        # Point on x-axis
        r, az, el = cart2sphere([1, 0, 0], system_type="az-el")
        assert np.isclose(r, 1.0)
        assert np.isclose(az, 0.0)
        assert np.isclose(el, 0.0)

        # Point on y-axis
        r, az, el = cart2sphere([0, 1, 0], system_type="az-el")
        assert np.isclose(r, 1.0)
        assert np.isclose(az, np.pi / 2)
        assert np.isclose(el, 0.0)

        # Point on z-axis
        r, az, el = cart2sphere([0, 0, 1], system_type="az-el")
        assert np.isclose(r, 1.0)
        assert np.isclose(el, np.pi / 2)

    def test_cart2sphere_standard_convention(self):
        """Test standard physics convention (polar from +z)."""
        # Point on z-axis
        r, az, el = cart2sphere([0, 0, 1], system_type="standard")
        assert np.isclose(r, 1.0)
        assert np.isclose(el, 0.0)  # Polar angle = 0 at +z

        # Point on x-axis
        r, az, el = cart2sphere([1, 0, 0], system_type="standard")
        assert np.isclose(r, 1.0)
        assert np.isclose(el, np.pi / 2)  # Polar angle = 90 deg at xy-plane

    def test_sphere2cart_basic(self):
        """Test basic spherical to Cartesian conversion."""
        # Point at r=1, az=0, el=0 (on x-axis)
        cart = sphere2cart(1.0, 0.0, 0.0, system_type="az-el")
        np.testing.assert_allclose(cart, [1, 0, 0], atol=1e-10)

        # Point at r=1, az=90deg, el=0 (on y-axis)
        cart = sphere2cart(1.0, np.pi / 2, 0.0, system_type="az-el")
        np.testing.assert_allclose(cart, [0, 1, 0], atol=1e-10)

        # Point at r=1, el=90deg (on z-axis)
        cart = sphere2cart(1.0, 0.0, np.pi / 2, system_type="az-el")
        np.testing.assert_allclose(cart, [0, 0, 1], atol=1e-10)

    def test_spherical_roundtrip(self):
        """Test that cart->sphere->cart gives original point."""
        original = np.array([1.0, 2.0, 3.0])
        r, az, el = cart2sphere(original, system_type="az-el")
        recovered = sphere2cart(r, az, el, system_type="az-el")
        np.testing.assert_allclose(recovered, original, rtol=1e-10)

    def test_spherical_multiple_points(self):
        """Test conversion with multiple points."""
        points = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T  # Shape (3, 3)
        r, az, el = cart2sphere(points, system_type="az-el")
        assert r.shape == (3,)
        np.testing.assert_allclose(r, [1, 1, 1], atol=1e-10)


class TestPolarConversions:
    """Tests for Cartesian <-> polar coordinate conversions."""

    def test_cart2pol_basic(self):
        """Test basic Cartesian to polar conversion."""
        r, theta = cart2pol([1, 0])
        assert np.isclose(r, 1.0)
        assert np.isclose(theta, 0.0)

        r, theta = cart2pol([0, 1])
        assert np.isclose(r, 1.0)
        assert np.isclose(theta, np.pi / 2)

        r, theta = cart2pol([1, 1])
        assert np.isclose(r, np.sqrt(2))
        assert np.isclose(theta, np.pi / 4)

    def test_pol2cart_basic(self):
        """Test basic polar to Cartesian conversion."""
        cart = pol2cart(1.0, 0.0)
        np.testing.assert_allclose(cart, [1, 0], atol=1e-10)

        cart = pol2cart(1.0, np.pi / 2)
        np.testing.assert_allclose(cart, [0, 1], atol=1e-10)

    def test_polar_roundtrip(self):
        """Test that cart->pol->cart gives original point."""
        original = np.array([3.0, 4.0])
        r, theta = cart2pol(original)
        recovered = pol2cart(r, theta)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


class TestCylindricalConversions:
    """Tests for Cartesian <-> cylindrical coordinate conversions."""

    def test_cart2cyl_basic(self):
        """Test basic Cartesian to cylindrical conversion."""
        rho, phi, z = cart2cyl([1, 0, 5])
        assert np.isclose(rho, 1.0)
        assert np.isclose(phi, 0.0)
        assert np.isclose(z, 5.0)

    def test_cyl2cart_basic(self):
        """Test basic cylindrical to Cartesian conversion."""
        cart = cyl2cart(1.0, 0.0, 5.0)
        np.testing.assert_allclose(cart, [1, 0, 5], atol=1e-10)

    def test_cylindrical_roundtrip(self):
        """Test that cart->cyl->cart gives original point."""
        original = np.array([2.0, 3.0, 4.0])
        rho, phi, z = cart2cyl(original)
        recovered = cyl2cart(rho, phi, z)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


class TestRUVConversions:
    """Tests for r-u-v (direction cosine) conversions."""

    def test_cart2ruv_basic(self):
        """Test basic Cartesian to r-u-v conversion."""
        # Point on x-axis
        r, u, v = cart2ruv([10, 0, 0])
        assert np.isclose(r, 10.0)
        assert np.isclose(u, 1.0)
        assert np.isclose(v, 0.0)

    def test_ruv2cart_basic(self):
        """Test basic r-u-v to Cartesian conversion."""
        # u=1, v=0 means on x-axis (w=0)
        cart = ruv2cart(10.0, 1.0, 0.0)
        np.testing.assert_allclose(cart, [10, 0, 0], atol=1e-10)

    def test_ruv_roundtrip(self):
        """Test that cart->ruv->cart gives original point."""
        original = np.array([3.0, 4.0, 5.0])
        r, u, v = cart2ruv(original)
        recovered = ruv2cart(r, u, v)
        np.testing.assert_allclose(recovered, original, rtol=1e-10)


class TestBasicRotations:
    """Tests for basic rotation matrices rotx, roty, rotz."""

    def test_rotx_90_degrees(self):
        """Test 90 degree rotation about x-axis."""
        R = rotx(np.pi / 2)
        assert is_rotation_matrix(R)
        # y-axis maps to z-axis
        result = R @ np.array([0, 1, 0])
        np.testing.assert_allclose(result, [0, 0, 1], atol=1e-10)

    def test_roty_90_degrees(self):
        """Test 90 degree rotation about y-axis."""
        R = roty(np.pi / 2)
        assert is_rotation_matrix(R)
        # z-axis maps to x-axis
        result = R @ np.array([0, 0, 1])
        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-10)

    def test_rotz_90_degrees(self):
        """Test 90 degree rotation about z-axis."""
        R = rotz(np.pi / 2)
        assert is_rotation_matrix(R)
        # x-axis maps to y-axis
        result = R @ np.array([1, 0, 0])
        np.testing.assert_allclose(result, [0, 1, 0], atol=1e-10)

    def test_rotx_identity(self):
        """Test zero rotation gives identity."""
        R = rotx(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_roty_identity(self):
        """Test zero rotation gives identity."""
        R = roty(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotz_identity(self):
        """Test zero rotation gives identity."""
        R = rotz(0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)


class TestEulerAngles:
    """Tests for Euler angle conversions."""

    def test_euler2rotmat_identity(self):
        """Test zero angles give identity matrix."""
        R = euler2rotmat([0, 0, 0], "ZYX")
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_euler2rotmat_zyx(self):
        """Test ZYX (aerospace) convention."""
        yaw = np.pi / 4
        R = euler2rotmat([yaw, 0, 0], "ZYX")
        assert is_rotation_matrix(R)
        # Should be equivalent to rotz(yaw)
        np.testing.assert_allclose(R, rotz(yaw), atol=1e-10)

    def test_euler_roundtrip(self):
        """Test euler->rotmat->euler roundtrip."""
        angles = np.array([0.3, 0.2, 0.1])  # yaw, pitch, roll
        R = euler2rotmat(angles, "ZYX")
        recovered = rotmat2euler(R, "ZYX")
        np.testing.assert_allclose(recovered, angles, rtol=1e-10)

    def test_euler_xyz_sequence(self):
        """Test XYZ sequence produces valid rotation matrix."""
        angles = np.array([0.1, 0.2, 0.3])
        R = euler2rotmat(angles, "XYZ")
        # Verify it's a valid rotation matrix
        assert is_rotation_matrix(R)
        # Verify it equals Rx @ Ry @ Rz applied in sequence
        expected = rotx(angles[0]) @ roty(angles[1]) @ rotz(angles[2])
        np.testing.assert_allclose(R, expected, atol=1e-10)

    def test_invalid_sequence(self):
        """Test that invalid sequence raises error."""
        with pytest.raises(ValueError):
            euler2rotmat([0, 0, 0], "AB")


class TestAxisAngle:
    """Tests for axis-angle representation."""

    def test_axisangle_x_rotation(self):
        """Test axis-angle rotation about x-axis."""
        axis = np.array([1, 0, 0])
        angle = np.pi / 2
        R = axisangle2rotmat(axis, angle)
        np.testing.assert_allclose(R, rotx(angle), atol=1e-10)

    def test_axisangle_roundtrip(self):
        """Test axis-angle roundtrip."""
        axis = np.array([1, 1, 1]) / np.sqrt(3)
        angle = np.pi / 3
        R = axisangle2rotmat(axis, angle)
        axis_r, angle_r = rotmat2axisangle(R)
        np.testing.assert_allclose(np.abs(axis_r), np.abs(axis), atol=1e-10)
        assert np.isclose(angle_r, angle, atol=1e-10)

    def test_axisangle_identity(self):
        """Test zero angle gives identity."""
        axis = np.array([0, 0, 1])
        R = axisangle2rotmat(axis, 0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rotmat2axisangle_identity(self):
        """Test identity matrix gives zero rotation."""
        axis, angle = rotmat2axisangle(np.eye(3))
        assert np.isclose(angle, 0.0, atol=1e-10)


class TestQuaternions:
    """Tests for quaternion operations."""

    def test_identity_quaternion(self):
        """Test identity quaternion gives identity rotation."""
        q = np.array([1, 0, 0, 0])
        R = quat2rotmat(q)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_quat2rotmat_x_rotation(self):
        """Test quaternion for 90 deg rotation about x."""
        # q = cos(45) + sin(45)*i
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), np.sin(angle / 2), 0, 0])
        R = quat2rotmat(q)
        np.testing.assert_allclose(R, rotx(angle), atol=1e-10)

    def test_quat_roundtrip(self):
        """Test quaternion <-> rotation matrix roundtrip."""
        # Start with a rotation matrix
        R = euler2rotmat([0.3, 0.2, 0.1], "ZYX")
        q = rotmat2quat(R)
        R_recovered = quat2rotmat(q)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)

    def test_quat_multiply_identity(self):
        """Test multiplication with identity quaternion."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        identity = np.array([1, 0, 0, 0])
        result = quat_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-10)

    def test_quat_multiply_inverse(self):
        """Test q * q_inv = identity."""
        q = np.array([0.5, 0.5, 0.5, 0.5])
        q_inv = quat_inverse(q)
        result = quat_multiply(q, q_inv)
        # Should be identity (or close to it)
        np.testing.assert_allclose(np.abs(result[0]), 1.0, atol=1e-10)
        np.testing.assert_allclose(result[1:], [0, 0, 0], atol=1e-10)

    def test_quat_conjugate(self):
        """Test quaternion conjugate."""
        q = np.array([1, 2, 3, 4])
        q_conj = quat_conjugate(q)
        np.testing.assert_allclose(q_conj, [1, -2, -3, -4])

    def test_quat_rotate_vector(self):
        """Test quaternion rotation of a vector."""
        # 90 degree rotation about z-axis
        angle = np.pi / 2
        q = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        v = np.array([1, 0, 0])
        v_rotated = quat_rotate(q, v)
        np.testing.assert_allclose(v_rotated, [0, 1, 0], atol=1e-10)


class TestEulerQuatConversion:
    """Tests for Euler <-> quaternion conversions."""

    def test_euler2quat_identity(self):
        """Test zero Euler angles give identity quaternion."""
        q = euler2quat([0, 0, 0], "ZYX")
        np.testing.assert_allclose(q, [1, 0, 0, 0], atol=1e-10)

    def test_euler_quat_roundtrip(self):
        """Test euler->quat->euler roundtrip."""
        angles = np.array([0.3, 0.2, 0.1])
        q = euler2quat(angles, "ZYX")
        recovered = quat2euler(q, "ZYX")
        np.testing.assert_allclose(recovered, angles, rtol=1e-10)


class TestSlerp:
    """Tests for spherical linear interpolation."""

    def test_slerp_endpoints(self):
        """Test SLERP at t=0 and t=1."""
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])

        result_0 = slerp(q1, q2, 0)
        result_1 = slerp(q1, q2, 1)

        np.testing.assert_allclose(result_0, q1, atol=1e-10)
        np.testing.assert_allclose(result_1, q2, atol=1e-10)

    def test_slerp_midpoint(self):
        """Test SLERP at t=0.5."""
        # Identity to 90 deg rotation about z
        q1 = np.array([1, 0, 0, 0])
        angle = np.pi / 2
        q2 = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        result = slerp(q1, q2, 0.5)

        # Should be 45 deg rotation about z
        expected_angle = angle / 2
        expected = np.array(
            [np.cos(expected_angle / 2), 0, 0, np.sin(expected_angle / 2)]
        )
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestRodrigues:
    """Tests for Rodrigues vector representation."""

    def test_rodrigues_identity(self):
        """Test zero Rodrigues vector gives identity."""
        rvec = np.array([0, 0, 0])
        R = rodrigues2rotmat(rvec)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_rodrigues_x_rotation(self):
        """Test Rodrigues vector for x-axis rotation."""
        angle = np.pi / 3
        rvec = np.array([angle, 0, 0])  # angle * x-axis
        R = rodrigues2rotmat(rvec)
        np.testing.assert_allclose(R, rotx(angle), atol=1e-10)

    def test_rodrigues_roundtrip(self):
        """Test rodrigues->rotmat->rodrigues roundtrip."""
        rvec = np.array([0.3, 0.2, 0.1])
        R = rodrigues2rotmat(rvec)
        rvec_recovered = rotmat2rodrigues(R)
        np.testing.assert_allclose(rvec_recovered, rvec, rtol=1e-10)


class TestDCMRate:
    """Tests for direction cosine matrix time derivative."""

    def test_dcm_rate_zero_omega(self):
        """Test zero angular velocity gives zero derivative."""
        R = np.eye(3)
        omega = np.array([0, 0, 0])
        R_dot = dcm_rate(R, omega)
        np.testing.assert_allclose(R_dot, np.zeros((3, 3)), atol=1e-10)

    def test_dcm_rate_shape(self):
        """Test output shape is correct."""
        R = euler2rotmat([0.1, 0.2, 0.3], "ZYX")
        omega = np.array([0.1, 0.2, 0.3])
        R_dot = dcm_rate(R, omega)
        assert R_dot.shape == (3, 3)


class TestIsRotationMatrix:
    """Tests for rotation matrix validation."""

    def test_identity_is_rotation(self):
        """Test identity is a valid rotation matrix."""
        assert is_rotation_matrix(np.eye(3))

    def test_rotx_is_rotation(self):
        """Test rotx output is a valid rotation matrix."""
        assert is_rotation_matrix(rotx(0.5))

    def test_invalid_shape(self):
        """Test non-3x3 matrix is not a rotation matrix."""
        assert not is_rotation_matrix(np.eye(4))
        assert not is_rotation_matrix(np.eye(2))

    def test_non_orthogonal_not_rotation(self):
        """Test non-orthogonal matrix is not a rotation matrix."""
        M = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        assert not is_rotation_matrix(M)

    def test_reflection_not_rotation(self):
        """Test reflection (det=-1) is not a rotation matrix."""
        # Reflection about xy-plane
        M = np.diag([1, 1, -1])
        assert not is_rotation_matrix(M)


class TestRandomRotations:
    """Property-based tests using random rotations."""

    @pytest.fixture
    def random_angles(self):
        """Generate random Euler angles."""
        np.random.seed(42)
        return np.random.uniform(-np.pi, np.pi, 3)

    def test_rotation_orthogonality(self, random_angles):
        """Test that rotation matrices are orthogonal."""
        R = euler2rotmat(random_angles, "ZYX")
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)

    def test_rotation_determinant(self, random_angles):
        """Test that rotation matrices have det=1."""
        R = euler2rotmat(random_angles, "ZYX")
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_rotation_preserves_norm(self, random_angles):
        """Test that rotations preserve vector norms."""
        R = euler2rotmat(random_angles, "ZYX")
        v = np.array([1, 2, 3])
        v_rotated = R @ v
        assert np.isclose(np.linalg.norm(v_rotated), np.linalg.norm(v), atol=1e-10)
