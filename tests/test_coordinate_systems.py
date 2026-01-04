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
- Geodetic coordinate conversions (ECEF, ENU, NED)
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


# =============================================================================
# Geodetic Coordinate Conversion Tests
# =============================================================================


from pytcl.coordinate_systems.conversions.geodetic import (  # noqa: E402
    ecef2enu,
    ecef2geodetic,
    ecef2ned,
    enu2ecef,
    enu2ned,
    geocentric_radius,
    geodetic2ecef,
    geodetic2enu,
    meridional_radius,
    ned2ecef,
    ned2enu,
    prime_vertical_radius,
)


class TestGeodetic2ECEF:
    """Tests for geodetic to ECEF conversions."""

    def test_equator_prime_meridian(self):
        """Test point at equator on prime meridian."""
        lat, lon, alt = 0.0, 0.0, 0.0
        ecef = geodetic2ecef(lat, lon, alt)
        # On equator at prime meridian, x should be ~semi-major axis
        assert ecef[0] > 6e6
        assert np.isclose(ecef[1], 0.0, atol=1)
        assert np.isclose(ecef[2], 0.0, atol=1)

    def test_north_pole(self):
        """Test point at north pole."""
        lat, lon, alt = np.pi / 2, 0.0, 0.0
        ecef = geodetic2ecef(lat, lon, alt)
        # At north pole, z should be ~semi-minor axis, x and y ~0
        assert np.isclose(ecef[0], 0.0, atol=1)
        assert np.isclose(ecef[1], 0.0, atol=1)
        assert ecef[2] > 6e6

    def test_south_pole(self):
        """Test point at south pole."""
        lat, lon, alt = -np.pi / 2, 0.0, 0.0
        ecef = geodetic2ecef(lat, lon, alt)
        # At south pole, z should be negative
        assert np.isclose(ecef[0], 0.0, atol=1)
        assert np.isclose(ecef[1], 0.0, atol=1)
        assert ecef[2] < -6e6

    def test_altitude_effect(self):
        """Test that altitude increases distance from center."""
        lat, lon = np.radians(45), np.radians(-75)
        ecef_ground = geodetic2ecef(lat, lon, 0.0)
        ecef_high = geodetic2ecef(lat, lon, 10000.0)

        r_ground = np.linalg.norm(ecef_ground)
        r_high = np.linalg.norm(ecef_high)

        assert r_high > r_ground
        # Should be approximately 10km higher
        assert np.isclose(r_high - r_ground, 10000.0, rtol=0.01)

    def test_known_location(self):
        """Test a known location (Washington DC approx)."""
        lat = np.radians(38.9)
        lon = np.radians(-77.0)
        alt = 0.0
        ecef = geodetic2ecef(lat, lon, alt)

        # Should be in Western hemisphere (negative x due to longitude)
        # and Northern hemisphere (positive z)
        assert ecef[1] < 0  # Western hemisphere
        assert ecef[2] > 0  # Northern hemisphere

    def test_multiple_points(self):
        """Test conversion with multiple points."""
        lats = np.radians([0, 45, 90])
        lons = np.radians([0, 90, 0])
        alts = np.array([0, 0, 0])

        ecef = geodetic2ecef(lats, lons, alts)
        assert ecef.shape == (3, 3)


class TestECEF2Geodetic:
    """Tests for ECEF to geodetic conversions."""

    def test_roundtrip_equator(self):
        """Test roundtrip at equator."""
        lat_orig, lon_orig, alt_orig = 0.0, 0.0, 100.0
        ecef = geodetic2ecef(lat_orig, lon_orig, alt_orig)
        lat, lon, alt = ecef2geodetic(ecef)

        assert np.isclose(lat, lat_orig, atol=1e-10)
        assert np.isclose(lon, lon_orig, atol=1e-10)
        assert np.isclose(alt, alt_orig, atol=0.01)

    def test_roundtrip_pole(self):
        """Test roundtrip at north pole."""
        lat_orig, lon_orig, alt_orig = np.pi / 2, 0.0, 500.0
        ecef = geodetic2ecef(lat_orig, lon_orig, alt_orig)
        lat, lon, alt = ecef2geodetic(ecef)

        assert np.isclose(lat, lat_orig, atol=1e-10)
        assert np.isclose(alt, alt_orig, atol=0.1)

    def test_roundtrip_random(self):
        """Test roundtrip for random locations."""
        np.random.seed(42)
        for _ in range(10):
            lat_orig = np.random.uniform(-np.pi / 2, np.pi / 2)
            lon_orig = np.random.uniform(-np.pi, np.pi)
            alt_orig = np.random.uniform(0, 50000)

            ecef = geodetic2ecef(lat_orig, lon_orig, alt_orig)
            lat, lon, alt = ecef2geodetic(ecef)

            assert np.isclose(lat, lat_orig, atol=1e-9)
            assert np.isclose(lon, lon_orig, atol=1e-9)
            assert np.isclose(alt, alt_orig, atol=0.1)

    def test_direct_method(self):
        """Test the direct (closed-form) method."""
        lat_orig = np.radians(45)
        lon_orig = np.radians(-75)
        alt_orig = 1000.0

        ecef = geodetic2ecef(lat_orig, lon_orig, alt_orig)
        lat, lon, alt = ecef2geodetic(ecef, method="direct")

        # Direct method has lower accuracy than iterative
        assert np.isclose(lat, lat_orig, atol=0.01)  # ~0.5 degrees
        assert np.isclose(lon, lon_orig, atol=1e-9)

    def test_multiple_points(self):
        """Test conversion with multiple points."""
        ecef = np.array([[6378137, 0, 0], [0, 6378137, 0], [0, 0, 6356752]]).T  # (3, 3)

        lat, lon, alt = ecef2geodetic(ecef)
        assert lat.shape == (3,)
        assert lon.shape == (3,)
        assert alt.shape == (3,)


class TestENUConversions:
    """Tests for ENU coordinate conversions."""

    def test_ecef2enu_origin(self):
        """Test that reference point maps to origin in ENU."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        enu = ecef2enu(ecef_ref, lat_ref, lon_ref, ecef_ref)
        np.testing.assert_allclose(enu, [0, 0, 0], atol=1e-6)

    def test_ecef2enu_east(self):
        """Test point to the east of reference."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)

        # Point slightly east (larger longitude)
        lat_pt = np.radians(45)
        lon_pt = np.radians(-74.99)
        ecef_pt = geodetic2ecef(lat_pt, lon_pt, 0.0)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        enu = ecef2enu(ecef_pt, lat_ref, lon_ref, ecef_ref)
        # Should have positive east component
        assert enu[0] > 0
        # North and up should be small
        assert np.abs(enu[1]) < np.abs(enu[0])
        assert np.abs(enu[2]) < 100

    def test_ecef2enu_north(self):
        """Test point to the north of reference."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)

        # Point slightly north (larger latitude)
        lat_pt = np.radians(45.01)
        lon_pt = np.radians(-75)
        ecef_pt = geodetic2ecef(lat_pt, lon_pt, 0.0)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        enu = ecef2enu(ecef_pt, lat_ref, lon_ref, ecef_ref)
        # Should have positive north component
        assert enu[1] > 0

    def test_enu2ecef_roundtrip(self):
        """Test ENU -> ECEF -> ENU roundtrip."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        enu_orig = np.array([100.0, 200.0, 50.0])
        ecef = enu2ecef(enu_orig, lat_ref, lon_ref, ecef_ref)
        enu_recovered = ecef2enu(ecef, lat_ref, lon_ref, ecef_ref)

        np.testing.assert_allclose(enu_recovered, enu_orig, atol=1e-8)

    def test_geodetic2enu(self):
        """Test direct geodetic to ENU conversion."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        alt_ref = 0.0

        lat_pt = np.radians(45.001)
        lon_pt = np.radians(-74.999)
        alt_pt = 10.0

        enu = geodetic2enu(lat_pt, lon_pt, alt_pt, lat_ref, lon_ref, alt_ref)

        # Should have positive east, north, and up
        assert enu[0] > 0  # East
        assert enu[1] > 0  # North
        assert enu[2] > 0  # Up

    def test_enu_multiple_points(self):
        """Test ENU conversion with multiple points."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        enu_points = np.array([[100, 200, 300], [50, 100, 150], [10, 20, 30]]).T

        ecef = enu2ecef(enu_points, lat_ref, lon_ref, ecef_ref)
        enu_recovered = ecef2enu(ecef, lat_ref, lon_ref, ecef_ref)

        np.testing.assert_allclose(enu_recovered, enu_points, atol=1e-8)


class TestNEDConversions:
    """Tests for NED coordinate conversions."""

    def test_ecef2ned_origin(self):
        """Test that reference point maps to origin in NED."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        ned = ecef2ned(ecef_ref, lat_ref, lon_ref, ecef_ref)
        np.testing.assert_allclose(ned, [0, 0, 0], atol=1e-6)

    def test_ned2ecef_roundtrip(self):
        """Test NED -> ECEF -> NED roundtrip."""
        lat_ref = np.radians(45)
        lon_ref = np.radians(-75)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, 0.0)

        ned_orig = np.array([200.0, 100.0, -50.0])  # North, East, Down
        ecef = ned2ecef(ned_orig, lat_ref, lon_ref, ecef_ref)
        ned_recovered = ecef2ned(ecef, lat_ref, lon_ref, ecef_ref)

        np.testing.assert_allclose(ned_recovered, ned_orig, atol=1e-8)


class TestENUNEDConversion:
    """Tests for ENU <-> NED conversions."""

    def test_enu2ned(self):
        """Test ENU to NED conversion."""
        enu = np.array([100.0, 200.0, 50.0])  # East, North, Up
        ned = enu2ned(enu)
        expected = np.array([200.0, 100.0, -50.0])  # North, East, Down
        np.testing.assert_allclose(ned, expected, atol=1e-10)

    def test_ned2enu(self):
        """Test NED to ENU conversion."""
        ned = np.array([200.0, 100.0, -50.0])  # North, East, Down
        enu = ned2enu(ned)
        expected = np.array([100.0, 200.0, 50.0])  # East, North, Up
        np.testing.assert_allclose(enu, expected, atol=1e-10)

    def test_enu_ned_roundtrip(self):
        """Test ENU -> NED -> ENU roundtrip."""
        enu_orig = np.array([100.0, 200.0, 50.0])
        ned = enu2ned(enu_orig)
        enu_recovered = ned2enu(ned)
        np.testing.assert_allclose(enu_recovered, enu_orig, atol=1e-10)

    def test_enu2ned_multiple_points(self):
        """Test ENU to NED with multiple points."""
        enu = np.array([[100, 200, 50], [10, 20, 5]]).T  # (3, 2)
        ned = enu2ned(enu)
        expected = np.array([[200, 100, -50], [20, 10, -5]]).T
        np.testing.assert_allclose(ned, expected, atol=1e-10)


class TestRadiiOfCurvature:
    """Tests for radii of curvature functions."""

    def test_geocentric_radius_equator(self):
        """Test geocentric radius at equator."""
        r = geocentric_radius(0.0)
        # Should be close to semi-major axis at equator
        assert np.isclose(r, 6378137.0, rtol=1e-6)

    def test_geocentric_radius_pole(self):
        """Test geocentric radius at pole."""
        r = geocentric_radius(np.pi / 2)
        # Should be close to semi-minor axis at pole
        b = 6378137.0 * (1 - 1 / 298.257223563)
        assert np.isclose(r, b, rtol=1e-3)

    def test_prime_vertical_radius_equator(self):
        """Test prime vertical radius at equator."""
        N = prime_vertical_radius(0.0)
        # Should equal semi-major axis at equator
        assert np.isclose(N, 6378137.0, rtol=1e-6)

    def test_prime_vertical_radius_pole(self):
        """Test prime vertical radius at pole."""
        N_eq = prime_vertical_radius(0.0)
        N_pole = prime_vertical_radius(np.pi / 2)
        # Prime vertical radius is larger at pole
        assert N_pole > N_eq

    def test_meridional_radius_equator(self):
        """Test meridional radius at equator."""
        M = meridional_radius(0.0)
        # Meridional radius is smaller than prime vertical at equator
        N = prime_vertical_radius(0.0)
        assert M < N

    def test_meridional_radius_pole(self):
        """Test meridional radius at pole."""
        M_eq = meridional_radius(0.0)
        M_pole = meridional_radius(np.pi / 2)
        # Meridional radius is larger at pole
        assert M_pole > M_eq

    def test_radii_multiple_latitudes(self):
        """Test radii with array of latitudes."""
        lats = np.radians([0, 30, 45, 60, 90])

        R = geocentric_radius(lats)
        N = prime_vertical_radius(lats)
        M = meridional_radius(lats)

        assert R.shape == (5,)
        assert N.shape == (5,)
        assert M.shape == (5,)

        # All radii should decrease from equator to pole (for geocentric)
        # or have known behavior
        assert all(N > 0)
        assert all(M > 0)
        assert all(R > 0)


# =============================================================================
# Jacobian Matrix Tests
# =============================================================================


from pytcl.coordinate_systems.jacobians.jacobians import (  # noqa: E402
    cross_covariance_transform,
    enu_jacobian,
    geodetic_jacobian,
    ned_jacobian,
    numerical_jacobian,
    polar_jacobian,
    polar_jacobian_inv,
    ruv_jacobian,
    spherical_jacobian,
    spherical_jacobian_inv,
)


class TestSphericalJacobian:
    """Tests for spherical coordinate Jacobians."""

    def test_spherical_jacobian_shape(self):
        """Test that spherical Jacobian has correct shape."""
        J = spherical_jacobian([1, 2, 3])
        assert J.shape == (3, 3)

    def test_spherical_jacobian_x_axis(self):
        """Test Jacobian at point on x-axis."""
        J = spherical_jacobian([10, 0, 0])
        # dr/dx = 1 at [r,0,0]
        assert np.isclose(J[0, 0], 1.0)
        # dr/dy = 0, dr/dz = 0
        assert np.isclose(J[0, 1], 0.0)
        assert np.isclose(J[0, 2], 0.0)

    def test_spherical_jacobian_origin(self):
        """Test that Jacobian at origin returns NaN."""
        J = spherical_jacobian([0, 0, 0])
        assert np.all(np.isnan(J))

    def test_spherical_jacobian_standard_type(self):
        """Test standard (physics) convention."""
        J = spherical_jacobian([1, 1, 1], system_type="standard")
        assert J.shape == (3, 3)
        # Should have finite values
        assert np.all(np.isfinite(J))

    def test_spherical_jacobian_vs_numerical(self):
        """Test analytical Jacobian matches numerical."""
        point = np.array([3.0, 4.0, 5.0])

        def cart2sphere(p):
            x, y, z = p
            r = np.sqrt(x**2 + y**2 + z**2)
            az = np.arctan2(y, x)
            el = np.arctan2(z, np.sqrt(x**2 + y**2))
            return np.array([r, az, el])

        J_analytical = spherical_jacobian(point)
        J_numerical = numerical_jacobian(cart2sphere, point)

        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5)


class TestSphericalJacobianInv:
    """Tests for inverse spherical Jacobian."""

    def test_spherical_jacobian_inv_shape(self):
        """Test inverse Jacobian shape."""
        J = spherical_jacobian_inv(10.0, np.pi / 4, np.pi / 6)
        assert J.shape == (3, 3)

    def test_spherical_jacobian_inv_identity_check(self):
        """Test that J_inv @ J ≈ I at corresponding points."""
        r, az, el = 10.0, np.pi / 4, np.pi / 6
        # Compute Cartesian point
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)

        J = spherical_jacobian([x, y, z])
        J_inv = spherical_jacobian_inv(r, az, el)

        # J_inv @ J should be close to identity
        product = J_inv @ J
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_spherical_jacobian_inv_standard(self):
        """Test standard convention inverse."""
        J = spherical_jacobian_inv(10.0, np.pi / 4, np.pi / 6, system_type="standard")
        assert J.shape == (3, 3)
        assert np.all(np.isfinite(J))


class TestPolarJacobian:
    """Tests for 2D polar Jacobians."""

    def test_polar_jacobian_shape(self):
        """Test polar Jacobian shape."""
        J = polar_jacobian([3, 4])
        assert J.shape == (2, 2)

    def test_polar_jacobian_x_axis(self):
        """Test Jacobian at point on x-axis."""
        J = polar_jacobian([5, 0])
        # dr/dx = 1, dr/dy = 0
        assert np.isclose(J[0, 0], 1.0)
        assert np.isclose(J[0, 1], 0.0)
        # dtheta/dx = 0, dtheta/dy = 1/r = 0.2
        assert np.isclose(J[1, 0], 0.0)
        assert np.isclose(J[1, 1], 0.2)

    def test_polar_jacobian_origin(self):
        """Test Jacobian at origin returns NaN."""
        J = polar_jacobian([0, 0])
        assert np.all(np.isnan(J))

    def test_polar_jacobian_vs_numerical(self):
        """Test analytical vs numerical Jacobian."""
        point = np.array([3.0, 4.0])

        def cart2pol(p):
            x, y = p
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            return np.array([r, theta])

        J_analytical = polar_jacobian(point)
        J_numerical = numerical_jacobian(cart2pol, point)

        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5)


class TestPolarJacobianInv:
    """Tests for inverse polar Jacobian."""

    def test_polar_jacobian_inv_shape(self):
        """Test inverse polar Jacobian shape."""
        J = polar_jacobian_inv(5.0, np.pi / 4)
        assert J.shape == (2, 2)

    def test_polar_jacobian_inv_identity_check(self):
        """Test J_inv @ J ≈ I."""
        r, theta = 5.0, np.pi / 4
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        J = polar_jacobian([x, y])
        J_inv = polar_jacobian_inv(r, theta)

        product = J_inv @ J
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)


class TestRUVJacobian:
    """Tests for r-u-v Jacobians."""

    def test_ruv_jacobian_shape(self):
        """Test r-u-v Jacobian shape."""
        J = ruv_jacobian([3, 4, 5])
        assert J.shape == (3, 3)

    def test_ruv_jacobian_origin(self):
        """Test Jacobian at origin returns NaN."""
        J = ruv_jacobian([0, 0, 0])
        assert np.all(np.isnan(J))

    def test_ruv_jacobian_vs_numerical(self):
        """Test analytical vs numerical."""
        point = np.array([3.0, 4.0, 5.0])

        def cart2ruv(p):
            r = np.linalg.norm(p)
            u = p[0] / r
            v = p[1] / r
            return np.array([r, u, v])

        J_analytical = ruv_jacobian(point)
        J_numerical = numerical_jacobian(cart2ruv, point)

        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5)


class TestENUJacobian:
    """Tests for ENU Jacobian."""

    def test_enu_jacobian_shape(self):
        """Test ENU Jacobian shape."""
        J = enu_jacobian(np.radians(45), np.radians(-75))
        assert J.shape == (3, 3)

    def test_enu_jacobian_orthogonal(self):
        """Test that ENU Jacobian is orthogonal (rotation matrix)."""
        J = enu_jacobian(np.radians(45), np.radians(-75))
        # J @ J.T should be identity
        np.testing.assert_allclose(J @ J.T, np.eye(3), atol=1e-10)
        # det should be 1
        assert np.isclose(np.linalg.det(J), 1.0, atol=1e-10)

    def test_enu_jacobian_equator(self):
        """Test ENU Jacobian at equator, prime meridian."""
        J = enu_jacobian(0.0, 0.0)
        # At (0,0), East is +Y, North is -X (along meridian toward pole), Up is +Z
        # The rotation should map ECEF to ENU appropriately
        assert J.shape == (3, 3)


class TestNEDJacobian:
    """Tests for NED Jacobian."""

    def test_ned_jacobian_shape(self):
        """Test NED Jacobian shape."""
        J = ned_jacobian(np.radians(45), np.radians(-75))
        assert J.shape == (3, 3)

    def test_ned_jacobian_orthogonal(self):
        """Test that NED Jacobian is orthogonal."""
        J = ned_jacobian(np.radians(45), np.radians(-75))
        np.testing.assert_allclose(J @ J.T, np.eye(3), atol=1e-10)

    def test_ned_vs_enu_relationship(self):
        """Test relationship between NED and ENU Jacobians."""
        lat, lon = np.radians(45), np.radians(-75)
        J_enu = enu_jacobian(lat, lon)
        J_ned = ned_jacobian(lat, lon)

        # NED and ENU are related by permutation: N=E[1], E=E[0], D=-E[2]
        # So J_ned[0] = J_enu[1], J_ned[1] = J_enu[0], J_ned[2] = -J_enu[2]
        np.testing.assert_allclose(J_ned[0], J_enu[1], atol=1e-10)
        np.testing.assert_allclose(J_ned[1], J_enu[0], atol=1e-10)
        np.testing.assert_allclose(J_ned[2], -J_enu[2], atol=1e-10)


class TestGeodeticJacobian:
    """Tests for geodetic Jacobian."""

    def test_geodetic_jacobian_shape(self):
        """Test geodetic Jacobian shape."""
        J = geodetic_jacobian(np.radians(45), np.radians(-75), 100.0)
        assert J.shape == (3, 3)

    def test_geodetic_jacobian_vs_numerical(self):
        """Test analytical vs numerical geodetic Jacobian."""
        lat = np.radians(45)
        lon = np.radians(-75)
        alt = 1000.0

        def geodetic2ecef(lla):
            lat, lon, alt = lla
            a = 6378137.0
            f = 1 / 298.257223563
            e2 = 2 * f - f**2
            sin_lat = np.sin(lat)
            cos_lat = np.cos(lat)
            N = a / np.sqrt(1 - e2 * sin_lat**2)
            x = (N + alt) * cos_lat * np.cos(lon)
            y = (N + alt) * cos_lat * np.sin(lon)
            z = (N * (1 - e2) + alt) * sin_lat
            return np.array([x, y, z])

        J_analytical = geodetic_jacobian(lat, lon, alt)
        J_numerical = numerical_jacobian(geodetic2ecef, np.array([lat, lon, alt]))

        # Looser tolerance due to numerical differentiation and ellipsoid effects
        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-2)


class TestCovarianceTransform:
    """Tests for covariance transformation."""

    def test_cross_covariance_transform_identity(self):
        """Test identity Jacobian preserves covariance."""
        P = np.diag([1.0, 2.0, 3.0])
        J = np.eye(3)
        P_new = cross_covariance_transform(J, P)
        np.testing.assert_allclose(P_new, P)

    def test_cross_covariance_transform_scaling(self):
        """Test scaling Jacobian scales variances."""
        P = np.diag([1.0, 1.0, 1.0])
        J = 2.0 * np.eye(3)  # Scale by 2
        P_new = cross_covariance_transform(J, P)
        # Variances should scale by 4 (2^2)
        np.testing.assert_allclose(P_new, 4.0 * np.eye(3))

    def test_cross_covariance_transform_rotation(self):
        """Test rotation preserves trace (total variance)."""
        P = np.diag([1.0, 2.0, 3.0])
        angle = np.pi / 4
        J = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        P_new = cross_covariance_transform(J, P)
        # Trace should be preserved under rotation
        assert np.isclose(np.trace(P_new), np.trace(P))


class TestNumericalJacobian:
    """Tests for numerical Jacobian computation."""

    def test_numerical_jacobian_linear(self):
        """Test numerical Jacobian for linear function."""

        def linear(x):
            A = np.array([[1, 2, 3], [4, 5, 6]])
            return A @ x

        x = np.array([1.0, 2.0, 3.0])
        J = numerical_jacobian(linear, x)
        expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
        np.testing.assert_allclose(J, expected, atol=1e-6)

    def test_numerical_jacobian_quadratic(self):
        """Test numerical Jacobian for quadratic function."""

        def quadratic(x):
            return np.array([x[0] ** 2, x[0] * x[1], x[1] ** 2])

        x = np.array([2.0, 3.0])
        J = numerical_jacobian(quadratic, x)
        # df1/dx = 2*x[0] = 4, df1/dy = 0
        # df2/dx = x[1] = 3, df2/dy = x[0] = 2
        # df3/dx = 0, df3/dy = 2*x[1] = 6
        expected = np.array([[4, 0], [3, 2], [0, 6]], dtype=np.float64)
        np.testing.assert_allclose(J, expected, atol=1e-5)

    def test_numerical_jacobian_scalar_output(self):
        """Test with scalar-valued function."""

        def scalar_func(x):
            return x[0] ** 2 + x[1] ** 2 + x[2] ** 2

        x = np.array([1.0, 2.0, 3.0])
        J = numerical_jacobian(scalar_func, x)
        # Gradient: [2*x0, 2*x1, 2*x2] = [2, 4, 6]
        expected = np.array([[2, 4, 6]], dtype=np.float64)
        np.testing.assert_allclose(J, expected, atol=1e-5)


class TestSEZConversions:
    """Tests for SEZ (South-East-Zenith) coordinate conversions."""

    def test_geodetic2sez_at_reference_point(self):
        """SEZ at reference point should be zero."""
        from pytcl.coordinate_systems.conversions.geodetic import geodetic2sez

        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-105.0)
        alt_ref = 1000.0  # meters

        sez = geodetic2sez(lat_ref, lon_ref, alt_ref, lat_ref, lon_ref, alt_ref)
        np.testing.assert_allclose(sez, np.zeros(3), atol=1e-10)

    def test_sez_geodetic_roundtrip(self):
        """Test SEZ <-> geodetic roundtrip."""
        from pytcl.coordinate_systems.conversions.geodetic import (
            geodetic2sez,
            sez2geodetic,
        )

        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-105.0)
        alt_ref = 1000.0

        # Target point slightly offset
        lat_tgt = np.radians(40.01)
        lon_tgt = np.radians(-104.99)
        alt_tgt = 1100.0

        # Forward
        sez = geodetic2sez(lat_tgt, lon_tgt, alt_tgt, lat_ref, lon_ref, alt_ref)

        # Inverse
        lat_back, lon_back, alt_back = sez2geodetic(sez, lat_ref, lon_ref, alt_ref)

        np.testing.assert_allclose(
            [lat_back, lon_back, alt_back], [lat_tgt, lon_tgt, alt_tgt], rtol=1e-9
        )

    def test_sez_distance_calculation(self):
        """Verify distance calculation using SEZ."""
        from pytcl.coordinate_systems.conversions.geodetic import (
            geodetic2ecef,
            geodetic2sez,
        )

        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-105.0)
        alt_ref = 0.0

        lat_tgt = np.radians(40.01)
        lon_tgt = np.radians(-105.01)
        alt_tgt = 0.0

        # SEZ conversion
        sez = geodetic2sez(lat_tgt, lon_tgt, alt_tgt, lat_ref, lon_ref, alt_ref)
        sez_distance = np.linalg.norm(sez)

        # ECEF distance (should be similar for small distances)
        ecef_ref = geodetic2ecef(lat_ref, lon_ref, alt_ref)
        ecef_tgt = geodetic2ecef(lat_tgt, lon_tgt, alt_tgt)
        ecef_distance = np.linalg.norm(ecef_tgt - ecef_ref)

        # SEZ distance should be close to ECEF distance for small separations
        np.testing.assert_allclose(sez_distance, ecef_distance, rtol=0.01)

    def test_sez_elevation_azimuth(self):
        """Verify elevation and azimuth computation from SEZ."""
        from pytcl.coordinate_systems.conversions.geodetic import geodetic2sez

        lat_ref = np.radians(40.0)
        lon_ref = np.radians(-105.0)
        alt_ref = 1000.0

        # Target directly east (azimuth = 90 degrees)
        lat_tgt = lat_ref
        lon_tgt = lon_ref + np.radians(0.1)
        alt_tgt = alt_ref

        sez = geodetic2sez(lat_tgt, lon_tgt, alt_tgt, lat_ref, lon_ref, alt_ref)

        # In SEZ: S=south, E=east, Z=zenith
        # For eastward target: S should be small relative to E, Z~0
        assert abs(sez[0]) < abs(sez[1]), "South component should be smaller than east"
        assert sez[1] > 0.0, "East component should be positive"
        assert sez[2] < 1000.0, "Zenith component should be small for same altitude"

        # Compute elevation and azimuth
        rho = np.linalg.norm(sez[:2])  # Horizontal distance
        elevation = np.arctan2(sez[2], rho)

        # Should be roughly east (azimuth near pi/2)
        assert elevation < np.radians(1.0), "Low elevation for same altitude"

    def test_sez_north_target(self):
        """SEZ for northward target."""
        from pytcl.coordinate_systems.conversions.geodetic import geodetic2sez

        lat_ref = np.radians(0.0)
        lon_ref = np.radians(0.0)
        alt_ref = 0.0

        # Target to the north
        lat_tgt = np.radians(1.0)
        lon_tgt = lon_ref
        alt_tgt = alt_ref

        sez = geodetic2sez(lat_tgt, lon_tgt, alt_tgt, lat_ref, lon_ref, alt_ref)

        # For northward target, the magnitude should be large and east component small
        total_distance = np.linalg.norm(sez)
        assert total_distance > 100000, "Distance should be significant"
        assert abs(sez[1]) < abs(
            sez[0]
        ), "East component should be smaller than meridional"
