"""
Coordinate Systems Example.

This example demonstrates:
1. Cartesian to spherical coordinate conversions
2. Geodetic (WGS84) to ECEF transformations
3. Local tangent plane (ENU/NED) coordinates
4. Rotation matrices and quaternions
5. Jacobian-based covariance transformations

Run with: python examples/coordinate_systems.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from pytcl.coordinate_systems import (  # noqa: E402; Spherical conversions; Geodetic conversions; Rotation matrices; Quaternions; Jacobians
    cart2sphere,
    cross_covariance_transform,
    ecef2enu,
    ecef2geodetic,
    ecef2ned,
    enu2ecef,
    euler2rotmat,
    geodetic2ecef,
    quat2rotmat,
    quat_multiply,
    rotmat2euler,
    rotmat2quat,
    rotz,
    slerp,
    sphere2cart,
    spherical_jacobian_inv,
)


def spherical_conversions_demo() -> None:
    """Demonstrate spherical coordinate conversions."""
    print("=" * 60)
    print("1. SPHERICAL COORDINATE CONVERSIONS")
    print("=" * 60)

    # Define a point in Cartesian coordinates
    cart_point = np.array([1000.0, 2000.0, 500.0])  # meters
    print(
        f"\nCartesian point: x={cart_point[0]:.1f}, y={cart_point[1]:.1f}, "
        f"z={cart_point[2]:.1f} m"
    )

    # Convert to spherical (range, azimuth, elevation)
    r, az, el = cart2sphere(cart_point)
    print("\nSpherical coordinates:")
    print(f"  Range:     {r:.2f} m")
    print(f"  Azimuth:   {np.degrees(az):.2f} deg")
    print(f"  Elevation: {np.degrees(el):.2f} deg")

    # Convert back to Cartesian
    cart_recovered = sphere2cart(r, az, el)
    print(
        f"\nRecovered Cartesian: x={cart_recovered[0]:.1f}, "
        f"y={cart_recovered[1]:.1f}, z={cart_recovered[2]:.1f} m"
    )

    # Verify roundtrip
    error = np.linalg.norm(cart_point - cart_recovered)
    print(f"Roundtrip error: {error:.2e} m")


def geodetic_conversions_demo() -> None:
    """Demonstrate geodetic coordinate conversions."""
    print("\n" + "=" * 60)
    print("2. GEODETIC (WGS84) COORDINATE CONVERSIONS")
    print("=" * 60)

    # Define a geodetic point (Washington DC area)
    lat = np.radians(38.9072)  # Latitude in radians
    lon = np.radians(-77.0369)  # Longitude in radians
    alt = 100.0  # Altitude in meters

    print("\nGeodetic coordinates:")
    print(f"  Latitude:  {np.degrees(lat):.4f} deg")
    print(f"  Longitude: {np.degrees(lon):.4f} deg")
    print(f"  Altitude:  {alt:.1f} m")

    # Convert to ECEF
    ecef = geodetic2ecef(lat, lon, alt)
    print("\nECEF coordinates:")
    print(f"  X: {ecef[0]/1000:.3f} km")
    print(f"  Y: {ecef[1]/1000:.3f} km")
    print(f"  Z: {ecef[2]/1000:.3f} km")

    # Convert back to geodetic
    lat_r, lon_r, alt_r = ecef2geodetic(ecef)
    print("\nRecovered geodetic:")
    print(f"  Latitude:  {np.degrees(lat_r):.4f} deg")
    print(f"  Longitude: {np.degrees(lon_r):.4f} deg")
    print(f"  Altitude:  {alt_r:.1f} m")


def local_tangent_plane_demo() -> None:
    """Demonstrate local tangent plane (ENU/NED) conversions."""
    print("\n" + "=" * 60)
    print("3. LOCAL TANGENT PLANE (ENU/NED) CONVERSIONS")
    print("=" * 60)

    # Reference point (origin of local frame)
    ref_lat = np.radians(38.9072)
    ref_lon = np.radians(-77.0369)

    # Compute ECEF reference point for altitude = 0
    ref_ecef = geodetic2ecef(ref_lat, ref_lon, 0.0)

    # Target point 1 km East, 2 km North, 100 m Up from reference
    enu_offset = np.array([1000.0, 2000.0, 100.0])  # East, North, Up
    print("\nENU offset from reference:")
    print(f"  East:  {enu_offset[0]:.1f} m")
    print(f"  North: {enu_offset[1]:.1f} m")
    print(f"  Up:    {enu_offset[2]:.1f} m")

    # Convert ENU to ECEF
    target_ecef = enu2ecef(enu_offset, ref_lat, ref_lon, ref_ecef)
    print("\nTarget ECEF coordinates:")
    print(f"  X: {target_ecef[0]/1000:.3f} km")
    print(f"  Y: {target_ecef[1]/1000:.3f} km")
    print(f"  Z: {target_ecef[2]/1000:.3f} km")

    # Convert ECEF back to ENU
    enu_recovered = ecef2enu(target_ecef, ref_lat, ref_lon, ref_ecef)
    print("\nRecovered ENU:")
    print(f"  East:  {enu_recovered[0]:.1f} m")
    print(f"  North: {enu_recovered[1]:.1f} m")
    print(f"  Up:    {enu_recovered[2]:.1f} m")

    # Also show NED (North, East, Down) - common in aviation
    ned = ecef2ned(target_ecef, ref_lat, ref_lon, ref_ecef)
    print("\nNED coordinates (aviation convention):")
    print(f"  North: {ned[0]:.1f} m")
    print(f"  East:  {ned[1]:.1f} m")
    print(f"  Down:  {ned[2]:.1f} m")


def rotation_demo() -> None:
    """Demonstrate rotation matrices and Euler angles."""
    print("\n" + "=" * 60)
    print("4. ROTATION MATRICES AND EULER ANGLES")
    print("=" * 60)

    # Create individual rotation matrices
    roll = np.radians(10.0)  # Roll about X
    pitch = np.radians(20.0)  # Pitch about Y
    yaw = np.radians(30.0)  # Yaw about Z

    print("\nEuler angles (ZYX convention):")
    print(f"  Roll (X):  {np.degrees(roll):.1f} deg")
    print(f"  Pitch (Y): {np.degrees(pitch):.1f} deg")
    print(f"  Yaw (Z):   {np.degrees(yaw):.1f} deg")

    # Combined rotation (ZYX order: yaw, then pitch, then roll)
    # euler2rotmat expects [angle1, angle2, angle3] for the sequence
    R = euler2rotmat([yaw, pitch, roll], sequence="ZYX")
    print("\nRotation matrix (3x3):")
    print(R)

    # Verify it's a proper rotation (det = 1, orthogonal)
    print(f"\nDeterminant: {np.linalg.det(R):.6f} (should be 1)")
    print(f"R @ R.T = I check: {np.allclose(R @ R.T, np.eye(3))}")

    # Extract Euler angles back
    angles_recovered = rotmat2euler(R, sequence="ZYX")
    yaw_r, pitch_r, roll_r = angles_recovered
    print("\nRecovered Euler angles:")
    print(f"  Roll:  {np.degrees(roll_r):.1f} deg")
    print(f"  Pitch: {np.degrees(pitch_r):.1f} deg")
    print(f"  Yaw:   {np.degrees(yaw_r):.1f} deg")


def quaternion_demo() -> None:
    """Demonstrate quaternion operations and interpolation."""
    print("\n" + "=" * 60)
    print("5. QUATERNIONS AND SLERP INTERPOLATION")
    print("=" * 60)

    # Create a rotation matrix (ZYX = yaw, pitch, roll order)
    roll, pitch, yaw = np.radians(15.0), np.radians(25.0), np.radians(45.0)
    R = euler2rotmat([yaw, pitch, roll], sequence="ZYX")

    # Convert to quaternion [w, x, y, z]
    q = rotmat2quat(R)
    print("\nQuaternion [w, x, y, z]:")
    print(f"  q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
    print(f"  Norm: {np.linalg.norm(q):.6f} (should be 1)")

    # Convert back to rotation matrix
    R_recovered = quat2rotmat(q)
    print(f"\nRotation matrix roundtrip check: {np.allclose(R, R_recovered)}")

    # Quaternion multiplication (composing rotations)
    q2 = rotmat2quat(rotz(np.radians(90.0)))  # 90 deg yaw rotation
    q_composed = quat_multiply(q, q2)
    print("\nComposed quaternion (original + 90 deg yaw):")
    print(
        f"  q = [{q_composed[0]:.4f}, {q_composed[1]:.4f}, "
        f"{q_composed[2]:.4f}, {q_composed[3]:.4f}]"
    )

    # SLERP interpolation between two orientations
    print("\nSLERP interpolation (identity to 90 deg yaw):")
    q_start = np.array([1.0, 0.0, 0.0, 0.0])  # Identity
    q_end = rotmat2quat(rotz(np.radians(90.0)))

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = slerp(q_start, q_end, t)
        R_interp = quat2rotmat(q_interp)
        # rotmat2euler returns [angle1, angle2, angle3] for ZYX = [yaw, pitch, roll]
        angles = rotmat2euler(R_interp, sequence="ZYX")
        yaw_interp = angles[0]
        print(f"  t={t:.2f}: yaw = {np.degrees(yaw_interp):.1f} deg")


def jacobian_covariance_demo() -> None:
    """Demonstrate Jacobian-based covariance transformation."""
    print("\n" + "=" * 60)
    print("6. JACOBIAN-BASED COVARIANCE TRANSFORMATION")
    print("=" * 60)

    # Sensor measures in spherical coordinates with uncertainty
    r = 5000.0  # Range in meters
    az = np.radians(45.0)  # Azimuth
    el = np.radians(10.0)  # Elevation

    # Measurement covariance in spherical coordinates
    sigma_r = 10.0  # Range std (meters)
    sigma_az = np.radians(0.5)  # Azimuth std (radians)
    sigma_el = np.radians(0.5)  # Elevation std (radians)

    P_spherical = np.diag([sigma_r**2, sigma_az**2, sigma_el**2])

    print("\nSpherical measurement:")
    print(f"  Range:     {r:.1f} +/- {sigma_r:.1f} m")
    print(f"  Azimuth:   {np.degrees(az):.1f} +/- {np.degrees(sigma_az):.2f} deg")
    print(f"  Elevation: {np.degrees(el):.1f} +/- {np.degrees(sigma_el):.2f} deg")

    # Get Jacobian of Cartesian w.r.t. spherical at this point
    # spherical_jacobian_inv: d[x,y,z] = J @ d[r,az,el]
    J = spherical_jacobian_inv(r, az, el)

    print("\nJacobian (dCartesian/dSpherical):")
    print(J)

    # Transform covariance to Cartesian
    P_cartesian = cross_covariance_transform(P_spherical, J)

    print("\nCartesian covariance matrix:")
    print(P_cartesian)

    # Extract position uncertainties
    sigma_x = np.sqrt(P_cartesian[0, 0])
    sigma_y = np.sqrt(P_cartesian[1, 1])
    sigma_z = np.sqrt(P_cartesian[2, 2])

    # Convert mean to Cartesian
    cart = sphere2cart(r, az, el)
    print("\nCartesian position with uncertainties:")
    print(f"  x = {cart[0]:.1f} +/- {sigma_x:.1f} m")
    print(f"  y = {cart[1]:.1f} +/- {sigma_y:.1f} m")
    print(f"  z = {cart[2]:.1f} +/- {sigma_z:.1f} m")


def main() -> None:
    """Run all coordinate system demonstrations."""
    print("\nCoordinate Systems Examples")
    print("=" * 60)
    print("Demonstrating pytcl coordinate transformation capabilities")

    spherical_conversions_demo()
    geodetic_conversions_demo()
    local_tangent_plane_demo()
    rotation_demo()
    quaternion_demo()
    jacobian_covariance_demo()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
