"""
INS/GNSS Navigation Example.

This example demonstrates:
1. INS mechanization (strapdown navigation)
2. IMU data processing with coning/sculling corrections
3. Loosely-coupled INS/GNSS integration
4. Tightly-coupled INS/GNSS integration
5. DOP computation and GNSS outage detection

Run with: python examples/ins_gnss_navigation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.navigation import (  # noqa: E402
    WGS84,
    GNSSMeasurement,
    IMUData,
    coarse_alignment,
    compute_dop,
    coning_correction,
    earth_rate_ned,
    geodetic_to_ecef,
    gnss_outage_detection,
    gravity_ned,
    initialize_ins_gnss,
    initialize_ins_state,
    loose_coupled_predict,
    loose_coupled_update,
    mechanize_ins_ned,
    normal_gravity,
    radii_of_curvature,
    satellite_elevation_azimuth,
    sculling_correction,
    transport_rate_ned,
)


def ins_basics_demo() -> None:
    """Demonstrate basic INS concepts."""
    print("=" * 60)
    print("1. INS FUNDAMENTALS")
    print("=" * 60)

    # Define a location (San Francisco)
    lat = np.radians(37.7749)
    lon = np.radians(-122.4194)
    alt = 100.0  # meters

    print("\nLocation: San Francisco")
    print(f"  Latitude: {np.degrees(lat):.4f} deg")
    print(f"  Longitude: {np.degrees(lon):.4f} deg")
    print(f"  Altitude: {alt:.1f} m")

    # Normal gravity
    g = normal_gravity(lat, alt)
    print(f"\nNormal gravity: {g:.6f} m/s^2")

    # Gravity in NED frame
    g_ned = gravity_ned(lat, alt)
    print(
        f"Gravity vector (NED): [{g_ned[0]:.6f}, {g_ned[1]:.6f}, {g_ned[2]:.6f}] m/s^2"
    )

    # Earth rate in NED
    omega_ie = earth_rate_ned(lat)
    print("\nEarth rotation (NED):")
    print(f"  North: {omega_ie[0] * 1e6:.3f} micro-rad/s")
    print(f"  East:  {omega_ie[1] * 1e6:.3f} micro-rad/s")
    print(f"  Down:  {omega_ie[2] * 1e6:.3f} micro-rad/s")
    print(f"  Total: {np.linalg.norm(omega_ie) * 180 / np.pi * 3600:.4f} deg/hr")

    # Radii of curvature
    R_n, R_e = radii_of_curvature(lat)
    print("\nRadii of curvature:")
    print(f"  Meridian (R_n): {R_n / 1e6:.3f} Mm")
    print(f"  Prime vertical (R_e): {R_e / 1e6:.3f} Mm")


def imu_processing_demo() -> None:
    """Demonstrate IMU data processing."""
    print("\n" + "=" * 60)
    print("2. IMU DATA PROCESSING")
    print("=" * 60)

    np.random.seed(42)

    # Simulated IMU data for a stationary sensor
    dt = 0.01  # 100 Hz IMU

    # Small biases and noise (typical MEMS IMU)
    gyro_bias = np.array([0.001, -0.0005, 0.002])  # rad/s
    accel_bias = np.array([0.01, -0.02, 0.015])  # m/s^2

    # Generate IMU samples
    n_samples = 100
    print(f"\nSimulating {n_samples} IMU samples at {1/dt:.0f} Hz")
    print(
        f"  Gyro bias: [{gyro_bias[0]*1e3:.2f}, {gyro_bias[1]*1e3:.2f}, {gyro_bias[2]*1e3:.2f}] mrad/s"
    )
    print(
        f"  Accel bias: [{accel_bias[0]*1e3:.1f}, {accel_bias[1]*1e3:.1f}, {accel_bias[2]*1e3:.1f}] mm/s^2"
    )

    # Stationary sensor: gyro measures Earth rate, accel measures gravity
    lat = np.radians(37.7749)
    omega_ie = earth_rate_ned(lat)
    g_ned = gravity_ned(lat, 0)

    # Transform to body frame (assume level, north-pointing)
    omega_body = omega_ie + gyro_bias + 1e-4 * np.random.randn(3)
    accel_body = -g_ned + accel_bias + 0.01 * np.random.randn(3)

    print("\nRaw IMU readings (stationary):")
    print(
        f"  Gyro: [{omega_body[0]*1e3:.3f}, {omega_body[1]*1e3:.3f}, {omega_body[2]*1e3:.3f}] mrad/s"
    )
    print(
        f"  Accel: [{accel_body[0]:.4f}, {accel_body[1]:.4f}, {accel_body[2]:.4f}] m/s^2"
    )

    # Coning correction (for high-frequency angular motion)
    # Simulate angular increments
    alpha_prev = omega_body * dt + 1e-6 * np.random.randn(3)
    alpha_curr = omega_body * dt + 1e-6 * np.random.randn(3)

    coning = coning_correction(alpha_prev, alpha_curr)
    print(f"\nConing correction magnitude: {np.linalg.norm(coning)*1e9:.3f} nano-rad")

    # Sculling correction
    dv_prev = accel_body * dt
    dv_curr = accel_body * dt

    sculling = sculling_correction(alpha_prev, alpha_curr, dv_prev, dv_curr)
    print(f"Sculling correction magnitude: {np.linalg.norm(sculling)*1e9:.3f} nano-m/s")


def coarse_alignment_demo() -> None:
    """Demonstrate INS coarse alignment (leveling)."""
    print("\n" + "=" * 60)
    print("3. INS COARSE ALIGNMENT (LEVELING)")
    print("=" * 60)

    # Location
    lat = np.radians(37.7749)
    alt = 100.0

    # Simulated accelerometer measurements (stationary)
    g_ned = gravity_ned(lat, alt)

    # True attitude: small roll and pitch
    true_roll = np.radians(2.0)
    true_pitch = np.radians(-1.5)

    print("\nTrue attitude:")
    print(f"  Roll: {np.degrees(true_roll):.2f} deg")
    print(f"  Pitch: {np.degrees(true_pitch):.2f} deg")
    print("  (Heading cannot be determined from accelerometers alone)")

    # Construct rotation matrix (NED to body)
    cr, sr = np.cos(true_roll), np.sin(true_roll)
    cp, sp = np.cos(true_pitch), np.sin(true_pitch)

    # Rotation matrices (heading assumed 0 for simplicity)
    R_roll = np.array([[1, 0, 0], [0, cr, sr], [0, -sr, cr]])
    R_pitch = np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])

    C_bn = R_roll @ R_pitch  # Body from NED (no heading rotation)

    # Body-frame measurements
    # Accelerometer measures reaction to gravity (specific force)
    accel_body = C_bn @ (-g_ned)

    # Add small noise
    np.random.seed(42)
    accel_body += 0.005 * np.random.randn(3)

    print("\nBody-frame accelerometer reading:")
    print(
        f"  Accel: [{accel_body[0]:.4f}, {accel_body[1]:.4f}, {accel_body[2]:.4f}] m/s^2"
    )
    print(f"  (For level vehicle: [0, 0, {-normal_gravity(lat, alt):.4f}] m/s^2)")

    # Coarse alignment (leveling only - uses accelerometer to find roll/pitch)
    roll_est, pitch_est = coarse_alignment(accel_body, lat)

    print("\nEstimated attitude (coarse leveling):")
    print(
        f"  Roll: {np.degrees(roll_est):.2f} deg (error: {np.degrees(roll_est - true_roll):.3f} deg)"
    )
    print(
        f"  Pitch: {np.degrees(pitch_est):.2f} deg (error: {np.degrees(pitch_est - true_pitch):.3f} deg)"
    )

    # Note about gyrocompassing
    print("\nNote: Heading estimation requires gyrocompassing (sensing Earth")
    print("rotation with gyroscopes), which is separate from coarse leveling.")


def ins_mechanization_demo() -> None:
    """Demonstrate INS mechanization."""
    print("\n" + "=" * 60)
    print("4. INS MECHANIZATION")
    print("=" * 60)

    # Initialize INS state
    lat = np.radians(37.7749)
    lon = np.radians(-122.4194)
    alt = 100.0
    vN, vE, vD = 10.0, 5.0, -1.0  # Moving NE, slight descent

    state = initialize_ins_state(lat, lon, alt, vN=vN, vE=vE, vD=vD)

    print("\nInitial state:")
    print(
        f"  Position: {np.degrees(state.latitude):.6f}N, {np.degrees(state.longitude):.6f}E, {state.altitude:.1f}m"
    )
    print(
        f"  Velocity (NED): [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}] m/s"
    )

    # Simulate forward motion with IMU
    dt = 0.01  # 100 Hz
    n_steps = 100

    # IMU measurements for constant velocity (no acceleration)
    g_ned = gravity_ned(lat, alt)
    omega_ie = earth_rate_ned(lat)
    omega_en = transport_rate_ned(
        state.velocity[0], state.velocity[1], state.latitude, state.altitude
    )

    # Body-frame measurements (assuming level flight, north heading)
    accel_body = -g_ned  # Only gravity
    gyro_body = omega_ie + omega_en  # Earth rate + transport rate

    print(f"\nSimulating {n_steps} navigation steps at {1/dt:.0f} Hz")

    # Integrate
    for _ in range(n_steps):
        imu = IMUData(
            gyro=gyro_body,
            accel=accel_body,
            dt=dt,
        )
        state = mechanize_ins_ned(state, imu)

    print(f"\nFinal state after {n_steps * dt:.1f} seconds:")
    print(
        f"  Position: {np.degrees(state.latitude):.6f}N, {np.degrees(state.longitude):.6f}E, {state.altitude:.1f}m"
    )
    print(
        f"  Velocity (NED): [{state.velocity[0]:.2f}, {state.velocity[1]:.2f}, {state.velocity[2]:.2f}] m/s"
    )

    # Expected position change
    R_n, R_e = radii_of_curvature(lat)
    _expected_dlat = vN * n_steps * dt / (R_n + alt)  # noqa: F841
    _expected_dlon = vE * n_steps * dt / ((R_e + alt) * np.cos(lat))  # noqa: F841

    print("\nPosition change:")
    print(
        f"  North: {np.degrees(state.latitude - lat) * 111000:.1f} m (expected: {vN * n_steps * dt:.1f} m)"
    )
    print(
        f"  East: {np.degrees(state.longitude - lon) * 111000 * np.cos(lat):.1f} m "
        f"(expected: {vE * n_steps * dt:.1f} m)"
    )


def gnss_geometry_demo() -> None:
    """Demonstrate GNSS geometry and DOP."""
    print("\n" + "=" * 60)
    print("5. GNSS GEOMETRY AND DOP")
    print("=" * 60)

    # User position
    user_lat = np.radians(37.7749)
    user_lon = np.radians(-122.4194)
    user_alt = 100.0
    user_lla = np.array([user_lat, user_lon, user_alt])

    user_ecef = np.array(geodetic_to_ecef(user_lat, user_lon, user_alt, WGS84))
    print(f"\nUser position: {np.degrees(user_lat):.4f}N, {np.degrees(user_lon):.4f}E")

    # Simulated satellite positions (GPS constellation subset)
    sat_positions = [
        (30, 45, 20200e3),  # Lat, lon, alt (deg, deg, m)
        (45, -90, 20200e3),
        (15, -150, 20200e3),
        (60, 0, 20200e3),
        (-30, 90, 20200e3),
        (0, 180, 20200e3),
    ]

    print("\nSatellite visibility:")
    print("-" * 50)

    visible_sats = []
    for i, (lat_deg, lon_deg, alt) in enumerate(sat_positions):
        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)
        pos_ecef = np.array(geodetic_to_ecef(lat, lon, alt, WGS84))

        # Compute elevation and azimuth
        el, az = satellite_elevation_azimuth(user_lla, pos_ecef)

        if el > 0:  # Above horizon
            visible_sats.append(pos_ecef)
            print(
                f"  PRN {i+1}: El={np.degrees(el):.1f} deg, Az={np.degrees(az):.1f} deg"
            )
        else:
            print(f"  PRN {i+1}: Below horizon (El={np.degrees(el):.1f} deg)")

    # Compute DOP from geometry matrix
    if len(visible_sats) >= 4:
        # Build geometry matrix (line-of-sight unit vectors + clock column)
        H = np.zeros((len(visible_sats), 4))
        for i, sat_ecef in enumerate(visible_sats):
            los = sat_ecef - user_ecef
            range_val = np.linalg.norm(los)
            H[i, :3] = -los / range_val  # Unit vector toward satellite
            H[i, 3] = 1.0  # Clock column

        GDOP, PDOP, HDOP, VDOP = compute_dop(H)
        print(f"\nDilution of Precision (DOP) with {len(visible_sats)} satellites:")
        print(f"  GDOP: {GDOP:.2f}")
        print(f"  PDOP: {PDOP:.2f}")
        print(f"  HDOP: {HDOP:.2f}")
        print(f"  VDOP: {VDOP:.2f}")

        # Interpret DOP
        if PDOP < 2:
            quality = "Excellent"
        elif PDOP < 5:
            quality = "Good"
        elif PDOP < 10:
            quality = "Moderate"
        else:
            quality = "Poor"
        print(f"  Position accuracy: {quality}")
    else:
        print(f"\nInsufficient satellites ({len(visible_sats)}) for DOP computation")


def loose_coupling_demo() -> None:
    """Demonstrate loosely-coupled INS/GNSS integration."""
    print("\n" + "=" * 60)
    print("6. LOOSELY-COUPLED INS/GNSS INTEGRATION")
    print("=" * 60)

    np.random.seed(42)

    # Initialize INS state
    lat = np.radians(37.7749)
    lon = np.radians(-122.4194)
    alt = 100.0
    vN, vE, vD = 10.0, 5.0, 0.0

    ins_state = initialize_ins_state(lat, lon, alt, vN=vN, vE=vE, vD=vD)

    # Initialize INS/GNSS integrated state
    pos_std = 5.0  # meters
    vel_std = 0.1  # m/s
    att_std = np.radians(0.5)  # rad

    state = initialize_ins_gnss(
        ins_state, position_std=pos_std, velocity_std=vel_std, attitude_std=att_std
    )

    print("\nInitial state:")
    print(
        f"  Position: {np.degrees(state.ins_state.latitude):.6f}N, "
        f"{np.degrees(state.ins_state.longitude):.6f}E"
    )
    print(f"  Position std: {np.sqrt(state.error_cov[0, 0]):.2f} m")
    print(f"  Velocity std: {np.sqrt(state.error_cov[3, 3]):.3f} m/s")

    # Simulate navigation with GNSS updates
    dt_ins = 0.01  # INS rate
    dt_gnss = 1.0  # GNSS rate
    n_gnss_epochs = 5

    print(f"\nSimulating {n_gnss_epochs} GNSS epochs ({n_gnss_epochs} seconds)")
    print("-" * 60)

    # IMU measurements (constant velocity)
    g_ned = gravity_ned(lat, alt)
    accel_body = -g_ned
    gyro_body = earth_rate_ned(lat)

    for epoch in range(n_gnss_epochs):
        # INS propagation between GNSS updates
        for _ in range(int(dt_gnss / dt_ins)):
            imu = IMUData(
                gyro=gyro_body + 1e-5 * np.random.randn(3),
                accel=accel_body + 0.01 * np.random.randn(3),
                dt=dt_ins,
            )
            state = loose_coupled_predict(state, imu)

        # GNSS measurement (with noise)
        gnss_pos = np.array(
            [
                state.ins_state.latitude + np.random.randn() * 2e-6,
                state.ins_state.longitude + np.random.randn() * 2e-6,
                state.ins_state.altitude + np.random.randn() * 5.0,
            ]
        )
        gnss_vel = np.array(
            [
                state.ins_state.velocity[0] + np.random.randn() * 0.05,
                state.ins_state.velocity[1] + np.random.randn() * 0.05,
                state.ins_state.velocity[2] + np.random.randn() * 0.1,
            ]
        )

        gnss_meas = GNSSMeasurement(
            position=gnss_pos,
            velocity=gnss_vel,
            position_cov=np.diag([3.0**2, 3.0**2, 6.0**2]),
            velocity_cov=np.diag([0.05**2, 0.05**2, 0.1**2]),
            time=epoch * dt_gnss,
        )

        # GNSS update
        result = loose_coupled_update(state, gnss_meas)
        state = result.state

        print(
            f"  Epoch {epoch + 1}: Position std = {np.sqrt(state.error_cov[0, 0]):.2f} m, "
            f"Velocity std = {np.sqrt(state.error_cov[3, 3]):.4f} m/s"
        )

    print("\nFinal uncertainties:")
    print(f"  Position (N): {np.sqrt(state.error_cov[0, 0]):.2f} m")
    print(f"  Position (E): {np.sqrt(state.error_cov[1, 1]):.2f} m")
    print(f"  Position (D): {np.sqrt(state.error_cov[2, 2]):.2f} m")
    print(f"  Velocity (N): {np.sqrt(state.error_cov[3, 3]):.4f} m/s")


def gnss_outage_demo() -> None:
    """Demonstrate GNSS outage detection."""
    print("\n" + "=" * 60)
    print("7. GNSS OUTAGE DETECTION")
    print("=" * 60)

    np.random.seed(42)

    # GNSS outage detection uses chi-squared test on innovations
    # To detect measurement faults (spoofing, multipath, etc.)
    innovation_cov = np.diag([3.0**2, 3.0**2, 6.0**2])

    print("\nGNSS measurement fault detection using chi-squared test")
    print("  Innovation covariance: diag([9, 9, 36]) m^2")

    # Chi-squared threshold for 3 DOF (position), 95% confidence
    threshold_95 = 7.815  # chi2.ppf(0.95, 3)

    # Test with normal innovations
    print("\nNormal innovations (should pass):")
    for i in range(3):
        innovation = np.random.multivariate_normal(np.zeros(3), innovation_cov)
        fault = gnss_outage_detection(
            innovation, innovation_cov, threshold=threshold_95
        )
        nis = innovation @ np.linalg.solve(innovation_cov, innovation)
        print(f"  Sample {i+1}: NIS={nis:.2f}, Fault={fault}")

    # Test with biased innovations (simulating fault)
    print("\nBiased innovations (should detect fault):")
    fault_bias = np.array([15.0, -10.0, 20.0])  # Large bias
    for i in range(3):
        innovation = fault_bias + np.random.multivariate_normal(
            np.zeros(3), innovation_cov
        )
        fault = gnss_outage_detection(
            innovation, innovation_cov, threshold=threshold_95
        )
        nis = innovation @ np.linalg.solve(innovation_cov, innovation)
        print(f"  Sample {i+1}: NIS={nis:.2f}, Fault={fault}")

    print(
        "\nNote: NIS (Normalized Innovation Squared) should follow chi-squared distribution"
    )
    print(
        f"      with {len(innovation)} DOF. Threshold={threshold_95:.2f} (95% confidence)"
    )


def main() -> None:
    """Run INS/GNSS navigation demonstrations."""
    print("\nINS/GNSS Navigation Examples")
    print("=" * 60)
    print("Demonstrating pytcl navigation capabilities")

    ins_basics_demo()
    imu_processing_demo()
    coarse_alignment_demo()
    ins_mechanization_demo()
    gnss_geometry_demo()
    loose_coupling_demo()
    gnss_outage_demo()

    # Visualization
    visualize_navigation_trajectory()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


def visualize_navigation_trajectory() -> None:
    """Visualize INS navigation trajectory."""
    print("\nGenerating navigation trajectory visualization...")

    # Simulate a trajectory
    np.random.seed(42)
    n_steps = 200

    # Create a circular trajectory
    t = np.linspace(0, 2 * np.pi, n_steps)
    x = 1000 * np.cos(t)
    y = 1000 * np.sin(t)
    z = 50 * np.sin(2 * t)

    # Add noise
    x_noisy = x + 5 * np.random.randn(n_steps)
    y_noisy = y + 5 * np.random.randn(n_steps)
    z_noisy = z + 2 * np.random.randn(n_steps)

    # Create 3D trajectory plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color="blue", width=3),
            name="True Trajectory",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_noisy,
            y=y_noisy,
            z=z_noisy,
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.5),
            name="Measured Position",
        )
    )

    fig.update_layout(
        title="INS Navigation Trajectory: True vs Measured",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
        ),
        height=600,
        width=800,
    )

    fig.show()


if __name__ == "__main__":
    main()
