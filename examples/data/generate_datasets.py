#!/usr/bin/env python
"""Generate sample datasets for pyTCL tutorials and examples.

This script creates synthetic datasets for use in Jupyter notebooks
and example scripts. All generated data is reproducible via fixed seeds.

Usage:
    python generate_datasets.py

Output files:
    - tracking_scenarios.npz: Multi-target tracking scenarios
    - navigation_trajectory.npz: INS/GNSS navigation data
    - satellite_tle_samples.txt: Sample TLE data
"""

from pathlib import Path

import numpy as np

# Reproducibility
np.random.seed(42)

OUTPUT_DIR = Path(__file__).parent


def generate_tracking_scenarios() -> None:
    """Generate multi-target tracking scenarios."""
    print("Generating tracking scenarios...")

    data = {}

    # Scenario 1: 3 targets, linear motion, low clutter
    n_targets = 3
    n_steps = 100
    dt = 1.0

    # State: [x, vx, y, vy]
    states_1 = np.zeros((n_targets, n_steps, 4))
    # Initial states
    states_1[0, 0] = [0, 10, 0, 5]
    states_1[1, 0] = [100, -5, 50, 8]
    states_1[2, 0] = [50, 3, 100, -6]

    # Propagate with constant velocity
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    for t in range(1, n_steps):
        for i in range(n_targets):
            states_1[i, t] = F @ states_1[i, t - 1]
            # Add small process noise
            states_1[i, t, 0] += np.random.normal(0, 0.5)
            states_1[i, t, 2] += np.random.normal(0, 0.5)

    # Generate measurements (position only)
    meas_noise = 5.0
    measurements_1 = []
    for t in range(n_steps):
        meas_t = []
        for i in range(n_targets):
            # Detection probability 0.95
            if np.random.rand() < 0.95:
                pos = states_1[i, t, [0, 2]]
                meas = pos + np.random.normal(0, meas_noise, 2)
                meas_t.append(meas)
        # Add 2 clutter measurements per frame
        for _ in range(2):
            clutter = np.random.uniform(-50, 200, 2)
            meas_t.append(clutter)
        measurements_1.append(np.array(meas_t))

    data["scenario_1_states"] = states_1
    data["scenario_1_measurements"] = np.array(measurements_1, dtype=object)
    data["scenario_1_timestamps"] = np.arange(n_steps) * dt

    # Scenario 2: 5 targets, maneuvering, crossing paths
    n_targets = 5
    n_steps = 150

    states_2 = np.zeros((n_targets, n_steps, 4))
    states_2[0, 0] = [0, 8, 50, 0]
    states_2[1, 0] = [100, -6, 0, 6]
    states_2[2, 0] = [50, 0, 0, 10]
    states_2[3, 0] = [0, 5, 100, -3]
    states_2[4, 0] = [80, -4, 80, -4]

    for t in range(1, n_steps):
        for i in range(n_targets):
            # Add coordinated turn for some targets
            if i in [1, 3] and 40 < t < 80:
                omega = 0.02  # Turn rate
                v = np.sqrt(states_2[i, t - 1, 1] ** 2 + states_2[i, t - 1, 3] ** 2)
                heading = np.arctan2(states_2[i, t - 1, 3], states_2[i, t - 1, 1])
                heading += omega * dt
                states_2[i, t, 0] = states_2[i, t - 1, 0] + v * np.cos(heading) * dt
                states_2[i, t, 1] = v * np.cos(heading)
                states_2[i, t, 2] = states_2[i, t - 1, 2] + v * np.sin(heading) * dt
                states_2[i, t, 3] = v * np.sin(heading)
            else:
                states_2[i, t] = F @ states_2[i, t - 1]
            states_2[i, t, 0] += np.random.normal(0, 0.3)
            states_2[i, t, 2] += np.random.normal(0, 0.3)

    measurements_2 = []
    for t in range(n_steps):
        meas_t = []
        for i in range(n_targets):
            if np.random.rand() < 0.90:
                pos = states_2[i, t, [0, 2]]
                meas = pos + np.random.normal(0, meas_noise, 2)
                meas_t.append(meas)
        for _ in range(3):
            clutter = np.random.uniform(-50, 200, 2)
            meas_t.append(clutter)
        measurements_2.append(np.array(meas_t))

    data["scenario_2_states"] = states_2
    data["scenario_2_measurements"] = np.array(measurements_2, dtype=object)
    data["scenario_2_timestamps"] = np.arange(n_steps) * dt

    # Scenario 3: 10 targets, dense clutter
    n_targets = 10
    n_steps = 200

    states_3 = np.zeros((n_targets, n_steps, 4))
    for i in range(n_targets):
        states_3[i, 0] = [
            np.random.uniform(0, 100),
            np.random.uniform(-5, 5),
            np.random.uniform(0, 100),
            np.random.uniform(-5, 5),
        ]

    for t in range(1, n_steps):
        for i in range(n_targets):
            states_3[i, t] = F @ states_3[i, t - 1]
            states_3[i, t, 0] += np.random.normal(0, 0.5)
            states_3[i, t, 2] += np.random.normal(0, 0.5)

    measurements_3 = []
    for t in range(n_steps):
        meas_t = []
        for i in range(n_targets):
            if np.random.rand() < 0.85:
                pos = states_3[i, t, [0, 2]]
                meas = pos + np.random.normal(0, meas_noise, 2)
                meas_t.append(meas)
        # Dense clutter: 10 per frame
        for _ in range(10):
            clutter = np.random.uniform(-20, 150, 2)
            meas_t.append(clutter)
        measurements_3.append(np.array(meas_t))

    data["scenario_3_states"] = states_3
    data["scenario_3_measurements"] = np.array(measurements_3, dtype=object)
    data["scenario_3_timestamps"] = np.arange(n_steps) * dt

    # Save
    np.savez_compressed(OUTPUT_DIR / "tracking_scenarios.npz", **data)
    print(f"  Saved: {OUTPUT_DIR / 'tracking_scenarios.npz'}")


def generate_navigation_trajectory() -> None:
    """Generate INS/GNSS navigation trajectory."""
    print("Generating navigation trajectory...")

    # Simulate 5-minute flight
    duration = 300.0  # seconds
    imu_rate = 100.0  # Hz
    gnss_rate = 1.0  # Hz

    n_imu = int(duration * imu_rate)
    n_gnss = int(duration * gnss_rate)

    imu_time = np.linspace(0, duration, n_imu)
    gnss_time = np.linspace(0, duration, n_gnss)

    # Generate flight profile (figure-8 pattern with altitude changes)
    # Position in NED frame (North, East, Down relative to start)
    omega = 2 * np.pi / 120  # Complete figure-8 in 2 minutes

    true_pos = np.zeros((n_imu, 3))
    true_vel = np.zeros((n_imu, 3))

    for i, t in enumerate(imu_time):
        # Figure-8 in horizontal plane
        true_pos[i, 0] = 500 * np.sin(omega * t)  # North
        true_pos[i, 1] = 250 * np.sin(2 * omega * t)  # East
        true_pos[i, 2] = -1000 - 50 * np.sin(0.5 * omega * t)  # Down (negative = up)

        # Velocities (analytical derivatives)
        true_vel[i, 0] = 500 * omega * np.cos(omega * t)
        true_vel[i, 1] = 500 * omega * np.cos(2 * omega * t)
        true_vel[i, 2] = -25 * omega * np.cos(0.5 * omega * t)

    # Generate IMU measurements
    # Accelerometer: specific force in body frame
    # For simplicity, assume level flight with body = NED
    g = 9.81

    accel = np.zeros((n_imu, 3))
    gyro = np.zeros((n_imu, 3))

    # Numerical differentiation for acceleration
    accel[1:, 0] = np.diff(true_vel[:, 0]) * imu_rate
    accel[1:, 1] = np.diff(true_vel[:, 1]) * imu_rate
    accel[1:, 2] = np.diff(true_vel[:, 2]) * imu_rate + g  # Add gravity

    # Add IMU noise
    accel_noise = 0.01  # m/s^2
    gyro_noise = 0.001  # rad/s
    accel_bias = np.array([0.02, -0.01, 0.015])
    gyro_bias = np.array([0.0001, -0.0002, 0.00015])

    accel += accel_bias + np.random.normal(0, accel_noise, (n_imu, 3))

    # Gyro: approximate rotation rates from trajectory curvature
    gyro[:, 2] = omega * (1 + np.cos(2 * omega * imu_time))  # Yaw rate
    gyro += gyro_bias + np.random.normal(0, gyro_noise, (n_imu, 3))

    # GNSS measurements with outages
    gnss_pos = np.zeros((n_gnss, 3))
    gnss_noise = 2.5  # meters

    # Convert NED to LLA (approximate, starting from lat=37, lon=-122)
    lat0, lon0, alt0 = 37.0, -122.0, 1000.0  # degrees, degrees, meters

    for i, t in enumerate(gnss_time):
        idx = int(t * imu_rate)
        if idx >= n_imu:
            idx = n_imu - 1

        # NED to LLA (approximate)
        lat = lat0 + true_pos[idx, 0] / 111000  # ~111km per degree
        lon = lon0 + true_pos[idx, 1] / (111000 * np.cos(np.radians(lat0)))
        alt = alt0 - true_pos[idx, 2]  # Down to altitude

        # Add noise
        gnss_pos[i, 0] = lat + np.random.normal(0, gnss_noise / 111000)
        gnss_pos[i, 1] = lon + np.random.normal(
            0, gnss_noise / (111000 * np.cos(np.radians(lat0)))
        )
        gnss_pos[i, 2] = alt + np.random.normal(0, gnss_noise)

    # Create GNSS outages (mask certain time periods)
    gnss_available = np.ones(n_gnss, dtype=bool)
    # Outage from t=100 to t=130
    gnss_available[(gnss_time >= 100) & (gnss_time <= 130)] = False
    # Outage from t=200 to t=210
    gnss_available[(gnss_time >= 200) & (gnss_time <= 210)] = False

    # Save
    np.savez_compressed(
        OUTPUT_DIR / "navigation_trajectory.npz",
        imu_time=imu_time,
        accel=accel,
        gyro=gyro,
        gnss_time=gnss_time[gnss_available],
        gnss_pos=gnss_pos[gnss_available],
        gnss_available=gnss_available,
        true_pos=true_pos,
        true_vel=true_vel,
        start_lla=np.array([lat0, lon0, alt0]),
        accel_bias=accel_bias,
        gyro_bias=gyro_bias,
    )
    print(f"  Saved: {OUTPUT_DIR / 'navigation_trajectory.npz'}")


def generate_tle_samples() -> None:
    """Generate sample TLE data."""
    print("Generating TLE samples...")

    # Real TLE data (public domain, from Celestrak)
    tle_data = """# Sample TLE Data for pyTCL Examples
# Source: Celestrak (public domain)
# Note: These TLEs are for educational purposes only and may be outdated

# International Space Station (ISS)
ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9003
2 25544  51.6400 247.4627 0006703 130.5360 325.0288 15.49653601428391

# GPS BIIR-2 (PRN 13)
GPS BIIR-2 (PRN 13)
1 24876U 97035A   24001.50000000  .00000038  00000-0  00000+0 0  9992
2 24876  55.7162  37.8143 0057387  56.8631 303.7342  2.00562073193687

# GPS BIIR-3 (PRN 11)
GPS BIIR-3 (PRN 11)
1 25933U 99055A   24001.50000000 -.00000048  00000-0  00000+0 0  9997
2 25933  51.2877 155.0871 0158247 271.2576  87.1148  2.00571158177851

# NOAA 19 (Polar orbiting weather satellite)
NOAA 19
1 33591U 09005A   24001.50000000  .00000092  00000-0  76561-4 0  9999
2 33591  99.1917 321.7563 0014106 138.2891 221.9391 14.12447823775938

# Landsat 8
LANDSAT 8
1 39084U 13008A   24001.50000000  .00000597  00000-0  13755-3 0  9998
2 39084  98.2104  83.0987 0001257  95.9880 264.1432 14.57112037595437

# Sentinel-2A (Earth observation)
SENTINEL-2A
1 40697U 15028A   24001.50000000  .00000014  00000-0  21165-4 0  9991
2 40697  98.5693 282.4961 0001082  90.7987 269.3331 14.30817997454312

# STARLINK-1007
STARLINK-1007
1 44713U 19074A   24001.50000000  .00008893  00000-0  57988-3 0  9994
2 44713  53.0538 256.1821 0001551  80.6598 279.4544 15.06409251234589
"""

    with open(OUTPUT_DIR / "satellite_tle_samples.txt", "w") as f:
        f.write(tle_data)
    print(f"  Saved: {OUTPUT_DIR / 'satellite_tle_samples.txt'}")


def main() -> None:
    """Generate all sample datasets."""
    print("=" * 60)
    print("pyTCL Sample Dataset Generator")
    print("=" * 60)
    print()

    generate_tracking_scenarios()
    generate_navigation_trajectory()
    generate_tle_samples()

    print()
    print("=" * 60)
    print("All datasets generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
