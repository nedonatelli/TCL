"""
Test all tutorial code snippets to verify they execute without errors.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_kalman_filtering():
    """Test Kalman Filtering Tutorial."""
    print("Testing: kalman_filtering.rst")

    import numpy as np

    from pytcl.dynamic_estimation import kf_predict, kf_update

    # System parameters
    dt = 0.1
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    q = 0.1
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0, 0],
            [dt**2 / 2, dt, 0, 0],
            [0, 0, dt**3 / 3, dt**2 / 2],
            [0, 0, dt**2 / 2, dt],
        ]
    )
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.eye(2) * 0.5

    # Initialize
    x = np.array([0.0, 1.0, 0.0, 0.5])
    P = np.eye(4) * 10.0

    # Generate data
    np.random.seed(42)
    x_true = np.array([0.0, 1.0, 0.0, 0.5])
    true_states, measurements = [], []
    for _ in range(100):
        true_states.append(x_true.copy())
        measurements.append(H @ x_true + np.random.multivariate_normal(np.zeros(2), R))
        x_true = F @ x_true + np.random.multivariate_normal(np.zeros(4), Q)

    # Filter
    estimates = []
    for z in measurements:
        pred = kf_predict(x, P, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x, P = upd.x, upd.P
        estimates.append(x.copy())

    # Results
    true_states = np.array(true_states)
    estimates = np.array(estimates)
    rmse = np.sqrt(
        np.mean(
            (true_states[:, 0] - estimates[:, 0]) ** 2
            + (true_states[:, 2] - estimates[:, 2]) ** 2
        )
    )
    print(f"  Position RMSE: {rmse:.3f}")
    print("  PASSED")


def test_nonlinear_filtering():
    """Test Nonlinear Filtering Tutorial."""
    print("\nTesting: nonlinear_filtering.rst")

    import numpy as np

    from pytcl.dynamic_estimation import (
        ckf_predict,
        ckf_update,
        ekf_predict,
        ekf_update,
        ukf_predict,
        ukf_update,
    )

    # Setup
    dt = 0.1
    n_steps = 100
    np.random.seed(42)

    def f(x):
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        return F @ x

    def F_jac(x):
        return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

    def h(x):
        r = np.sqrt(x[0] ** 2 + x[2] ** 2)
        theta = np.arctan2(x[2], x[0])
        return np.array([r, theta])

    def H_jac(x):
        r = np.sqrt(x[0] ** 2 + x[2] ** 2)
        return np.array([[x[0] / r, 0, x[2] / r, 0], [-x[2] / r**2, 0, x[0] / r**2, 0]])

    Q = np.diag([0.01, 0.1, 0.01, 0.1])
    R = np.diag([1.0, 0.02])

    # Generate truth and measurements
    x_true = np.array([100.0, -5.0, 50.0, 2.0])
    truth, measurements = [], []
    for _ in range(n_steps):
        truth.append(x_true.copy())
        z_true = h(x_true)
        z_noisy = z_true + np.random.multivariate_normal(np.zeros(2), R)
        measurements.append(z_noisy)
        x_true = f(x_true) + np.random.multivariate_normal(np.zeros(4), Q)

    # Run EKF - evaluate Jacobians at current state
    x_ekf = np.array([100.0, -5.0, 50.0, 2.0])
    P_ekf = np.diag([10.0, 1.0, 10.0, 1.0])
    ekf_est = []
    for z in measurements:
        F = F_jac(x_ekf)  # Evaluate Jacobian at current state
        pred = ekf_predict(x_ekf, P_ekf, f, F, Q)
        H = H_jac(pred.x)  # Evaluate Jacobian at predicted state
        upd = ekf_update(pred.x, pred.P, z, h, H, R)
        x_ekf, P_ekf = upd.x, upd.P
        ekf_est.append(x_ekf.copy())

    # Run UKF
    x_ukf = np.array([100.0, -5.0, 50.0, 2.0])
    P_ukf = np.diag([10.0, 1.0, 10.0, 1.0])
    ukf_est = []
    for z in measurements:
        pred = ukf_predict(x_ukf, P_ukf, f, Q)
        upd = ukf_update(pred.x, pred.P, z, h, R)
        x_ukf, P_ukf = upd.x, upd.P
        ukf_est.append(x_ukf.copy())

    # Run CKF
    x_ckf = np.array([100.0, -5.0, 50.0, 2.0])
    P_ckf = np.diag([10.0, 1.0, 10.0, 1.0])
    ckf_est = []
    for z in measurements:
        pred = ckf_predict(x_ckf, P_ckf, f, Q)
        upd = ckf_update(pred.x, pred.P, z, h, R)
        x_ckf, P_ckf = upd.x, upd.P
        ckf_est.append(x_ckf.copy())

    # Compare
    truth = np.array(truth)
    ekf_est = np.array(ekf_est)
    ukf_est = np.array(ukf_est)
    ckf_est = np.array(ckf_est)

    ekf_rmse = np.sqrt(
        np.mean((truth[:, 0] - ekf_est[:, 0]) ** 2 + (truth[:, 2] - ekf_est[:, 2]) ** 2)
    )
    ukf_rmse = np.sqrt(
        np.mean((truth[:, 0] - ukf_est[:, 0]) ** 2 + (truth[:, 2] - ukf_est[:, 2]) ** 2)
    )
    ckf_rmse = np.sqrt(
        np.mean((truth[:, 0] - ckf_est[:, 0]) ** 2 + (truth[:, 2] - ckf_est[:, 2]) ** 2)
    )

    print(f"  EKF Position RMSE: {ekf_rmse:.3f}")
    print(f"  UKF Position RMSE: {ukf_rmse:.3f}")
    print(f"  CKF Position RMSE: {ckf_rmse:.3f}")
    print("  PASSED")


def test_signal_processing():
    """Test Signal Processing Tutorial."""
    print("\nTesting: signal_processing.rst")

    import numpy as np

    from pytcl.mathematical_functions.signal_processing import (
        apply_filter,
        butter_design,
        cfar_ca,
        matched_filter,
    )
    from pytcl.mathematical_functions.transforms import power_spectrum

    # Filter design
    fs = 1000.0
    filt = butter_design(order=4, cutoff=50.0, fs=fs, btype="low")
    print(f"  Butterworth filter: {len(filt.b)} coefficients")

    # Apply filter
    t = np.linspace(0, 1, int(fs), endpoint=False)
    signal = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
    filtered = apply_filter(filt, signal)
    print(f"  Filtered signal length: {len(filtered)}")

    # CFAR detection
    np.random.seed(42)
    noise = np.abs(np.random.randn(200))
    noise[50] = 15.0
    noise[100] = 10.0
    result = cfar_ca(noise, guard_cells=2, ref_cells=8, pfa=1e-4)
    print(f"  CFAR detections: {len(result.detection_indices)}")

    # Matched filter
    pulse = np.sin(2 * np.pi * 100 * np.linspace(0, 0.01, 100))
    received = np.zeros(500)
    received[200:300] = pulse
    received += 0.5 * np.random.randn(500)
    mf_result = matched_filter(received, pulse)
    print(f"  Matched filter peak at: {mf_result.peak_index}")

    # Power spectrum
    ps = power_spectrum(signal, fs)
    print(f"  Power spectrum: {len(ps.frequencies)} bins")
    print("  PASSED")


def test_radar_detection():
    """Test Radar Detection Tutorial."""
    print("\nTesting: radar_detection.rst")

    import numpy as np

    from pytcl.mathematical_functions.signal_processing import (
        cfar_ca,
        cfar_go,
        cfar_os,
        cfar_so,
        detection_probability,
        threshold_factor,
    )

    np.random.seed(42)

    # Create range profile
    n_cells = 200
    noise = np.abs(np.random.randn(n_cells))
    noise[50] = 15.0
    noise[100] = 8.0
    noise[150] = 5.0

    # Threshold factor
    alpha = threshold_factor(pfa=1e-4, n_ref=16, method="ca")
    print(f"  CA-CFAR threshold factor: {alpha:.2f}")

    # Run different CFAR algorithms
    ca_result = cfar_ca(noise, guard_cells=2, ref_cells=8, pfa=1e-4)
    go_result = cfar_go(noise, guard_cells=2, ref_cells=8, pfa=1e-4)
    so_result = cfar_so(noise, guard_cells=2, ref_cells=8, pfa=1e-4)
    os_result = cfar_os(noise, guard_cells=2, ref_cells=8, pfa=1e-4, k=6)

    print(f"  CA-CFAR detections: {len(ca_result.detection_indices)}")
    print(f"  GO-CFAR detections: {len(go_result.detection_indices)}")
    print(f"  SO-CFAR detections: {len(so_result.detection_indices)}")
    print(f"  OS-CFAR detections: {len(os_result.detection_indices)}")

    # Detection probability
    pd = detection_probability(snr=15, pfa=1e-4, n_ref=16, method="ca")
    print(f"  Detection probability at SNR=15dB: {pd:.4f}")
    print("  PASSED")


def test_ins_gnss_integration():
    """Test INS/GNSS Integration Tutorial."""
    print("\nTesting: ins_gnss_integration.rst")

    import numpy as np

    from pytcl.coordinate_systems import euler2rotmat
    from pytcl.navigation import (
        ecef_to_enu,
        ecef_to_geodetic,
        geodetic_to_ecef,
    )

    # Test coordinate conversions
    lat = np.radians(40.0)
    lon = np.radians(-74.0)
    alt = 100.0

    ecef = geodetic_to_ecef(lat, lon, alt)
    lat2, lon2, alt2 = ecef_to_geodetic(ecef[0], ecef[1], ecef[2])
    print(f"  Geodetic roundtrip error: {abs(lat - lat2):.2e} rad")

    # ENU conversion
    ref_lat, ref_lon, ref_alt = lat, lon, 0.0
    enu = ecef_to_enu(ecef[0], ecef[1], ecef[2], ref_lat, ref_lon, ref_alt)
    print(f"  ENU position: [{enu[0]:.2f}, {enu[1]:.2f}, {enu[2]:.2f}]")

    # Rotation matrix
    roll, pitch, yaw = np.radians([5.0, 10.0, 45.0])
    R = euler2rotmat([yaw, pitch, roll], "ZYX")
    print(f"  Rotation matrix shape: {R.shape}")
    print("  PASSED")


def test_multi_target_tracking():
    """Test Multi-Target Tracking Tutorial."""
    print("\nTesting: multi_target_tracking.rst")

    import numpy as np

    from pytcl.assignment_algorithms import gnn_association, hungarian
    from pytcl.performance_evaluation import ospa
    from pytcl.trackers import MultiTargetTracker

    np.random.seed(42)

    # Test assignment algorithm - hungarian returns (row_ind, col_ind, cost)
    cost = np.array([[1.0, 10.0], [10.0, 2.0]])
    row_ind, col_ind, total_cost = hungarian(cost)
    print(f"  Hungarian assignment cost: {total_cost:.2f}")

    # Test GNN association - returns AssociationResult
    cost_matrix = np.array([[1.0, 5.0, 10.0], [10.0, 2.0, 5.0]])
    gnn_result = gnn_association(cost_matrix)
    # track_to_measurement[i] gives measurement index for track i
    assignments = [
        (i, gnn_result.track_to_measurement[i])
        for i in range(len(gnn_result.track_to_measurement))
    ]
    print(f"  GNN assignments: {assignments}")

    # Test tracker
    def F(dt):
        return np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]], dtype=np.float64
        )

    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)

    def Q(dt):
        q = 0.5
        return (
            np.array(
                [
                    [dt**4 / 4, dt**3 / 2, 0, 0],
                    [dt**3 / 2, dt**2, 0, 0],
                    [0, 0, dt**4 / 4, dt**3 / 2],
                    [0, 0, dt**3 / 2, dt**2],
                ]
            )
            * q**2
        )

    R = np.eye(2) * 2.0
    P0 = np.diag([10.0, 5.0, 10.0, 5.0])

    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=R,
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=P0,
    )

    # Run tracker with some measurements
    for _ in range(10):
        meas = [
            np.array([10.0 + np.random.randn(), 20.0 + np.random.randn()]),
            np.array([50.0 + np.random.randn(), 60.0 + np.random.randn()]),
        ]
        tracks = tracker.process(meas, dt=1.0)

    print(f"  Active tracks: {len(tracks)}")

    # Test OSPA
    true_targets = [np.array([10.0, 20.0]), np.array([50.0, 60.0])]
    estimates = [np.array([11.0, 21.0]), np.array([49.0, 59.0])]
    ospa_result = ospa(true_targets, estimates, c=10.0, p=2)
    print(f"  OSPA distance: {ospa_result.ospa:.3f}")
    print("  PASSED")


def main():
    """Run all tutorial tests."""
    print("=" * 60)
    print("Testing Tutorial Code Snippets")
    print("=" * 60)

    results = {}
    tests = [
        ("kalman_filtering", test_kalman_filtering),
        ("nonlinear_filtering", test_nonlinear_filtering),
        ("signal_processing", test_signal_processing),
        ("radar_detection", test_radar_detection),
        ("ins_gnss_integration", test_ins_gnss_integration),
        ("multi_target_tracking", test_multi_target_tracking),
    ]

    for name, test_func in tests:
        try:
            test_func()
            results[name] = "PASSED"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v == "PASSED")
    failed = len(results) - passed

    for name, result in results.items():
        status = "PASS" if result == "PASSED" else "FAIL"
        print(f"  [{status}] {name}")
        if result != "PASSED":
            print(f"         {result}")

    print(f"\nTotal: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
