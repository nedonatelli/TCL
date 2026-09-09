"""
Bistatic Measurement Jacobians and Hessians

Demonstrates the v2.10.0 measurement-derivative suite: the analytic
Jacobians an EKF needs to linearize bistatic radar measurements, the
converted-measurement form used to transform measurement covariances,
the second-order (Hessian) terms, and a sensor time-allocation coda
with the interval-scheduling algorithms.

Key scenarios:
1. EKF linearization of a bistatic [range; az; el; range-rate] radar
2. Converted-measurement covariance via the inverse Jacobian
3. Second-order correction size from the measurement Hessian
4. Dwell scheduling with the weighted-interval dynamic program
"""

import numpy as np

from pytcl.coordinate_systems.hessians import calc_spher_hessian
from pytcl.coordinate_systems.jacobians import (
    calc_spher_conv_jacob,
    calc_spher_rr_jacob,
    numerical_jacobian,
)
from pytcl.scheduling import schedule_intervals, schedule_weighted_intervals

L_TX = np.array([-12e3, 8e3, 5e3, 5.0, 3.0, -2.0])  # moving transmitter state
L_RX = np.array([4e3, -6e3, 12.0, -4.0, 2.0, 6.0])  # moving receiver state


def ekf_linearization_demo() -> None:
    """The Jacobian an EKF uses for a bistatic radar measurement."""
    print("\n" + "=" * 60)
    print("1. EKF LINEARIZATION OF A BISTATIC RADAR")
    print("=" * 60)

    x = np.array([-3e3, -2e3, -1e3, 30.0, -20.0, 10.0])  # target state
    jac = calc_spher_rr_jacob(x, 0, False, L_TX, L_RX)
    print("\nTarget state: position (-3, -2, -1) km, velocity (30, -20, 10) m/s")
    print("Measurement: [bistatic range; azimuth; elevation; range rate]")
    print(f"Jacobian shape: {jac.shape} (4 measurements x 6 state components)")
    print(f"  d(range)/d(position)      = {np.array2string(jac[0, :3], precision=3)}")
    print(f"  d(range rate)/d(velocity) = {np.array2string(jac[3, 3:], precision=3)}")

    # The analytic Jacobian matches central differences of the
    # measurement function -- this is what the validation suite pins.
    def meas_range(p):
        return [np.linalg.norm(p - L_RX[:3]) + np.linalg.norm(p - L_TX[:3])]

    num = numerical_jacobian(meas_range, x[:3])
    err = np.abs(num - jac[0, :3]).max()
    print(f"  |analytic - numeric| on the range row: {err:.2e}")


def converted_measurement_demo() -> None:
    """Transform a measurement covariance to Cartesian."""
    print("\n" + "=" * 60)
    print("2. CONVERTED-MEASUREMENT COVARIANCE")
    print("=" * 60)

    z = np.array([9e3, 0.5, 0.2])  # a monostatic [r; az; el] measurement
    r_meas = np.diag([25.0, 1e-6, 1e-6])  # sensor noise covariance
    jac = calc_spher_conv_jacob(z, 0)  # evaluated at the measurement itself
    j_inv = np.linalg.inv(jac)
    p_cart = j_inv @ r_meas @ j_inv.T
    sig = np.sqrt(np.diag(p_cart))
    print("\nMeasurement (9 km, 28.6 deg az, 11.5 deg el),")
    print("noise: 5 m range, 1 mrad angles")
    print(f"Cartesian 1-sigma: ({sig[0]:.1f}, {sig[1]:.1f}, {sig[2]:.1f}) m")
    print("(cross-range uncertainty dominates at long range)")


def hessian_demo() -> None:
    """How curved is the measurement function across the uncertainty?"""
    print("\n" + "=" * 60)
    print("3. SECOND-ORDER (HESSIAN) CORRECTION SIZE")
    print("=" * 60)

    x = np.array([-3e3, -2e3, -1e3])
    hess = calc_spher_hessian(x, 0, True)
    sigma = 100.0  # 100 m position uncertainty
    # Quadratic-term scale: 0.5 * sigma^2 * ||H|| per measurement row.
    print(f"\nWith {sigma:.0f} m position uncertainty at 3.7 km range:")
    for name, k in (("range", 0), ("azimuth", 1), ("elevation", 2)):
        scale = 0.5 * sigma**2 * np.abs(hess[:, :, k]).max()
        unit = "m" if k == 0 else "rad"
        print(f"  {name:9s}: second-order term ~ {scale:.2e} {unit}")
    print("(when these rival the sensor noise, use the second-order EKF")
    print(" or the cubature conversions instead of a first-order EKF)")


def scheduling_demo() -> None:
    """Allocate a sensor's time among candidate dwells."""
    print("\n" + "=" * 60)
    print("4. DWELL SCHEDULING")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n = 12
    starts = rng.uniform(0, 10, n)
    intervals = np.vstack([starts, starts + rng.uniform(0.5, 3.0, n)])
    priorities = rng.uniform(1.0, 10.0, n)

    most = schedule_intervals(intervals)
    weight, best = schedule_weighted_intervals(intervals, priorities)
    print(f"\n{n} candidate dwells over a 13 s window:")
    print(f"  Most dwells (greedy):      {len(most)} scheduled -> {sorted(most)}")
    print(
        f"  Max priority (weighted DP): {len(best)} scheduled -> "
        f"{sorted(best)}, total priority {weight:.1f}"
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Bistatic Measurement Jacobians and Hessians")
    print("=" * 60)
    ekf_linearization_demo()
    converted_measurement_demo()
    hessian_demo()
    scheduling_demo()
    print("\nAll demonstrations complete!")
