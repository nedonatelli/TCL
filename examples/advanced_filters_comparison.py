"""Advanced filters comparison demonstration.

Demonstrates three advanced filtering techniques:
1. Constrained Extended Kalman Filter (CEKF): Enforces state constraints
2. Gaussian Sum Filter (GSF): Models multi-modal posterior distributions
3. Rao-Blackwellized Particle Filter (RBPF): Combines particles with Kalman filters

Scenario: Nonlinear target tracking with constraints on valid state region.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from pytcl.dynamic_estimation.gaussian_sum_filter import (
    GaussianComponent,
    GaussianSumFilter,
)
from pytcl.dynamic_estimation.kalman.constrained import (
    ConstrainedEKF,
    ConstraintFunction,
)
from pytcl.dynamic_estimation.rbpf import RBPFFilter


class TargetTrackingScenario:
    """Nonlinear target tracking scenario.

    Target moves in 2D with nonlinear dynamics. Measurements are range and
    bearing from a fixed observer.
    """

    def __init__(self, seed: int = 42):
        """Initialize scenario.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)

        # State: [x, y, vx, vy] (position and velocity in Cartesian coords)
        self.state_dim = 4
        self.measurement_dim = 2  # range and bearing

        # System matrices
        self.dt = 0.1
        self.F = np.array(
            [
                [1, 0, self.dt, 0],
                [0, 1, 0, self.dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        self.Q = np.diag([0, 0, 0.001, 0.001])  # Process noise

        # Measurement observer position
        self.observer = np.array([0.0, 0.0])

        # Measurement noise
        self.R = np.diag([0.1, 0.01])  # range error, bearing error (radians)

        # Initial state
        self.x0 = np.array([10.0, 10.0, -1.0, -0.5])
        self.P0 = np.diag([1.0, 1.0, 0.5, 0.5])

    def f(self, x: np.ndarray) -> np.ndarray:
        """Nonlinear state transition with friction.

        Parameters
        ----------
        x : ndarray
            State vector [x, y, vx, vy]

        Returns
        -------
        ndarray
            Next state with velocity friction
        """
        x_next = self.F @ x
        # Add friction to velocity
        x_next[2] *= 0.95
        x_next[3] *= 0.95
        return x_next

    def h(self, x: np.ndarray) -> np.ndarray:
        """Measurement function: range and bearing.

        Parameters
        ----------
        x : ndarray
            State vector [x, y, vx, vy]

        Returns
        -------
        ndarray
            Measurement [range, bearing]
        """
        pos = x[:2]
        delta = pos - self.observer

        # Range
        r = np.linalg.norm(delta)

        # Bearing (angle from East)
        bearing = np.arctan2(delta[1], delta[0])

        return np.array([r, bearing])

    def h_jacobian(self, x: np.ndarray) -> np.ndarray:
        """Jacobian of measurement function.

        Parameters
        ----------
        x : ndarray
            State vector

        Returns
        -------
        ndarray
            Jacobian dh/dx
        """
        pos = x[:2]
        delta = pos - self.observer
        r = np.linalg.norm(delta)

        if r < 0.01:
            # Avoid singularity
            return np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ]
            )

        H = np.zeros((2, 4))

        # dr/dx = delta[0] / r
        H[0, 0] = delta[0] / r
        H[0, 1] = delta[1] / r

        # dbearing/dx = -delta[1] / r^2, dbearing/dy = delta[0] / r^2
        H[1, 0] = -delta[1] / r**2
        H[1, 1] = delta[0] / r**2

        return H

    def generate_trajectory(self, steps: int = 50):
        """Generate synthetic true trajectory and measurements.

        Parameters
        ----------
        steps : int
            Number of time steps

        Returns
        -------
        x_true : ndarray (steps, 4)
            True state trajectory
        measurements : ndarray (steps, 2)
            Noisy range/bearing measurements
        """
        x_true = np.zeros((steps, 4))
        measurements = np.zeros((steps, 2))

        x_true[0] = self.x0

        for k in range(1, steps):
            # True dynamics
            x_true[k] = self.f(x_true[k - 1])
            x_true[k] += np.random.multivariate_normal(np.zeros(4), self.Q)

            # Measurement
            z_true = self.h(x_true[k])
            measurements[k] = z_true + np.random.multivariate_normal(
                np.zeros(2), self.R
            )

        return x_true, measurements


def run_cekf_filter(
    scenario: TargetTrackingScenario,
    measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Constrained EKF with position constraint.

    Parameters
    ----------
    scenario : TargetTrackingScenario
        Tracking scenario
    measurements : ndarray
        Measurements

    Returns
    -------
    x_est : ndarray
        State estimates
    P_est : ndarray
        Covariance estimates
    """
    cekf = ConstrainedEKF()

    # Add constraint: target must stay within region
    # Constraint: (x-5)^2 + (y-5)^2 <= 100 (circle centered at (5,5) with radius 10)
    def g_circle(x):
        # Negative means inside region
        center = np.array([5.0, 5.0])
        radius = 10.0
        return (x[0] - center[0]) ** 2 + (x[1] - center[1]) ** 2 - radius**2

    # Jacobian
    def G_circle(x):
        center = np.array([5.0, 5.0])
        jac = np.zeros((1, 4))
        jac[0, 0] = 2 * (x[0] - center[0])
        jac[0, 1] = 2 * (x[1] - center[1])
        return jac

    cekf.add_constraint(ConstraintFunction(g_circle, G=G_circle))

    # Initialize
    x = scenario.x0.copy()
    P = scenario.P0.copy()

    x_est = np.zeros((len(measurements), 4))
    P_est = np.zeros((len(measurements), 4, 4))

    for k, z in enumerate(measurements):
        # Predict
        def f_wrapper(x_):
            return scenario.f(x_)

        pred = cekf.predict(x, P, f_wrapper, scenario.F, scenario.Q)
        x = pred.x
        P = pred.P

        # Update
        def h_wrapper(x_):
            return scenario.h(x_)

        upd = cekf.update(x, P, z, h_wrapper, scenario.h_jacobian(x), scenario.R)
        x = upd.x
        P = upd.P

        x_est[k] = x
        P_est[k] = P

    return x_est, P_est


def run_gsf_filter(
    scenario: TargetTrackingScenario,
    measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Gaussian Sum Filter.

    Parameters
    ----------
    scenario : TargetTrackingScenario
        Tracking scenario
    measurements : ndarray
        Measurements

    Returns
    -------
    x_est : ndarray
        State estimates
    P_est : ndarray
        Covariance estimates
    """
    gsf = GaussianSumFilter(max_components=5)

    # Initialize with multiple modes
    gsf.initialize(scenario.x0, scenario.P0, num_components=3)

    x_est = np.zeros((len(measurements), 4))
    P_est = np.zeros((len(measurements), 4, 4))

    for k, z in enumerate(measurements):
        # Predict
        def f_wrapper(x_):
            return scenario.f(x_)

        gsf.predict(f_wrapper, scenario.F, scenario.Q)

        # Update
        def h_wrapper(x_):
            return scenario.h(x_)

        gsf.update(z, h_wrapper, scenario.h_jacobian(None), scenario.R)

        # Estimate
        x, P = gsf.estimate()
        x_est[k] = x
        P_est[k] = P

    return x_est, P_est


def run_rbpf_filter(
    scenario: TargetTrackingScenario,
    measurements: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run Rao-Blackwellized Particle Filter.

    Parameters
    ----------
    scenario : TargetTrackingScenario
        Tracking scenario
    measurements : ndarray
        Measurements

    Returns
    -------
    x_est : ndarray
        State estimates
    P_est : ndarray
        Covariance estimates
    """
    rbpf = RBPFFilter(max_particles=50)

    # Partition: nonlinear (position), linear (velocity)
    y0 = scenario.x0[:2]  # position
    x0 = scenario.x0[2:]  # velocity
    P0 = scenario.P0[2:, 2:]

    rbpf.initialize(y0, x0, P0, num_particles=30)

    x_est = np.zeros((len(measurements), 4))
    P_est = np.zeros((len(measurements), 4, 4))

    for k, z in enumerate(measurements):
        # Predict nonlinear: position dynamics
        def g(y):
            return y + scenario.dt * np.array(
                [
                    np.random.normal(0, 0.1),  # noise
                    np.random.normal(0, 0.1),
                ]
            )

        G = np.eye(2)
        Qy = np.eye(2) * 0.001

        # Predict linear: velocity dynamics
        F_v = np.eye(2) * 0.95  # friction
        Qx = np.eye(2) * 0.0001

        def f_linear(v, y):
            # Next position depends on current velocity
            # For RBPF, we need x[k+1] = f(x[k], y[k])
            return F_v @ v

        rbpf.predict(g, G, Qy, f_linear, F_v, Qx)

        # Update
        def h_rbpf(v, y):
            # Full state from position and velocity
            x_full = np.concatenate([y, v])
            return scenario.h(x_full)

        def H_rbpf_func(y):
            # For measurement jacobian, need position
            return scenario.h_jacobian(np.concatenate([y, np.zeros(2)]))

        # Get H for first particle
        if rbpf.particles:
            H = H_rbpf_func(rbpf.particles[0].y)
        else:
            H = scenario.h_jacobian(scenario.x0)

        rbpf.update(z, h_rbpf, H, scenario.R)

        # Estimate
        y_est, v_est, P_v = rbpf.estimate()
        x_est[k] = np.concatenate([y_est, v_est])

        # Full covariance (approximate)
        P_est[k, :2, :2] = np.eye(2) * 0.1
        P_est[k, 2:, 2:] = P_v
        P_est[k, :2, 2:] = 0
        P_est[k, 2:, :2] = 0

    return x_est, P_est


def main():
    """Run comparison and generate plots."""
    # Create scenario
    scenario = TargetTrackingScenario()

    # Generate data
    print("Generating synthetic trajectory...")
    x_true, measurements = scenario.generate_trajectory(steps=50)

    # Run filters
    print("Running CEKF...")
    x_cekf, P_cekf = run_cekf_filter(scenario, measurements)

    print("Running GSF...")
    x_gsf, P_gsf = run_gsf_filter(scenario, measurements)

    print("Running RBPF...")
    x_rbpf, P_rbpf = run_rbpf_filter(scenario, measurements)

    # Compute errors
    err_cekf = np.linalg.norm(x_cekf - x_true, axis=1)
    err_gsf = np.linalg.norm(x_gsf - x_true, axis=1)
    err_rbpf = np.linalg.norm(x_rbpf - x_true, axis=1)

    # Compute average uncertainties (trace of covariance)
    unc_cekf = np.array([np.trace(P_cekf[k]) for k in range(len(x_true))])
    unc_gsf = np.array([np.trace(P_gsf[k]) for k in range(len(x_true))])
    unc_rbpf = np.array([np.trace(P_rbpf[k]) for k in range(len(x_true))])

    # Print statistics
    print("\n" + "=" * 60)
    print("FILTER COMPARISON RESULTS")
    print("=" * 60)
    print(
        f"CEKF - Mean Error: {np.mean(err_cekf):.4f}, Mean Uncertainty: {np.mean(unc_cekf):.4f}"
    )
    print(
        f"GSF  - Mean Error: {np.mean(err_gsf):.4f}, Mean Uncertainty: {np.mean(unc_gsf):.4f}"
    )
    print(
        f"RBPF - Mean Error: {np.mean(err_rbpf):.4f}, Mean Uncertainty: {np.mean(unc_rbpf):.4f}"
    )
    print("=" * 60)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Trajectories
    ax = axes[0, 0]
    ax.plot(x_true[:, 0], x_true[:, 1], "k-", linewidth=2, label="True")
    ax.plot(x_cekf[:, 0], x_cekf[:, 1], "b--", alpha=0.7, label="CEKF")
    ax.plot(x_gsf[:, 0], x_gsf[:, 1], "g--", alpha=0.7, label="GSF")
    ax.plot(x_rbpf[:, 0], x_rbpf[:, 1], "r--", alpha=0.7, label="RBPF")
    circle = plt.Circle(
        (5, 5), 10, fill=False, color="gray", linestyle=":", label="Constraint"
    )
    ax.add_patch(circle)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Estimated Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    # Plot 2: Position errors
    ax = axes[0, 1]
    time = np.arange(len(x_true))
    ax.plot(time, err_cekf, "b-", label="CEKF", linewidth=2)
    ax.plot(time, err_gsf, "g-", label="GSF", linewidth=2)
    ax.plot(time, err_rbpf, "r-", label="RBPF", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Position Error (Norm)")
    ax.set_title("State Estimation Error")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Uncertainty estimates
    ax = axes[1, 0]
    ax.plot(time, unc_cekf, "b-", label="CEKF", linewidth=2)
    ax.plot(time, unc_gsf, "g-", label="GSF", linewidth=2)
    ax.plot(time, unc_rbpf, "r-", label="RBPF", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Covariance Trace")
    ax.set_title("Estimated Uncertainty (Lower is Better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Error distribution
    ax = axes[1, 1]
    ax.boxplot(
        [err_cekf, err_gsf, err_rbpf],
        labels=["CEKF", "GSF", "RBPF"],
    )
    ax.set_ylabel("Position Error")
    ax.set_title("Error Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    # Save output to examples/output directory instead of root
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "advanced_filters_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to '{output_path}'")

    plt.show()


if __name__ == "__main__":
    main()
