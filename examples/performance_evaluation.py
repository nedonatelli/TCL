"""
Performance Evaluation Example.

This example demonstrates:
1. OSPA (Optimal Sub-Pattern Assignment) metric for multi-target tracking
2. NEES (Normalized Estimation Error Squared) for filter consistency
3. NIS (Normalized Innovation Squared) for measurement consistency
4. Monte Carlo simulation for tracker evaluation
5. Track quality metrics (purity, fragmentation)

Run with: python examples/performance_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List  # noqa: E402

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from pytcl.dynamic_estimation import (  # noqa: E402
    kf_predict,
    kf_update,
)
from pytcl.dynamic_models import (  # noqa: E402
    f_constant_velocity,
    q_constant_velocity,
)
from pytcl.performance_evaluation import (  # noqa: E402; Track metrics
    ospa,
)


def ospa_demo() -> None:
    """Demonstrate OSPA metric for multi-target tracking evaluation."""
    print("=" * 60)
    print("1. OSPA METRIC FOR MULTI-TARGET TRACKING")
    print("=" * 60)

    print("\nOSPA (Optimal Sub-Pattern Assignment) measures the distance")
    print("between two sets of targets, accounting for:")
    print("  - Localization errors (how far are matched targets?)")
    print("  - Cardinality errors (how many targets are missed/false?)")

    # Ground truth targets (2D positions)
    truth = np.array(
        [
            [10.0, 20.0],
            [50.0, 30.0],
            [80.0, 60.0],
        ]
    )

    print(f"\nGround truth: {len(truth)} targets")
    for i, pos in enumerate(truth):
        print(f"  Target {i+1}: ({pos[0]:.1f}, {pos[1]:.1f})")

    # Different estimate scenarios
    scenarios = [
        ("Perfect tracking", np.array([[10.0, 20.0], [50.0, 30.0], [80.0, 60.0]])),
        ("Small errors", np.array([[12.0, 22.0], [48.0, 32.0], [82.0, 58.0]])),
        ("One missed target", np.array([[10.0, 20.0], [50.0, 30.0]])),
        (
            "One false target",
            np.array([[10.0, 20.0], [50.0, 30.0], [80.0, 60.0], [100.0, 100.0]]),
        ),
        ("Wrong positions", np.array([[0.0, 0.0], [100.0, 100.0], [50.0, 50.0]])),
    ]

    # OSPA parameters
    c = 50.0  # Cutoff distance (max penalty per target)
    p = 2  # Order parameter

    print(f"\nOSPA parameters: c={c}, p={p}")
    print("-" * 60)

    for name, estimates in scenarios:
        result = ospa(truth, estimates, c=c, p=p)

        print(f"\n{name}:")
        print(f"  Estimates: {len(estimates)} targets")
        print(f"  OSPA distance: {result.ospa:.2f}")
        print(f"  Localization: {result.localization:.2f}")
        print(f"  Cardinality:  {result.cardinality:.2f}")


def nees_consistency_demo() -> None:
    """Demonstrate NEES for filter consistency evaluation."""
    print("\n" + "=" * 60)
    print("2. NEES FOR FILTER CONSISTENCY")
    print("=" * 60)

    print("\nNEES (Normalized Estimation Error Squared) tests if the filter's")
    print("covariance estimate matches the actual estimation errors.")
    print("For a consistent filter, NEES should average to the state dimension.")

    np.random.seed(42)

    # Simulate a simple 2D tracking scenario
    n_steps = 100
    dt = 1.0

    # System matrices
    F = f_constant_velocity(dt, 2)  # 4-state CV model
    Q = q_constant_velocity(dt, 0.1, 2)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])  # Measure position
    R = np.diag([5.0**2, 5.0**2])

    # True initial state
    x_true = np.array([0.0, 2.0, 0.0, 1.0])

    # Run filter with correct process noise (consistent)
    print("\n--- Correctly Tuned Filter ---")
    nees_correct = run_filter_get_nees(x_true, F, Q, H, R, Q, n_steps)

    # Run filter with underestimated process noise (optimistic)
    print("\n--- Underestimated Process Noise (Optimistic) ---")
    Q_low = q_constant_velocity(dt, 0.01, 2)  # Too low
    nees_optimistic = run_filter_get_nees(x_true, F, Q, H, R, Q_low, n_steps)

    # Run filter with overestimated process noise (conservative)
    print("\n--- Overestimated Process Noise (Conservative) ---")
    Q_high = q_constant_velocity(dt, 1.0, 2)  # Too high
    nees_conservative = run_filter_get_nees(x_true, F, Q, H, R, Q_high, n_steps)

    # Statistical test
    state_dim = 4
    print(f"\n{'Filter Type':<30} {'Mean NEES':<12} {'Expected':<12} {'Consistent?'}")
    print("-" * 70)

    for name, nees_vals in [
        ("Correctly tuned", nees_correct),
        ("Optimistic (low Q)", nees_optimistic),
        ("Conservative (high Q)", nees_conservative),
    ]:
        mean_nees = np.mean(nees_vals)
        # Chi-squared bounds (95% confidence)
        lower = state_dim * 0.5  # Rough approximation
        upper = state_dim * 1.5
        consistent = lower <= mean_nees <= upper
        status = "Yes" if consistent else "No"

        print(f"{name:<30} {mean_nees:<12.2f} {state_dim:<12} {status}")


def run_filter_get_nees(
    x_true_init: np.ndarray,
    F: np.ndarray,
    Q_true: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
    Q_filter: np.ndarray,
    n_steps: int,
) -> List[float]:
    """Run Kalman filter and compute NEES at each step."""
    # Generate true trajectory
    x_true = x_true_init.copy()
    true_states = [x_true.copy()]

    for _ in range(n_steps - 1):
        # Process noise
        w = np.random.multivariate_normal(np.zeros(4), Q_true)
        x_true = F @ x_true + w
        true_states.append(x_true.copy())

    # Generate measurements
    measurements = []
    for x in true_states:
        v = np.random.multivariate_normal(np.zeros(2), R)
        z = H @ x + v
        measurements.append(z)

    # Run filter
    x = np.array([measurements[0][0], 0.0, measurements[0][1], 0.0])
    P = np.diag([25.0, 10.0, 25.0, 10.0])

    nees_values = []

    for k in range(n_steps):
        if k > 0:
            x, P = kf_predict(x, P, F, Q_filter)
            result = kf_update(x, P, measurements[k], H, R)
            x, P = result.x, result.P

        # Compute NEES
        err = x - true_states[k]
        nees_val = float(err.T @ np.linalg.solve(P, err))
        nees_values.append(nees_val)

    return nees_values


def monte_carlo_demo() -> None:
    """Demonstrate Monte Carlo evaluation of tracker performance."""
    print("\n" + "=" * 60)
    print("3. MONTE CARLO TRACKER EVALUATION")
    print("=" * 60)

    print("\nMonte Carlo simulation runs multiple trials to get")
    print("statistically meaningful performance metrics.")

    np.random.seed(123)

    n_runs = 50
    n_steps = 100
    dt = 1.0

    # System setup
    F = f_constant_velocity(dt, 2)
    Q = q_constant_velocity(dt, 0.1, 2)
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    R = np.diag([5.0**2, 5.0**2])

    print(f"\nRunning {n_runs} Monte Carlo trials...")
    print(f"  {n_steps} time steps per trial")

    # Collect metrics across runs
    all_rmse = []
    all_nees = []
    all_nis = []

    for run in range(n_runs):
        # Random initial state
        x_true = np.array(
            [
                np.random.uniform(-50, 50),  # x
                np.random.uniform(1, 3),  # vx
                np.random.uniform(-50, 50),  # y
                np.random.uniform(1, 3),  # vy
            ]
        )

        # Generate trajectory and measurements
        true_states = [x_true.copy()]
        measurements = []

        for _ in range(n_steps):
            # Measurement
            v = np.random.multivariate_normal(np.zeros(2), R)
            z = H @ x_true + v
            measurements.append(z)

            # Propagate
            w = np.random.multivariate_normal(np.zeros(4), Q)
            x_true = F @ x_true + w
            true_states.append(x_true.copy())

        true_states = true_states[:-1]  # Match measurement count

        # Run filter
        x = np.array([measurements[0][0], 0.0, measurements[0][1], 0.0])
        P = np.diag([25.0, 10.0, 25.0, 10.0])

        run_errors = []
        run_nees = []
        run_nis = []

        for k in range(n_steps):
            if k > 0:
                x, P = kf_predict(x, P, F, Q)
                z_pred = H @ x
                S = H @ P @ H.T + R
                innovation = measurements[k] - z_pred

                # NIS
                nis_val = float(innovation.T @ np.linalg.solve(S, innovation))
                run_nis.append(nis_val)

                result = kf_update(x, P, measurements[k], H, R)
                x, P = result.x, result.P

            # Position error
            err = np.sqrt(
                (x[0] - true_states[k][0]) ** 2 + (x[2] - true_states[k][2]) ** 2
            )
            run_errors.append(err)

            # NEES
            state_err = x - true_states[k]
            nees_val = float(state_err.T @ np.linalg.solve(P, state_err))
            run_nees.append(nees_val)

        all_rmse.append(np.sqrt(np.mean(np.array(run_errors) ** 2)))
        all_nees.append(np.mean(run_nees))
        all_nis.append(np.mean(run_nis))

    # Report statistics
    print("\nResults across all Monte Carlo runs:")
    print("-" * 60)

    print("\nPosition RMSE (m):")
    print(f"  Mean:   {np.mean(all_rmse):.2f}")
    print(f"  Std:    {np.std(all_rmse):.2f}")
    print(f"  Min:    {np.min(all_rmse):.2f}")
    print(f"  Max:    {np.max(all_rmse):.2f}")

    print("\nNEES (expected: 4.0 for 4-state filter):")
    print(f"  Mean:   {np.mean(all_nees):.2f}")
    print(f"  Std:    {np.std(all_nees):.2f}")

    print("\nNIS (expected: 2.0 for 2-measurement filter):")
    print(f"  Mean:   {np.mean(all_nis):.2f}")
    print(f"  Std:    {np.std(all_nis):.2f}")

    # Chi-squared test
    print("\nConsistency check:")
    nees_pass = 3.0 <= np.mean(all_nees) <= 5.0
    nis_pass = 1.5 <= np.mean(all_nis) <= 2.5
    print(f"  NEES within bounds: {'PASS' if nees_pass else 'FAIL'}")
    print(f"  NIS within bounds:  {'PASS' if nis_pass else 'FAIL'}")

    return all_rmse, all_nees, all_nis


def ospa_over_time_demo() -> None:
    """Demonstrate OSPA metric evolution over time."""
    print("\n" + "=" * 60)
    print("4. OSPA OVER TIME FOR TRACKER EVALUATION")
    print("=" * 60)

    np.random.seed(456)

    n_steps = 50

    # Simulate ground truth: 2 targets, one appears at t=10, one disappears at t=40
    print("\nSimulating scenario with target birth/death:")
    print("  - Target 1: present throughout")
    print("  - Target 2: appears at t=10, disappears at t=40")

    # Generate trajectories
    truth_history = []
    for t in range(n_steps):
        targets = []
        # Target 1: always present, moving right
        targets.append(np.array([10.0 + t * 2, 30.0 + t * 0.5]))

        # Target 2: appears at t=10, disappears at t=40
        if 10 <= t < 40:
            targets.append(np.array([80.0 - t * 1.5, 20.0 + t * 1.0]))

        truth_history.append(
            np.array(targets) if targets else np.array([]).reshape(0, 2)
        )

    # Simulate tracker estimates (with some noise and occasional misses)
    estimate_history = []
    for t, truth in enumerate(truth_history):
        estimates = []
        for target in truth:
            # 90% detection probability
            if np.random.rand() < 0.9:
                noise = np.random.randn(2) * 5.0
                estimates.append(target + noise)

        # 10% false alarm probability
        if np.random.rand() < 0.1:
            false_alarm = np.array(
                [np.random.uniform(0, 100), np.random.uniform(0, 80)]
            )
            estimates.append(false_alarm)

        estimate_history.append(
            np.array(estimates) if estimates else np.array([]).reshape(0, 2)
        )

    # Compute OSPA over time
    c = 50.0
    p = 2
    ospa_history = []
    loc_history = []
    card_history = []

    for truth, estimates in zip(truth_history, estimate_history):
        if len(truth) == 0 and len(estimates) == 0:
            ospa_history.append(0.0)
            loc_history.append(0.0)
            card_history.append(0.0)
        else:
            result = ospa(truth, estimates, c=c, p=p)
            ospa_history.append(result.ospa)
            loc_history.append(result.localization)
            card_history.append(result.cardinality)

    # Summary
    print(f"\nOSPA statistics over {n_steps} time steps:")
    print(f"  Mean OSPA:         {np.mean(ospa_history):.2f}")
    print(f"  Mean Localization: {np.mean(loc_history):.2f}")
    print(f"  Mean Cardinality:  {np.mean(card_history):.2f}")

    return ospa_history, loc_history, card_history


def plot_results(
    mc_rmse: List[float],
    mc_nees: List[float],
    mc_nis: List[float],
    ospa_hist: List[float],
    loc_hist: List[float],
    card_hist: List[float],
) -> None:
    """Create performance evaluation plots."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Monte Carlo RMSE Distribution",
            "NEES/NIS Distribution",
            "OSPA Over Time",
            "OSPA Components",
        ),
    )

    # RMSE histogram
    fig.add_trace(
        go.Histogram(x=mc_rmse, nbinsx=20, name="Position RMSE", marker_color="blue"),
        row=1,
        col=1,
    )

    # NEES/NIS histograms
    fig.add_trace(
        go.Histogram(
            x=mc_nees,
            nbinsx=20,
            name="NEES",
            marker_color="green",
            opacity=0.7,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Histogram(
            x=mc_nis,
            nbinsx=20,
            name="NIS",
            marker_color="orange",
            opacity=0.7,
        ),
        row=1,
        col=2,
    )

    # OSPA over time
    time = list(range(len(ospa_hist)))
    fig.add_trace(
        go.Scatter(x=time, y=ospa_hist, name="OSPA", line=dict(color="red", width=2)),
        row=2,
        col=1,
    )

    # OSPA components
    fig.add_trace(
        go.Scatter(
            x=time,
            y=loc_hist,
            name="Localization",
            line=dict(color="blue", width=1.5),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=card_hist,
            name="Cardinality",
            line=dict(color="green", width=1.5),
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        title="Performance Evaluation Metrics",
        height=800,
        width=1200,
        showlegend=True,
        barmode="overlay",
    )

    fig.update_xaxes(title_text="RMSE (m)", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Value", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Time Step", row=2, col=1)
    fig.update_yaxes(title_text="OSPA", row=2, col=1)
    fig.update_xaxes(title_text="Time Step", row=2, col=2)
    fig.update_yaxes(title_text="Component Value", row=2, col=2)

    fig.write_html("performance_evaluation.html")
    print("\nInteractive plot saved to performance_evaluation.html")
    fig.show()


def main() -> None:
    """Run performance evaluation demonstrations."""
    print("\nPerformance Evaluation Examples")
    print("=" * 60)
    print("Demonstrating pytcl tracker and filter evaluation metrics")

    ospa_demo()
    nees_consistency_demo()
    mc_rmse, mc_nees, mc_nis = monte_carlo_demo()
    ospa_hist, loc_hist, card_hist = ospa_over_time_demo()

    # Generate plots
    try:
        plot_results(mc_rmse, mc_nees, mc_nis, ospa_hist, loc_hist, card_hist)
    except Exception as e:
        print(f"\nCould not generate plots: {e}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
