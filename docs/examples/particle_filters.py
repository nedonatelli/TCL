"""
Particle Filters Example
========================

This example demonstrates particle filtering (Sequential Monte Carlo)
algorithms in PyTCL:

- Bootstrap particle filter
- Importance sampling and resampling
- Different resampling strategies (multinomial, systematic, residual)
- Effective sample size monitoring
- Particle statistics computation
- Comparison with Kalman filter for linear systems
- Nonlinear system tracking

Particle filters are essential for nonlinear, non-Gaussian state estimation
where Kalman filters cannot be directly applied.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.dynamic_estimation.particle_filters import (
    ParticleState,
    bootstrap_pf_predict,
    bootstrap_pf_step,
    bootstrap_pf_update,
    effective_sample_size,
    gaussian_likelihood,
    initialize_particles,
    particle_covariance,
    particle_mean,
    resample_multinomial,
    resample_residual,
    resample_systematic,
)


def demo_particle_basics():
    """Demonstrate basic particle filter operations."""
    print("=" * 70)
    print("Particle Filter Basics Demo")
    print("=" * 70)

    np.random.seed(42)

    # Initialize particles for a 2D state [x, y]
    n_particles = 1000
    state_dim = 2

    # Initial distribution: Gaussian centered at origin
    mean = np.array([0.0, 0.0])
    cov = np.eye(2) * 2.0

    # initialize_particles returns a ParticleState object
    state = initialize_particles(mean, cov, n_particles)
    particles = state.particles
    weights = state.weights

    print(f"\nInitialized {n_particles} particles")
    print(f"State dimension: {state_dim}")
    print(f"Initial mean: {particle_mean(particles, weights)}")
    print(f"Initial std: {np.sqrt(np.diag(particle_covariance(particles, weights)))}")

    # Effective sample size
    ess = effective_sample_size(weights)
    print(f"Initial ESS: {ess:.1f} (should be ~{n_particles})")

    # Demonstrate weight degeneracy
    print("\n--- Weight Degeneracy Example ---")
    # Create skewed weights
    skewed_weights = np.ones(n_particles)
    skewed_weights[0] = 100.0  # One dominant particle
    skewed_weights /= skewed_weights.sum()

    ess_skewed = effective_sample_size(skewed_weights)
    print(f"With one dominant particle, ESS: {ess_skewed:.1f}")
    print("This indicates severe weight degeneracy - resampling needed!")


def demo_resampling_methods():
    """Demonstrate different resampling strategies."""
    print("\n" + "=" * 70)
    print("Resampling Methods Demo")
    print("=" * 70)

    np.random.seed(42)

    n_particles = 1000

    # Create particles with non-uniform weights
    particles = np.random.randn(n_particles, 2)
    weights = np.exp(-np.sum(particles**2, axis=1) / 4)  # Higher near origin
    weights /= weights.sum()

    print(f"\nOriginal particle distribution:")
    print(f"  Mean: {particle_mean(particles, weights)}")
    print(f"  ESS: {effective_sample_size(weights):.1f}")

    # Multinomial resampling - returns resampled particles directly
    particles_multi = resample_multinomial(particles, weights)
    weights_multi = np.ones(n_particles) / n_particles

    print("\n--- Multinomial Resampling ---")
    print(f"  Mean: {particle_mean(particles_multi, weights_multi)}")
    print(f"  ESS: {effective_sample_size(weights_multi):.1f}")

    # Systematic resampling (lower variance)
    particles_sys = resample_systematic(particles, weights)
    weights_sys = np.ones(n_particles) / n_particles

    print("\n--- Systematic Resampling ---")
    print(f"  Mean: {particle_mean(particles_sys, weights_sys)}")
    print(f"  ESS: {effective_sample_size(weights_sys):.1f}")

    # Residual resampling
    particles_res = resample_residual(particles, weights)
    weights_res = np.ones(n_particles) / n_particles

    print("\n--- Residual Resampling ---")
    print(f"  Mean: {particle_mean(particles_res, weights_res)}")
    print(f"  ESS: {effective_sample_size(weights_res):.1f}")

    print("\nNote: Systematic resampling typically preserves more diversity")
    print("and has lower variance than multinomial resampling.")

    # Plot resampling comparison
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Original Particles (ESS={effective_sample_size(weights):.0f})",
                "After Multinomial Resampling",
                "After Systematic Resampling",
                "After Residual Resampling",
            ),
        )

        # Original particles with weights
        fig.add_trace(
            go.Scatter(
                x=particles[:, 0],
                y=particles[:, 1],
                mode="markers",
                marker=dict(size=5, color=weights, colorscale="Viridis", opacity=0.6),
                name="Original",
            ),
            row=1,
            col=1,
        )

        # Multinomial resampling
        fig.add_trace(
            go.Scatter(
                x=particles_multi[:, 0],
                y=particles_multi[:, 1],
                mode="markers",
                marker=dict(size=5, color="blue", opacity=0.6),
                name="Multinomial",
            ),
            row=1,
            col=2,
        )

        # Systematic resampling
        fig.add_trace(
            go.Scatter(
                x=particles_sys[:, 0],
                y=particles_sys[:, 1],
                mode="markers",
                marker=dict(size=5, color="green", opacity=0.6),
                name="Systematic",
            ),
            row=2,
            col=1,
        )

        # Residual resampling
        fig.add_trace(
            go.Scatter(
                x=particles_res[:, 0],
                y=particles_res[:, 1],
                mode="markers",
                marker=dict(size=5, color="red", opacity=0.6),
                name="Residual",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800,
            width=1000,
            title_text="Comparison of Resampling Methods",
            showlegend=False,
        )
        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="y")

        fig.write_html("particle_resampling_comparison.html")
        print("\n  [Plot saved to particle_resampling_comparison.html]")


def demo_linear_tracking():
    """Compare particle filter to Kalman filter for linear system."""
    print("\n" + "=" * 70)
    print("Linear System Tracking Demo")
    print("=" * 70)

    np.random.seed(42)

    # Linear constant-velocity model
    dt = 1.0
    F = np.array(
        [
            [1, dt, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 1],
        ]
    )

    # Process noise
    q = 0.1
    Q = q * np.array(
        [
            [dt**3 / 3, dt**2 / 2, 0, 0],
            [dt**2 / 2, dt, 0, 0],
            [0, 0, dt**3 / 3, dt**2 / 2],
            [0, 0, dt**2 / 2, dt],
        ]
    )

    # Measurement model (observe position only)
    H = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    R = np.eye(2) * 1.0

    # True trajectory
    n_steps = 20
    true_states = np.zeros((n_steps, 4))
    true_states[0] = [0, 1, 0, 0.5]  # Start at origin, moving diagonally

    for k in range(1, n_steps):
        true_states[k] = F @ true_states[k - 1] + np.random.multivariate_normal(
            np.zeros(4), Q * 0.1
        )

    # Generate measurements
    measurements = [
        H @ true_states[k] + np.random.multivariate_normal(np.zeros(2), R)
        for k in range(n_steps)
    ]

    print(f"\nSimulating {n_steps} time steps")
    print("True initial state: [x=0, vx=1, y=0, vy=0.5]")

    # Kalman filter
    x_kf = np.array([0.0, 0.0, 0.0, 0.0])
    P_kf = np.eye(4) * 10.0
    kf_estimates = []

    for z in measurements:
        pred = kf_predict(x_kf, P_kf, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x_kf, P_kf = upd.x, upd.P
        kf_estimates.append(x_kf.copy())

    # Particle filter
    n_particles = 500
    state = initialize_particles(np.zeros(4), np.eye(4) * 10.0, n_particles)
    particles = state.particles
    weights = state.weights.copy()
    pf_estimates = []

    def process_fn(x):
        return F @ x + np.random.multivariate_normal(np.zeros(4), Q)

    def likelihood_fn(z, x):
        z_pred = H @ x
        return gaussian_likelihood(z, z_pred, R)

    for z in measurements:
        # Predict
        particles = np.array([process_fn(p) for p in particles])

        # Update weights
        likelihoods = np.array([likelihood_fn(z, p) for p in particles])
        weights = weights * likelihoods
        weights /= weights.sum()

        # Estimate
        pf_estimates.append(particle_mean(particles, weights))

        # Resample if needed
        ess = effective_sample_size(weights)
        if ess < n_particles / 2:
            particles = resample_systematic(particles, weights)
            weights = np.ones(n_particles) / n_particles

    # Compare RMSE
    kf_estimates = np.array(kf_estimates)
    pf_estimates = np.array(pf_estimates)

    kf_rmse = np.sqrt(np.mean((kf_estimates[:, [0, 2]] - true_states[:, [0, 2]]) ** 2))
    pf_rmse = np.sqrt(np.mean((pf_estimates[:, [0, 2]] - true_states[:, [0, 2]]) ** 2))

    print("\n--- Filter Comparison (Position RMSE) ---")
    print(f"  Kalman Filter: {kf_rmse:.4f}")
    print(f"  Particle Filter ({n_particles} particles): {pf_rmse:.4f}")
    print("\nNote: For linear Gaussian systems, KF is optimal.")
    print("PF approaches KF performance as particle count increases.")

    # Plot tracking comparison
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Trajectory Tracking: KF vs PF",
                "Position Error Over Time",
            ),
        )

        measurements_arr = np.array(measurements)

        # Trajectory plot
        fig.add_trace(
            go.Scatter(
                x=true_states[:, 0],
                y=true_states[:, 2],
                mode="lines",
                name="True trajectory",
                line=dict(color="black", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=kf_estimates[:, 0],
                y=kf_estimates[:, 2],
                mode="lines",
                name=f"Kalman Filter (RMSE={kf_rmse:.3f})",
                line=dict(color="blue", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pf_estimates[:, 0],
                y=pf_estimates[:, 2],
                mode="lines",
                name=f"Particle Filter (RMSE={pf_rmse:.3f})",
                line=dict(color="red", width=1.5, dash="dot"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=measurements_arr[:, 0],
                y=measurements_arr[:, 1],
                mode="markers",
                name="Measurements",
                marker=dict(size=5, color="gray", opacity=0.5),
            ),
            row=1,
            col=1,
        )

        # Error comparison
        time = np.arange(n_steps)
        kf_pos_err = np.sqrt(
            (kf_estimates[:, 0] - true_states[:, 0]) ** 2
            + (kf_estimates[:, 2] - true_states[:, 2]) ** 2
        )
        pf_pos_err = np.sqrt(
            (pf_estimates[:, 0] - true_states[:, 0]) ** 2
            + (pf_estimates[:, 2] - true_states[:, 2]) ** 2
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=kf_pos_err,
                mode="lines",
                name="Kalman Filter",
                line=dict(color="blue"),
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=time,
                y=pf_pos_err,
                mode="lines",
                name="Particle Filter",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="x position", row=1, col=1)
        fig.update_yaxes(title_text="y position", row=1, col=1)
        fig.update_xaxes(title_text="Time step", row=1, col=2)
        fig.update_yaxes(title_text="Position error", row=1, col=2)

        fig.update_layout(height=500, width=1200)
        fig.write_html("particle_linear_tracking.html")
        print("\n  [Plot saved to particle_linear_tracking.html]")


def demo_nonlinear_tracking():
    """Demonstrate particle filter for nonlinear system."""
    print("\n" + "=" * 70)
    print("Nonlinear System Tracking Demo")
    print("=" * 70)

    np.random.seed(42)

    # Nonlinear dynamics: polar to Cartesian (range-bearing sensor)
    # State: [x, y, vx, vy]
    # Measurement: [range, bearing] (nonlinear!)

    dt = 0.1
    n_steps = 50
    n_particles = 1000

    # True trajectory: circular motion
    omega = 0.5  # angular velocity
    radius = 10.0
    true_states = np.zeros((n_steps, 4))

    for k in range(n_steps):
        t = k * dt
        true_states[k] = [
            radius * np.cos(omega * t),
            radius * np.sin(omega * t),
            -radius * omega * np.sin(omega * t),
            radius * omega * np.cos(omega * t),
        ]

    # Measurement noise
    sigma_range = 0.5
    sigma_bearing = np.radians(2.0)

    def measurement_model(state):
        """Nonlinear measurement: range and bearing from origin."""
        x, y = state[0], state[1]
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x)
        return np.array([r, theta])

    # Generate measurements
    measurements = []
    for k in range(n_steps):
        z_true = measurement_model(true_states[k])
        noise = np.array(
            [np.random.randn() * sigma_range, np.random.randn() * sigma_bearing]
        )
        measurements.append(z_true + noise)

    print(f"\nSimulating circular motion with range-bearing sensor")
    print(f"  Radius: {radius} m, Angular velocity: {omega} rad/s")
    print(
        f"  Measurement noise: sigma_r={sigma_range} m, "
        f"sigma_theta={np.degrees(sigma_bearing):.1f} deg"
    )

    # Initialize particle filter
    state = initialize_particles(
        np.array([radius, 0.0, 0.0, radius * omega]),  # Near true initial
        np.diag([1.0, 1.0, 0.5, 0.5]),
        n_particles,
    )
    particles = state.particles
    weights = state.weights.copy()

    R = np.diag([sigma_range**2, sigma_bearing**2])

    def process_fn(state):
        """Constant velocity motion model with noise."""
        x, y, vx, vy = state
        q = 0.1
        return np.array(
            [
                x + vx * dt + np.random.randn() * q * dt,
                y + vy * dt + np.random.randn() * q * dt,
                vx + np.random.randn() * q,
                vy + np.random.randn() * q,
            ]
        )

    # Run particle filter
    pf_estimates = []
    ess_history = []

    for k, z in enumerate(measurements):
        # Predict
        particles = np.array([process_fn(p) for p in particles])

        # Update weights using range-bearing likelihood
        for i in range(n_particles):
            z_pred = measurement_model(particles[i])
            # Handle angle wraparound for bearing
            z_wrapped = z.copy()
            z_pred_wrapped = z_pred.copy()
            # Normalize bearing difference
            bearing_diff = np.arctan2(
                np.sin(z[1] - z_pred[1]), np.cos(z[1] - z_pred[1])
            )
            z_wrapped[1] = z_pred[1] + bearing_diff
            likelihood = gaussian_likelihood(z_wrapped, z_pred_wrapped, R)
            weights[i] *= likelihood

        # Normalize
        if weights.sum() > 0:
            weights /= weights.sum()
        else:
            weights = np.ones(n_particles) / n_particles

        # Estimate
        pf_estimates.append(particle_mean(particles, weights))
        ess_history.append(effective_sample_size(weights))

        # Resample
        if ess_history[-1] < n_particles / 2:
            particles = resample_systematic(particles, weights)
            weights = np.ones(n_particles) / n_particles

    pf_estimates = np.array(pf_estimates)

    # Compute errors
    pos_errors = np.sqrt(
        (pf_estimates[:, 0] - true_states[:, 0]) ** 2
        + (pf_estimates[:, 1] - true_states[:, 1]) ** 2
    )

    print("\n--- Tracking Results ---")
    print(f"  Mean position error: {np.mean(pos_errors):.3f} m")
    print(f"  Max position error: {np.max(pos_errors):.3f} m")
    print(f"  Min ESS: {np.min(ess_history):.1f}")
    print(f"  Mean ESS: {np.mean(ess_history):.1f}")

    # Show trajectory snapshots
    print("\n--- Trajectory Snapshots ---")
    times = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]
    for t in times:
        true_pos = true_states[t, :2]
        est_pos = pf_estimates[t, :2]
        err = pos_errors[t]
        print(
            f"  t={t*dt:.1f}s: True=({true_pos[0]:.2f}, {true_pos[1]:.2f}), "
            f"Est=({est_pos[0]:.2f}, {est_pos[1]:.2f}), Err={err:.3f}m"
        )

    # Plot nonlinear tracking results
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Circular Motion Tracking with Range-Bearing Sensor",
                "Position Error Over Time",
                "ESS History (resampling when ESS < N/2)",
                "Range-Bearing Measurements (color=time)",
            ),
        )

        # Trajectory plot
        fig.add_trace(
            go.Scatter(
                x=true_states[:, 0],
                y=true_states[:, 1],
                mode="lines",
                name="True trajectory",
                line=dict(color="black", width=2),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pf_estimates[:, 0],
                y=pf_estimates[:, 1],
                mode="lines",
                name="PF estimate",
                line=dict(color="red", width=1.5, dash="dash"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[true_states[0, 0]],
                y=[true_states[0, 1]],
                mode="markers",
                name="Start",
                marker=dict(size=15, color="green", symbol="circle"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[true_states[-1, 0]],
                y=[true_states[-1, 1]],
                mode="markers",
                name="End",
                marker=dict(size=15, color="blue", symbol="square"),
            ),
            row=1,
            col=1,
        )

        # Position error over time
        time_axis = np.arange(n_steps) * dt
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=pos_errors,
                mode="lines",
                line=dict(color="blue", width=1.5),
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=np.mean(pos_errors),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean={np.mean(pos_errors):.3f}",
            row=1,
            col=2,
        )

        # ESS history
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=ess_history,
                mode="lines",
                line=dict(color="green", width=1.5),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=n_particles / 2,
            line_dash="dash",
            line_color="red",
            annotation_text="Resampling threshold",
            row=2,
            col=1,
        )

        # Measurements in polar form
        meas_arr = np.array(measurements)
        fig.add_trace(
            go.Scatter(
                x=np.degrees(meas_arr[:, 1]),
                y=meas_arr[:, 0],
                mode="markers",
                marker=dict(
                    size=5,
                    color=time_axis,
                    colorscale="Viridis",
                    colorbar=dict(title="Time (s)", x=1.0),
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_xaxes(title_text="x position (m)", row=1, col=1)
        fig.update_yaxes(title_text="y position (m)", row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Position error (m)", row=1, col=2)
        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Effective Sample Size", row=2, col=1)
        fig.update_xaxes(title_text="Bearing (degrees)", row=2, col=2)
        fig.update_yaxes(title_text="Range (m)", row=2, col=2)

        fig.update_layout(height=800, width=1000, showlegend=True)
        fig.write_html("particle_nonlinear_tracking.html")
        print("\n  [Plot saved to particle_nonlinear_tracking.html]")


def demo_multimodal():
    """Demonstrate particle filter advantage for multimodal distributions."""
    print("\n" + "=" * 70)
    print("Multimodal Distribution Demo")
    print("=" * 70)

    np.random.seed(42)

    # Scenario: Target could be at one of two locations
    # This is impossible for a Kalman filter but natural for particle filters

    n_particles = 2000

    # Prior: mixture of two Gaussians
    mode1 = np.array([5.0, 0.0])
    mode2 = np.array([-5.0, 0.0])
    cov = np.eye(2) * 0.5

    # Initialize with bimodal distribution
    state1 = initialize_particles(mode1, cov, n_particles // 2)
    state2 = initialize_particles(mode2, cov, n_particles // 2)
    particles = np.vstack([state1.particles, state2.particles])
    weights = np.ones(n_particles) / n_particles

    print("\nBimodal prior distribution:")
    print(f"  Mode 1: {mode1}")
    print(f"  Mode 2: {mode2}")
    print(f"  Mean: {particle_mean(particles, weights)}")
    print("  (Mean is between modes - not representative!)")

    # Measurement that confirms mode 2
    z = np.array([-4.8, 0.1])
    R = np.eye(2) * 0.2

    print(f"\nMeasurement received: {z}")

    # Save prior particles for plotting
    prior_particles = particles.copy()

    # Update weights
    for i in range(n_particles):
        z_pred = particles[i]  # Direct position observation
        weights[i] *= gaussian_likelihood(z, z_pred, R)
    weights /= weights.sum()

    # After update
    print("\nAfter measurement update:")
    print(f"  Mean: {particle_mean(particles, weights)}")
    print(f"  ESS: {effective_sample_size(weights):.1f}")

    # Analyze particle distribution
    near_mode1 = np.sum(particles[:, 0] > 0)
    near_mode2 = np.sum(particles[:, 0] < 0)
    weight_mode1 = np.sum(weights[particles[:, 0] > 0])
    weight_mode2 = np.sum(weights[particles[:, 0] < 0])

    print(f"\n  Particles near mode 1: {near_mode1} (weight: {weight_mode1:.4f})")
    print(f"  Particles near mode 2: {near_mode2} (weight: {weight_mode2:.4f})")
    print("\nNote: PF correctly concentrates probability on mode 2")
    print("after receiving the confirming measurement.")

    # Plot multimodal distribution
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Prior: Bimodal Distribution",
                "Posterior: After Measurement Update",
            ),
        )

        # Prior distribution
        fig.add_trace(
            go.Scatter(
                x=prior_particles[:, 0],
                y=prior_particles[:, 1],
                mode="markers",
                marker=dict(size=3, color="blue", opacity=0.3),
                name="Prior particles",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[mode1[0], mode2[0]],
                y=[mode1[1], mode2[1]],
                mode="markers",
                marker=dict(size=15, color="green", symbol="x", line=dict(width=3)),
                name="Modes",
            ),
            row=1,
            col=1,
        )

        # Posterior distribution
        fig.add_trace(
            go.Scatter(
                x=particles[:, 0],
                y=particles[:, 1],
                mode="markers",
                marker=dict(
                    size=weights * n_particles * 50,
                    color=weights,
                    colorscale="Reds",
                    opacity=0.5,
                ),
                name="Posterior particles",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[z[0]],
                y=[z[1]],
                mode="markers",
                marker=dict(size=20, color="blue", symbol="star"),
                name="Measurement",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(range=[-10, 10], row=1, col=1)
        fig.update_yaxes(range=[-5, 5], row=1, col=1)
        fig.update_xaxes(range=[-10, 10], row=1, col=2)
        fig.update_yaxes(range=[-5, 5], row=1, col=2)

        fig.update_layout(
            height=500,
            width=1200,
            title_text="Particle Filter for Multimodal Distribution (Point size proportional to weight)",
        )
        fig.write_html("particle_multimodal.html")
        print("\n  [Plot saved to particle_multimodal.html]")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Particle Filters Example")
    print("#" * 70)

    demo_particle_basics()
    demo_resampling_methods()
    demo_linear_tracking()
    demo_nonlinear_tracking()
    demo_multimodal()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: particle_resampling_comparison.html, ")
        print("             particle_linear_tracking.html,")
        print("             particle_nonlinear_tracking.html,")
        print("             particle_multimodal.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
