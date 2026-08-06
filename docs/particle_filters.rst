Particle Filters & Non-Gaussian Estimation
==========================================

*Comprehensive guide to sequential Monte Carlo (SMC) methods for tracking non-Gaussian, nonlinear systems.*

Particle filters represent probability distributions as weighted samples (particles), enabling estimation in highly nonlinear and non-Gaussian scenarios where Kalman filters fail. pytcl ships a complete functional particle filter toolkit in ``pytcl.dynamic_estimation.particle_filters``.

**Table of Contents:**

- Particle Filter Fundamentals
- Bootstrap Particle Filter
- Resampling Strategies
- Degeneracy Detection & Solutions
- Sequential Importance Sampling
- Practical Implementation
- Performance & Efficiency
- Common Issues & Solutions
- Best Practices

Particle Filter Fundamentals
----------------------------

**Why Particle Filters?**

Kalman filters assume:
- Linear systems (or EKF/UKF with differentiability)
- Gaussian noise
- Unimodal distributions

Particle filters work with:
- Arbitrary nonlinear models
- Non-Gaussian and multimodal distributions
- Discontinuous measurement likelihoods
- Heavy-tailed distributions and outliers

**Particle Filter Concept:**

Represent state uncertainty as a set of :math:`N_p` weighted samples:

.. math::

   p(x_k \mid z_{1:k}) \approx \sum_{i=1}^{N_p} w_k^{(i)} \, \delta(x_k - x_k^{(i)})

Where:

- :math:`x_k^{(i)}`: i-th particle (state sample)
- :math:`w_k^{(i)}`: weight of i-th particle
- :math:`\delta(\cdot)`: Dirac delta function
- Weights normalize to 1: :math:`\sum_i w_k^{(i)} = 1`

**Key Equations:**

1. **Prediction**: Sample from motion model

   .. math::

      x_k^{(i)} \sim p(x_k \mid x_{k-1}^{(i)})

2. **Update**: Compute importance weights

   .. math::

      w_k^{(i)} \propto p(z_k \mid x_k^{(i)}) \cdot w_{k-1}^{(i)}

3. **Normalize**:

   .. math::

      \hat{w}_k^{(i)} = \frac{w_k^{(i)}}{\sum_j w_k^{(j)}}

4. **Resample** (if needed): Generate new particles from weighted distribution


Bootstrap Particle Filter
-------------------------

**Algorithm: Sequential Importance Resampling (SIR)**

pytcl provides the complete SIR cycle as ``bootstrap_pf_step``, which performs
predict, update, and adaptive resampling in one call. Particles and weights
travel together in a ``ParticleState`` named tuple.

.. code-block:: python

    import numpy as np
    from pytcl.dynamic_estimation.particle_filters import (
        bootstrap_pf_step,
        initialize_particles,
        particle_covariance,
        particle_mean,
    )

    rng = np.random.default_rng(42)

    # 1D constant-velocity state: [position, velocity]
    dt = 0.1

    def f(x):
        """Motion model: x_{k+1} = f(x_k)."""
        return np.array([x[0] + x[1] * dt, x[1]])

    def h(x):
        """Measurement model: observe position only."""
        return np.array([x[0]])

    def Q_sample(n, rng):
        """Draw n process-noise samples, shape (n, 2)."""
        return rng.normal(0.0, [0.02, 0.10], size=(n, 2))

    R = np.array([[0.25]])  # measurement noise covariance

    # Initialize 500 particles from a Gaussian prior
    state = initialize_particles(
        x0=np.array([0.0, 1.0]), P0=np.diag([1.0, 0.25]), N=500, rng=rng
    )

    # Simulate and filter
    true_x = np.array([0.0, 1.0])
    for k in range(50):
        true_x = f(true_x)
        z = h(true_x) + rng.normal(0.0, 0.5, size=1)
        state = bootstrap_pf_step(
            state.particles, state.weights, z, f, h, Q_sample, R, rng=rng
        )

    x_hat = particle_mean(state.particles, state.weights)
    P_hat = particle_covariance(state.particles, state.weights)
    print(f"true:     pos={true_x[0]:.3f} vel={true_x[1]:.3f}")
    print(f"estimate: pos={x_hat[0]:.3f} vel={x_hat[1]:.3f}")
    print(f"pos std:  {np.sqrt(P_hat[0, 0]):.3f}")
    # true:     pos=5.000 vel=1.000
    # estimate: pos=5.081 vel=1.137
    # pos std:  0.213

**Predict and Update as Separate Steps**

When you need finer control (e.g. multiple measurements per scan, custom
likelihoods), use the individual building blocks. ``bootstrap_pf_update``
takes any likelihood function ``likelihood_func(z, x) -> float``;
``gaussian_likelihood`` covers the common Gaussian-measurement case.

.. code-block:: python

    from pytcl.dynamic_estimation.particle_filters import (
        bootstrap_pf_predict,
        bootstrap_pf_update,
        gaussian_likelihood,
    )

    # Predict: propagate particles through f and add process noise
    particles = bootstrap_pf_predict(state.particles, f, Q_sample, rng=rng)

    # Update: reweight particles by measurement likelihood
    z = np.array([5.2])

    def likelihood(z, x):
        return gaussian_likelihood(z, h(x), R)

    weights, log_lik = bootstrap_pf_update(particles, state.weights, z, likelihood)
    print(f"weights sum to {weights.sum():.6f}")
    print(f"log marginal likelihood: {log_lik:.3f}")
    # weights sum to 1.000000
    # log marginal likelihood: -0.325


Resampling Strategies
---------------------

**Problem: Particle Degeneracy**

As the filter runs, most particles get zero weight (weight collapse).
Solution: resample. pytcl ships three resamplers; each takes ``(particles,
weights)`` and returns a new equally-weighted particle set.

Set up a degenerate particle set to compare them:

.. code-block:: python

    from pytcl.dynamic_estimation.particle_filters import (
        effective_sample_size,
        resample_multinomial,
        resample_residual,
        resample_systematic,
    )

    # Highly non-uniform weights: most mass on a few particles
    N = 500
    demo_particles = rng.normal(0.0, 1.0, size=(N, 2))
    raw = np.exp(-0.5 * (demo_particles[:, 0] - 2.0) ** 2 / 0.1)
    demo_weights = raw / raw.sum()
    print(f"ESS before resampling: {effective_sample_size(demo_weights):.1f}")
    # ESS before resampling: 43.0

**1. Multinomial Resampling (Simple)**

Sample with replacement proportionally to weight. Highest variance.

.. code-block:: python

    resampled = resample_multinomial(demo_particles, demo_weights, rng=rng)
    print(f"unique ancestors kept: {len(np.unique(resampled[:, 0]))}")
    # unique ancestors kept: 71

**2. Systematic Resampling (Best for Most Cases)**

Deterministic positions with a single random offset. Low variance, O(N).

.. code-block:: python

    resampled = resample_systematic(demo_particles, demo_weights, rng=rng)
    print(f"unique ancestors kept: {len(np.unique(resampled[:, 0]))}")
    # unique ancestors kept: 70

**3. Stratified Resampling**

.. note::

   pytcl does not ship a stratified resampler. The implementation below is a
   self-contained reference; in practice ``resample_systematic`` is the
   recommended low-variance choice.

.. code-block:: python

    def resample_stratified(particles, weights, rng):
        """Stratified resampling: one uniform draw per stratum."""
        N = len(weights)
        u = (np.arange(N) + rng.uniform(size=N)) / N
        indices = np.searchsorted(np.cumsum(weights), u)
        return particles[indices]

    resampled = resample_stratified(demo_particles, demo_weights, rng)
    print(f"unique ancestors kept: {len(np.unique(resampled[:, 0]))}")
    # unique ancestors kept: 70

**4. Residual Resampling (Semi-Deterministic)**

Keep ``floor(N * w_i)`` copies of each particle deterministically, then fill
the remaining slots stochastically from the fractional remainders.

.. code-block:: python

    resampled = resample_residual(demo_particles, demo_weights, rng=rng)
    print(f"unique ancestors kept: {len(np.unique(resampled[:, 0]))}")
    # unique ancestors kept: 72

**Resampling Comparison:**

============================ ================ ============== ================
Function                     Complexity       Variance       Stability
============================ ================ ============== ================
``resample_multinomial``     O(N log N)       High           Unstable
``resample_systematic``      O(N)             Lowest         Robust
stratified (hand-rolled)     O(N)             Low            Very Good
``resample_residual``        O(N log N)       Very low       Excellent
============================ ================ ============== ================

**Recommended:** ``resample_systematic`` for most applications.


Degeneracy Detection & Solutions
--------------------------------

**Effective Sample Size (ESS):**

.. math::

   N_{\text{eff}} = \frac{1}{\sum_i \left(w^{(i)}\right)^2}

ESS ranges from 1 (all weight on one particle) to :math:`N` (uniform
weights). Resample when :math:`N_{\text{eff}} < \tau N` (typically
:math:`\tau = 0.5`). pytcl computes it with ``effective_sample_size``:

.. code-block:: python

    def resample_if_needed(particles, weights, threshold=0.5, rng=None):
        """Adaptive resampling: resample only when degenerate."""
        N = len(weights)
        if effective_sample_size(weights) < threshold * N:
            particles = resample_systematic(particles, weights, rng=rng)
            weights = np.full(N, 1.0 / N)
        return particles, weights

    particles, weights = resample_if_needed(
        demo_particles, demo_weights, rng=rng
    )
    print(f"ESS after adaptive resampling: {effective_sample_size(weights):.1f}")
    # ESS after adaptive resampling: 500.0

**Solutions to Degeneracy:**

1. **Adaptive Resampling**: Resample only when needed (above). This is what
   ``bootstrap_pf_step`` does internally via its ``resample_threshold``
   argument.

2. **Regularization**: Add jitter to particles after resampling

.. code-block:: python

    def resample_with_regularization(particles, weights, rng, bandwidth=1.0):
        """Resample and add regularization noise."""
        resampled = resample_systematic(particles, weights, rng=rng)

        # Adaptive bandwidth from particle spread
        h_bw = bandwidth * particles.std(axis=0)
        return resampled + rng.normal(0.0, h_bw, size=resampled.shape)

    regularized = resample_with_regularization(
        demo_particles, demo_weights, rng
    )
    print(f"distinct particles: {len(np.unique(regularized[:, 0]))}")
    # distinct particles: 500


Sequential Importance Sampling (SIS)
------------------------------------

**More General Particle Filter Framework:**

The bootstrap filter is SIS with the prior :math:`p(x_k \mid x_{k-1})` as
proposal plus adaptive resampling. Composing the pytcl building blocks by
hand exposes each SIS stage, which is useful when you need a custom proposal
or likelihood:

.. code-block:: python

    def sis_step(particles, weights, z, likelihood, rng):
        """One SIS cycle: propose, reweight, adaptively resample."""
        # 1. Propose from the motion model (bootstrap proposal)
        particles = bootstrap_pf_predict(particles, f, Q_sample, rng=rng)

        # 2. Reweight by measurement likelihood
        weights, _ = bootstrap_pf_update(particles, weights, z, likelihood)

        # 3. Resample if degenerate
        return resample_if_needed(particles, weights, rng=rng)

    state = initialize_particles(
        x0=np.array([0.0, 1.0]), P0=np.diag([1.0, 0.25]), N=500, rng=rng
    )
    particles, weights = state.particles, state.weights

    true_x = np.array([0.0, 1.0])
    for k in range(30):
        true_x = f(true_x)
        z = h(true_x) + rng.normal(0.0, 0.5, size=1)
        particles, weights = sis_step(particles, weights, z, likelihood, rng)

    x_hat = particle_mean(particles, weights)
    print(f"true pos={true_x[0]:.3f}  estimate pos={x_hat[0]:.3f}")
    # true pos=3.000  estimate pos=2.954


Practical Implementation
------------------------

**Complete Tracking System:**

Range-only tracking is a classic nonlinear problem where the measurement
function makes EKF linearization fragile:

.. code-block:: python

    # 2D constant-velocity state: [x, y, vx, vy]
    def f_cv(x):
        dt = 0.5
        return np.array([x[0] + x[2] * dt, x[1] + x[3] * dt, x[2], x[3]])

    def h_range(x):
        """Range from origin: strongly nonlinear in state."""
        return np.array([np.hypot(x[0], x[1])])

    def Q_sample_cv(n, rng):
        return rng.normal(0.0, [0.05, 0.05, 0.20, 0.20], size=(n, 4))

    R_range = np.array([[4.0]])

    rng = np.random.default_rng(7)
    state = initialize_particles(
        x0=np.array([100.0, 50.0, -2.0, 1.0]),
        P0=np.diag([100.0, 100.0, 1.0, 1.0]),
        N=2000,
        rng=rng,
    )

    truth = np.array([105.0, 45.0, -2.5, 1.2])
    pos_errors, range_errors = [], []
    for k in range(60):
        truth = f_cv(truth)
        z = h_range(truth) + rng.normal(0.0, 2.0, size=1)
        state = bootstrap_pf_step(
            state.particles, state.weights, z, f_cv, h_range,
            Q_sample_cv, R_range, rng=rng,
        )
        x_hat = particle_mean(state.particles, state.weights)
        pos_errors.append(np.hypot(*(x_hat[:2] - truth[:2])))
        range_errors.append(abs(np.hypot(*x_hat[:2]) - np.hypot(*truth[:2])))

    print(f"mean position error: {np.mean(pos_errors):.2f}")
    print(f"mean range error:    {np.mean(range_errors):.2f}")
    # mean position error: 7.39
    # mean range error:    2.72

Range-only measurements observe bearing only weakly (through motion over
time), so the position error stays several times larger than the range
error, which converges to the measurement noise floor. The particle cloud
spreads along the range arc: exactly the non-Gaussian behavior a Kalman
filter cannot represent.

**Multi-Target Extension:**

Run one ``ParticleState`` per target and update each with its associated
measurement:

.. code-block:: python

    targets = {
        tid: initialize_particles(x0, np.diag([25.0, 25.0, 1.0, 1.0]),
                                  N=1000, rng=rng)
        for tid, x0 in {
            0: np.array([100.0, 50.0, -2.0, 1.0]),
            1: np.array([-40.0, 80.0, 1.5, -0.5]),
        }.items()
    }

    # associations: list of (measurement, target_id) pairs from a data
    # association stage (e.g. GNN); see :doc:`assignment_association`
    associations = [(np.array([111.0]), 0), (np.array([90.0]), 1)]

    for z, tid in associations:
        s = targets[tid]
        targets[tid] = bootstrap_pf_step(
            s.particles, s.weights, z, f_cv, h_range,
            Q_sample_cv, R_range, rng=rng,
        )

    for tid, s in targets.items():
        x_hat = particle_mean(s.particles, s.weights)
        print(f"target {tid}: range estimate {np.hypot(*x_hat[:2]):.1f}")
    # target 0: range estimate 111.0
    # target 1: range estimate 89.7


Performance & Efficiency
------------------------

**Computational Complexity:**

- **Prediction**: O(N_p x d) where d = state dimension
- **Update**: O(N_p) for likelihood evaluation
- **Resampling**: O(N_p) to O(N_p log N_p) depending on method
- **Overall**: O(N_p) per cycle (most efficient SMC method)

**Memory Considerations:**

.. code-block:: python

    def estimate_memory_usage(num_particles, state_dim, num_timesteps):
        """Estimate memory for storing particle histories."""
        bytes_per_particle = state_dim * 8  # double precision
        bytes_per_weight = 8

        bytes_per_step = (bytes_per_particle + bytes_per_weight) * num_particles
        return bytes_per_step * num_timesteps / (1024**2)

    mb = estimate_memory_usage(1000, 10, 1000)
    print(f"1000 particles, 10 states, 1000 steps: {mb:.0f} MB")
    # 1000 particles, 10 states, 1000 steps: 84 MB

**Scaling with Particle Count:**

Monte Carlo error shrinks like :math:`1/\sqrt{N_p}`. Measure it directly on
the 1D tracking problem from the first section:

.. code-block:: python

    # One fixed measurement sequence shared by every trial, so that
    # trial-to-trial spread isolates the Monte Carlo error
    data_rng = np.random.default_rng(123)
    true_x = np.array([0.0, 1.0])
    truths, zs = [], []
    for _ in range(30):
        true_x = f(true_x)
        truths.append(true_x.copy())
        zs.append(h(true_x) + data_rng.normal(0.0, 0.5, size=1))

    def run_trial(n_particles, rng):
        """Run the 1D tracker once, return mean position error."""
        state = initialize_particles(
            np.array([0.0, 1.0]), np.diag([1.0, 0.25]), n_particles, rng=rng
        )
        errs = []
        for tx, z in zip(truths, zs):
            state = bootstrap_pf_step(
                state.particles, state.weights, z, f, h, Q_sample, R, rng=rng
            )
            x_hat = particle_mean(state.particles, state.weights)
            errs.append(abs(x_hat[0] - tx[0]))
        return np.mean(errs)

    for n in [50, 200, 1000]:
        errors = [run_trial(n, np.random.default_rng(seed))
                  for seed in range(20)]
        print(f"N={n:5d}: mean_error={np.mean(errors):.4f} "
              f"std={np.std(errors):.4f}")
    # N=   50: mean_error=0.1592 std=0.0171
    # N=  200: mean_error=0.1480 std=0.0121
    # N= 1000: mean_error=0.1473 std=0.0050

The mean error approaches the noise-limited floor while the trial-to-trial
standard deviation drops roughly as :math:`1/\sqrt{N_p}`.


Common Issues & Solutions
-------------------------

**Problem: Particle Collapse (All Weight on Few Particles)**

Symptom: ESS drops to < 10% of N

Solutions:

1. Increase resampling threshold (e.g., resample at 0.5*N instead of 0.2*N)
2. Use regularization/jitter
3. Increase number of particles
4. Check measurement likelihood (may be too informative)

.. code-block:: python

    def diagnose_collapse(weights):
        """Analyze weight distribution."""
        sorted_w = np.sort(weights)[::-1]
        top_10_percent = np.sum(sorted_w[: len(weights) // 10])

        print(f"Top 10% have {top_10_percent * 100:.1f}% of weight")
        if top_10_percent > 0.8:
            print("WARNING: Significant weight collapse")
            print("  - Consider more particles")
            print("  - Add regularization noise")
            print("  - Increase resampling threshold")

    diagnose_collapse(demo_weights)
    # Top 10% have 93.9% of weight
    # WARNING: Significant weight collapse
    #   - Consider more particles
    #   - Add regularization noise
    #   - Increase resampling threshold

**Problem: Particle Impoverishment (After Heavy Resampling)**

Symptom: Many identical particles, filter stuck

Solutions:

1. Use regularization with adaptive bandwidth
2. Mix in samples from proposal distribution
3. Lower resampling threshold (resample less often)

.. code-block:: python

    def regularized_resample(particles, weights, rng, scale=1.0):
        """Resample with adaptive regularization (Silverman's rule)."""
        new_particles = resample_systematic(particles, weights, rng=rng)

        n = len(particles)
        h_bw = scale * n ** (-1 / 5) * new_particles.std(axis=0)
        return new_particles + rng.normal(0.0, h_bw, size=new_particles.shape)

    jittered = regularized_resample(demo_particles, demo_weights, rng)
    print(f"distinct particles: {len(np.unique(jittered[:, 0]))}")
    # distinct particles: 500

**Problem: Filter Diverges (State Estimates Far from Reality)**

Solutions:

1. Check likelihood calculation for bugs
2. Verify process noise is appropriate
3. Increase particles
4. Use likelihood annealing (temper a sharp likelihood in stages)

.. code-block:: python

    def annealed_update(particles, weights, z, likelihood_fn, rng,
                        num_stages=5):
        """Incorporate a sharp likelihood gradually (tempering).

        At each stage, apply the likelihood raised to an increment of
        beta so that the full likelihood is absorbed after all stages,
        resampling between stages to keep the particle set healthy.
        """
        beta_prev = 0.0
        for stage in range(1, num_stages + 1):
            beta = stage / num_stages
            lik = np.array([likelihood_fn(z, p) for p in particles])
            weights = weights * lik ** (beta - beta_prev)
            weights = weights / weights.sum()

            particles, weights = resample_if_needed(
                particles, weights, rng=rng
            )
            beta_prev = beta
        return particles, weights

    def sharp_likelihood(z, x):
        return gaussian_likelihood(z, x[:1], np.array([[0.01]]))

    z_sharp = np.array([2.0])
    annealed, w = annealed_update(
        demo_particles, np.full(len(demo_particles), 1 / len(demo_particles)),
        z_sharp, sharp_likelihood, rng,
    )
    print(f"ESS after annealed update: {effective_sample_size(w):.1f}")
    # ESS after annealed update: 284.4


Best Practices
--------------

1. **Choose Particle Count Wisely**

   - Start with N=500 for 3D problems
   - Rule of thumb: N ~ 10^(d/2) for quick estimates
   - Monitor ESS; if < 100 typically, increase N

2. **Adaptive Parameters**

   - Resampling threshold (``resample_threshold`` in ``bootstrap_pf_step``): 0.3-0.7 of N
   - Regularization bandwidth: use Silverman's rule
   - Process noise: match actual motion uncertainty

3. **Monitor Filter Health**

   - Track ``effective_sample_size`` over time (should oscillate, not decay monotonically)
   - Check innovation sequences (should be white noise)
   - Visualize particle cloud (should spread over state space)

4. **Use Systematic Resampling**

   - Better variance than multinomial
   - Negligible overhead vs bootstrap
   - The default ``resample_method`` in ``bootstrap_pf_step``

5. **Combine with Other Methods**

   - Use Kalman filter as proposal distribution (proposal-guided PF)
   - Use particle filter for initialization of EKF
   - Hybrid systems for mixed Gaussian/non-Gaussian

6. **For Real-Time Systems**

   - Use smallest N that maintains accuracy (test offline first)
   - Compute ESS during update, resample only when needed
   - Profile particle filter speed vs system requirements


Troubleshooting
---------------

**Diagnostic Checklist:**

.. code-block:: python

    def diagnose_particle_filter(particles, weights, spread_floor=0.01):
        """Basic particle filter health checks."""
        issues = []

        # Check 1: effective sample size
        ess = effective_sample_size(weights)
        if ess < 0.2 * len(weights):
            issues.append("Severe weight collapse - increase particles")

        # Check 2: particle spread
        spread = particles.std(axis=0)
        if np.any(spread < spread_floor):
            issues.append("Particles too concentrated - increase noise")

        return issues

    print(diagnose_particle_filter(demo_particles, demo_weights))
    # ['Severe weight collapse - increase particles']

    print(diagnose_particle_filter(state.particles, state.weights))
    # []

Beyond these structural checks, validate estimates against ground truth in
simulation and check innovation whiteness; see :doc:`troubleshooting`.


See Also
~~~~~~~~

- :doc:`recipes` - Ready-to-use particle filter examples
- :doc:`troubleshooting` - Common particle filter issues
- :doc:`kalman_filter_tuning` - Parameter selection strategies
- API: ``pytcl.dynamic_estimation.particle_filters`` module for implementations
