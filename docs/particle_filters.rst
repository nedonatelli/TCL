Particle Filters & Non-Gaussian Estimation
==========================================

*Comprehensive guide to sequential Monte Carlo (SMC) methods for tracking non-Gaussian, nonlinear systems.*

Particle filters represent probability distributions as weighted samples (particles), enabling estimation in highly nonlinear and non-Gaussian scenarios where Kalman filters fail.

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

Represent state uncertainty as a set of $N_p$ weighted samples:

$$
p(x_k | z_{1:k}) \approx \sum_{i=1}^{N_p} w_k^{(i)} \delta(x_k - x_k^{(i)})
$$

Where:
- $x_k^{(i)}$: i-th particle (state sample)
- $w_k^{(i)}$: weight of i-th particle
- $\delta()$: Dirac delta function
- Weights normalize to 1: $\sum_i w_k^{(i)} = 1$

**Key Equations:**

1. **Prediction**: Sample from motion model
   $$x_k^{(i)} \sim p(x_k | x_{k-1}^{(i)})$$

2. **Update**: Compute importance weights
   $$w_k^{(i)} \propto p(z_k | x_k^{(i)}) \cdot w_{k-1}^{(i)}$$

3. **Normalize**: 
   $$\hat{w}_k^{(i)} = \frac{w_k^{(i)}}{\sum_j w_k^{(j)}}$$

4. **Resample** (if needed): Generate new particles from weighted distribution


Bootstrap Particle Filter
-------------------------

**Algorithm: Sequential Importance Resampling (SIR)**

.. code-block:: python

    import numpy as np
    from scipy.stats import norm
    
    class BootstrapParticleFilter:
        """Sequential Importance Resampling particle filter."""
        
        def __init__(self, num_particles=1000, process_noise=1.0, 
                     measurement_noise=1.0):
            """
            Args:
                num_particles: number of particles
                process_noise: standard deviation of motion model noise
                measurement_noise: standard deviation of measurement noise
            """
            self.N = num_particles
            self.q = process_noise  # motion model noise std
            self.r = measurement_noise  # measurement noise std
            
            # Initialize particles uniformly
            self.particles = np.random.normal(0, 10, (self.N, 3))
            self.weights = np.ones(self.N) / self.N
        
        def predict(self, dt):
            """Predict step: propagate particles using motion model."""
            # Simple constant velocity model: x = x + v*dt
            # Velocity is state[1]
            self.particles[:, 0] += self.particles[:, 1] * dt
            
            # Add process noise (motion uncertainty)
            self.particles += np.random.normal(0, self.q, self.particles.shape)
        
        def update(self, measurement):
            """Update step: weight particles by measurement likelihood."""
            # Measurement likelihood: how well each particle explains measurement
            # Measurement is position only (state[0])
            predicted_meas = self.particles[:, 0]
            
            # Gaussian likelihood
            likelihood = norm.pdf(measurement, loc=predicted_meas, 
                                 scale=self.r)
            
            # Update weights
            self.weights *= likelihood
            self.weights /= np.sum(self.weights)  # normalize
            
            # Resample if needed (effective sample size check)
            self._resample_if_needed()
        
        def _resample_if_needed(self):
            """Check effective sample size and resample if degenerate."""
            # Effective sample size: N_eff = 1 / sum(w^2)
            n_eff = 1.0 / np.sum(self.weights**2)
            
            # Threshold: resample if N_eff < 0.5 * N
            if n_eff < 0.5 * self.N:
                self._resample()
        
        def _resample(self):
            """Resample particles (systematic resampling)."""
            # Copy particles with probability proportional to weight
            indices = self._systematic_resample()
            
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N
        
        def _systematic_resample(self):
            """Systematic resampling: reduced variance."""
            N = self.N
            indices = np.zeros(N, dtype=int)
            
            cumsum = np.cumsum(self.weights)
            i, j = 0, 0
            u = np.random.uniform(0, 1/N)
            
            while i < N:
                if u < cumsum[j]:
                    indices[i] = j
                    i += 1
                    u += 1/N
                else:
                    j += 1
            
            return indices
        
        def estimate_state(self):
            """Compute weighted mean estimate."""
            return np.average(self.particles, weights=self.weights, axis=0)
        
        def estimate_covariance(self):
            """Compute weighted covariance."""
            mean = self.estimate_state()
            centered = self.particles - mean
            cov = np.average(centered**2, weights=self.weights, axis=0)
            return np.diag(cov)


**Tracking Example:**

.. code-block:: python

    def track_with_particle_filter():
        """Example: track object with nonlinear motion."""
        pf = BootstrapParticleFilter(num_particles=500)
        
        # Simulated measurements (nonlinear target with outliers)
        true_state = np.array([0, 5, 0.1])  # [position, velocity, accel]
        measurements = []
        for t in np.arange(0, 10, 0.1):
            true_state[0] += true_state[1] * 0.1
            true_state[1] += true_state[2] * 0.1
            
            # Nonlinear measurement: range = sqrt(x^2 + y^2) - only see distance
            meas = np.sqrt(true_state[0]**2 + 1) + np.random.normal(0, 0.1)
            measurements.append(meas)
        
        # Filter
        estimates = []
        for z in measurements:
            pf.predict(0.1)
            pf.update(z)
            estimates.append(pf.estimate_state())
        
        return np.array(estimates)


Resampling Strategies
--------------------

**Problem: Particle Degeneracy**

As filter runs, most particles get zero weight (weight collapse). Solution: resample.

**1. Multinomial Resampling (Simple)**

.. code-block:: python

    def multinomial_resample(particles, weights):
        """Basic resampling: sample with replacement per weight."""
        N = len(particles)
        indices = np.random.choice(N, size=N, p=weights)
        return particles[indices]
    
    # Variance: high (sample variance ~O(1))


**2. Systematic Resampling (Best for Most Cases)**

.. code-block:: python

    def systematic_resample(particles, weights):
        """
        Reduced variance resampling.
        
        Deterministic positions with random offset reduce variance.
        """
        N = len(particles)
        indices = np.zeros(N, dtype=int)
        
        cumsum = np.cumsum(weights)
        i, j = 0, 0
        u = np.random.uniform(0, 1/N)
        
        while i < N:
            if u < cumsum[j]:
                indices[i] = j
                i += 1
                u += 1/N
            else:
                j += 1
        
        return particles[indices]
    
    # Variance: low (sample variance ~O(1/N²))


**3. Stratified Resampling**

.. code-block:: python

    def stratified_resample(particles, weights):
        """
        Similar to systematic but with stratified sampling.
        Divides probability into strata.
        """
        N = len(particles)
        indices = np.zeros(N, dtype=int)
        
        cumsum = np.cumsum(weights)
        i, j = 0, 0
        
        for i in range(N):
            u = (i + np.random.uniform(0, 1)) / N
            while u > cumsum[j]:
                j += 1
            indices[i] = j
        
        return particles[indices]
    
    # Variance: intermediate
    

**4. Residual Resampling (Deterministic)**

.. code-block:: python

    def residual_resample(particles, weights):
        """
        Deterministic: keep floor(N*w_i) copies of each particle,
        then randomly fill remaining slots.
        """
        N = len(particles)
        indices = []
        
        # Deterministic: keep low-variance copies
        for i, w in enumerate(weights):
            num_copies = int(np.floor(N * w))
            indices.extend([i] * num_copies)
        
        # Stochastic: fill remaining slots
        residual_weights = (N * weights) % 1  # fractional parts
        residual_weights /= np.sum(residual_weights)
        
        remaining = N - len(indices)
        remaining_indices = np.random.choice(
            N, size=remaining, p=residual_weights
        )
        indices.extend(remaining_indices)
        
        return particles[np.array(indices)]
    
    # Variance: very low (completely deterministic except for remainder)


**Resampling Comparison:**

=========== ================ ============== ================
Method      Complexity       Variance       Stability
=========== ================ ============== ================
Multinomial O(N log N)       O(1)           Unstable
Systematic  O(N)             O(1/N²)        Robust
Stratified  O(N)             O(1/N)         Very Good
Residual    O(N log N)       ~0             Excellent
=========== ================ ============== ================

**Recommended:** Systematic or stratified for most applications.


Degeneracy Detection & Solutions
--------------------------------

**Effective Sample Size (ESS):**

.. code-block:: python

    def compute_ess(weights):
        """
        Effective sample size: N_eff = 1 / sum(w_i^2)
        
        Ranges from 1 (all weight on one particle) to N (uniform).
        Resample when N_eff < threshold * N (typically 0.5).
        """
        return 1.0 / np.sum(weights**2)
    
    def should_resample(weights, threshold=0.5):
        """Check if resampling is needed."""
        N = len(weights)
        n_eff = compute_ess(weights)
        return n_eff < threshold * N


**Solutions to Degeneracy:**

1. **Adaptive Resampling**: Resample only when needed (above)

2. **Regularization**: Add jitter to particles after resampling

.. code-block:: python

    def resample_with_regularization(particles, weights, bandwidth=1.0):
        """Resample and add regularization noise."""
        # Standard resampling
        indices = systematic_resample(particles, weights)
        particles_resampled = particles[indices]
        
        # Add regularization: kernel density estimation
        h = bandwidth * particles.std(axis=0)  # adaptive bandwidth
        noise = np.random.normal(0, h, particles_resampled.shape)
        
        particles_resampled += noise
        return particles_resampled


3. **Look-Ahead Resampling**: Predict degeneracy and resample early

.. code-block:: python

    def predictive_ess(weights, measurement_likelihood_variance):
        """Estimate ESS after next measurement."""
        # Rough heuristic: shrinkage based on variance
        predicted_weights = weights * (1 - measurement_likelihood_variance)
        predicted_weights /= np.sum(predicted_weights)
        return compute_ess(predicted_weights)


Sequential Importance Sampling (SIS)
------------------------------------

**More General Particle Filter Framework:**

.. code-block:: python

    class SequentialImportanceSampling:
        """SIS: generalization of bootstrap PF."""
        
        def __init__(self, proposal, likelihood, num_particles=1000):
            """
            Args:
                proposal: function(state) -> new_state (motion model)
                likelihood: function(measurement, state) -> scalar
                num_particles: number of particles
            """
            self.proposal = proposal
            self.likelihood = likelihood
            self.N = num_particles
            
            self.particles = np.random.randn(self.N, 3)
            self.weights = np.ones(self.N) / self.N
        
        def step(self, measurement):
            """One SIS update step."""
            # Predict from proposal distribution
            self.particles = np.array([
                self.proposal(p) for p in self.particles
            ])
            
            # Update weights by likelihood
            likelihoods = np.array([
                self.likelihood(measurement, p) for p in self.particles
            ])
            
            self.weights *= likelihoods
            self.weights /= np.sum(self.weights)
            
            # Optional: resample if degenerate
            if compute_ess(self.weights) < 0.5 * self.N:
                self._resample()
        
        def _resample(self):
            """Internal resampling."""
            indices = systematic_resample(self.particles, self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.N) / self.N


Practical Implementation
------------------------

**Complete Tracking System:**

.. code-block:: python

    class NonlinearTracker:
        """Full particle filter tracker for nonlinear, non-Gaussian systems."""
        
        def __init__(self, num_particles=500, process_noise=0.1, 
                     measurement_noise=1.0):
            self.pf = BootstrapParticleFilter(num_particles, 
                                             process_noise, 
                                             measurement_noise)
            self.history = {
                'states': [],
                'measurements': [],
                'uncertainties': []
            }
        
        def filter_step(self, measurement, dt=0.1):
            """One complete filter cycle."""
            # Predict
            self.pf.predict(dt)
            
            # Update
            self.pf.update(measurement)
            
            # Estimate
            state = self.pf.estimate_state()
            cov = self.pf.estimate_covariance()
            
            # Log
            self.history['states'].append(state)
            self.history['measurements'].append(measurement)
            self.history['uncertainties'].append(cov)
            
            return state, cov
        
        def get_particle_cloud(self):
            """Return current particles for visualization."""
            return self.pf.particles
        
        def get_particle_weights(self):
            """Return particle importance weights."""
            return self.pf.weights


**Multi-Target Extension:**

.. code-block:: python

    class MultiTargetParticleFilter:
        """Manage filters for multiple targets."""
        
        def __init__(self, num_targets, particles_per_target=500):
            self.filters = [
                BootstrapParticleFilter(num_particles=particles_per_target)
                for _ in range(num_targets)
            ]
        
        def predict_all(self, dt):
            """Predict all targets."""
            for pf in self.filters:
                pf.predict(dt)
        
        def update_with_associations(self, measurements, associations):
            """
            Update targets with measurements.
            
            Args:
                measurements: list of measurement vectors
                associations: list of (measurement_idx, target_idx) tuples
            """
            # Mark which targets got measurements
            updated = set()
            
            for meas_idx, target_idx in associations:
                self.filters[target_idx].update(measurements[meas_idx])
                updated.add(target_idx)
            
            # Targets without measurements: just predict next step
            for target_idx in range(len(self.filters)):
                if target_idx not in updated:
                    self.filters[target_idx].predict(0.1)


Performance & Efficiency
------------------------

**Computational Complexity:**

- **Prediction**: O(N_p × d) where d = state dimension
- **Update**: O(N_p) for likelihood evaluation
- **Resampling**: O(N_p) to O(N_p log N_p) depending on method
- **Overall**: O(N_p) per cycle (most efficient SMC method)

**Memory Considerations:**

.. code-block:: python

    def estimate_memory_usage(num_particles, state_dim, num_timesteps):
        """Estimate memory for storing particle histories."""
        bytes_per_particle = state_dim * 8  # double precision
        bytes_per_weight = 8
        
        # Per timestep
        bytes_per_step = (bytes_per_particle + bytes_per_weight) * num_particles
        
        # Total
        total_bytes = bytes_per_step * num_timesteps
        total_mb = total_bytes / (1024**2)
        
        return total_mb
    
    # Example: 1000 particles, 10 states, 1000 timesteps
    # ~ 80 MB (manageable)


**Scaling with Particle Count:**

.. code-block:: python

    def performance_analysis():
        """Analyze filter accuracy vs particle count."""
        num_trials = 100
        particle_counts = [100, 500, 1000, 5000]
        errors = {n: [] for n in particle_counts}
        
        for n_particles in particle_counts:
            for trial in range(num_trials):
                pf = BootstrapParticleFilter(num_particles=n_particles)
                # Run filter, compute error
                error = run_filter(pf, ground_truth)
                errors[n_particles].append(error)
        
        # Plot error vs particles (should be ~1/sqrt(N))
        for n, errs in errors.items():
            print(f"N={n}: mean_error={np.mean(errs):.4f} " +
                  f"std={np.std(errs):.4f}")


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
        top_10_percent = np.sum(sorted_w[:len(weights)//10])
        
        print(f"Top 10% have {top_10_percent*100:.1f}% of weight")
        if top_10_percent > 0.8:
            print("WARNING: Significant weight collapse")
            print("  - Consider more particles")
            print("  - Add regularization noise")
            print("  - Increase resampling threshold")


**Problem: Particle Impoverishment (After Heavy Resampling)**

Symptom: Many identical particles, filter stuck

Solutions:
1. Use regularization with adaptive bandwidth
2. Mix in samples from proposal distribution
3. Lower resampling threshold (resample less often)

.. code-block:: python

    def regularized_resample(particles, weights, num_particles, scale=1.0):
        """Resample with adaptive regularization."""
        # Standard resampling
        indices = systematic_resample(particles, weights)
        new_particles = particles[indices].copy()
        
        # Adaptive bandwidth (Silverman's rule)
        h = scale * np.power(num_particles, -1/5) * new_particles.std(axis=0)
        
        # Add regularization noise
        new_particles += np.random.normal(0, h, new_particles.shape)
        
        return new_particles


**Problem: Filter Diverges (State Estimates Far from Reality)**

Solutions:
1. Check likelihood calculation for bugs
2. Verify process noise is appropriate
3. Increase particles
4. Use likelihood annealing (gradually increase impact)

.. code-block:: python

    def annealed_update(particles, weights, measurements, 
                       likelihood_fn, num_iterations=5):
        """Anneal measurement incorporation over iterations."""
        for iteration in range(num_iterations):
            # Fractional power of likelihood (annealing)
            beta = (iteration + 1) / num_iterations
            
            # Apply fractional likelihood
            partial_likelihood = np.array([
                likelihood_fn(measurements[p])**beta 
                for p in particles
            ])
            
            weights *= partial_likelihood
            weights /= np.sum(weights)
        
        return weights


Best Practices
--------------

1. **Choose Particle Count Wisely**
   - Start with N=500 for 3D problems
   - Rule of thumb: N ~ 10^(d/2) for quick estimates
   - Monitor ESS; if < 100 typically, increase N

2. **Adaptive Parameters**
   - Resampling threshold: 0.3-0.7 of N
   - Regularization bandwidth: use Silverman's rule
   - Process noise: match actual motion uncertainty

3. **Monitor Filter Health**
   - Track ESS over time (should be oscillating, not monotone)
   - Check innovation sequences (should be white noise)
   - Visualize particle cloud (should spread over state space)

4. **Use Systematic Resampling**
   - Better variance than multinomial
   - Negligible overhead vs bootstrap
   - More stable than stratified for most cases

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

    def diagnose_particle_filter(pf, measurements, ground_truth):
        """Comprehensive filter diagnostics."""
        
        issues = []
        
        # Check 1: ESS trajectory
        ess_history = []
        for z in measurements:
            pf.update(z)
            ess_history.append(compute_ess(pf.weights))
        
        if np.mean(ess_history) < 0.2 * pf.N:
            issues.append("Severe weight collapse - increase particles")
        
        # Check 2: Particle spread
        spread = np.std(pf.particles, axis=0)
        if np.any(spread < 0.01):
            issues.append("Particles too concentrated - increase noise")
        
        # Check 3: Estimate error
        estimates = [pf.estimate_state() for _ in measurements]
        errors = np.array([np.linalg.norm(e[:3] - gt[:3]) 
                          for e, gt in zip(estimates, ground_truth)])
        
        if np.mean(errors) > np.std(ground_truth) * 10:
            issues.append("Large estimation errors - check likelihood")
        
        return issues


See Also
~~~~~~~~

- :doc:`recipes` - Ready-to-use particle filter examples
- :doc:`troubleshooting` - Common particle filter issues
- :doc:`kalman_filter_tuning` - Parameter selection strategies
- API: `tcl.particle_filters` module for implementations
