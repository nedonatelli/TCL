"""
Additional tests to boost coverage for low-coverage modules.

This file targets modules with coverage below 50%:
- Particle filters (bootstrap.py)
- Singer dynamic model (singer.py)
- Statistics estimators
- Combinatorics
- Numerical integration (quadrature)
"""

import numpy as np

# =============================================================================
# Particle Filter Tests
# =============================================================================


class TestParticleFilters:
    """Tests for particle filter functions."""

    def test_resample_multinomial(self):
        """Test multinomial resampling."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            resample_multinomial,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        resampled = resample_multinomial(particles, weights, rng)

        assert resampled.shape == particles.shape
        # Higher weight particles should appear more often on average
        assert np.any(np.all(resampled == particles[3], axis=1))

    def test_resample_systematic(self):
        """Test systematic resampling."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            resample_systematic,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        resampled = resample_systematic(particles, weights, rng)

        assert resampled.shape == particles.shape

    def test_resample_residual(self):
        """Test residual resampling."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            resample_residual,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        # Give one particle very high weight to test deterministic copies
        weights = np.array([0.1, 0.6, 0.2, 0.1])

        resampled = resample_residual(particles, weights, rng)

        assert resampled.shape == particles.shape
        # Particle 1 (weight 0.6) should appear at least twice deterministically
        count = np.sum(np.all(resampled == particles[1], axis=1))
        assert count >= 2

    def test_effective_sample_size(self):
        """Test ESS computation."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            effective_sample_size,
        )

        # Uniform weights -> ESS = N
        N = 100
        uniform_weights = np.ones(N) / N
        ess_uniform = effective_sample_size(uniform_weights)
        assert np.isclose(ess_uniform, N)

        # One particle has all weight -> ESS = 1
        degenerate_weights = np.zeros(N)
        degenerate_weights[0] = 1.0
        ess_degen = effective_sample_size(degenerate_weights)
        assert np.isclose(ess_degen, 1.0)

    def test_bootstrap_pf_predict(self):
        """Test particle filter prediction step."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_predict,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

        def f(x):
            return x + np.array([0.1, 0.1])

        def Q_sample(N, rng):
            return rng.normal(0, 0.01, size=(N, 2))

        predicted = bootstrap_pf_predict(particles, f, Q_sample, rng)

        assert predicted.shape == particles.shape
        # Particles should have moved approximately by [0.1, 0.1]
        assert np.allclose(predicted, particles + 0.1, atol=0.1)

    def test_bootstrap_pf_update(self):
        """Test particle filter update step."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_update,
            gaussian_likelihood,
        )

        particles = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        weights = np.ones(3) / 3
        z = np.array([1.0, 1.0])
        R = np.eye(2) * 0.5

        def likelihood_func(z, x):
            return gaussian_likelihood(z, x, R)

        new_weights, log_lik = bootstrap_pf_update(
            particles, weights, z, likelihood_func
        )

        assert new_weights.shape == weights.shape
        assert np.isclose(np.sum(new_weights), 1.0)
        # Particle closest to measurement should have highest weight
        assert new_weights[1] > new_weights[0]
        assert new_weights[1] > new_weights[2]

    def test_gaussian_likelihood(self):
        """Test Gaussian likelihood computation."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            gaussian_likelihood,
        )

        z = np.array([0.0, 0.0])
        z_pred = np.array([0.0, 0.0])
        R = np.eye(2)

        # Zero innovation should give maximum likelihood
        lik = gaussian_likelihood(z, z_pred, R)
        assert lik > 0
        expected = 1.0 / (2 * np.pi)  # For 2D unit covariance at zero
        assert np.isclose(lik, expected)

        # Large innovation should give low likelihood
        z_pred_far = np.array([10.0, 10.0])
        lik_far = gaussian_likelihood(z, z_pred_far, R)
        assert lik_far < lik

    def test_gaussian_likelihood_singular(self):
        """Test Gaussian likelihood with singular covariance."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            gaussian_likelihood,
        )

        z = np.array([0.0, 0.0])
        z_pred = np.array([0.0, 0.0])
        R_singular = np.array([[1.0, 0.0], [0.0, 0.0]])  # Singular

        lik = gaussian_likelihood(z, z_pred, R_singular)
        assert lik == 0.0

    def test_bootstrap_pf_step(self):
        """Test complete particle filter step."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_step,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        weights = np.ones(5) / 5
        z = np.array([1.0])

        def f(x):
            return x

        def h(x):
            return x

        def Q_sample(N, rng):
            return rng.normal(0, 0.1, size=(N, 1))

        R = np.array([[0.1]])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            f,
            h,
            Q_sample,
            R,
            resample_threshold=0.5,
            resample_method="systematic",
            rng=rng,
        )

        assert state.particles.shape == particles.shape
        assert state.weights.shape == weights.shape
        assert np.isclose(np.sum(state.weights), 1.0)

    def test_bootstrap_pf_step_multinomial(self):
        """Test particle filter with multinomial resampling."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_step,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [1.0], [2.0], [3.0]])
        weights = np.ones(4) / 4
        z = np.array([1.5])

        def f(x):
            return x

        def h(x):
            return x

        def Q_sample(N, rng):
            return np.zeros((N, 1))

        R = np.array([[0.1]])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            f,
            h,
            Q_sample,
            R,
            resample_threshold=0.99,  # Force resampling
            resample_method="multinomial",
            rng=rng,
        )

        assert state.particles.shape == particles.shape

    def test_bootstrap_pf_step_residual(self):
        """Test particle filter with residual resampling."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_step,
        )

        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [1.0], [2.0], [3.0]])
        weights = np.ones(4) / 4
        z = np.array([1.5])

        def f(x):
            return x

        def h(x):
            return x

        def Q_sample(N, rng):
            return np.zeros((N, 1))

        R = np.array([[0.1]])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            f,
            h,
            Q_sample,
            R,
            resample_threshold=0.99,
            resample_method="residual",
            rng=rng,
        )

        assert state.particles.shape == particles.shape

    def test_bootstrap_pf_no_resample(self):
        """Test particle filter when ESS is high (no resampling needed)."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            bootstrap_pf_step,
        )

        rng = np.random.default_rng(42)
        # All particles at similar locations
        particles = np.array([[0.9], [1.0], [1.0], [1.1]])
        weights = np.ones(4) / 4
        z = np.array([1.0])

        def f(x):
            return x

        def h(x):
            return x

        def Q_sample(N, rng):
            return rng.normal(0, 0.01, size=(N, 1))

        R = np.array([[1.0]])  # Large noise -> uniform weights

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            f,
            h,
            Q_sample,
            R,
            resample_threshold=0.1,  # Low threshold -> no resampling
            resample_method="systematic",
            rng=rng,
        )

        assert state.particles.shape == particles.shape
        # Weights should not be uniform since no resampling occurred
        assert not np.allclose(state.weights, 0.25)

    def test_particle_mean(self):
        """Test weighted mean of particles."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import particle_mean

        particles = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        weights = np.array([0.5, 0.3, 0.2])

        mean = particle_mean(particles, weights)

        expected = (
            0.5 * np.array([0, 0]) + 0.3 * np.array([1, 1]) + 0.2 * np.array([2, 2])
        )
        assert np.allclose(mean, expected)

    def test_particle_covariance(self):
        """Test weighted covariance of particles."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            particle_covariance,
            particle_mean,
        )

        particles = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = np.ones(4) / 4

        cov = particle_covariance(particles, weights)

        assert cov.shape == (2, 2)
        # With precomputed mean
        mean = particle_mean(particles, weights)
        cov2 = particle_covariance(particles, weights, mean=mean)
        assert np.allclose(cov, cov2)

    def test_initialize_particles(self):
        """Test particle initialization from Gaussian prior."""
        from pytcl.dynamic_estimation.particle_filters.bootstrap import (
            initialize_particles,
        )

        rng = np.random.default_rng(42)
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.1
        N = 1000

        state = initialize_particles(x0, P0, N, rng)

        assert state.particles.shape == (N, 2)
        assert state.weights.shape == (N,)
        assert np.allclose(state.weights, 1.0 / N)
        # Mean should be close to x0
        assert np.allclose(np.mean(state.particles, axis=0), x0, atol=0.1)


# =============================================================================
# Singer Model Tests
# =============================================================================


class TestSingerModel:
    """Tests for Singer acceleration model."""

    def test_f_singer_1d(self):
        """Test 1D Singer model."""
        from pytcl.dynamic_models.discrete_time.singer import f_singer

        T = 1.0
        tau = 10.0

        F = f_singer(T, tau)

        assert F.shape == (3, 3)
        # Check structure
        assert F[0, 0] == 1.0
        assert F[0, 1] == T
        assert F[1, 0] == 0.0
        assert F[1, 1] == 1.0
        assert F[2, 0] == 0.0
        assert F[2, 1] == 0.0

        # Check alpha = exp(-T/tau)
        alpha = np.exp(-T / tau)
        assert np.isclose(F[2, 2], alpha)

    def test_f_singer_2d(self):
        """Test 2D Singer model."""
        from pytcl.dynamic_models.discrete_time.singer import f_singer, f_singer_2d

        T = 0.5
        tau = 5.0

        F = f_singer_2d(T, tau)
        F_expected = f_singer(T, tau, num_dims=2)

        assert F.shape == (6, 6)
        assert np.allclose(F, F_expected)
        # Check block diagonal structure
        assert np.allclose(F[:3, :3], F[3:6, 3:6])
        assert np.allclose(F[:3, 3:6], 0)
        assert np.allclose(F[3:6, :3], 0)

    def test_f_singer_3d(self):
        """Test 3D Singer model."""
        from pytcl.dynamic_models.discrete_time.singer import f_singer, f_singer_3d

        T = 0.1
        tau = 20.0

        F = f_singer_3d(T, tau)
        F_expected = f_singer(T, tau, num_dims=3)

        assert F.shape == (9, 9)
        assert np.allclose(F, F_expected)

    def test_f_singer_state_propagation(self):
        """Test state propagation with Singer model."""
        from pytcl.dynamic_models.discrete_time.singer import f_singer

        T = 0.1
        tau = 5.0
        F = f_singer(T, tau)

        # Initial state: [pos=0, vel=10, acc=2]
        x = np.array([0.0, 10.0, 2.0])

        # Propagate
        x_next = F @ x

        # Position should increase by approximately vel * T + 0.5 * acc * T^2
        expected_pos_approx = x[0] + x[1] * T + 0.5 * x[2] * T**2
        assert x_next[0] > x[0]
        assert np.isclose(x_next[0], expected_pos_approx, atol=0.1)

        # Velocity should increase by approximately acc * T
        expected_vel_approx = x[1] + x[2] * T
        assert np.isclose(x_next[1], expected_vel_approx, atol=0.5)

        # Acceleration should decay toward zero
        assert abs(x_next[2]) < abs(x[2])


# =============================================================================
# Statistics Estimators Tests
# =============================================================================


class TestStatisticsEstimators:
    """Tests for statistical estimator functions."""

    def test_weighted_mean(self):
        """Test weighted mean computation."""
        from pytcl.mathematical_functions.statistics.estimators import weighted_mean

        x = np.array([1.0, 2.0, 3.0, 4.0])
        weights = np.array([1.0, 1.0, 1.0, 1.0])

        mean = weighted_mean(x, weights)
        assert np.isclose(mean, 2.5)

        # Non-uniform weights
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        mean = weighted_mean(x, weights)
        expected = 0.1 * 1 + 0.2 * 2 + 0.3 * 3 + 0.4 * 4
        assert np.isclose(mean, expected)

    def test_weighted_var(self):
        """Test weighted variance computation."""
        from pytcl.mathematical_functions.statistics.estimators import weighted_var

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)

        var = weighted_var(x, weights)
        assert var > 0

        # With ddof correction
        var_corrected = weighted_var(x, weights, ddof=1)
        assert var_corrected > var

    def test_weighted_cov(self):
        """Test weighted covariance matrix."""
        from pytcl.mathematical_functions.statistics.estimators import weighted_cov

        np.random.seed(42)
        x = np.random.randn(100, 3)
        weights = np.ones(100)

        cov = weighted_cov(x, weights)

        assert cov.shape == (3, 3)
        # Symmetric
        assert np.allclose(cov, cov.T)
        # Positive semi-definite
        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals >= -1e-10)

    def test_weighted_cov_1d(self):
        """Test weighted covariance with 1D input."""
        from pytcl.mathematical_functions.statistics.estimators import weighted_cov

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = np.ones(5)

        cov = weighted_cov(x, weights)
        assert cov.shape == (1, 1)

    def test_sample_mean(self):
        """Test sample mean."""
        from pytcl.mathematical_functions.statistics.estimators import sample_mean

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean = sample_mean(x)
        assert np.isclose(mean, 3.0)

    def test_sample_var(self):
        """Test sample variance."""
        from pytcl.mathematical_functions.statistics.estimators import sample_var

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = sample_var(x, ddof=0)  # Population variance
        assert np.isclose(var, 2.0)

        var_sample = sample_var(x, ddof=1)  # Sample variance
        assert np.isclose(var_sample, 2.5)

    def test_sample_cov(self):
        """Test sample covariance."""
        from pytcl.mathematical_functions.statistics.estimators import sample_cov

        np.random.seed(42)
        x = np.random.randn(100, 2)
        cov = sample_cov(x)

        assert cov.shape == (2, 2)
        assert np.allclose(cov, cov.T)

    def test_sample_cov_1d(self):
        """Test sample covariance with 1D input."""
        from pytcl.mathematical_functions.statistics.estimators import sample_cov

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        var = sample_cov(x)
        assert np.isclose(var, 2.5)  # Sample variance with ddof=1

    def test_sample_cov_cross(self):
        """Test cross-covariance."""
        from pytcl.mathematical_functions.statistics.estimators import sample_cov

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        cov = sample_cov(x, y)

        assert cov.shape == (2, 2)
        # Perfect positive correlation
        assert cov[0, 1] > 0

    def test_sample_corr(self):
        """Test sample correlation."""
        from pytcl.mathematical_functions.statistics.estimators import sample_corr

        np.random.seed(42)
        x = np.random.randn(100, 3)
        corr = sample_corr(x)

        assert corr.shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(corr), 1.0)
        # Symmetric
        assert np.allclose(corr, corr.T)
        # All values between -1 and 1
        assert np.all(corr >= -1) and np.all(corr <= 1)

    def test_median(self):
        """Test median computation."""
        from pytcl.mathematical_functions.statistics.estimators import median

        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        med = median(x)
        assert med == 3.0

        x_even = np.array([1.0, 2.0, 3.0, 4.0])
        med_even = median(x_even)
        assert med_even == 2.5

    def test_mad(self):
        """Test median absolute deviation."""
        from pytcl.mathematical_functions.statistics.estimators import mad

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mad(x)
        assert result > 0

        # For normal data, MAD * 1.4826 ≈ std
        np.random.seed(42)
        normal_data = np.random.randn(10000)
        mad_scaled = mad(normal_data)
        assert np.isclose(mad_scaled, 1.0, atol=0.1)

    def test_iqr(self):
        """Test interquartile range."""
        from pytcl.mathematical_functions.statistics.estimators import iqr

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        result = iqr(x)
        # Q3 = 6.25, Q1 = 2.75, IQR = 3.5
        assert result > 0

    def test_skewness(self):
        """Test skewness computation."""
        from pytcl.mathematical_functions.statistics.estimators import skewness

        # Symmetric distribution -> skewness ≈ 0
        symmetric = np.array([-2, -1, 0, 1, 2])
        skew = skewness(symmetric)
        assert np.isclose(skew, 0.0, atol=1e-10)

        # Right-skewed distribution
        right_skewed = np.array([1, 2, 2, 3, 3, 3, 10])
        skew_right = skewness(right_skewed)
        assert skew_right > 0

    def test_kurtosis(self):
        """Test kurtosis computation."""
        from pytcl.mathematical_functions.statistics.estimators import kurtosis

        np.random.seed(42)
        normal_data = np.random.randn(10000)

        # Normal distribution has excess kurtosis ≈ 0
        kurt = kurtosis(normal_data, fisher=True)
        assert np.isclose(kurt, 0.0, atol=0.2)

        # Pearson kurtosis for normal is 3
        kurt_pearson = kurtosis(normal_data, fisher=False)
        assert np.isclose(kurt_pearson, 3.0, atol=0.2)

    def test_moment(self):
        """Test moment computation."""
        from pytcl.mathematical_functions.statistics.estimators import moment

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Second central moment = variance (population)
        m2 = moment(x, order=2, central=True)
        var = np.var(x, ddof=0)
        assert np.isclose(m2, var)

        # Raw moment
        m2_raw = moment(x, order=2, central=False)
        assert np.isclose(m2_raw, np.mean(x**2))

    def test_nees(self):
        """Test Normalized Estimation Error Squared."""
        from pytcl.mathematical_functions.statistics.estimators import nees

        error = np.array([1.0, 0.0])
        cov = np.eye(2)

        result = nees(error, cov)
        assert np.isclose(result, 1.0)

        # Multiple errors
        errors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        results = nees(errors, cov)
        assert results.shape == (3,)
        assert np.allclose(results, [1.0, 1.0, 2.0])

    def test_nis(self):
        """Test Normalized Innovation Squared."""
        from pytcl.mathematical_functions.statistics.estimators import nis

        innovation = np.array([0.5, 0.5])
        S = np.eye(2) * 0.25

        result = nis(innovation, S)
        expected = 0.5**2 / 0.25 + 0.5**2 / 0.25
        assert np.isclose(result, expected)


# =============================================================================
# Combinatorics Tests
# =============================================================================


class TestCombinatorics:
    """Tests for combinatorics functions."""

    def test_factorial(self):
        """Test factorial computation."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import factorial

        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800

    def test_binomial(self):
        """Test binomial coefficient."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import n_choose_k

        assert n_choose_k(5, 0) == 1
        assert n_choose_k(5, 5) == 1
        assert n_choose_k(5, 2) == 10
        assert n_choose_k(10, 3) == 120

    def test_multinomial(self):
        """Test multinomial coefficient."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            multinomial_coefficient,
        )

        # (3+2+1)! / (3! * 2! * 1!) = 720 / (6 * 2 * 1) = 60
        result = multinomial_coefficient(3, 2, 1)
        assert result == 60

    def test_permutations(self):
        """Test permutation count."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            n_permute_k,
        )

        # P(5, 3) = 5! / (5-3)! = 60
        assert n_permute_k(5, 3) == 60
        assert n_permute_k(5, 5) == 120
        assert n_permute_k(5, 0) == 1

    def test_combinations(self):
        """Test combination count (alias)."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            combinations,
        )

        # Just verify the iterator works
        result = list(combinations([1, 2, 3, 4, 5], 3))
        assert len(result) == 10

    def test_stirling_second(self):
        """Test Stirling numbers of the second kind."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            stirling_second,
        )

        # S(n, 1) = 1
        assert stirling_second(5, 1) == 1
        # S(n, n) = 1
        assert stirling_second(5, 5) == 1
        # S(4, 2) = 7
        assert stirling_second(4, 2) == 7

    def test_bell_number(self):
        """Test Bell numbers."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            bell_number,
        )

        # B(0) = 1, B(1) = 1, B(2) = 2, B(3) = 5, B(4) = 15, B(5) = 52
        assert bell_number(0) == 1
        assert bell_number(1) == 1
        assert bell_number(2) == 2
        assert bell_number(3) == 5
        assert bell_number(4) == 15
        assert bell_number(5) == 52

    def test_catalan_number(self):
        """Test Catalan numbers."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            catalan_number,
        )

        # C_0 = 1, C_1 = 1, C_2 = 2, C_3 = 5, C_4 = 14
        assert catalan_number(0) == 1
        assert catalan_number(1) == 1
        assert catalan_number(2) == 2
        assert catalan_number(3) == 5
        assert catalan_number(4) == 14

    def test_derangements(self):
        """Test derangement count (subfactorial)."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            subfactorial,
        )

        # D(0) = 1, D(1) = 0, D(2) = 1, D(3) = 2, D(4) = 9
        assert subfactorial(0) == 1
        assert subfactorial(1) == 0
        assert subfactorial(2) == 1
        assert subfactorial(3) == 2
        assert subfactorial(4) == 9

    def test_partition_count(self):
        """Test integer partition count."""
        from pytcl.mathematical_functions.combinatorics.combinatorics import (
            partition_count,
        )

        # p(0) = 1, p(1) = 1, p(2) = 2, p(3) = 3, p(4) = 5, p(5) = 7
        assert partition_count(0) == 1
        assert partition_count(1) == 1
        assert partition_count(4) == 5
        assert partition_count(5) == 7


# =============================================================================
# Numerical Integration (Quadrature) Tests
# =============================================================================


class TestQuadrature:
    """Tests for numerical integration functions."""

    def test_gauss_legendre(self):
        """Test Gauss-Legendre quadrature."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            gauss_legendre,
        )

        # Integrate x^2 from -1 to 1
        n = 5
        nodes, weights = gauss_legendre(n)

        assert len(nodes) == n
        assert len(weights) == n
        assert np.isclose(np.sum(weights), 2.0)  # Integral of 1 from -1 to 1

        # x^2 integrated from -1 to 1 = 2/3
        integral = np.sum(weights * nodes**2)
        assert np.isclose(integral, 2.0 / 3.0)

    def test_gauss_hermite(self):
        """Test Gauss-Hermite quadrature."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            gauss_hermite,
        )

        n = 5
        nodes, weights = gauss_hermite(n)

        assert len(nodes) == n
        assert len(weights) == n

        # Integral of x^2 * exp(-x^2) = sqrt(pi)/2
        integral = np.sum(weights * nodes**2)
        assert np.isclose(integral, np.sqrt(np.pi) / 2, atol=1e-10)

    def test_gauss_laguerre(self):
        """Test Gauss-Laguerre quadrature."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            gauss_laguerre,
        )

        n = 10
        nodes, weights = gauss_laguerre(n)

        assert len(nodes) == n
        assert len(weights) == n
        assert np.all(nodes >= 0)  # Nodes are non-negative

        # Integral of exp(-x) from 0 to inf = 1
        integral = np.sum(weights)
        assert np.isclose(integral, 1.0, atol=1e-10)

    def test_dblquad(self):
        """Test 2D integration with dblquad."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            dblquad,
        )

        # Integrate 1 over [0, 1] x [0, 1] = 1
        def f(y, x):
            return 1.0

        result, error = dblquad(f, 0, 1, 0, 1)
        assert np.isclose(result, 1.0, atol=1e-6)

    def test_quad(self):
        """Test 1D numerical integration."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import quad

        # Integrate x^2 from 0 to 1 = 1/3
        def f(x):
            return x**2

        result, error = quad(f, 0, 1)
        assert np.isclose(result, 1.0 / 3.0, atol=1e-6)

    def test_simpson(self):
        """Test Simpson's rule."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            simpson,
        )

        # Integrate x^2 from 0 to 1 using pre-computed y values
        x = np.linspace(0, 1, 101)
        y = x**2

        result = simpson(y, x)
        assert np.isclose(result, 1.0 / 3.0, atol=1e-6)

    def test_trapezoid(self):
        """Test trapezoidal rule."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            trapezoid,
        )

        # Integrate sin(x) from 0 to pi using pre-computed y values
        x = np.linspace(0, np.pi, 1001)
        y = np.sin(x)

        result = trapezoid(y, x)
        assert np.isclose(result, 2.0, atol=1e-4)

    def test_romberg(self):
        """Test Romberg integration."""
        from pytcl.mathematical_functions.numerical_integration.quadrature import (
            romberg,
        )

        # Integrate exp(x) from 0 to 1 = e - 1
        def f(x):
            return np.exp(x)

        result = romberg(f, 0, 1)
        assert np.isclose(result, np.e - 1, atol=1e-10)


# =============================================================================
# Great Circle / Rhumb Navigation Coverage Boost
# =============================================================================


class TestNavigationCoverage:
    """Additional tests to boost navigation coverage."""

    def test_great_circle_waypoints(self):
        """Test great circle waypoint generation."""
        from pytcl.navigation.great_circle import great_circle_waypoints

        # NYC to London
        lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
        lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)

        lats, lons = great_circle_waypoints(lat1, lon1, lat2, lon2, n_points=5)

        assert len(lats) == 5
        assert len(lons) == 5
        # First waypoint should be start
        assert np.isclose(lats[0], lat1)
        assert np.isclose(lons[0], lon1)
        # Last waypoint should be end
        assert np.isclose(lats[-1], lat2)
        assert np.isclose(lons[-1], lon2)

    def test_great_circle_waypoint_single(self):
        """Test single great circle waypoint."""
        from pytcl.navigation.great_circle import great_circle_waypoint

        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(0), np.radians(90)

        # Get midpoint (fraction=0.5)
        result = great_circle_waypoint(lat1, lon1, lat2, lon2, 0.5)

        # Midpoint on equator between 0 and 90 should be at 45
        assert np.isclose(result.lat, 0, atol=1e-10)
        assert np.isclose(result.lon, np.radians(45), atol=1e-10)

    def test_great_circle_tdoa(self):
        """Test TDOA localization on sphere."""
        from pytcl.navigation.great_circle import great_circle_tdoa_loc

        # Three receivers
        lat1, lon1 = 0.0, 0.0  # Receiver 1 at origin
        lat2, lon2 = 0.0, np.radians(10)  # Receiver 2 at 0, 10E
        lat3, lon3 = np.radians(10), np.radians(5)  # Receiver 3 at 10N, 5E

        # Zero TDOAs -> source equidistant from all receivers
        tdoa12 = 0.0
        tdoa13 = 0.0

        result1, result2 = great_circle_tdoa_loc(
            lat1, lon1, lat2, lon2, lat3, lon3, tdoa12, tdoa13
        )

        # Should return at least one solution
        assert result1 is not None or result2 is not None

    def test_rhumb_bearing_meridian(self):
        """Test rhumb bearing along meridian."""
        from pytcl.navigation.rhumb import rhumb_bearing

        lat1, lon1 = np.radians(0), np.radians(0)
        lat2, lon2 = np.radians(45), np.radians(0)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)

        # Due north
        assert np.isclose(bearing, 0, atol=1e-10)

    def test_rhumb_bearing_parallel(self):
        """Test rhumb bearing along parallel."""
        from pytcl.navigation.rhumb import rhumb_bearing

        lat1, lon1 = np.radians(45), np.radians(0)
        lat2, lon2 = np.radians(45), np.radians(45)

        bearing = rhumb_bearing(lat1, lon1, lat2, lon2)

        # Due east
        assert np.isclose(bearing, np.pi / 2)
