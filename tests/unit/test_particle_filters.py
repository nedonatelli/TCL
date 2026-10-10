"""
Tests for bootstrap particle filter functions.

Tests cover:
- Resampling methods (multinomial, systematic, residual)
- Effective sample size
- Prediction and update steps
- Gaussian likelihood
- Complete particle filter step
- Particle mean and covariance
- Particle initialization
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.particle_filters.bootstrap import (
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


class TestResampling:
    """Tests for particle resampling methods."""

    def test_resample_multinomial(self):
        """Test multinomial resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        resampled = resample_multinomial(particles, weights, rng)

        assert resampled.shape == particles.shape
        assert np.any(np.all(resampled == particles[3], axis=1))

    def test_resample_systematic(self):
        """Test systematic resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        weights = np.array([0.1, 0.2, 0.3, 0.4])

        resampled = resample_systematic(particles, weights, rng)

        assert resampled.shape == particles.shape

    def test_resample_residual(self):
        """Test residual resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        weights = np.array([0.1, 0.6, 0.2, 0.1])

        resampled = resample_residual(particles, weights, rng)

        assert resampled.shape == particles.shape
        count = np.sum(np.all(resampled == particles[1], axis=1))
        assert count >= 2

    BAD_WEIGHTS = [np.full(4, 0.125), np.array([0.5, 0.5, 0.5, 0.5]), np.zeros(4)]

    @pytest.mark.parametrize("w", BAD_WEIGHTS)
    @pytest.mark.parametrize(
        "fn", [resample_systematic, resample_multinomial, resample_residual]
    )
    def test_every_resampler_rejects_unnormalized_weights(self, fn, w):
        """Defect test: all three resamplers must reject weights that do
        not sum to 1, instead of two of them silently returning a biased
        sample (resample_systematic) or one warning then raising for the
        wrong reason (resample_residual's divide-by-zero on all-zero
        weights)."""
        with pytest.raises(ValueError, match="sum to 1"):
            fn(np.arange(4.0), w, np.random.default_rng(0))

    @pytest.mark.parametrize(
        "fn", [resample_systematic, resample_multinomial, resample_residual]
    )
    def test_normalization_tolerance_boundary(self, fn):
        """Regression guard: the band is sqrt(eps) = 1.4901e-8, the one
        numpy.random.Generator.choice applies (measured: it accepts a
        1.4e-8 deviation and rejects 2.3e-8). 1e-9 and 1.4e-8 are inside
        and must be accepted -- the former was rejected by an earlier
        2.22e-13 band although it resamples correctly. 2.3e-8 and 1e-6
        are outside and must be rejected.
        """
        particles = np.arange(4.0).reshape(4, 1)
        for dev in (6.661e-16, 1e-9, 1.4e-8):
            w = np.full(4, 0.25)
            w[-1] += dev
            fn(particles, w, np.random.default_rng(0))

        for dev in (2.3e-8, 1e-6):
            w = np.full(4, 0.25)
            w[-1] += dev
            with pytest.raises(ValueError, match="sum to 1"):
                fn(particles, w, np.random.default_rng(0))

    def test_systematic_accepts_1e9_deviation_with_correct_result(self):
        """Defect test: [0.25]*3 + [0.25 + 1e-9] raised ValueError under the
        2.22e-13 band; pre-guard it returned [0, 1, 2, 3], which is correct."""
        w = np.array([0.25, 0.25, 0.25, 0.25 + 1e-9])
        out = resample_systematic(np.arange(4.0), w, np.random.default_rng(0))
        np.testing.assert_array_equal(out, [0.0, 1.0, 2.0, 3.0])

    @pytest.mark.parametrize(
        "fn", [resample_systematic, resample_multinomial, resample_residual]
    )
    def test_validation_message_points_at_a_real_notes_section(self, fn):
        """Defect test: the error promised a Notes section none of the
        resamplers had."""
        assert "Notes\n    -----" in fn.__doc__
        with pytest.raises(ValueError, match="Notes section"):
            fn(np.arange(4.0), np.zeros(4), np.random.default_rng(0))

    def test_systematic_resampling_is_unbiased_on_normalized_weights(self):
        """PROPERTY: expected counts track the weights.

        resample_systematic returns a copy of the selected particle rows,
        not their indices, so np.bincount only recovers the selection
        counts here because the particle values happen to equal their
        own indices (0.0, 1.0, 2.0, 3.0).
        """
        w = np.array([0.1, 0.2, 0.3, 0.4])
        counts = np.zeros(4)
        for seed in range(2000):
            idx = resample_systematic(
                np.arange(4.0), w, rng=np.random.default_rng(seed)
            )
            counts += np.bincount(idx.astype(int), minlength=4)
        assert np.allclose(counts / counts.sum(), w, atol=0.01)


class TestEffectiveSampleSize:
    """Tests for ESS computation."""

    def test_uniform_weights(self):
        """Test ESS with uniform weights equals N."""
        N = 100
        uniform_weights = np.ones(N) / N
        ess_uniform = effective_sample_size(uniform_weights)
        assert np.isclose(ess_uniform, N)

    def test_degenerate_weights(self):
        """Test ESS with degenerate weights equals 1."""
        N = 100
        degenerate_weights = np.zeros(N)
        degenerate_weights[0] = 1.0
        ess_degen = effective_sample_size(degenerate_weights)
        assert np.isclose(ess_degen, 1.0)


class TestGaussianLikelihood:
    """Tests for Gaussian likelihood computation."""

    def test_zero_innovation(self):
        """Test Gaussian likelihood at zero innovation."""
        z = np.array([0.0, 0.0])
        z_pred = np.array([0.0, 0.0])
        R = np.eye(2)

        lik = gaussian_likelihood(z, z_pred, R)
        assert lik > 0
        expected = 1.0 / (2 * np.pi)
        assert np.isclose(lik, expected)

    def test_large_innovation(self):
        """Test likelihood decreases with distance."""
        z = np.array([0.0, 0.0])
        R = np.eye(2)

        lik_near = gaussian_likelihood(z, np.array([0.0, 0.0]), R)
        lik_far = gaussian_likelihood(z, np.array([10.0, 10.0]), R)
        assert lik_far < lik_near

    def test_singular_covariance(self):
        """Test Gaussian likelihood with singular covariance."""
        z = np.array([0.0, 0.0])
        z_pred = np.array([0.0, 0.0])
        R_singular = np.array([[1.0, 0.0], [0.0, 0.0]])

        lik = gaussian_likelihood(z, z_pred, R_singular)
        assert lik == 0.0


class TestBootstrapPF:
    """Tests for bootstrap particle filter prediction and update."""

    def test_predict(self):
        """Test particle filter prediction step."""
        rng = np.random.default_rng(42)
        particles = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])

        def f(x):
            return x + np.array([0.1, 0.1])

        def Q_sample(N, rng):
            return rng.normal(0, 0.01, size=(N, 2))

        predicted = bootstrap_pf_predict(particles, f, Q_sample, rng)

        assert predicted.shape == particles.shape
        assert np.allclose(predicted, particles + 0.1, atol=0.1)

    def test_update(self):
        """Test particle filter update step."""
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
        assert new_weights[1] > new_weights[0]
        assert new_weights[1] > new_weights[2]

    def test_step_systematic(self):
        """Test complete particle filter step with systematic resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        weights = np.ones(5) / 5
        z = np.array([1.0])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            lambda x: x,
            lambda x: x,
            lambda N, rng: rng.normal(0, 0.1, size=(N, 1)),
            np.array([[0.1]]),
            resample_threshold=0.5,
            resample_method="systematic",
            rng=rng,
        )

        assert state.particles.shape == particles.shape
        assert state.weights.shape == weights.shape
        assert np.isclose(np.sum(state.weights), 1.0)

    def test_step_multinomial(self):
        """Test particle filter with multinomial resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [1.0], [2.0], [3.0]])
        weights = np.ones(4) / 4
        z = np.array([1.5])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            lambda x: x,
            lambda x: x,
            lambda N, rng: np.zeros((N, 1)),
            np.array([[0.1]]),
            resample_threshold=0.99,
            resample_method="multinomial",
            rng=rng,
        )

        assert state.particles.shape == particles.shape

    def test_step_residual(self):
        """Test particle filter with residual resampling."""
        rng = np.random.default_rng(42)
        particles = np.array([[0.0], [1.0], [2.0], [3.0]])
        weights = np.ones(4) / 4
        z = np.array([1.5])

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            lambda x: x,
            lambda x: x,
            lambda N, rng: np.zeros((N, 1)),
            np.array([[0.1]]),
            resample_threshold=0.99,
            resample_method="residual",
            rng=rng,
        )

        assert state.particles.shape == particles.shape

    def test_no_resample_high_ess(self):
        """Test particle filter when ESS is high (no resampling needed)."""
        rng = np.random.default_rng(42)
        particles = np.array([[0.9], [1.0], [1.0], [1.1]])
        weights = np.ones(4) / 4
        z = np.array([1.0])

        def Q_sample(N, rng):
            return rng.normal(0, 0.01, size=(N, 1))

        state = bootstrap_pf_step(
            particles,
            weights,
            z,
            lambda x: x,
            lambda x: x,
            Q_sample,
            np.array([[1.0]]),
            resample_threshold=0.1,
            resample_method="systematic",
            rng=rng,
        )

        assert state.particles.shape == particles.shape
        assert not np.allclose(state.weights, 0.25)


class TestParticleStatistics:
    """Tests for particle mean, covariance, and initialization."""

    def test_particle_mean(self):
        """Test weighted mean of particles."""
        particles = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        weights = np.array([0.5, 0.3, 0.2])

        mean = particle_mean(particles, weights)

        expected = (
            0.5 * np.array([0, 0]) + 0.3 * np.array([1, 1]) + 0.2 * np.array([2, 2])
        )
        assert np.allclose(mean, expected)

    def test_particle_covariance(self):
        """Test weighted covariance of particles."""
        particles = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        weights = np.ones(4) / 4

        cov = particle_covariance(particles, weights)
        assert cov.shape == (2, 2)

        mean = particle_mean(particles, weights)
        cov2 = particle_covariance(particles, weights, mean=mean)
        assert np.allclose(cov, cov2)

    def test_initialize_particles(self):
        """Test particle initialization from Gaussian prior."""
        rng = np.random.default_rng(42)
        x0 = np.array([0.0, 0.0])
        P0 = np.eye(2) * 0.1
        N = 1000

        state = initialize_particles(x0, P0, N, rng)

        assert state.particles.shape == (N, 2)
        assert state.weights.shape == (N,)
        assert np.allclose(state.weights, 1.0 / N)
        assert np.allclose(np.mean(state.particles, axis=0), x0, atol=0.1)
