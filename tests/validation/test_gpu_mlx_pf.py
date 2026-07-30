"""Validation of the backend-dispatched GPU particle filter on MLX.

``pytcl.gpu.particle_filter`` used to be hardwired to CuPy and raised
``DependencyError`` on Apple Silicon. It is now written against the
:mod:`pytcl.gpu._backend` operation surface and executes for real here.

Ground truth is threefold:

1. The reference-validated CPU implementations in
   ``pytcl.dynamic_estimation.particle_filters`` and ``scipy.stats``.
2. Analytic properties of the resampling schemes -- notably the defining
   systematic-resampling bound ``|count_i - n * w_i| < 1``, which is an exact
   statement, not a statistical one.
3. Statistical tests (chi-squared, CLT bounds) for the schemes whose output is
   genuinely random.

The MLX backend computes in float32, so comparisons against float64 CPU
references are made at float32 tolerance. Observed relative error on the
weight update is ~1e-6.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

mx = pytest.importorskip("mlx.core", reason="MLX required for the GPU PF tests")

from pytcl.dynamic_estimation.particle_filters import (  # noqa: E402
    bootstrap_pf_update,
    gaussian_likelihood,
)
from pytcl.gpu import particle_filter as gpu_pf  # noqa: E402
from pytcl.gpu._backend import get_compute_backend  # noqa: E402

BACKEND = get_compute_backend()
FLOAT32 = not BACKEND.supports_float64

# Tolerances against the float64 CPU references.
RTOL = 3e-5 if FLOAT32 else 1e-12
ATOL = 1e-7 if FLOAT32 else 1e-14
# Slack on the exact systematic-resampling count bound, for float32 CDF
# rounding only (float64 needs none).
COUNT_SLACK = 1e-4 if FLOAT32 else 1e-9


def _normalized_weights(rng, n):
    w = rng.random(n) + 1e-3
    return w / w.sum()


def _gaussian_likelihood_fn(R):
    """Batched Gaussian likelihood, evaluated on the host in float64."""

    def fn(particles, measurement):
        d = np.asarray(particles, dtype=np.float64) - np.asarray(
            measurement, dtype=np.float64
        )
        quad = np.einsum("ni,ij,nj->n", d, np.linalg.inv(R), d)
        norm = 1.0 / np.sqrt((2 * np.pi) ** R.shape[0] * np.linalg.det(R))
        return norm * np.exp(-0.5 * quad)

    return fn


# ---------------------------------------------------------------------------
# The port itself
# ---------------------------------------------------------------------------


def test_particle_filter_runs_without_cupy():
    """Regression for issue #12: no CuPy on this machine, PF still runs."""
    assert BACKEND.name in ("mlx", "cupy")
    idx = np.asarray(gpu_pf.gpu_resample_systematic(np.full(4, 0.25), seed=0))
    assert idx.shape == (4,)
    assert idx.min() >= 0 and idx.max() < 4


# ---------------------------------------------------------------------------
# Weight computation
# ---------------------------------------------------------------------------


def test_weights_match_scipy_multivariate_normal():
    """PF weight update vs per-particle scipy Gaussian pdf."""
    scipy_stats = pytest.importorskip("scipy.stats")
    n, dim = 400, 2
    pf = gpu_pf.CuPyParticleFilter(n_particles=n, state_dim=dim)
    np.random.seed(5)
    pf.initialize(np.zeros(dim), np.eye(dim))
    particles = np.asarray(pf.particles, dtype=np.float64)

    z = np.array([0.3, -0.2])
    R = np.diag([0.5, 0.8])

    pf.resample_threshold = 0.0  # no resampling, compare the weights directly
    log_lik = pf.update(z, _gaussian_likelihood_fn(R))

    ref = scipy_stats.multivariate_normal(mean=z, cov=R).pdf(particles)
    ref_norm = ref / ref.sum()

    weights = np.asarray(pf.weights, dtype=np.float64)
    assert_allclose(weights, ref_norm, rtol=RTOL, atol=ATOL)
    assert_allclose(weights.sum(), 1.0, rtol=RTOL)
    assert_allclose(log_lik, np.log(ref.mean()), rtol=RTOL, atol=1e-5)

    est = np.asarray(pf.get_estimate(), dtype=np.float64)
    assert_allclose(est, ref_norm @ particles, rtol=1e-4, atol=1e-5)

    diff = particles - ref_norm @ particles
    cov = np.asarray(pf.get_covariance(), dtype=np.float64)
    assert_allclose(cov, np.einsum("n,ni,nj->ij", ref_norm, diff, diff), atol=1e-4)


def test_weights_match_cpu_bootstrap_update():
    """PF weight update vs the reference CPU bootstrap_pf_update."""
    rng = np.random.default_rng(11)
    n, dim = 200, 2
    particles = rng.normal(size=(n, dim))
    z = np.array([0.4, 1.1])
    R = np.diag([0.3, 0.7])

    pf = gpu_pf.CuPyParticleFilter(n_particles=n, state_dim=dim)
    pf.particles = BACKEND.asarray(particles)
    pf.resample_threshold = 0.0
    pf.update(z, _gaussian_likelihood_fn(R))

    w_ref, _ = bootstrap_pf_update(
        particles,
        np.full(n, 1.0 / n),
        z,
        lambda zz, x: gaussian_likelihood(zz, x, R),
    )
    assert_allclose(
        np.asarray(pf.weights, dtype=np.float64), w_ref, rtol=RTOL, atol=ATOL
    )


def test_normalize_weights_vs_log_sum_exp():
    """gpu_normalize_weights vs a direct log-sum-exp reference."""
    rng = np.random.default_rng(2)
    log_w = rng.normal(size=500) * 10  # wide dynamic range

    weights, log_lik = gpu_pf.gpu_normalize_weights(log_w)
    weights = np.asarray(weights, dtype=np.float64)

    shifted = log_w - log_w.max()
    ref = np.exp(shifted) / np.exp(shifted).sum()
    ref_ll = log_w.max() + np.log(np.exp(shifted).sum())

    assert_allclose(weights, ref, rtol=RTOL, atol=ATOL)
    assert_allclose(weights.sum(), 1.0, rtol=RTOL)
    assert_allclose(log_lik, ref_ll, rtol=RTOL, atol=1e-4)


def test_normalize_weights_handles_zero_likelihoods():
    """A particle with zero likelihood gets weight 0, not NaN."""
    lik = np.array([0.0, 1.0, 2.0, 0.0])
    pf = gpu_pf.CuPyParticleFilter(n_particles=4, state_dim=1)
    pf.resample_threshold = 0.0
    pf.update(np.zeros(1), lambda p, z: lik)
    w = np.asarray(pf.weights, dtype=np.float64)
    assert np.isfinite(w).all()
    assert_allclose(w, np.array([0.0, 1 / 3, 2 / 3, 0.0]), rtol=RTOL, atol=1e-6)


def test_effective_sample_size():
    n = 100
    assert_allclose(gpu_pf.gpu_effective_sample_size(np.full(n, 1.0 / n)), n, rtol=RTOL)
    degenerate = np.zeros(n)
    degenerate[0] = 1.0
    assert_allclose(gpu_pf.gpu_effective_sample_size(degenerate), 1.0, rtol=RTOL)


# ---------------------------------------------------------------------------
# Resampling properties
# ---------------------------------------------------------------------------


def test_systematic_resampling_count_bound():
    """The defining property: |count_i - n*w_i| < 1 for every particle."""
    rng = np.random.default_rng(3)
    n = 50
    w = _normalized_weights(rng, n)
    for seed in range(40):
        idx = np.asarray(gpu_pf.gpu_resample_systematic(w, seed=seed))
        assert idx.shape == (n,)
        assert idx.min() >= 0 and idx.max() < n
        counts = np.bincount(idx, minlength=n)
        assert counts.sum() == n
        assert np.abs(counts - n * w).max() < 1.0 + COUNT_SLACK


def test_stratified_resampling_count_bound():
    """Stratified sampling keeps every count within 2 of its expectation."""
    rng = np.random.default_rng(5)
    n = 40
    w = _normalized_weights(rng, n)
    for seed in range(40):
        idx = np.asarray(gpu_pf.gpu_resample_stratified(w, seed=seed))
        counts = np.bincount(idx, minlength=n)
        assert np.abs(counts - n * w).max() < 2.0 + COUNT_SLACK


def test_multinomial_resampling_chi_squared():
    """Aggregated multinomial counts vs expectation, chi-squared test."""
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(4)
    n, trials = 20, 500
    w = _normalized_weights(rng, n)

    counts = np.zeros(n)
    for seed in range(trials):
        idx = np.asarray(gpu_pf.gpu_resample_multinomial(w, seed=seed))
        assert idx.shape == (n,)
        counts += np.bincount(idx, minlength=n)

    total = trials * n
    _, p_value = scipy_stats.chisquare(counts, f_exp=w * total)
    assert p_value > 1e-3  # seeded; fails only if the distribution is wrong


@pytest.mark.parametrize(
    "resample", ["gpu_resample_systematic", "gpu_resample_multinomial"]
)
def test_resampling_preserves_weighted_mean(resample):
    """E[mean of resampled values] equals the weighted mean (seeded CLT)."""
    fn = getattr(gpu_pf, resample)
    rng = np.random.default_rng(6)
    n, trials = 200, 300
    values = rng.normal(size=n) * 3.0
    w = _normalized_weights(rng, n)
    target = float(np.dot(w, values))

    means = [values[np.asarray(fn(w, seed=seed))].mean() for seed in range(trials)]
    est = float(np.mean(means))

    # Multinomial std of a single-trial mean; it also bounds systematic, whose
    # variance is strictly smaller.
    sigma_trial = np.sqrt(np.dot(w, (values - target) ** 2) / n)
    assert abs(est - target) < 4 * sigma_trial / np.sqrt(trials)


@pytest.mark.parametrize("method", ["systematic", "multinomial", "stratified"])
def test_post_resample_weights_are_uniform(method):
    """Resampling resets the weights to 1/N and keeps the particle count."""
    n = 100
    pf = gpu_pf.CuPyParticleFilter(n_particles=n, state_dim=1, resample_method=method)
    np.random.seed(6)
    pf.initialize(np.zeros(1), np.eye(1))

    # A sharp likelihood collapses the ESS and forces a resample.
    def sharp(p, z):
        return np.exp(-50.0 * (np.asarray(p, dtype=np.float64)[:, 0] - 5.0) ** 2)

    pf.update(np.array([5.0]), sharp)

    w = np.asarray(pf.weights, dtype=np.float64)
    assert w.shape == (n,)
    assert_allclose(w, np.full(n, 1.0 / n), rtol=RTOL, atol=ATOL)
    assert np.asarray(pf.particles).shape == (n, 1)
    assert_allclose(pf.get_ess(), float(n), rtol=RTOL)


def test_resample_selects_only_existing_particles():
    """Resampled particles are a multiset drawn from the pre-resample set."""
    rng = np.random.default_rng(7)
    n = 64
    pf = gpu_pf.CuPyParticleFilter(n_particles=n, state_dim=2)
    particles = rng.normal(size=(n, 2))
    pf.particles = BACKEND.asarray(particles)
    pf.weights = BACKEND.asarray(_normalized_weights(rng, n))
    pf._resample()

    after = np.asarray(pf.particles, dtype=np.float64)
    before = np.asarray(BACKEND.asarray(particles), dtype=np.float64)
    for row in after:
        assert np.isclose(before, row).all(axis=1).any()


def test_unknown_resample_method_raises():
    with pytest.raises(ValueError, match="Unknown resample method"):
        gpu_pf.CuPyParticleFilter(n_particles=4, state_dim=1, resample_method="nope")


# ---------------------------------------------------------------------------
# Batch update
# ---------------------------------------------------------------------------


def test_batch_particle_filter_update():
    rng = np.random.default_rng(9)
    n_filters, n_particles, dim = 4, 100, 2
    particles = rng.normal(size=(n_filters, n_particles, dim))
    weights = np.full((n_filters, n_particles), 1.0 / n_particles)
    measurements = rng.normal(size=(n_filters, dim))

    def likelihood(p, z):
        d = np.asarray(p, dtype=np.float64) - np.asarray(z, dtype=np.float64)
        return np.exp(-0.5 * np.sum(d**2, axis=1))

    w_upd, log_liks, ess = gpu_pf.batch_particle_filter_update(
        particles, weights, measurements, likelihood
    )
    w_upd = np.asarray(w_upd, dtype=np.float64)
    log_liks = np.asarray(log_liks, dtype=np.float64)
    ess = np.asarray(ess, dtype=np.float64)

    assert w_upd.shape == (n_filters, n_particles)
    assert log_liks.shape == (n_filters,)
    assert ess.shape == (n_filters,)

    for i in range(n_filters):
        ref = weights[i] * likelihood(particles[i], measurements[i])
        ref_norm = ref / ref.sum()
        assert_allclose(w_upd[i], ref_norm, rtol=RTOL, atol=ATOL)
        assert_allclose(log_liks[i], np.log(ref.sum()), rtol=RTOL, atol=1e-5)
        assert_allclose(ess[i], 1.0 / np.sum(ref_norm**2), rtol=1e-4)


def test_initialize_uniform_bounds():
    pf = gpu_pf.CuPyParticleFilter(n_particles=500, state_dim=3)
    low = np.array([-1.0, 0.0, 2.0])
    high = np.array([1.0, 4.0, 3.0])
    pf.initialize_uniform(low, high)
    p = np.asarray(pf.particles, dtype=np.float64)
    assert p.shape == (500, 3)
    assert (p >= low).all() and (p <= high).all()
    assert_allclose(
        np.asarray(pf.weights, dtype=np.float64), np.full(500, 1 / 500), rtol=RTOL
    )


def test_get_state_and_cpu_transfer():
    pf = gpu_pf.CuPyParticleFilter(n_particles=32, state_dim=2)
    np.random.seed(1)
    pf.initialize(np.zeros(2), np.eye(2))
    state = pf.get_state()
    assert np.asarray(state.particles).shape == (32, 2)
    assert np.asarray(state.weights).shape == (32,)
    assert_allclose(state.ess, 32.0, rtol=RTOL)
    assert pf.get_particles_cpu().shape == (32, 2)
    assert pf.get_weights_cpu().shape == (32,)
