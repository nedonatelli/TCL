"""Tests for Gaussian mixture operations."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pytcl.clustering import (
    GaussianComponent,
    GaussianMixture,
    merge_gaussians,
    moment_match,
    prune_mixture,
    reduce_mixture_runnalls,
    reduce_mixture_west,
    runnalls_merge_cost,
)


class TestGaussianComponent:
    """Tests for GaussianComponent."""

    def test_creation(self):
        """Test basic component creation."""
        c = GaussianComponent(
            weight=0.5,
            mean=np.array([1.0, 2.0]),
            covariance=np.eye(2),
        )
        assert c.weight == 0.5
        assert_array_equal(c.mean, [1.0, 2.0])
        assert_array_equal(c.covariance, np.eye(2))


class TestMomentMatch:
    """Tests for moment matching."""

    def test_single_component(self):
        """Single component moment match returns same mean/cov."""
        mean = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.2], [0.2, 1.0]])

        m, P = moment_match([1.0], [mean], [cov])

        assert_allclose(m, mean)
        assert_allclose(P, cov)

    def test_equal_weight_two_components(self):
        """Two equal-weight components average correctly."""
        m1 = np.array([0.0, 0.0])
        m2 = np.array([2.0, 0.0])
        P = np.eye(2) * 0.1

        m, P_out = moment_match([0.5, 0.5], [m1, m2], [P, P])

        # Mean should be midpoint
        assert_allclose(m, [1.0, 0.0])

        # Covariance should include spread of means
        # P_spread = 0.5*outer([-1,0],[-1,0]) + 0.5*outer([1,0],[1,0]) = [[1,0],[0,0]]
        # P_out = 0.1*I + [[1,0],[0,0]]
        expected_cov = np.array([[1.1, 0.0], [0.0, 0.1]])
        assert_allclose(P_out, expected_cov)

    def test_unequal_weights(self):
        """Unequal weights produce weighted mean."""
        m1 = np.array([0.0])
        m2 = np.array([10.0])
        P = np.array([[1.0]])

        m, P_out = moment_match([0.9, 0.1], [m1, m2], [P, P])

        # Mean = 0.9 * 0 + 0.1 * 10 = 1.0
        assert_allclose(m, [1.0])


class TestRunnallsMergeCost:
    """Tests for Runnalls' merge cost."""

    def test_identical_components(self):
        """Identical components have zero merge cost."""
        c = GaussianComponent(0.5, np.array([0.0, 0.0]), np.eye(2))
        cost = runnalls_merge_cost(c, c)
        assert cost >= 0
        # Cost should be small for identical components
        assert cost < 1e-10

    def test_cost_symmetry(self):
        """Merge cost is symmetric."""
        c1 = GaussianComponent(0.3, np.array([0.0, 0.0]), np.eye(2) * 0.1)
        c2 = GaussianComponent(0.2, np.array([1.0, 0.0]), np.eye(2) * 0.2)

        cost_12 = runnalls_merge_cost(c1, c2)
        cost_21 = runnalls_merge_cost(c2, c1)

        assert_allclose(cost_12, cost_21)

    def test_cost_positivity(self):
        """Merge cost is non-negative."""
        c1 = GaussianComponent(0.3, np.array([0.0, 0.0]), np.eye(2))
        c2 = GaussianComponent(0.7, np.array([5.0, 5.0]), np.eye(2))

        cost = runnalls_merge_cost(c1, c2)
        assert cost >= 0

    def test_similar_vs_distant(self):
        """Similar components have lower cost than distant ones."""
        c_center = GaussianComponent(0.5, np.array([0.0, 0.0]), np.eye(2))
        c_close = GaussianComponent(0.5, np.array([0.1, 0.0]), np.eye(2))
        c_far = GaussianComponent(0.5, np.array([10.0, 0.0]), np.eye(2))

        cost_close = runnalls_merge_cost(c_center, c_close)
        cost_far = runnalls_merge_cost(c_center, c_far)

        assert cost_close < cost_far


class TestMergeGaussians:
    """Tests for Gaussian merging."""

    def test_weight_sum(self):
        """Merged weight is sum of component weights."""
        c1 = GaussianComponent(0.3, np.array([0.0]), np.array([[1.0]]))
        c2 = GaussianComponent(0.2, np.array([1.0]), np.array([[1.0]]))

        result = merge_gaussians(c1, c2)

        assert_allclose(result.component.weight, 0.5)

    def test_mean_weighted_average(self):
        """Merged mean is weighted average."""
        c1 = GaussianComponent(0.6, np.array([0.0, 0.0]), np.eye(2))
        c2 = GaussianComponent(0.4, np.array([10.0, 0.0]), np.eye(2))

        result = merge_gaussians(c1, c2)

        # Mean = (0.6 * 0 + 0.4 * 10) / 1.0 = 4.0
        assert_allclose(result.component.mean, [4.0, 0.0])


class TestPruneMixture:
    """Tests for mixture pruning."""

    def test_remove_low_weight(self):
        """Low-weight components are removed."""
        comps = [
            GaussianComponent(0.9, np.array([0.0]), np.array([[1.0]])),
            GaussianComponent(1e-8, np.array([10.0]), np.array([[1.0]])),
        ]

        pruned = prune_mixture(comps, weight_threshold=1e-5)

        assert len(pruned) == 1
        assert_allclose(pruned[0].weight, 1.0)  # Renormalized

    def test_keep_all_above_threshold(self):
        """Components above threshold are kept."""
        comps = [
            GaussianComponent(0.5, np.array([0.0]), np.array([[1.0]])),
            GaussianComponent(0.5, np.array([1.0]), np.array([[1.0]])),
        ]

        pruned = prune_mixture(comps, weight_threshold=0.1)

        assert len(pruned) == 2

    def test_keep_at_least_one(self):
        """At least one component is kept even if all below threshold."""
        comps = [
            GaussianComponent(1e-10, np.array([0.0]), np.array([[1.0]])),
            GaussianComponent(1e-11, np.array([1.0]), np.array([[1.0]])),
        ]

        pruned = prune_mixture(comps, weight_threshold=0.1)

        assert len(pruned) == 1


class TestReduceMixtureRunnalls:
    """Tests for Runnalls' mixture reduction."""

    def test_no_reduction_needed(self):
        """No reduction if already below target."""
        comps = [
            GaussianComponent(0.5, np.array([0.0, 0.0]), np.eye(2)),
            GaussianComponent(0.5, np.array([5.0, 5.0]), np.eye(2)),
        ]

        result = reduce_mixture_runnalls(comps, max_components=3)

        assert result.n_reduced == 2
        assert result.total_cost == 0.0

    def test_reduction_to_target(self):
        """Reduction reaches target number."""
        comps = [
            GaussianComponent(0.25, np.array([0.0, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([0.1, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([10.0, 10.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([10.1, 10.0]), np.eye(2) * 0.1),
        ]

        result = reduce_mixture_runnalls(comps, max_components=2)

        assert len(result.components) == 2
        assert result.n_original == 4
        assert result.n_reduced == 2

    def test_weights_sum_to_one(self):
        """Weights sum to 1 after reduction."""
        comps = [
            GaussianComponent(0.2, np.array([i * 0.1, 0.0]), np.eye(2) * 0.1)
            for i in range(5)
        ]

        result = reduce_mixture_runnalls(comps, max_components=2)

        total_weight = sum(c.weight for c in result.components)
        assert_allclose(total_weight, 1.0)

    def test_merges_similar_first(self):
        """Algorithm merges similar components first."""
        # Two clusters: near origin and far away
        comps = [
            GaussianComponent(0.25, np.array([0.0, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([0.05, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([100.0, 100.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([100.05, 100.0]), np.eye(2) * 0.1),
        ]

        result = reduce_mixture_runnalls(comps, max_components=2)

        # Should have one component near origin, one near (100, 100)
        means = [c.mean for c in result.components]
        has_near_origin = any(np.linalg.norm(m) < 1 for m in means)
        has_near_far = any(np.linalg.norm(m - [100, 100]) < 1 for m in means)

        assert has_near_origin
        assert has_near_far


class TestReduceMixtureWest:
    """Tests for West's mixture reduction."""

    def test_reduction_to_target(self):
        """Reduction reaches target number."""
        comps = [
            GaussianComponent(0.25, np.array([0.0, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([0.1, 0.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([10.0, 10.0]), np.eye(2) * 0.1),
            GaussianComponent(0.25, np.array([10.1, 10.0]), np.eye(2) * 0.1),
        ]

        result = reduce_mixture_west(comps, max_components=2)

        assert len(result.components) == 2
        assert result.n_reduced == 2


class TestGaussianMixture:
    """Tests for GaussianMixture class."""

    def test_empty_mixture(self):
        """Empty mixture has zero components."""
        gm = GaussianMixture()
        assert len(gm) == 0

    def test_add_component(self):
        """Adding components works."""
        gm = GaussianMixture()
        gm.add_component(0.5, np.array([0.0, 0.0]), np.eye(2))
        gm.add_component(0.5, np.array([1.0, 0.0]), np.eye(2))

        assert len(gm) == 2

    def test_normalize_weights(self):
        """Weight normalization works."""
        gm = GaussianMixture()
        gm.add_component(2.0, np.array([0.0]), np.array([[1.0]]))
        gm.add_component(3.0, np.array([1.0]), np.array([[1.0]]))

        gm.normalize_weights()

        assert_allclose(gm.weights, [0.4, 0.6])

    def test_mixture_mean(self):
        """Mixture mean is correct."""
        gm = GaussianMixture()
        gm.add_component(0.5, np.array([0.0, 0.0]), np.eye(2))
        gm.add_component(0.5, np.array([2.0, 2.0]), np.eye(2))

        assert_allclose(gm.mean, [1.0, 1.0])

    def test_pdf_evaluation(self):
        """PDF evaluates to reasonable values."""
        gm = GaussianMixture()
        gm.add_component(1.0, np.array([0.0, 0.0]), np.eye(2))

        # PDF at mode should be maximum
        pdf_at_mode = gm.pdf(np.array([0.0, 0.0]))
        pdf_away = gm.pdf(np.array([10.0, 10.0]))

        assert pdf_at_mode > pdf_away
        assert pdf_at_mode > 0

    def test_sample(self):
        """Sampling produces correct number of samples."""
        gm = GaussianMixture()
        gm.add_component(0.5, np.array([0.0, 0.0]), np.eye(2) * 0.1)
        gm.add_component(0.5, np.array([10.0, 10.0]), np.eye(2) * 0.1)

        rng = np.random.default_rng(42)
        samples = gm.sample(100, rng=rng)

        assert samples.shape == (100, 2)

    def test_reduce_methods(self):
        """Reduction methods work on class."""
        gm = GaussianMixture()
        for i in range(5):
            gm.add_component(0.2, np.array([i * 0.1, 0.0]), np.eye(2) * 0.1)

        reduced_runnalls = gm.reduce_runnalls(2)
        reduced_west = gm.reduce_west(2)

        assert len(reduced_runnalls) == 2
        assert len(reduced_west) == 2

    def test_copy(self):
        """Copy creates independent copy."""
        gm = GaussianMixture()
        gm.add_component(1.0, np.array([0.0, 0.0]), np.eye(2))

        gm_copy = gm.copy()
        gm_copy.add_component(1.0, np.array([1.0, 1.0]), np.eye(2))

        assert len(gm) == 1
        assert len(gm_copy) == 2


class TestNumericalStability:
    """Tests for numerical stability edge cases."""

    def test_small_weights(self):
        """Handling of very small weights."""
        comps = [
            GaussianComponent(1e-15, np.array([0.0]), np.array([[1.0]])),
            GaussianComponent(1.0 - 1e-15, np.array([1.0]), np.array([[1.0]])),
        ]

        # Should not crash
        result = reduce_mixture_runnalls(comps, max_components=1)
        assert len(result.components) == 1

    def test_near_singular_covariance(self):
        """Handling of near-singular covariances."""
        cov = np.array([[1.0, 0.99999], [0.99999, 1.0]])
        comps = [
            GaussianComponent(0.5, np.array([0.0, 0.0]), cov),
            GaussianComponent(0.5, np.array([1.0, 1.0]), cov),
        ]

        # Should not crash
        result = reduce_mixture_runnalls(comps, max_components=1)
        assert len(result.components) == 1

    def test_high_dimensional(self):
        """Handling of higher dimensions."""
        dim = 10
        comps = [
            GaussianComponent(0.5, np.zeros(dim), np.eye(dim)),
            GaussianComponent(0.5, np.ones(dim), np.eye(dim)),
        ]

        result = reduce_mixture_runnalls(comps, max_components=1)
        assert len(result.components) == 1
        assert result.components[0].mean.shape == (dim,)
