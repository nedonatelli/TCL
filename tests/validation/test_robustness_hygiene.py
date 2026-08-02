"""Robustness fixes from gh-26: determinism, recursion depth, shapes, models.

Four defects that share a character rather than a subsystem: none of them makes
a routine call return an obviously wrong answer, so none of them showed up in a
suite built from routine calls. They bite at scale, across runs, or at the
boundary of a documented contract.

- ``minimum_bounding_circle`` drew from the global ``np.random`` state and
  recursed once per point, so it was neither reproducible nor usable on a large
  set;
- ``q_discrete_white_noise`` silently switched to a *different noise model*
  above dimension 4;
- ``tria_sqrt`` returned a non-square factor when the product was rank
  deficient, contradicting its own docstring;
- ``viewshed`` marked the cell south-west of each sample rather than the
  nearest one.

Four further bullets on that issue turned out to be already fixed, verified
before writing anything here; see the pull request.
"""

import numpy as np
import pytest

from pytcl.dynamic_models.process_noise.polynomial import q_discrete_white_noise
from pytcl.mathematical_functions.basic_matrix.decompositions import tria_sqrt
from pytcl.mathematical_functions.geometry.geometry import minimum_bounding_circle


class TestMinimumBoundingCircleIsReproducible:
    """It shuffled with the global RNG, so results depended on unrelated code."""

    def test_the_same_seed_gives_the_same_circle(self):
        points = np.random.default_rng(0).normal(size=(200, 2))

        first = minimum_bounding_circle(points, rng=7)
        second = minimum_bounding_circle(points, rng=7)

        np.testing.assert_array_equal(first[0], second[0])
        assert first[1] == second[1]

    def test_a_generator_is_accepted_as_well_as_a_seed(self):
        points = np.random.default_rng(1).normal(size=(50, 2))

        from_seed = minimum_bounding_circle(points, rng=3)
        from_generator = minimum_bounding_circle(points, rng=np.random.default_rng(3))

        np.testing.assert_allclose(from_seed[0], from_generator[0])

    def test_the_global_random_state_is_not_consumed(self):
        """The old shuffle advanced ``np.random``, perturbing unrelated draws.

        A caller seeding the legacy global RNG for their own reproducibility
        would find their sequence changed by how many points they happened to
        bound.
        """
        points = np.random.default_rng(2).normal(size=(80, 2))

        np.random.seed(1234)
        before = np.random.rand(3)

        np.random.seed(1234)
        minimum_bounding_circle(points, rng=0)
        after = np.random.rand(3)

        np.testing.assert_array_equal(before, after)

    def test_the_seed_does_not_change_the_radius(self):
        """The minimum enclosing circle is unique, so seeding may only change
        which equally-valid representation comes back -- never the answer."""
        points = np.random.default_rng(3).normal(size=(120, 2))

        radii = [minimum_bounding_circle(points, rng=seed)[1] for seed in range(8)]
        assert max(radii) - min(radii) < 1e-9


class TestMinimumBoundingCircleScales:
    """One recursive call per point meant a few thousand points overflowed."""

    @pytest.mark.parametrize("n_points", [2, 3, 10, 5_000, 20_000])
    def test_large_point_sets_do_not_exhaust_the_stack(self, n_points):
        points = np.random.default_rng(4).normal(size=(n_points, 2))

        center, radius = minimum_bounding_circle(points, rng=0)

        assert np.all(np.isfinite(center))
        assert radius > 0.0

    def test_the_recursion_limit_is_not_a_factor(self):
        """Explicit about what changed: lowering the limit must not matter."""
        import sys

        points = np.random.default_rng(5).normal(size=(4_000, 2))
        original = sys.getrecursionlimit()
        sys.setrecursionlimit(200)
        try:
            center, radius = minimum_bounding_circle(points, rng=0)
        finally:
            sys.setrecursionlimit(original)

        assert np.all(np.linalg.norm(points - center, axis=1) <= radius + 1e-8)


class TestMinimumBoundingCircleIsStillCorrect:
    """A rewrite is only worth anything if the answer is unchanged."""

    @staticmethod
    def _brute_force(points):
        """Exhaustive search over every 2- and 3-point circle."""
        import itertools

        best = (None, np.inf)
        for i, j in itertools.combinations(range(len(points)), 2):
            center = (points[i] + points[j]) / 2
            radius = np.linalg.norm(points[i] - center)
            if (
                np.all(np.linalg.norm(points - center, axis=1) <= radius + 1e-9)
                and radius < best[1]
            ):
                best = (center, radius)
        for i, j, k in itertools.combinations(range(len(points)), 3):
            a, b, c = points[i], points[j], points[k]
            d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
            if abs(d) < 1e-12:
                continue
            ux = (
                (a @ a) * (b[1] - c[1])
                + (b @ b) * (c[1] - a[1])
                + (c @ c) * (a[1] - b[1])
            ) / d
            uy = (
                (a @ a) * (c[0] - b[0])
                + (b @ b) * (a[0] - c[0])
                + (c @ c) * (b[0] - a[0])
            ) / d
            center = np.array([ux, uy])
            radius = np.linalg.norm(a - center)
            if (
                np.all(np.linalg.norm(points - center, axis=1) <= radius + 1e-9)
                and radius < best[1]
            ):
                best = (center, radius)
        return best

    @pytest.mark.parametrize("trial", range(25))
    def test_the_radius_matches_exhaustive_search(self, trial):
        rng = np.random.default_rng(100 + trial)
        points = rng.normal(size=(int(rng.integers(2, 11)), 2)) * 5

        _, ours = minimum_bounding_circle(points, rng=trial)
        _, reference = self._brute_force(points)

        assert ours == pytest.approx(reference, abs=1e-9)

    @pytest.mark.parametrize("trial", range(25))
    def test_every_point_is_enclosed(self, trial):
        rng = np.random.default_rng(200 + trial)
        points = rng.normal(size=(int(rng.integers(1, 40)), 2)) * 10

        center, radius = minimum_bounding_circle(points, rng=trial)

        assert np.all(np.linalg.norm(points - center, axis=1) <= radius + 1e-8)

    def test_collinear_points_are_handled(self):
        """The degenerate case the three-point circle helper falls back on."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

        center, radius = minimum_bounding_circle(points, rng=0)

        np.testing.assert_allclose(center, [1.5, 0.0], atol=1e-9)
        assert radius == pytest.approx(1.5, abs=1e-9)

    def test_a_single_point_gives_a_zero_radius(self):
        center, radius = minimum_bounding_circle(np.array([[3.0, -4.0]]), rng=0)
        np.testing.assert_allclose(center, [3.0, -4.0])
        assert radius == 0.0


class TestDiscreteWhiteNoiseUsesOneModel:
    """Above dimension 4 it switched to the continuous-noise discretization."""

    @staticmethod
    def _gain_vector_model(dim, T, var):
        """``var * G G^T`` with ``G[i] = T^(dim-i)/(dim-i)!``.

        Written out from the definition rather than imported, so this is a
        statement of the model rather than a restatement of the code.
        """
        from math import factorial

        gain = np.array([T ** (dim - i) / factorial(dim - i) for i in range(dim)])
        return var * np.outer(gain, gain)

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 6, 8])
    def test_every_dimension_follows_the_same_model(self, dim):
        T, var = 0.7, 2.5
        np.testing.assert_allclose(
            q_discrete_white_noise(dim, T, var),
            self._gain_vector_model(dim, T, var),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_the_original_literal_blocks_are_unchanged(self, dim):
        """Dimensions 2 to 4 were hard-coded matrices; those were correct.

        Generalizing must reproduce them exactly, or this is a behavior change
        dressed up as a refactor.
        """
        T, var = 1.0, 1.0
        expected = {
            2: [[0.25, 0.5], [0.5, 1.0]],
            3: [[1 / 36, 1 / 12, 1 / 6], [1 / 12, 0.25, 0.5], [1 / 6, 0.5, 1.0]],
            4: [
                [1 / 576, 1 / 144, 1 / 48, 1 / 24],
                [1 / 144, 1 / 36, 1 / 12, 1 / 6],
                [1 / 48, 1 / 12, 0.25, 0.5],
                [1 / 24, 1 / 6, 0.5, 1.0],
            ],
        }[dim]
        np.testing.assert_allclose(q_discrete_white_noise(dim, T, var), expected)

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 6])
    def test_the_result_is_a_valid_covariance(self, dim):
        Q = q_discrete_white_noise(dim, 0.3, 1.7)
        np.testing.assert_allclose(Q, Q.T, rtol=1e-12)
        assert np.all(np.linalg.eigvalsh(Q) > -1e-12)

    @pytest.mark.parametrize("dim", [2, 3, 5])
    def test_it_is_rank_one_because_one_noise_source_enters(self, dim):
        """``G G^T`` has rank 1. A model built from a different mechanism --
        the continuous discretization it used to fall back on above dim 4 --
        is full rank, so this distinguishes them directly."""
        assert np.linalg.matrix_rank(q_discrete_white_noise(dim, 0.5, 1.0)) == 1

    def test_block_size_tiles_the_single_axis_block(self):
        T, var, dim, blocks = 0.4, 3.0, 5, 3
        Q = q_discrete_white_noise(dim, T, var, block_size=blocks)

        assert Q.shape == (dim * blocks, dim * blocks)
        single = q_discrete_white_noise(dim, T, var)
        for b in range(blocks):
            s = slice(b * dim, (b + 1) * dim)
            np.testing.assert_allclose(Q[s, s], single)

    def test_a_dimension_below_two_is_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            q_discrete_white_noise(1, 1.0, 1.0)


class TestTriaSqrtReturnsASquareFactor:
    """The docstring promised (n, n); a rank-deficient product gave (n, k)."""

    @pytest.mark.parametrize("shape", [(5, 2), (4, 4), (3, 7), (6, 1), (2, 2), (8, 3)])
    def test_the_factor_is_square_and_lower_triangular(self, shape):
        A = np.random.default_rng(0).normal(size=shape)

        S = tria_sqrt(A)

        assert S.shape == (shape[0], shape[0])
        np.testing.assert_allclose(S, np.tril(S), atol=1e-12)

    @pytest.mark.parametrize("shape", [(5, 2), (4, 4), (3, 7), (6, 1)])
    def test_the_reconstruction_still_holds(self, shape):
        """What already worked. Padding must not disturb it."""
        A = np.random.default_rng(1).normal(size=shape)

        S = tria_sqrt(A)

        np.testing.assert_allclose(S @ S.T, A @ A.T, atol=1e-10)

    def test_the_two_argument_form_is_square_too(self):
        rng = np.random.default_rng(2)
        A, B = rng.normal(size=(5, 2)), rng.normal(size=(5, 3))

        S = tria_sqrt(A, B)
        combined = np.hstack([A, B])

        assert S.shape == (5, 5)
        np.testing.assert_allclose(S @ S.T, combined @ combined.T, atol=1e-10)

    def test_the_padded_columns_are_exactly_zero(self):
        """A rank-2 product in 5 dimensions has three zero columns.

        Padding with anything else would still reconstruct, so checking the
        shape alone would not catch a wrong fill.
        """
        A = np.random.default_rng(3).normal(size=(5, 2))

        S = tria_sqrt(A)

        assert np.linalg.matrix_rank(S, tol=1e-10) == 2
        np.testing.assert_allclose(S[:, 2:], 0.0, atol=1e-12)

    def test_the_diagonal_is_non_negative(self):
        """The Cholesky sign convention, retained across the padding."""
        A = np.random.default_rng(4).normal(size=(6, 4))
        assert np.all(np.diag(tria_sqrt(A)) >= -1e-12)
