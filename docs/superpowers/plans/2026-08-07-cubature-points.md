# Estimation-Grade Gaussian Cubature Points Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the estimation-grade Gaussian-weight cubature rules (degree-5, degree-7, arbitrary-odd-degree spherical-radial) and let the CKF consume them.

**Architecture:** One new module `pytcl/mathematical_functions/numerical_integration/cubature_points.py` holding point generators that all return `(points, weights)` for the N(0, I) weight with `weights.sum() == 1`; `ckf_predict`/`ckf_update` gain optional `points`/`weights` kwargs defaulting to current behavior. Spec: `docs/superpowers/specs/2026-08-07-cubature-points-design.md`.

**Tech Stack:** numpy, scipy.special (`roots_genlaguerre`, `roots_jacobi`, `gamma`) — all already core deps. pytest for tests.

## Global Constraints

- All rules use the standard-normal N(0, I) convention: `points.shape == (num_points, n)`, `weights.sum() == 1`. (1-D `gauss_hermite` in quadrature.py is physicists' `exp(-x^2)` — do NOT mix conventions.)
- Docstrings: NumPy style with executable doctests (CI runs `--doctest-modules`); doctest outputs must be platform-robust (`round()`, `bool()`).
- Errors: `ValueError` for `n < 1`, even or `< 3` degree, shape-inconsistent points/weights. No fallbacks, no new dependencies.
- Style: ruff (88 cols), `snake_case`, type hints `Tuple[NDArray[np.floating], NDArray[np.floating]]` matching quadrature.py.
- Negative weights in degree-5 (n > 4) and degree-7 rules are documented behavior, never suppressed or "fixed".
- Every commit runs the prek hook (ruff + ty); keep each task's commit green.
- Run all commands with `uv run` from the repo root.

## Verified facts the plan relies on

- `cubature_gauss_hermite(n_dim, n_points_per_dim)` (quadrature.py:521) returns RAW physicists' tensor Gauss-Hermite: for N(0, I) expectations you must scale points by `sqrt(2)` and divide weight-sums by `pi**(n/2)`. The REFERENCE tests below do this explicitly.
- `ckf_spherical_cubature_points(n)` (unscented.py:429) returns `(2n, n)` unit points, weights `1/(2n)` — the existing probability convention this module adopts.
- `ckf_predict(x, P, f, Q)` / `ckf_update(x, P, z, h, R)` (unscented.py:473/555) compute covariances via `np.sqrt(weights)` — valid only for nonnegative weights; Task 5 replaces that with the sign-safe form.
- Degree-5 closed form (Stroud E_n^{r^2} 5-3), verified by hand for weight sum and moments through degree 5 (see Task 2 comments): center weight `2/(n+2)`; axis points `±sqrt(n+2)·e_i` weight `(4-n)/(2(n+2)^2)`; pair points `(±mu, ±mu)` on all coordinate pairs, `mu = sqrt((n+2)/2)`, weight `1/(n+2)^2`. Total `2n^2+1`.
- Exact Gaussian moments: `E[prod x_i^{a_i}] = prod (a_i - 1)!!` if all `a_i` even, else 0.

---

### Task 1: Test scaffolding + `transform_cubature_points`

**Files:**
- Create: `pytcl/mathematical_functions/numerical_integration/cubature_points.py`
- Create: `tests/unit/test_cubature_points.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `transform_cubature_points(points, weights, mean, sqrt_cov) -> Tuple[NDArray, NDArray]`; test helpers `gaussian_moment(alpha) -> float` and `rule_moment(points, weights, alpha) -> float` used by every later task.

- [ ] **Step 1: Write the failing tests (including the shared helpers)**

```python
# tests/unit/test_cubature_points.py
"""Tests for Gaussian-weight cubature point generators.

PROPERTY class: a degree-d rule must integrate every monomial of total
degree <= d against N(0, I) exactly, and must FAIL on some degree d+1
monomial (sharpness -- guards against a vacuous exactness loop).
"""

import itertools

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    transform_cubature_points,
)


def gaussian_moment(alpha):
    """E[prod x_i^alpha_i] for x ~ N(0, I): prod (a-1)!! over even a, else 0."""
    m = 1.0
    for a in alpha:
        if a % 2 == 1:
            return 0.0
        for k in range(a - 1, 0, -2):  # (a-1)!! = (a-1)(a-3)...1
            m *= k
    return m


def rule_moment(points, weights, alpha):
    """Sum_i w_i * prod_j points[i, j]**alpha_j."""
    return float(np.sum(weights * np.prod(points ** np.asarray(alpha), axis=1)))


def monomials_up_to(n, degree):
    """All exponent tuples of length n with total degree <= degree."""
    for total in range(degree + 1):
        for alpha in itertools.product(range(total + 1), repeat=n):
            if sum(alpha) == total:
                yield alpha


def assert_rule_exact(points, weights, n, degree):
    for alpha in monomials_up_to(n, degree):
        assert_allclose(
            rule_moment(points, weights, alpha),
            gaussian_moment(alpha),
            atol=1e-9,
            err_msg=f"monomial {alpha} not integrated exactly",
        )


class TestHelpers:
    def test_gaussian_moment_known_values(self):
        # E[x^2]=1, E[x^4]=3, E[x^6]=15, E[x^2 y^2]=1, odd -> 0
        assert gaussian_moment((2,)) == 1.0
        assert gaussian_moment((4,)) == 3.0
        assert gaussian_moment((6,)) == 15.0
        assert gaussian_moment((8,)) == 105.0
        assert gaussian_moment((2, 2)) == 1.0
        assert gaussian_moment((1, 2)) == 0.0


class TestTransformCubaturePoints:
    def test_affine_map_moments(self):
        # Any exact-through-degree-2 rule pushed through the transform must
        # reproduce the target mean and covariance.
        n = 3
        rng = np.random.default_rng(7)
        A = rng.normal(size=(n, n))
        cov = A @ A.T
        mean = rng.normal(size=n)
        sqrt_cov = np.linalg.cholesky(cov)

        # 3rd-degree unit rule: points +-sqrt(n) e_i, weights 1/(2n)
        unit = np.sqrt(n) * np.vstack([np.eye(n), -np.eye(n)])
        w = np.full(2 * n, 1.0 / (2 * n))

        pts, wts = transform_cubature_points(unit, w, mean, sqrt_cov)
        assert pts.shape == unit.shape
        assert_allclose(wts, w)  # weights unchanged
        assert_allclose(np.sum(wts[:, None] * pts, axis=0), mean, atol=1e-12)
        resid = pts - mean
        assert_allclose(resid.T @ (wts[:, None] * resid), cov, atol=1e-10)

    def test_shape_mismatch_raises(self):
        unit = np.zeros((4, 2))
        w = np.full(3, 1 / 3)  # wrong length
        with pytest.raises(ValueError):
            transform_cubature_points(unit, w, np.zeros(2), np.eye(2))
        with pytest.raises(ValueError):
            transform_cubature_points(
                np.zeros((4, 2)), np.full(4, 0.25), np.zeros(3), np.eye(3)
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: FAIL (ImportError: no module `cubature_points`)

- [ ] **Step 3: Write the module with `transform_cubature_points`**

```python
# pytcl/mathematical_functions/numerical_integration/cubature_points.py
"""
Cubature point sets for Gaussian-weighted integration.

Every generator in this module targets the standard multivariate normal
weight N(0, I): points have shape ``(num_points, n)`` and weights sum to 1,
so ``E[f(x)] ~= sum_i w_i f(x_i)`` directly. This matches
``ckf_spherical_cubature_points`` and differs from the 1-D
:func:`~pytcl.mathematical_functions.numerical_integration.gauss_hermite`,
which uses the physicists' ``exp(-x**2)`` weight (map with
``x -> sqrt(2) x`` and divide weights by ``sqrt(pi)`` per dimension).

Ported from the Tracker Component Library's Cubature_Points collection.

References
----------
.. [1] A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
   Prentice-Hall, 1971.
.. [2] J. McNamee and F. Stenger, "Construction of fully symmetric
   numerical integration formulas," Numerische Mathematik 10, 1967.
.. [3] D. F. Crouse, "The Tracker Component Library," IEEE AESS Magazine,
   2017.
"""

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray


def transform_cubature_points(
    points: ArrayLike,
    weights: ArrayLike,
    mean: ArrayLike,
    sqrt_cov: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Affinely map unit cubature points to a given mean and covariance.

    Parameters
    ----------
    points : array_like
        Unit points for N(0, I), shape (num_points, n).
    weights : array_like
        Weights, shape (num_points,).
    mean : array_like
        Target mean, shape (n,).
    sqrt_cov : array_like
        Square root of the target covariance (lower-triangular Cholesky
        factor S with S @ S.T = P), shape (n, n).

    Returns
    -------
    points : ndarray
        Transformed points ``mean + points @ sqrt_cov.T``.
    weights : ndarray
        Unchanged weights (copied).

    Examples
    --------
    >>> unit = np.array([[1.0], [-1.0]])
    >>> w = np.array([0.5, 0.5])
    >>> pts, wts = transform_cubature_points(unit, w, [10.0], [[2.0]])
    >>> pts.ravel().tolist()
    [12.0, 8.0]
    """
    points = np.asarray(points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64).ravel()
    sqrt_cov = np.asarray(sqrt_cov, dtype=np.float64)

    if points.ndim != 2:
        raise ValueError(f"points must be 2-D, got shape {points.shape}")
    num_points, n = points.shape
    if weights.shape != (num_points,):
        raise ValueError(
            f"weights shape {weights.shape} does not match {num_points} points"
        )
    if mean.shape != (n,) or sqrt_cov.shape != (n, n):
        raise ValueError(
            f"mean/sqrt_cov dimensions {mean.shape}/{sqrt_cov.shape} do not "
            f"match points dimension {n}"
        )

    return mean + points @ sqrt_cov.T, weights.copy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add pytcl/mathematical_functions/numerical_integration/cubature_points.py tests/unit/test_cubature_points.py
git commit -m "feat: cubature-points module scaffold with transform_cubature_points"
```

---

### Task 2: `fifth_order_cubature_points`

**Files:**
- Modify: `pytcl/mathematical_functions/numerical_integration/cubature_points.py`
- Modify: `tests/unit/test_cubature_points.py`

**Interfaces:**
- Consumes: test helpers from Task 1 (`assert_rule_exact`, `rule_moment`, `gaussian_moment`).
- Produces: `fifth_order_cubature_points(n) -> Tuple[NDArray, NDArray]` with `points.shape == (2*n*n + 1, n)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cubature_points.py` (add `fifth_order_cubature_points` to the module import):

```python
class TestFifthOrder:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_exact_through_degree_5(self, n):
        pts, w = fifth_order_cubature_points(n)
        assert pts.shape == (2 * n * n + 1, n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree=5)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
    def test_sharpness_degree_6_fails(self, n):
        # E[x1^6] = 15 must NOT be matched -- otherwise the exactness test
        # above could be vacuous.
        pts, w = fifth_order_cubature_points(n)
        alpha = (6,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - 15.0) > 1e-3

    def test_antipodal_symmetry(self):
        pts, _ = fifth_order_cubature_points(4)
        nonzero = pts[np.any(pts != 0.0, axis=1)]
        # every non-origin point's negation is also a point
        for p in nonzero:
            assert np.any(np.all(np.isclose(nonzero, -p), axis=1))

    def test_negative_axis_weights_above_n4_documented(self):
        # (4 - n)/(2 (n+2)^2) < 0 for n > 4: present, not suppressed.
        _, w = fifth_order_cubature_points(5)
        assert w.min() < 0.0

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            fifth_order_cubature_points(0)

    def test_published_values_n2(self):
        # REFERENCE: Stroud (1971), rule E_n^{r^2} 5-3 at n = 2:
        # lambda = sqrt(n+2) = 2, mu = sqrt((n+2)/2) = sqrt(2),
        # w_center = 2/(n+2) = 1/2, w_axis = (4-n)/(2(n+2)^2) = 1/16,
        # w_pair = 1/(n+2)^2 = 1/16.
        pts, w = fifth_order_cubature_points(2)
        assert_allclose(sorted(np.abs(pts).max(axis=1)), sorted(
            [0.0] + [2.0] * 4 + [np.sqrt(2.0)] * 4), atol=1e-12)
        assert_allclose(w[0], 0.5, atol=1e-15)          # center
        assert_allclose(w[1:5], 1.0 / 16.0, atol=1e-15)  # axis
        assert_allclose(w[5:], 1.0 / 16.0, atol=1e-15)   # pairs

    def test_reference_expectation_vs_tensor_gauss_hermite(self):
        # REFERENCE: E[cos(a.T x)] = exp(-|a|^2 / 2) has a closed form and is
        # not polynomial; compare rule vs dense tensor GH vs closed form.
        # cubature_gauss_hermite is RAW physicists' GH: scale by sqrt(2),
        # normalize by pi^(n/2).
        from pytcl.mathematical_functions.numerical_integration import (
            cubature_gauss_hermite,
        )

        n = 3
        a = np.array([0.3, -0.5, 0.2])
        exact = np.exp(-0.5 * float(a @ a))

        pts, w = fifth_order_cubature_points(n)
        rule_val = float(np.sum(w * np.cos(pts @ a)))

        gh_pts, gh_w = cubature_gauss_hermite(n, 10)
        gh_val = float(
            np.sum(gh_w * np.cos((np.sqrt(2.0) * gh_pts) @ a)) / np.pi ** (n / 2)
        )

        assert_allclose(gh_val, exact, atol=1e-9)   # oracle sanity
        assert_allclose(rule_val, exact, atol=5e-3)  # degree-5 approximation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: FAIL (ImportError: `fifth_order_cubature_points`)

- [ ] **Step 3: Implement**

```python
def fifth_order_cubature_points(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Degree-5 cubature points for the standard normal N(0, I).

    The 2n^2 + 1 point fully-symmetric rule E_n^{r^2} 5-3 of Stroud [1]_,
    the counterpart of the MATLAB TCL's ``fifthOrderCubPoints``. Exactly
    integrates every polynomial of total degree <= 5 against N(0, I).

    Parameters
    ----------
    n : int
        Dimension, n >= 1.

    Returns
    -------
    points : ndarray
        Shape (2*n*n + 1, n).
    weights : ndarray
        Shape (2*n*n + 1,), summing to 1. For n > 4 the axis-point weight
        (4 - n)/(2 (n+2)^2) is negative; this is inherent to the rule, not
        an error. Covariances assembled from these points must not use a
        sqrt-of-weights factorization.

    Examples
    --------
    >>> pts, w = fifth_order_cubature_points(3)
    >>> pts.shape
    (19, 3)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 12)  # E[x^4] = 3
    3.0
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")

    lam = np.sqrt(n + 2.0)
    mu = np.sqrt((n + 2.0) / 2.0)
    w_center = 2.0 / (n + 2.0)
    w_axis = (4.0 - n) / (2.0 * (n + 2.0) ** 2)
    w_pair = 1.0 / (n + 2.0) ** 2

    points = [np.zeros((1, n))]
    weights = [np.array([w_center])]

    axis = lam * np.eye(n)
    points.append(np.vstack([axis, -axis]))
    weights.append(np.full(2 * n, w_axis))

    pair_pts = []
    for i in range(n):
        for j in range(i + 1, n):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    p = np.zeros(n)
                    p[i] = si * mu
                    p[j] = sj * mu
                    pair_pts.append(p)
    if pair_pts:
        points.append(np.array(pair_pts))
        weights.append(np.full(len(pair_pts), w_pair))

    return np.vstack(points), np.concatenate(weights)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: PASS. If an exactness assertion fails, the weight/generator
constants are wrong — re-derive from the moment identities in the class
docstring comments; do NOT loosen tolerances.

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: degree-5 Gaussian cubature points (Stroud E_n^r2 5-3)"
```

---

### Task 3: `spherical_radial_points(n, degree)`

**Files:**
- Modify: `pytcl/mathematical_functions/numerical_integration/cubature_points.py`
- Modify: `tests/unit/test_cubature_points.py`

**Interfaces:**
- Consumes: test helpers from Task 1.
- Produces: `spherical_radial_points(n, degree) -> Tuple[NDArray, NDArray]` for any `n >= 1` and odd `degree >= 3`; private `_sphere_surface_points(n, degree)` (uniform-measure surface rule, weights sum to 1).

- [ ] **Step 1: Write the failing tests**

Append (add `spherical_radial_points` to imports):

```python
class TestSphericalRadial:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    @pytest.mark.parametrize("degree", [3, 5, 7, 9])
    def test_exact_through_degree(self, n, degree):
        pts, w = spherical_radial_points(n, degree)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree)

    @pytest.mark.parametrize("n,degree", [(2, 3), (3, 5), (2, 7)])
    def test_sharpness(self, n, degree):
        pts, w = spherical_radial_points(n, degree)
        alpha = (degree + 1,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - gaussian_moment(alpha)) > 1e-4

    def test_degree_3_matches_known_radius(self):
        # Single radial node at r = sqrt(n) -- same radius the CKF uses.
        pts, _ = spherical_radial_points(3, 3)
        radii = np.linalg.norm(pts, axis=1)
        assert_allclose(radii, np.sqrt(3.0), atol=1e-12)

    def test_invalid_degree_raises(self):
        with pytest.raises(ValueError):
            spherical_radial_points(2, 4)  # even
        with pytest.raises(ValueError):
            spherical_radial_points(2, 1)  # < 3
        with pytest.raises(ValueError):
            spherical_radial_points(0, 3)  # bad dim
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: FAIL (ImportError: `spherical_radial_points`)

- [ ] **Step 3: Implement**

Add to the module imports: `from scipy.special import gamma, roots_genlaguerre, roots_jacobi`

```python
def _sphere_surface_points(
    n: int, degree: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Degree-``degree`` rule for the uniform measure on S^(n-1).

    Dimension-recursive spherical-coordinate product construction:
    x = (t, sqrt(1 - t^2) * y) with t from Gauss-Jacobi quadrature with
    weight (1 - t^2)^((n-3)/2) and y a degree-``degree`` rule on S^(n-2).
    Weights are normalized to sum to 1.
    """
    if n == 1:
        return np.array([[1.0], [-1.0]]), np.array([0.5, 0.5])
    if n == 2:
        m = 2 * ((degree + 1) // 2 + 1)  # uniform points, exact for trig deg < m
        theta = 2.0 * np.pi * np.arange(m) / m
        return np.column_stack([np.cos(theta), np.sin(theta)]), np.full(m, 1.0 / m)

    m = (degree + 1) // 2 + 1  # Gauss-Jacobi exact through poly degree 2m-1
    t, wt = roots_jacobi(m, (n - 3.0) / 2.0, (n - 3.0) / 2.0)
    sub_pts, sub_w = _sphere_surface_points(n - 1, degree)

    pts = []
    wts = []
    for tk, wk in zip(t, wt):
        s = np.sqrt(1.0 - tk * tk)
        block = np.column_stack([np.full(len(sub_pts), tk), s * sub_pts])
        pts.append(block)
        wts.append(wk * sub_w)
    points = np.vstack(pts)
    weights = np.concatenate(wts)
    return points, weights / weights.sum()


def spherical_radial_points(
    n: int, degree: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Arbitrary-odd-degree spherical-radial cubature points for N(0, I).

    Product of a generalized Gauss-Laguerre radial rule (exact for all
    required even powers of r) with a dimension-recursive surface rule on
    the unit sphere. Generalizes the 3rd-degree spherical-radial rule of
    the CKF to any odd degree.

    The point count grows roughly as ``(degree/2)^(n-1)`` from the surface
    rule; for the common degrees 5 and 7 prefer
    :func:`fifth_order_cubature_points` and
    :func:`seventh_order_cubature_points`, which grow polynomially in n.

    Parameters
    ----------
    n : int
        Dimension, n >= 1.
    degree : int
        Odd polynomial degree >= 3 the rule integrates exactly.

    Returns
    -------
    points : ndarray, shape (num_points, n)
    weights : ndarray, shape (num_points,), summing to 1.

    Examples
    --------
    >>> pts, w = spherical_radial_points(2, 5)
    >>> round(float(w.sum()), 12)
    1.0
    >>> round(float(np.sum(w * pts[:, 0] ** 4)), 10)  # E[x^4] = 3
    3.0
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if degree < 3 or degree % 2 == 0:
        raise ValueError(f"degree must be an odd integer >= 3, got {degree}")

    # Radial part: substitute t = r^2/2 in the integral of g(r) r^(n-1)
    # exp(-r^2/2); Gauss-Laguerre with alpha = n/2 - 1 handles t^j exactly.
    # Even powers r^(2j) with 2j <= degree - 1 must be exact => j <=
    # (degree-1)/2 => m_r points with 2*m_r - 1 >= (degree-1)/2.
    # Checked: degree 3 -> 1 node (r = sqrt(n), the CKF radius); 5,7 -> 2;
    # 9 -> 3.
    m_r = (degree + 1) // 4 + 1
    t, wt = roots_genlaguerre(m_r, n / 2.0 - 1.0)
    radii = np.sqrt(2.0 * t)
    w_rad = wt / gamma(n / 2.0)

    surf_pts, surf_w = _sphere_surface_points(n, degree)

    points = np.vstack([r * surf_pts for r in radii])
    weights = np.concatenate([wr * surf_w for wr in w_rad])
    return points, weights / weights.sum()
```

Note on `m_r`: the exactness tests are the arbiter. If the exactness grid
fails at high even radial powers, increase `m_r` by one and re-run — but
`test_degree_3_matches_known_radius` must keep passing (degree 3 must stay
a single radial node at `sqrt(n)`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: PASS. Failure modes: (a) degree-3 radius test fails -> adjust
`m_r` per the note; (b) exactness fails only for n >= 3 -> the Jacobi
weight exponent is wrong (must be `(n-3)/2` for S^(n-1)); (c) exactness
fails at odd monomials -> the n == 2 circle rule must have an even point
count (antipodal symmetry).

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: arbitrary-odd-degree spherical-radial cubature points"
```

---

### Task 4: `seventh_order_cubature_points`

**Files:**
- Modify: `pytcl/mathematical_functions/numerical_integration/cubature_points.py`
- Modify: `tests/unit/test_cubature_points.py`
- Reference (read-only, not committed): MATLAB `seventhOrderCubPoints.m`

**Interfaces:**
- Consumes: test helpers from Task 1.
- Produces: `seventh_order_cubature_points(n) -> Tuple[NDArray, NDArray]`, point count polynomial in n (O(n^3) family).

- [ ] **Step 1: Fetch the MATLAB reference**

The generator constants come from the MATLAB TCL, the library this project
ports. Fetch the source (do not commit it — MATLAB code is reference
material only):

```bash
curl -sL "https://raw.githubusercontent.com/USNavalResearchLaboratory/TrackerComponentLibrary/master/Mathematical_Functions/Numerical_Integration/Cubature_Points/Gaussian_Points/Full_Dimensional_Sets/seventhOrderCubPoints.m" -o /tmp/seventhOrderCubPoints.m
```

If that path 404s, locate the file with
`gh api "search/code?q=seventhOrderCubPoints+repo:USNavalResearchLaboratory/TrackerComponentLibrary" --jq '.items[].path'`
and adjust. Read the file; identify the fully-symmetric generator sets
(origin / axis / pair / triple points with one or two distinct generator
magnitudes) and their weight formulas as functions of n.

- [ ] **Step 2: Write the failing tests**

Append (add `seventh_order_cubature_points` to imports):

```python
class TestSeventhOrder:
    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_exact_through_degree_7(self, n):
        pts, w = seventh_order_cubature_points(n)
        assert_allclose(w.sum(), 1.0, atol=1e-12)
        assert_rule_exact(pts, w, n, degree=7)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_sharpness_degree_8_fails(self, n):
        pts, w = seventh_order_cubature_points(n)
        alpha = (8,) + (0,) * (n - 1)
        assert abs(rule_moment(pts, w, alpha) - 105.0) > 1e-3

    @pytest.mark.parametrize("n", [3, 4, 5, 6])
    def test_point_count_polynomial(self, n):
        # The O(n^3) fully-symmetric family, not an exponential product rule.
        pts, _ = seventh_order_cubature_points(n)
        assert len(pts) < 8 * n**3 + 1

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            seventh_order_cubature_points(0)
```

Note the parametrization starts at n = 3: consult the MATLAB header for
the rule's minimum supported dimension and align the test and the
`ValueError` bound to it (if the MATLAB rule supports n >= 2, extend the
tests to n = 2).

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: FAIL (ImportError: `seventh_order_cubature_points`)

- [ ] **Step 4: Port the implementation**

Port the MATLAB generator table to a `seventh_order_cubature_points(n)`
function in the same style as `fifth_order_cubature_points` (explicit
fully-symmetric point-set construction; weights as closed-form functions
of n taken from the MATLAB source; cite the MATLAB file and its stated
reference in the docstring; document negative weights in the Returns
section exactly as done for the fifth-order rule; include a doctest that
checks `w.sum()` rounds to 1.0 and `E[x^6]` rounds to 15.0). Raise
`ValueError` below the rule's minimum dimension.

The monomial-exactness grid in Step 2 is the correctness gate: every
monomial through total degree 7 in n = 3..6, plus sharpness at degree 8.
A transcription error in any constant fails the grid — do not weaken the
grid to make a port pass.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "feat: degree-7 Gaussian cubature points (McNamee-Stenger family)"
```

---

### Task 5: CKF hookup — optional `points`/`weights`

**Files:**
- Modify: `pytcl/dynamic_estimation/kalman/unscented.py:473-640` (`ckf_predict`, `ckf_update`)
- Modify: `tests/unit/test_cubature_points.py`

**Interfaces:**
- Consumes: `fifth_order_cubature_points` (Task 2); existing `KalmanPrediction`/`KalmanUpdate` named tuples (unchanged).
- Produces: `ckf_predict(x, P, f, Q, points=None, weights=None)`, `ckf_update(x, P, z, h, R, points=None, weights=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_cubature_points.py`:

```python
class TestCKFWithCustomPoints:
    def _setup(self):
        x = np.array([1.0, -0.5, 0.2])
        P = np.diag([0.20, 0.30, 0.15])
        Q = np.eye(3) * 0.01

        def f(v):  # mildly nonlinear dynamics
            return np.array([v[0] + 0.1 * v[1], v[1] + 0.05 * v[0] ** 2, v[2]])

        return x, P, Q, f

    def test_default_path_unchanged(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict

        x, P, Q, f = self._setup()
        base = ckf_predict(x, P, f, Q)
        again = ckf_predict(x, P, f, Q, points=None, weights=None)
        assert_allclose(again.x, base.x, atol=0)
        assert_allclose(again.P, base.P, atol=0)

    def test_fifth_order_matches_tensor_gauss_hermite_moments(self):
        # REFERENCE: dense tensor GH (scaled to N(0,I) convention) is the
        # oracle for the propagated mean/covariance.
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict
        from pytcl.mathematical_functions.numerical_integration import (
            cubature_gauss_hermite,
        )
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        x, P, Q, f = self._setup()
        pts5, w5 = fifth_order_cubature_points(3)
        pred5 = ckf_predict(x, P, f, Q, points=pts5, weights=w5)

        gh_pts, gh_w = cubature_gauss_hermite(3, 8)
        unit = np.sqrt(2.0) * gh_pts
        wts = gh_w / np.pi ** 1.5
        S = np.linalg.cholesky(P)
        prop = np.array([f(x + S @ p) for p in unit])
        mean_ref = np.sum(wts[:, None] * prop, axis=0)
        resid = prop - mean_ref
        cov_ref = resid.T @ (wts[:, None] * resid) + Q

        # f is quadratic, so a degree-5 rule propagates mean and covariance
        # of a quadratic map exactly up to the rule's degree; tight tol.
        assert_allclose(pred5.x, mean_ref, atol=1e-10)
        assert_allclose(pred5.P, cov_ref, atol=1e-8)

    def test_negative_weights_covariance_is_sign_safe(self):
        # n = 5 makes the fifth-order axis weight negative; the covariance
        # assembly must not sqrt() weights.
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        n = 5
        x = np.zeros(n)
        P = np.eye(n)
        Q = np.zeros((n, n))
        F = np.diag([1.0, 2.0, 0.5, 1.5, 1.0])
        pts, w = fifth_order_cubature_points(n)
        assert w.min() < 0
        pred = ckf_predict(x, P, lambda v: F @ v, Q, points=pts, weights=w)
        # Linear map: exact answers are F x and F P F.T
        assert_allclose(pred.x, F @ x, atol=1e-10)
        assert_allclose(pred.P, F @ P @ F.T, atol=1e-9)
        assert not np.any(np.isnan(pred.P))

    def test_update_with_custom_points(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_update
        from pytcl.mathematical_functions.numerical_integration.cubature_points import (
            fifth_order_cubature_points,
        )

        x = np.array([3.0, 1.0])
        P = np.eye(2) * 0.5
        z = np.array([3.1])
        R = np.array([[0.1]])
        pts, w = fifth_order_cubature_points(2)
        upd = ckf_update(x, P, z, lambda v: np.array([v[0]]), R, points=pts, weights=w)
        assert upd.x.shape == (2,)
        # Linear measurement: must agree with the default CKF closely.
        from pytcl.dynamic_estimation.kalman.unscented import (
            ckf_update as default_update,
        )

        base = default_update(x, P, z, lambda v: np.array([v[0]]), R)
        assert_allclose(upd.x, base.x, atol=1e-9)
        assert_allclose(upd.P, base.P, atol=1e-9)

    def test_shape_mismatch_raises(self):
        from pytcl.dynamic_estimation.kalman.unscented import ckf_predict

        x, P, Q, f = self._setup()
        pts, w = fifth_order_cubature_points(2)  # wrong dimension (2 != 3)
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts, weights=w)
        pts3, w3 = fifth_order_cubature_points(3)
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts3, weights=w3[:-1])
        with pytest.raises(ValueError):
            ckf_predict(x, P, f, Q, points=pts3)  # points without weights
```

(Also add `fifth_order_cubature_points` to the top-level import in the test
file if not already there.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_cubature_points.py::TestCKFWithCustomPoints -x -q`
Expected: FAIL (TypeError: unexpected keyword argument 'points')

- [ ] **Step 3: Modify `ckf_predict` and `ckf_update`**

In both functions:

1. Extend the signature: `def ckf_predict(x, P, f, Q, points=None, weights=None) -> KalmanPrediction:` (and the `ckf_update` equivalent). Type hints: `points: Optional[ArrayLike] = None, weights: Optional[ArrayLike] = None`.
2. Replace the hardwired point generation with:

```python
    if (points is None) != (weights is None):
        raise ValueError("points and weights must be provided together")
    if points is None:
        unit_pts, weights_arr = ckf_spherical_cubature_points(n)
    else:
        unit_pts = np.asarray(points, dtype=np.float64)
        weights_arr = np.asarray(weights, dtype=np.float64)
        if unit_pts.ndim != 2 or unit_pts.shape[1] != n:
            raise ValueError(
                f"points shape {unit_pts.shape} incompatible with state "
                f"dimension {n}"
            )
        if weights_arr.shape != (unit_pts.shape[0],):
            raise ValueError(
                f"weights shape {weights_arr.shape} does not match "
                f"{unit_pts.shape[0]} points"
            )
```

3. Replace every `np.sqrt(weights)`-based covariance assembly with the
   sign-safe equivalent (numerically identical for the all-positive default
   weights, correct for negative weights):

```python
    # ckf_predict:
    residuals = transformed - x_pred
    P_pred = residuals.T @ (weights_arr[:, np.newaxis] * residuals) + Q

    # ckf_update innovation covariance:
    z_residuals = transformed - z_pred
    S = z_residuals.T @ (weights_arr[:, np.newaxis] * z_residuals) + R
```

   (`Pxz` in `ckf_update` already uses the sign-safe form — leave it.)
4. Update both docstrings: new Parameters entries stating the N(0, I) unit
   convention, a See Also pointing at
   `pytcl.mathematical_functions.numerical_integration.cubature_points`, and
   delete the now-false comment "All CKF weights are equal and positive".

- [ ] **Step 4: Run the new tests AND the existing CKF regression tests**

Run: `uv run pytest tests/unit/test_cubature_points.py -x -q && uv run pytest tests/unit -k "ckf" -q && uv run pytest tests/validation -k "estimation or ckf" -q`
Expected: all PASS — the default path must be bit-for-bit or
numerically indistinguishable (the covariance refactor changes operation
order; if a strict-equality regression test exists and fails at the 1e-16
level, examine it: tolerance-based tests must pass unchanged).

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: ckf_predict/ckf_update accept arbitrary cubature points"
```

---

### Task 6: Exports, coverage ledger, docs, full verification

**Files:**
- Modify: `pytcl/mathematical_functions/numerical_integration/__init__.py`
- Modify: `CHANGELOG.md`
- Modify: `docs/matlab_parity_inventory.rst:103-105`
- Possibly modify: `tests/contract/test_public_api_coverage.py` ledger (follow its failure message)

**Interfaces:**
- Consumes: all four public functions from Tasks 1-4.
- Produces: package-level exports `from pytcl.mathematical_functions.numerical_integration import fifth_order_cubature_points, seventh_order_cubature_points, spherical_radial_points, transform_cubature_points`.

- [ ] **Step 1: Add exports**

In `pytcl/mathematical_functions/numerical_integration/__init__.py`, import the four public functions from `.cubature_points` and add them to `__all__`, following the existing pattern for `quadrature`.

- [ ] **Step 2: Run the public-API coverage contract**

Run: `uv run pytest tests/contract/test_public_api_coverage.py -q`
Expected: it may fail listing the new exports as unclassified. Follow the
failure message's instructions to register them (they are PROPERTY/REFERENCE
class via tests/unit/test_cubature_points.py). Do NOT register anything as
STRUCTURAL.

- [ ] **Step 3: CHANGELOG and parity inventory**

CHANGELOG under `## [Unreleased]` / `### Added`:

```markdown
- Gaussian cubature point library: degree-5 (`fifth_order_cubature_points`),
  degree-7 (`seventh_order_cubature_points`), arbitrary-odd-degree
  spherical-radial (`spherical_radial_points`) rules and
  `transform_cubature_points`, all validated by monomial exactness.
  `ckf_predict`/`ckf_update` accept optional cubature points, making
  higher-degree CKFs a one-liner.
```

`docs/matlab_parity_inventory.rst` line ~103: amend the parenthetical
"(~148 files ... pytcl has Gauss-Hermite and spherical cubature only)" to
reflect the new coverage, e.g. "pytcl now has the degree-5/7 Gaussian rules,
arbitrary-degree spherical-radial points, and tensor Gauss-Hermite; the
uniform-region rules remain unported".

- [ ] **Step 4: Full verification**

Run, in order:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check pytcl
uv run pytest tests/unit/test_cubature_points.py -q
uv run pytest --doctest-modules pytcl/mathematical_functions/numerical_integration/ -q
PYTCL_REQUIRE_MLX=1 uv run pytest -q
```

Expected: all green. The full suite is required (CLAUDE.md) because Task 5
touched the estimation stack. Revert any regenerated
`docs/_static/images/examples/*.html` before committing
(`git checkout -- docs/_static/images/examples/`).

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "feat: export cubature points, update ledger and parity docs"
```

---

## Final step: PR

```bash
git push -u origin feat/cubature-points
gh pr create --base main --title "feat: estimation-grade Gaussian cubature point library" \
  --body "Implements docs/superpowers/specs/2026-08-07-cubature-points-design.md: degree-5/degree-7/arbitrary-degree Gaussian cubature rules, transform helper, and optional cubature points in ckf_predict/ckf_update. All rules gated by monomial-exactness + sharpness tests and tensor-GH reference checks."
```
