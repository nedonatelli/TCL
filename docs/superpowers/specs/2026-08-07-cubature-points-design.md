# Estimation-Grade Gaussian Cubature Points

**Date:** 2026-08-07
**Branch:** `feat/cubature-points`
**Status:** Approved

## Problem

The MATLAB TCL's cubature-point library (~148 files) is called out in
`docs/matlab_parity_inventory.rst` as "a signature strength" and is almost
entirely unported: pytcl has tensor-product Gauss-Hermite
(`cubature_gauss_hermite`), the 3rd-degree spherical-radial CKF points
(`ckf_spherical_cubature_points`), and UT sigma points — nothing else. The
CKF is hardwired to its 3rd-degree rule, so no higher-order Gaussian
moment-matching is reachable from the filter stack at all.

Scope decision (approved): port the estimation-grade Gaussian-weight rules
that filters actually consume, not the full 148-file sweep. Uniform-region
rules (cube/simplex/sphere-interior) and sparse-grid/Smolyak machinery are
explicitly out of scope; Smolyak is a candidate follow-up branch if
high-dimensional states demand it.

## Deliverables

### 1. New module: `pytcl/mathematical_functions/numerical_integration/cubature_points.py`

All rules integrate against the standard-normal weight N(0, I) and return
`(points, weights)` with `points.shape == (num_points, n)` and
`weights.sum() == 1` — the probability convention
`ckf_spherical_cubature_points` already uses. This differs from the 1-D
`gauss_hermite` (physicists' `exp(-x^2)` weight); the module docstring
states the convention and the sqrt(2) mapping explicitly.

| Function | Rule | Points |
|----------|------|--------|
| `fifth_order_cubature_points(n)` | Degree-5 fully-symmetric (Stroud E_n^{r^2} 5-3 / McNamee-Stenger); MATLAB `fifthOrderCubPoints` counterpart | `2n^2 + 1` |
| `seventh_order_cubature_points(n)` | Degree-7 McNamee-Stenger fully-symmetric family | O(n^3); negative weights occur for some n and are documented, not suppressed |
| `spherical_radial_points(n, degree)` | Arbitrary odd degree: generalized Gauss-Laguerre radial rule x dimension-recursive spherical-coordinate surface rule | Grows fast with n and degree; docstring says so and points to the named rules as the efficient special cases |
| `transform_cubature_points(points, weights, mean, sqrt_cov)` | Affine map of unit points to given mean / lower-Cholesky sqrt-covariance (MATLAB `transformCubPoints`) | unchanged |

Exact generator tables/recursions are pinned at implementation time against
the monomial-exactness tests; the references are Stroud (1971), McNamee &
Stenger (1967), and the MATLAB TCL sources.

### 2. Filter hookup

`ckf_predict` / `ckf_update` in `pytcl/dynamic_estimation/kalman/unscented.py`
gain optional `points` / `weights` keyword arguments (unit points, N(0, I)
convention). Default `None` reproduces today's 3rd-degree behavior bit-for-bit.
Validation: points/weights shape consistency with the state dimension —
`ValueError` on mismatch, nothing more.

## Out of scope

- Uniform-weight regions (hypercube, simplex, sphere interior/surface as
  integration domains in their own right).
- Sparse-grid (Smolyak) Gauss-Hermite (follow-up candidate).
- Randomized/stochastic cubature.
- New filter classes; only the optional-points hookup to the existing CKF.

## Error handling

`ValueError` for `n < 1`, non-odd or `< 3` `degree`, or shape-inconsistent
`points`/`weights` passed to the CKF. No fallbacks, no new dependencies.

## Testing

Per the CONTRIBUTING validation classes (REFERENCE or PROPERTY required):

**PROPERTY**
- Each rule integrates every monomial of total degree <= its stated degree
  exactly, for n = 1..6.
- Sharpness: for each rule, at least one degree+1 monomial is *not*
  integrated exactly — guards against a vacuous exactness test.
- `weights.sum() == 1`; point-set symmetry (antipodal points present);
  odd-moment integrals vanish.

**REFERENCE**
- Gaussian expectations of non-polynomial functions (e.g. `E[sin(a.T x)]`,
  which has a closed form) cross-checked against dense tensor Gauss-Hermite —
  an independent in-repo method.
- 5th-order points/weights for small n spot-checked against published Stroud
  table values, source cited in the test.

**Filters**
- Existing CKF tests pass unchanged (default path is untouched).
- New test: `ckf_predict` with 5th-order points reproduces tensor-GH moment
  propagation on a nonlinear model to tight tolerance.

## Documentation

- NumPy-style docstrings with executable doctests (they run in CI).
- CHANGELOG entry under Unreleased/Added.
- `docs/matlab_parity_inventory.rst` cubature line updated to reflect the
  new coverage.
