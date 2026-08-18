"""Check-0 validation for the Gaussian LCD objective and analytic gradients.

This is the spec's check 0 (docs/superpowers/specs/2026-08-16-lcd-samples-design.md,
Section 5): finite-difference verification of every transcribed gradient routine
against its objective piece, with no optimizer in the loop. A subtly wrong
gradient can still descend into a plausible basin and pass optimizer-level
checks, so these tests gate everything downstream.

No MATLAB fixtures are used here by design: raw-coordinate comparison is
provably invalid for this problem (flat optimum manifold under O(n); spec
Section 3). Only gradient check-0 and structural invariants are tested.

Measured baselines (this grid, 10 seeded points per (n, num_points) case,
Apple Silicon macOS, 2026-08-18): worst central-difference relative
discrepancy was 1.9e-7 for the four gradient pieces at h=1e-5, and 9.3e-7
for the assembled objective at h=1e-4. The objective needs the larger step
because its value is O(2500) while -2*D2' + D3' can nearly cancel (gradient
norms down to ~0.04 on this grid), so the eps*|f|/h subtraction-roundoff
floor dominates at h=1e-5 (measured 8.3e-6 there, shrinking 10x per 10x
step increase -- pure FD noise, verified not a transcription error).
Assertions allow 1e-5 (margin >= 10x); anything worse than 1e-4 would
indicate a transcription error per the campaign brief.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from scipy.special import exp1

from pytcl.core.exceptions import ConvergenceError, SingularMatrixError
from pytcl.mathematical_functions.numerical_integration.lcd_samples import (
    _compute_d1,
    _compute_de2,
    _compute_de2_grad,
    _compute_de3,
    _compute_de3_grad,
    _compute_do2_cont_term,
    _compute_do2_grad,
    _compute_do2_simp,
    _compute_do3,
    _compute_do3_grad,
    _ei,
    _lcd_objective,
    gaussian_lcd_samples,
)

# The campaign grid: both parities of n (1, 3 odd; 2, 4 even) and of
# num_points (10, 20 even; 5, 15 odd).
GRID = [(1, 5), (2, 10), (2, 20), (3, 15), (4, 20)]

B_MAX = 70.0
ABS_TOL = 1e-14
REL_TOL = 1e-14

# Central-difference steps, calibrated per target (see module docstring):
# pieces have O(1) gradient norms so h=1e-5 suffices; the assembled
# objective's near-cancelling gradient needs h=1e-4 to stay above the
# eps*|f|/h roundoff floor.
FD_STEP = 1e-5
FD_STEP_OBJECTIVE = 1e-4

# Measured worst was 1.9e-7 (pieces) / 9.3e-7 (objective) on this grid;
# 1e-5 leaves >=10x margin, far below the 1e-4 investigate-threshold.
CHECK_GRAD_TOL = 1e-5

NUM_RANDOM_POINTS = 10

# The four analytic gradient routines, each paired with the objective piece
# it differentiates. Names match the MATLAB subfunctions they transcribe.
PIECES = {
    "De2": (
        lambda s, b: _compute_de2(s, b, ABS_TOL, REL_TOL),
        lambda s, b: _compute_de2_grad(s, b, ABS_TOL, REL_TOL),
    ),
    "De3": (
        lambda s, b: _compute_de3(s, b),
        lambda s, b: _compute_de3_grad(s, b),
    ),
    "Do2": (
        lambda s, b: _compute_do2_simp(s, b, ABS_TOL, REL_TOL),
        lambda s, b: _compute_do2_grad(s, b, ABS_TOL, REL_TOL),
    ),
    "Do3": (
        lambda s, b: _compute_do3(s, b),
        lambda s, b: _compute_do3_grad(s, b),
    ),
}


def _half_shape(n, num_points):
    return (n, num_points // 2)


def _random_s(n, num_points, k):
    rng = np.random.default_rng((n, num_points, k))
    return rng.standard_normal(_half_shape(n, num_points))


def _central_diff(f, x0, h=FD_STEP):
    """Central-difference gradient of scalar f at flat vector x0."""
    g = np.empty_like(x0)
    for k in range(x0.size):
        xp = x0.copy()
        xp[k] += h
        xm = x0.copy()
        xm[k] -= h
        g[k] = (f(xp) - f(xm)) / (2.0 * h)
    return g


def _rel_discrepancy(g_analytic, g_numeric):
    return np.linalg.norm(g_analytic - g_numeric) / np.linalg.norm(g_analytic)


class TestCheckGradPieces:
    """Check-0 for each of the four gradient routines individually."""

    @pytest.mark.parametrize("piece", sorted(PIECES))
    @pytest.mark.parametrize("n,num_points", GRID)
    def test_piece_gradient_matches_central_differences(self, piece, n, num_points):
        value_fn, grad_fn = PIECES[piece]
        shape = _half_shape(n, num_points)
        worst = 0.0
        for k in range(NUM_RANDOM_POINTS):
            s = _random_s(n, num_points, k)

            def f(x):
                return value_fn(x.reshape(shape, order="F"), B_MAX)

            g_ana = grad_fn(s, B_MAX).flatten(order="F")
            g_num = _central_diff(f, s.flatten(order="F"))
            worst = max(worst, _rel_discrepancy(g_ana, g_num))
        assert worst < CHECK_GRAD_TOL, (
            f"{piece} gradient check-0 failed at ({n},{num_points}): "
            f"worst relative discrepancy {worst:.3e}"
        )

    @pytest.mark.parametrize("piece", ["De3", "Do3"])
    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_piece_gradient_small_b_max(self, piece, n, num_points):
        """Repeat check-0 at b_max=5 so a mis-scaled b_max term cannot hide
        behind b_max=70's near-flat exp/Ei regime."""
        value_fn, grad_fn = PIECES[piece]
        shape = _half_shape(n, num_points)
        for k in range(NUM_RANDOM_POINTS):
            s = _random_s(n, num_points, k)

            def f(x):
                return value_fn(x.reshape(shape, order="F"), 5.0)

            g_ana = grad_fn(s, 5.0).flatten(order="F")
            g_num = _central_diff(f, s.flatten(order="F"))
            assert _rel_discrepancy(g_ana, g_num) < CHECK_GRAD_TOL


class TestCheckGradObjective:
    """Check-0 for the assembled objective/gradient pair (both parity
    branches of modCvMDist/modCvMDistGrad are exercised by the grid)."""

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_objective_gradient_matches_central_differences(self, n, num_points):
        worst = 0.0
        for k in range(NUM_RANDOM_POINTS):
            s_flat = _random_s(n, num_points, k).flatten(order="F")

            def f(x):
                return _lcd_objective(x, n, num_points)[0]

            g_ana = _lcd_objective(s_flat, n, num_points)[1]
            g_num = _central_diff(f, s_flat, h=FD_STEP_OBJECTIVE)
            worst = max(worst, _rel_discrepancy(g_ana, g_num))
        assert worst < CHECK_GRAD_TOL, (
            f"objective check-0 failed at ({n},{num_points}): "
            f"worst relative discrepancy {worst:.3e}"
        )


class TestOrthogonalInvariance:
    """cost(R @ s) == cost(s) for orthogonal R (spec Section 3 proof)."""

    @pytest.mark.parametrize("n,num_points", [(2, 10), (2, 20), (3, 15), (4, 20)])
    def test_cost_invariant_under_orthogonal_transform(self, n, num_points):
        for k in range(5):
            s = _random_s(n, num_points, k)
            rng = np.random.default_rng((7, n, num_points, k))
            q, r = np.linalg.qr(rng.standard_normal((n, n)))
            q = q * np.sign(np.diag(r))  # Haar-ish; det sign irrelevant

            cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
            cost_rot = _lcd_objective((q @ s).flatten(order="F"), n, num_points)[0]
            assert cost_rot == pytest.approx(cost, rel=1e-12, abs=1e-12)


class TestObjectiveSanity:
    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_cost_invariant_under_column_permutation(self, n, num_points):
        s = _random_s(n, num_points, 0)
        perm = np.random.default_rng(3).permutation(s.shape[1])
        cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
        cost_perm = _lcd_objective(s[:, perm].flatten(order="F"), n, num_points)[0]
        assert cost_perm == pytest.approx(cost, rel=1e-12)

    @pytest.mark.parametrize("n,num_points", [(2, 10), (3, 15)])
    def test_cost_invariant_under_single_column_negation(self, n, num_points):
        """Negating one column swaps the diff/sum roles in D3 and leaves
        column norms (D2) unchanged, so the cost is invariant."""
        s = _random_s(n, num_points, 1)
        s_neg = s.copy()
        s_neg[:, 0] *= -1.0
        cost = _lcd_objective(s.flatten(order="F"), n, num_points)[0]
        cost_neg = _lcd_objective(s_neg.flatten(order="F"), n, num_points)[0]
        assert cost_neg == pytest.approx(cost, rel=1e-12)

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_objective_returns_float_and_matching_gradient_shape(self, n, num_points):
        s_flat = _random_s(n, num_points, 2).flatten(order="F")
        value, grad = _lcd_objective(s_flat, n, num_points)
        assert isinstance(value, float)
        assert np.isfinite(value)
        assert grad.shape == s_flat.shape
        assert np.all(np.isfinite(grad))

    def test_odd_d2_is_rescaled_even_d2(self):
        """computeDo2Simp is (2L/(2L+1)) * computeDe2 (MATLAB lines 360-368)."""
        s = _random_s(3, 15, 3)
        length = s.shape[1]
        de2 = _compute_de2(s, B_MAX, ABS_TOL, REL_TOL)
        do2 = _compute_do2_simp(s, B_MAX, ABS_TOL, REL_TOL)
        assert do2 == pytest.approx(2 * length / (2 * length + 1) * de2, rel=1e-13)

    def test_constant_terms_positive_and_finite(self):
        """D1 and the odd-count D2 constant are positive integrals of
        positive integrands; they never touch the gradient."""
        for n in (1, 2, 3, 4):
            assert _compute_d1(B_MAX, n, ABS_TOL, REL_TOL) > 0.0
        assert _compute_do2_cont_term(3, 7, B_MAX, ABS_TOL, REL_TOL) > 0.0


class TestEi:
    def test_zero_maps_to_zero(self):
        """MATLAB redefines Ei(0)=0 to kill the 0*Inf diagonal terms."""
        assert _ei(np.array(0.0)) == 0.0
        assert np.all(_ei(np.array([0.0, 0.0])) == 0.0)

    def test_negative_arguments_match_exp1(self):
        """For x<0, Ei(x) = -real(expint(-x)) = -E1(-x)."""
        x = np.array([-2.0, -0.5, -1e-8])
        np.testing.assert_allclose(_ei(x), -exp1(-x), rtol=1e-14)

    def test_preserves_input_shape(self):
        x = -np.ones((3, 4))
        assert _ei(x).shape == (3, 4)


def _s_min_from_points(points, n, num_points):
    """Recover the raw (unmirrored) free-point matrix s from a
    force_cov_match=False `gaussian_lcd_samples` result, for tests that
    need to re-evaluate `_lcd_objective` at the converged point. Only
    valid when whitening was not applied (whitening is not invertible
    without keeping the transform matrix around, and isn't needed for
    these objective-value checks -- whitening never touches the cost)."""
    num_half = num_points // 2
    is_even = num_points % 2 == 0
    xi = points.T
    return xi[:, :num_half] if is_even else xi[:, 1 : 1 + num_half]


class TestGaussianLCDSamplesConvergence:
    """Convergence on the campaign grid: the call succeeds (no exception --
    this port's equivalent of MATLAB's exitCode>=0, per the design spec's
    validation contract, Section 5 check 1) and the optimized CvM objective
    is strictly below its value at the random initialization (spec's
    "objective decreased from init" contract).

    Measured on this grid (macOS/Apple Silicon, 2026-08-18): every case
    converges and the objective decreases by ~0.04-0.11 in absolute terms
    (the values themselves are O(-1500) to O(-2500) -- D1/D3 dominate and
    are large; the optimized D2 correction is comparatively small).
    DECREASE_MARGIN is far below the measured decrease and exists only to
    absorb float/quadrature noise: quad tolerance 1e-14 at O(2500)
    magnitude gives an absolute noise floor around 2.5e-11.
    """

    DECREASE_MARGIN = 1e-6

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_converges_and_objective_decreases(self, n, num_points):
        seed = (100, n, num_points)

        probe_rng = np.random.default_rng(seed)
        s_init = probe_rng.standard_normal(_half_shape(n, num_points))
        init_cost = _lcd_objective(s_init.flatten(order="F"), n, num_points)[0]

        run_rng = np.random.default_rng(seed)
        points, weights = gaussian_lcd_samples(
            n, num_points, force_cov_match=False, rng=run_rng
        )

        s_min = _s_min_from_points(points, n, num_points)
        final_cost = _lcd_objective(s_min.flatten(order="F"), n, num_points)[0]

        assert final_cost < init_cost - self.DECREASE_MARGIN
        assert points.shape == (num_points, n)
        assert weights.shape == (num_points,)


class TestGaussianLCDSamplesMoments:
    """Under force_cov_match=True (the default), the returned points' sample
    mean is 0 and sample covariance is I, exactly by construction of the
    Cholesky-whitening transform (spec Section 5 check 4 -- an exactness
    check, not a statistical one, since every grid case here satisfies
    num_points >= 2*n).

    Measured worst case on this grid (macOS/Apple Silicon, 2026-08-18):
    mean |max component| ~4.4e-17, covariance |max deviation from I|
    ~4.4e-16. MOMENT_TOL leaves >=1000x margin on both.
    """

    MOMENT_TOL = 1e-10

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_mean_and_covariance_match_identity(self, n, num_points):
        rng = np.random.default_rng((200, n, num_points))
        points, weights = gaussian_lcd_samples(n, num_points, rng=rng)

        mean = points.mean(axis=0)
        cov = (points * weights[:, None]).T @ points  # sum_i w_i x_i x_i^T

        assert np.max(np.abs(mean)) < self.MOMENT_TOL
        assert np.max(np.abs(cov - np.eye(n))) < self.MOMENT_TOL
        np.testing.assert_allclose(weights, 1.0 / num_points)
        assert weights.sum() == pytest.approx(1.0, rel=1e-12)


class TestGaussianLCDSamplesDeterminism:
    """Bit-exact output given a seeded numpy.random.Generator: two runs
    seeded identically must produce identical points and weights (scipy's
    L-BFGS-B, the objective, and the whitening step are all deterministic
    given identical inputs)."""

    @pytest.mark.parametrize("n,num_points", GRID)
    def test_bit_exact_given_seeded_rng(self, n, num_points):
        points1, weights1 = gaussian_lcd_samples(
            n, num_points, rng=np.random.default_rng((300, n, num_points))
        )
        points2, weights2 = gaussian_lcd_samples(
            n, num_points, rng=np.random.default_rng((300, n, num_points))
        )
        np.testing.assert_array_equal(points1, points2)
        np.testing.assert_array_equal(weights1, weights2)


class TestGaussianLCDSamplesOrthogonalFamily:
    """Two different seeds land on different points of the flat optimum
    manifold (spec Section 3): the objective is invariant under any global
    orthogonal transform for n >= 2, so two independently-converged
    solutions generically differ in raw coordinates but agree closely in
    objective value. n == 1 is excluded (O(1) is discrete, not a
    continuous symmetry group -- see TestOrthogonalInvariance above, which
    covers the underlying cost-invariance property directly).

    Measured on this grid (macOS/Apple Silicon, 2026-08-18): cost relative
    difference between two seeds was <=1.2e-12 in every case, while the
    max per-coordinate point difference was >=2.27 -- clearly different
    point clouds, clearly the same cost. COST_REL_TOL leaves >=1e6x margin.
    """

    COST_REL_TOL = 1e-6
    POINT_DIFF_FLOOR = 1e-3

    @pytest.mark.parametrize("n,num_points", [(2, 10), (2, 20), (3, 15), (4, 20)])
    def test_different_seeds_different_points_similar_cost(self, n, num_points):
        def run(seed):
            rng = np.random.default_rng(seed)
            points, _ = gaussian_lcd_samples(
                n, num_points, force_cov_match=False, rng=rng
            )
            s_min = _s_min_from_points(points, n, num_points)
            cost = _lcd_objective(s_min.flatten(order="F"), n, num_points)[0]
            return points, cost

        points_a, cost_a = run((400, n, num_points))
        points_b, cost_b = run((401, n, num_points))

        assert np.max(np.abs(points_a - points_b)) > self.POINT_DIFF_FLOOR
        assert cost_a == pytest.approx(cost_b, rel=self.COST_REL_TOL)


class TestGaussianLCDSamplesErrors:
    def test_rejects_dimension_below_one(self):
        with pytest.raises(ValueError):
            gaussian_lcd_samples(0, 10)

    def test_rejects_num_points_below_two(self):
        with pytest.raises(ValueError):
            gaussian_lcd_samples(2, 1)

    def test_singular_covariance_raises_when_undersampled(self):
        """num_points < 2*n: a symmetric point set cannot span n
        independent directions, so the sample covariance is generically
        singular and force_cov_match=True (the default) must raise rather
        than silently return an ill-posed whitened result."""
        with pytest.raises(SingularMatrixError):
            gaussian_lcd_samples(4, 4, rng=np.random.default_rng(0))

    def test_convergence_error_on_zero_iterations(self):
        """max_iter=0 gives L-BFGS-B no iterations to run; scipy reports
        this as a non-success exit, which must surface as ConvergenceError
        rather than silently returning the (unoptimized) starting point."""
        with pytest.raises(ConvergenceError):
            gaussian_lcd_samples(2, 10, rng=np.random.default_rng(0), max_iter=0)


class TestGaussianLCDSamplesPublicAPIExport:
    """Reaches the public-API coverage contract via the package-level
    export path (numerical_integration/__init__.py), not just the direct
    module import used by every other test class in this file."""

    def test_importable_from_package_init(self):
        from pytcl.mathematical_functions.numerical_integration import (
            gaussian_lcd_samples as exported,
        )

        points, weights = exported(2, 10, rng=np.random.default_rng(1))
        assert points.shape == (10, 2)
        assert weights.shape == (10,)


_MATLAB_FIXTURES = Path(__file__).parents[1] / "fixtures" / "matlab"


def _load_lcd_fixture(n, num_points):
    """Load one (n, num_points) LCD MATLAB fixture case.

    Skip-gated per the repo's established convention
    (tests/unit/test_cubature_points.py's ``_load_matlab_fixture`` /
    tests/unit/test_region_cubature.py's twin): skips gracefully if any of
    the case's three ``lcd_n<N>_pts<P>*.csv`` files is absent, or fails hard
    under ``PYTCL_REQUIRE_MATLAB_FIXTURES=1``. No lcd_* fixtures exist in
    this checkout as of writing -- every test in
    TestGaussianLCDSamplesMatlabFixtures skips until a maintainer with
    MATLAB access runs ``scripts/matlab_capture/capture_lcd.m`` (which
    itself requires that MATLAB checkout's ``CompileCLibraries.m`` to have
    already been run, so ``quasiNewtonLBFGS`` exists as a compiled MEX
    function rather than the unmexed stub -- capture_lcd.m's own preflight
    check fails immediately otherwise) and commits the resulting CSVs.

    Returns
    -------
    xi_fixture : ndarray, shape (num_points, n)
        MATLAB's captured, already-mirrored (and, per capture_lcd.m,
        already force-cov-match-whitened) point set.
    w_fixture : ndarray, shape (num_points,)
        MATLAB's captured weights (uniform, ``1/num_points``).
    s_init : ndarray, shape (n, num_points // 2)
        The exact sInit matrix MATLAB's optimizer started from (native,
        untransposed layout, per capture_lcd.m).
    meta : dict
        ``numDim``, ``numSamples``, ``CvMDistMin``, ``exitCode``,
        ``determinism_max_abs_diff``, all as Python floats.
    """
    base = f"lcd_n{n}_pts{num_points}"
    points_path = _MATLAB_FIXTURES / f"{base}.csv"
    sinit_path = _MATLAB_FIXTURES / f"{base}_sinit.csv"
    meta_path = _MATLAB_FIXTURES / f"{base}_meta.csv"

    missing = [p.name for p in (points_path, sinit_path, meta_path) if not p.exists()]
    if missing:
        reason = (
            f"MATLAB LCD fixture(s) {missing} for (n={n}, num_points="
            f"{num_points}) not captured yet -- run "
            "scripts/matlab_capture/capture_lcd.m in MATLAB, against a "
            "TrackerComponentLibrary checkout that has already had its "
            "CompileCLibraries.m run (capture_lcd.m requires "
            "quasiNewtonLBFGS compiled as a MEX function -- its own "
            "preflight check fails immediately otherwise), then commit "
            "the resulting CSVs to tests/fixtures/matlab/."
        )
        if os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1":
            pytest.fail(reason)
        pytest.skip(reason)

    points_data = np.loadtxt(points_path, delimiter=",", ndmin=2)
    xi_fixture = points_data[:, :-1]
    w_fixture = points_data[:, -1]

    s_init = np.loadtxt(sinit_path, delimiter=",", ndmin=2)

    meta_raw = np.genfromtxt(meta_path, delimiter=",", names=True)
    meta_fields = (
        "numDim",
        "numSamples",
        "CvMDistMin",
        "exitCode",
        "determinism_max_abs_diff",
    )
    meta = {field: float(meta_raw[field]) for field in meta_fields}

    return xi_fixture, w_fixture, s_init, meta


class _FixedStandardNormal:
    """Stand-in for ``numpy.random.Generator`` whose ``standard_normal()``
    always returns a pre-loaded matrix, regardless of the requested shape
    (asserted to match, as a sanity check).

    ``gaussian_lcd_samples`` only accepts an ``rng``, not a raw starting
    matrix -- by design, since ordinary callers should never need to inject
    one. The MATLAB-fixture comparison is the one place that legitimately
    does: per the design spec (Section 4), both sides must start from the
    literal same sInit numbers for the cross-implementation comparison to
    mean anything (spec Section 3's rotation-invariance argument is about
    what happens *after* two optimizers start from the same point, not
    about independently-seeded runs). This stand-in drives the port's
    public API from that captured sInit without adding a parameter to the
    production signature for a validation-only need.
    """

    def __init__(self, s_init: np.ndarray) -> None:
        self._s_init = s_init

    def standard_normal(self, size):
        assert tuple(size) == self._s_init.shape, (
            f"fixture sInit shape {self._s_init.shape} does not match "
            f"the requested draw shape {tuple(size)}"
        )
        return self._s_init


class TestGaussianLCDSamplesMatlabFixtures:
    """Cross-check against real MATLAB ``GaussianLCDSamples`` output
    (``scripts/matlab_capture/capture_lcd.m``), per the design spec's
    validation contract (docs/superpowers/specs/2026-08-16-lcd-samples-design.md,
    Section 5, checks 1/2/3/4 -- check 0, the finite-difference gradient
    check, has no fixture dependency and lives in TestCheckGradPieces /
    TestCheckGradObjective above).

    NEVER compares raw coordinates. For n >= 2 the CvM cost is provably
    invariant under any global orthogonal transform applied to every point
    simultaneously (spec Section 3), so a converged point set is one
    arbitrary representative of a continuous manifold of equally-optimal
    solutions -- a correctly-ported optimizer started from the identical
    sInit generically lands on a *different* representative of that same
    manifold (same CvM cost, different raw coordinates), and this is
    expected, not a bug. Only rotation-and-permutation-invariant statistics
    (the Gram matrix's sorted eigenvalue spectrum) and scalar/exact
    quantities (the CvM objective value, sample mean, sample covariance)
    are compared.

    Every test loads the fixture's own captured sInit and drives
    gaussian_lcd_samples's public API with it via _FixedStandardNormal, so
    the port's optimizer starts from the literal same numbers MATLAB's did
    -- the spec's stated precondition (Section 4) for the comparison to be
    meaningful at all: it puts both optimizers in the same basin, so
    results should agree more tightly than two independently-seeded runs
    would (contrast TestGaussianLCDSamplesOrthogonalFamily above, which
    uses independent seeds and checks only cost-value agreement, never
    spectrum agreement).

    No lcd_* fixtures exist in this checkout as of writing (2026-08-18), so
    every test below skips via _load_lcd_fixture. All numeric tolerances
    here are therefore PLACEHOLDERS per this repo's
    claims-inherit-measurement-range convention, not measurements -- they
    must be recalibrated against real captured values once a maintainer
    runs capture_lcd.m, exactly as the design spec itself flags (Section 9:
    "Cost-value tolerance ... is a placeholder pending real data").
    """

    CASES = [(1, 5), (2, 10), (2, 20), (3, 15), (4, 20)]

    B_MAX = 70.0
    ABS_TOL = 1e-14
    REL_TOL = 1e-14

    # Spec Section 5 check 2's own stated placeholder ("a reasoned starting
    # guess, not a measurement"; Section 9 repeats the caveat explicitly).
    CVM_VALUE_RTOL = 1e-4

    # The spec's check 3 says to calibrate the Gram-spectrum tolerance "the
    # same way as (2)" -- i.e. the same reasoned-placeholder methodology,
    # reusing the same 1e-4 relative starting point, not a separately
    # invented number.
    GRAM_SPECTRUM_RTOL = 1e-4
    # Absolute floor for the Gram spectrum only: a numSamples x numSamples
    # Gram matrix built from n-dimensional points has rank <= n, so
    # numSamples - n of its eigenvalues are (near-)exactly zero on both
    # sides in every grid case here (num_points > n throughout) -- a pure
    # relative tolerance is undefined/unstable there.
    GRAM_SPECTRUM_ATOL = 1e-6

    # force_cov_match=True whitening is exact-by-construction (see
    # TestGaussianLCDSamplesMoments above); applied here to the fixture's
    # own captured points, which capture_lcd.m always requests
    # (forceCovMatch=true, every grid case), so this is likewise an
    # exactness check, not a statistical one. Looser than
    # TestGaussianLCDSamplesMoments.MOMENT_TOL (1e-10) only to allow for
    # the fixture CSV's %.17g round-trip, not for any statistical slack.
    MOMENT_TOL = 1e-8

    @pytest.mark.parametrize("n,num_points", CASES)
    def test_cvm_value_matches_matlab(self, n, num_points):
        """Spec Section 5 checks 1 (exitCode>=0) and 2 (CvM value parity).

        The port's own CvMDistMin is assembled the same way
        GaussianLCDSamples.m assembles it (module docstring): the raw
        optimized cost from _lcd_objective, plus the constant D1 term,
        minus 2x the odd-count D2 constant (computeDo2ContTerm) when
        num_points is odd -- both never enter the optimized cost/gradient,
        so they are added back here exactly as the MATLAB main function
        does before reporting CvMDistMin.
        """
        _, _, s_init, meta = _load_lcd_fixture(n, num_points)
        assert int(meta["exitCode"]) >= 0

        # force_cov_match=False: whitening is a post-hoc transform applied
        # after the optimizer already committed to s_min, so it must not
        # enter a CvM-value comparison (the minimize() call itself is
        # identical either way -- see gaussian_lcd_samples's source).
        points, _ = gaussian_lcd_samples(
            n,
            num_points,
            force_cov_match=False,
            rng=_FixedStandardNormal(s_init),
        )
        s_min = _s_min_from_points(points, n, num_points)

        num_half = num_points // 2
        port_cost = _lcd_objective(s_min.flatten(order="F"), n, num_points)[0]
        port_cost += _compute_d1(self.B_MAX, n, self.ABS_TOL, self.REL_TOL)
        if num_points % 2 != 0:
            port_cost -= 2.0 * _compute_do2_cont_term(
                n, num_half, self.B_MAX, self.ABS_TOL, self.REL_TOL
            )

        assert port_cost == pytest.approx(meta["CvMDistMin"], rel=self.CVM_VALUE_RTOL)

    @pytest.mark.parametrize("n,num_points", CASES)
    def test_gram_spectrum_matches_matlab(self, n, num_points):
        """Spec Section 5 check 3: the rotation-and-permutation-invariant
        Gram-matrix eigenvalue spectrum of the FULL point set (fixture's
        xi vs the port's, both under force_cov_match=True, matching how
        capture_lcd.m captured the fixture) -- never raw coordinates."""
        xi_fixture, _, s_init, _ = _load_lcd_fixture(n, num_points)

        points, _ = gaussian_lcd_samples(
            n,
            num_points,
            force_cov_match=True,
            rng=_FixedStandardNormal(s_init),
        )

        eig_fixture = np.sort(np.linalg.eigvalsh(xi_fixture @ xi_fixture.T))
        eig_port = np.sort(np.linalg.eigvalsh(points @ points.T))

        np.testing.assert_allclose(
            eig_port,
            eig_fixture,
            rtol=self.GRAM_SPECTRUM_RTOL,
            atol=self.GRAM_SPECTRUM_ATOL,
        )

    @pytest.mark.parametrize("n,num_points", CASES)
    def test_moment_identities_on_fixture_points(self, n, num_points):
        """Spec Section 5 check 4, applied directly to the fixture's own
        captured points (no port run involved): capture_lcd.m always
        passes forceCovMatch=true, so the fixture's sample mean should be
        (near-)exactly 0 and its sample covariance (near-)exactly the
        identity -- a sanity check on this test module's own understanding
        of the fixture format, independent of the port's optimizer."""
        xi_fixture, w_fixture, _, _ = _load_lcd_fixture(n, num_points)

        mean = np.average(xi_fixture, axis=0, weights=w_fixture)
        cov = (xi_fixture * w_fixture[:, None]).T @ xi_fixture

        assert np.max(np.abs(mean)) < self.MOMENT_TOL
        assert np.max(np.abs(cov - np.eye(n))) < self.MOMENT_TOL
