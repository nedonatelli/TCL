"""
Tests for Joint Probabilistic Data Association (JPDA) algorithms.

This module contains tests for JPDA-based multi-target association and tracking.
Tests are migrated from v0.3.0 comprehensive test suite.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from pytcl.assignment_algorithms.gating import mahalanobis_distance
from pytcl.assignment_algorithms.jpda import (
    compute_likelihood_matrix,
    compute_measurement_likelihood,
)

# =============================================================================
# JPDA Basic Tests
# =============================================================================


class TestJPDA:
    """Tests for Joint Probabilistic Data Association."""

    def test_jpda_probabilities_single_track(self):
        """Test JPDA with single track."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Single track, two measurements
        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.01
        )

        assert beta.shape == (1, 3)  # 1 track, 2 meas + 1 for no-meas
        # Probabilities should sum to 1 for each track
        assert_allclose(np.sum(beta[0, :]), 1.0, rtol=1e-6)
        # Higher likelihood measurement should have higher probability
        assert beta[0, 0] > beta[0, 1]

    def test_jpda_update_basic(self):
        """Test basic JPDA update."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2], [10.0]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x1, x2], [P, P], measurements, H, R)

        assert len(result.states) == 2
        assert len(result.covariances) == 2
        assert result.association_probs.shape == (2, 4)  # 2 tracks, 3 meas + no-meas

    def test_jpda_no_measurements(self):
        """Test JPDA with no measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([]).reshape(0, 1)
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # With no measurements, state should be unchanged
        assert_allclose(result.states[0], x)

    def test_jpda_result_convenience(self):
        """Test JPDA convenience function."""
        from pytcl.assignment_algorithms import jpda

        x1 = np.array([0.0, 1.0])
        x2 = np.array([5.0, -1.0])
        P = np.eye(2) * 0.5

        measurements = np.array([[0.1], [5.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda([x1, x2], [P, P], measurements, H, R)

        assert result.association_probs.shape == (2, 3)
        assert len(result.marginal_probs) == 2
        assert result.likelihood_matrix.shape == (2, 2)


# =============================================================================
# JPDA Comprehensive Tests (v0.3.0)
# =============================================================================


class TestJPDAComprehensive:
    """Comprehensive tests for JPDA algorithm robustness and edge cases."""

    def test_jpda_probabilities_normalization(self):
        """Test JPDA probabilities sum to 1."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        # Multiple tracks, multiple measurements
        likelihood = np.array([[0.8, 0.1, 0.05], [0.1, 0.7, 0.1], [0.05, 0.1, 0.6]])
        gated = np.ones_like(likelihood, dtype=bool)

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Each track's probabilities should sum to 1
        for i in range(3):
            assert_allclose(np.sum(beta[i, :]), 1.0, rtol=1e-6)

    def test_jpda_high_clutter(self):
        """Test JPDA behavior with high clutter density."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.5, 0.3]])
        gated = np.array([[True, True]])

        # High clutter should increase miss probability
        beta_low_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.001
        )
        beta_high_clutter = jpda_probabilities(
            likelihood, gated, detection_prob=0.9, clutter_density=0.1
        )

        # With high clutter, more probability goes to "no measurement"
        assert beta_high_clutter[0, -1] > beta_low_clutter[0, -1]

    def test_jpda_low_detection_probability(self):
        """Test JPDA with low detection probability."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.1]])
        gated = np.array([[True, True]])

        beta_high_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.99, clutter_density=0.01
        )
        beta_low_pd = jpda_probabilities(
            likelihood, gated, detection_prob=0.5, clutter_density=0.01
        )

        # Low detection prob should increase miss probability
        assert beta_low_pd[0, -1] > beta_high_pd[0, -1]

    def test_jpda_gating_effect(self):
        """Test that gating correctly excludes measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_probabilities

        likelihood = np.array([[0.8, 0.7, 0.1]])
        gated = np.array([[True, False, True]])  # Middle measurement not gated

        beta = jpda_probabilities(likelihood, gated, detection_prob=0.9)

        # Gated-out measurement should have zero probability
        assert_allclose(beta[0, 1], 0.0)

    def test_jpda_update_with_ambiguous_measurements(self):
        """Test JPDA update handles ambiguous measurements."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        x = np.array([0.0, 1.0])
        P = np.eye(2) * 0.1

        # Two measurements, both plausible
        measurements = np.array([[0.1], [-0.1]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([x], [P], measurements, H, R)

        # JPDA should produce valid state and covariance
        assert result.states[0].shape == (2,)
        assert result.covariances[0].shape == (2, 2)
        # Covariance should remain positive definite
        eigvals = np.linalg.eigvalsh(result.covariances[0])
        assert np.all(eigvals > 0)

    def test_jpda_multiple_tracks(self):
        """Test JPDA with multiple tracks."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        # 3 tracks at different positions
        tracks = [
            np.array([0.0, 1.0]),
            np.array([5.0, 0.0]),
            np.array([10.0, -1.0]),
        ]
        covs = [np.eye(2) * 0.1 for _ in range(3)]

        # 3 measurements near each track
        measurements = np.array([[0.1], [5.1], [9.9]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update(tracks, covs, measurements, H, R)

        assert len(result.states) == 3
        assert result.association_probs.shape == (3, 4)  # 3 tracks, 3+1 columns

    def test_jpda_empty_tracks(self):
        """Test JPDA handles empty track list."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        measurements = np.array([[0.1], [0.2]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.1]])

        result = jpda_update([], [], measurements, H, R)

        assert len(result.states) == 0
        assert len(result.covariances) == 0

    def test_jpda_single_measurement_per_track(self):
        """Test JPDA when each track has exactly one measurement."""
        from pytcl.assignment_algorithms.jpda import jpda_update

        tracks = [np.array([0.0, 1.0]), np.array([10.0, -1.0])]
        covs = [np.eye(2) * 0.01, np.eye(2) * 0.01]  # Very small covariance

        # Measurements clearly associated with each track
        measurements = np.array([[0.01], [10.01]])
        H = np.array([[1.0, 0.0]])
        R = np.array([[0.001]])

        result = jpda_update(tracks, covs, measurements, H, R)

        # Each track should strongly associate with its measurement
        assert result.association_probs[0, 0] > 0.9
        assert result.association_probs[1, 1] > 0.9


# =============================================================================
# compute_likelihood_matrix behavior-equality (perf-levers Task 1)
# =============================================================================


def _reference_likelihood_matrix(
    track_states, track_covariances, measurements, H, R, detection_prob, gate_threshold
):
    """Frozen copy of the pre-restructure `compute_likelihood_matrix` loop
    body: per (track, measurement) pair, one `mahalanobis_distance` solve
    (gating.py's own dispatched solve/inv path) plus one independent
    `compute_measurement_likelihood` solve, and a per-pair `np.linalg.det`
    of a matrix that only depends on the track. This is the defect the
    Task 1 restructure removes -- the shipped version computes a single
    per-track `np.linalg.inv`/`np.linalg.det` and reuses them across all of
    that track's measurements via one batched `mahalanobis_batch` call
    (see jpda.py's `compute_likelihood_matrix` for the in-code note on why
    an initial `scipy.linalg.cho_factor`/`cho_solve` attempt was replaced
    -- measured slower, not faster). This reference is deliberately never
    touched by that restructure so it stays the ground truth for behavior
    equality.

    `mahalanobis_distance` returns the *squared* Mahalanobis distance (see
    its docstring), and the loop below gates on that squared value directly
    against `gate_threshold` (itself on the chi-squared/squared-distance
    scale, e.g. `chi2.ppf(...)`) -- there is no separate square root step.
    """
    n_tracks = len(track_states)
    n_meas = len(measurements)

    likelihood_matrix = np.zeros((n_tracks, n_meas))
    gated = np.zeros((n_tracks, n_meas), dtype=bool)

    for i in range(n_tracks):
        z_pred = H @ track_states[i]
        S = H @ track_covariances[i] @ H.T + R

        for j in range(n_meas):
            innovation = measurements[j] - z_pred
            mahal_dist = mahalanobis_distance(innovation, S)

            if gate_threshold is None or mahal_dist <= gate_threshold:
                gated[i, j] = True
                likelihood_matrix[i, j] = compute_measurement_likelihood(
                    innovation, S, detection_prob
                )

    return likelihood_matrix, gated


_N_TRACKS_CHOICES = [1, 3, 10, 40]
_N_MEAS_CHOICES = [0, 1, 7, 25]
_M_CHOICES = [2, 3, 4, 6]


def _make_likelihood_matrix_scenario(seed):
    """Build one seeded (rng spawn key `seed`) scenario for the
    behavior-equality sweep.

    n_tracks/n_meas/m are drawn from the plan's grids
    ({1,3,10,40}/{0,1,7,25}/{2,3,4,6}); state dimension is set equal to `m`
    (H = identity). Each track's covariance is `(A @ A.T + m*I) * scale`
    with `scale` log-uniform over [1e-6, 1e6] and R fixed at `0.1 * I`,
    covering the plan's named 1e-6..1e6 range in absolute magnitude.

    This does NOT produce ill-conditioned S: a uniform scalar `scale`
    multiplies every eigenvalue of `A @ A.T + m*I` equally, so it changes
    S's overall size but not its condition number. Measured across every
    track covariance in all 20 scenarios: max cond(S) = 6.31, min
    cond(S) = 1.00 -- every S here is well-conditioned regardless of
    `scale`. `TestLikelihoodMatrixIllConditioned` below covers genuine
    ill-conditioning (cond(S) up to ~1e10) via direct eigenvalue
    construction, which this recipe cannot reach.

    Odd seeds get a finite gate threshold set to 1.5x
    chi2.ppf(0.999, df=m). This does not keep every pair's mahal_sq safely
    clear of the threshold: measured across the 8 (of 10) finite-threshold
    scenarios that also have nonempty measurements, all 8 have at least one
    ungated pair, and the closest observed gate decision sits only 3.47e-4
    relative to its threshold (seed 17, track 3, measurement 6:
    mahal_sq=20.7161 vs threshold=20.7233). Gate decisions still cannot
    flip between the two computation routes despite that margin: their
    disagreement is ULP-scale (~1e-16 relative, see
    test_gating_mahalanobis_dispatch.py's measured well-conditioned
    bounds), eight orders of magnitude below the smallest observed margin
    in this sweep. Even seeds leave gate_threshold unset (ungated).
    """
    rng = np.random.default_rng(seed)
    n_tracks = int(rng.choice(_N_TRACKS_CHOICES))
    n_meas = int(rng.choice(_N_MEAS_CHOICES))
    m = int(rng.choice(_M_CHOICES))

    H = np.eye(m)
    R = 0.1 * np.eye(m)

    track_states = [rng.normal(size=m) for _ in range(n_tracks)]
    track_covariances = []
    for _ in range(n_tracks):
        A = rng.normal(size=(m, m))
        scale = 10.0 ** rng.uniform(-6.0, 6.0)
        track_covariances.append((A @ A.T + m * np.eye(m)) * scale)

    measurements = rng.normal(size=(n_meas, m)) if n_meas else np.zeros((0, m))

    gate_threshold = None
    if seed % 2 == 1:
        gate_threshold = float(chi2.ppf(0.999, df=m)) * 1.5

    return {
        "track_states": track_states,
        "track_covariances": track_covariances,
        "measurements": measurements,
        "H": H,
        "R": R,
        "detection_prob": 0.9,
        "gate_threshold": gate_threshold,
    }


_LIKELIHOOD_UNDERFLOW_FLOOR = 1e-280


def _assert_likelihoods_close(actual, expected, rtol, context):
    """Compare two likelihood arrays with a regime split at
    `_LIKELIHOOD_UNDERFLOW_FLOOR` (~1e-280 -- comfortably above float64's
    smallest normal, ~2.2e-308, and far below any physically meaningful
    likelihood).

    Above the floor: a pure relative comparison, atol=0. Every value
    compared in this branch is >= the floor, so `rtol * abs(expected)`
    alone is already a meaningful absolute bound; no atol cushion is
    needed. This replaces an earlier atol=1e-12, which let 831 of the
    20-scenario sweep's gated entries pass on the atol term alone
    regardless of relative agreement -- entirely masking seed 9's
    scenario, whose 75 entries are ALL below that old atol (down to
    ~1.26e-18).

    Below the floor: assert both sides are negligible rather than
    comparing them to each other. Two independently-computed likelihoods
    that have each underflowed toward zero via `exp(-0.5 * mahal_sq)` are
    not meaningfully comparable in relative terms -- a benign difference in
    exactly where each rounds to zero can look like an enormous relative
    disagreement despite being physically irrelevant. This branch also
    covers ungated entries (expected == 0.0 exactly): both `_reference_
    likelihood_matrix` and `compute_likelihood_matrix` leave those
    zero-initialized and never write them, so actual is exactly 0.0 too.

    Measured across the main 20-scenario sweep (`TestLikelihoodMatrixEquality`),
    the smallest nonzero reference likelihood is ~1.27e-68 -- well above
    the floor, so that sweep does not currently exercise the underflow
    branch. It exists defensively, so a future scenario landing there
    doesn't silently degrade into a meaningless relative comparison.
    """
    near_zero = np.abs(expected) < _LIKELIHOOD_UNDERFLOW_FLOOR
    if np.any(near_zero):
        assert np.all(np.abs(actual[near_zero]) < _LIKELIHOOD_UNDERFLOW_FLOOR), (
            f"{context}: actual likelihood not negligible where the "
            f"reference underflowed below {_LIKELIHOOD_UNDERFLOW_FLOOR:g}"
        )
    if np.any(~near_zero):
        assert_allclose(actual[~near_zero], expected[~near_zero], rtol=rtol, atol=0)


class TestLikelihoodMatrixEquality:
    """`compute_likelihood_matrix` must match `_reference_likelihood_matrix`
    (the frozen pre-restructure loop) across 20 seeded scenarios.

    Written and passing against the unmodified code path (the live function
    *is* the reference's logic at this point, so the match is exact). After
    the Task 1 restructure (a single per-track `np.linalg.inv` +
    `np.linalg.det`, reused across all of that track's measurements via one
    batched `mahalanobis_batch` call, replacing the per-pair double solve
    and per-pair `np.linalg.det`), likelihoods are compared via
    `_assert_likelihoods_close` at rtol=1e-9 (see that function's docstring
    for the atol=0 / underflow-floor split) -- `np.linalg.inv`'s roundings
    are not expected to exactly reproduce `mahalanobis_distance`'s
    dispatched solve/inv path -- while `gated` is asserted exactly equal
    (see `_make_likelihood_matrix_scenario`'s docstring for the measured
    margin bounding why gate decisions cannot flip).
    """

    @pytest.mark.parametrize("seed", list(range(20)))
    def test_matches_reference(self, seed):
        scenario = _make_likelihood_matrix_scenario(seed)

        expected_likelihood, expected_gated = _reference_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )
        actual_likelihood, actual_gated = compute_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )

        assert_allclose(actual_gated, expected_gated)
        _assert_likelihoods_close(
            actual_likelihood, expected_likelihood, rtol=1e-9, context=f"seed {seed}"
        )


def _near_singular_covariance(dim, seed, eigval_floor):
    """Build a `dim`x`dim` SPD matrix with one eigenvalue forced down to
    `eigval_floor`, following test_gating_mahalanobis_dispatch.py's
    `TestMahalanobisDispatchNearSingular` recipe -- this repo's established
    precedent for stress-testing Mahalanobis numerics at genuine
    ill-conditioning, unlike `_make_likelihood_matrix_scenario`'s
    scalar-scaled covariances (see that function's docstring: cond(S)
    tops out at 6.31 there).

    Returns (S, eigvals, eigvecs): the eigendecomposition is returned
    alongside S so `_make_ill_conditioned_scenario` can build measurements
    with a controlled component along S's eigenbasis.
    """
    rng = np.random.default_rng(seed * 7 + dim * 100)
    A = rng.normal(size=(dim, dim))
    S = A @ A.T
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.clip(eigvals, 1e-3, None)
    eigvals[0] = eigval_floor
    S = (eigvecs * eigvals) @ eigvecs.T
    return S, eigvals, eigvecs


def _make_ill_conditioned_scenario(dim, seed, eigval_floor, n_tracks=5, n_meas=10):
    """A `compute_likelihood_matrix` scenario with genuinely ill-conditioned
    S. H = I and R = 0 so S == the track covariance exactly, letting
    `_near_singular_covariance`'s eigenvalues control S's conditioning
    directly (a synthetic H/R pair chosen specifically to expose that S,
    not a realistic sensor model).

    Measurements are built as track 0's state plus a perturbation confined
    to S's well-conditioned eigenspace (excludes the near-null eigenvector
    at index 0): a perturbation with ANY component along the near-null
    direction blows mahal_sq up by a factor of ~1/eigval_floor, making
    every pair ungated and the likelihood comparison trivial (both sides
    exactly 0.0, as measured while developing this scenario). Confining
    the perturbation is what makes this scenario exercise S's
    ill-conditioned inverse on a comparison that produces real, nonzero,
    gated likelihoods -- the inverse itself is still computed over all of
    S, near-null direction included, so this remains a genuine test of the
    ill-conditioned inversion, not a trivially well-conditioned subspace.
    """
    track_covariances = []
    track_states = []
    eigvals0 = eigvecs0 = None
    for k in range(n_tracks):
        S, eigvals, eigvecs = _near_singular_covariance(dim, seed + k, eigval_floor)
        track_covariances.append(S)
        rng = np.random.default_rng(seed * 100 + k)
        track_states.append(rng.normal(size=dim))
        if k == 0:
            eigvals0, eigvecs0 = eigvals, eigvecs

    rng_meas = np.random.default_rng(seed + 999)
    measurements = np.array(
        [
            track_states[0]
            + eigvecs0[:, 1:]
            @ (rng_meas.normal(size=dim - 1) * np.sqrt(eigvals0[1:]) * 0.3)
            for _ in range(n_meas)
        ]
    )

    H = np.eye(dim)
    R = np.zeros((dim, dim))
    gate_threshold = float(chi2.ppf(0.999, df=dim)) * 1.5

    return {
        "track_states": track_states,
        "track_covariances": track_covariances,
        "measurements": measurements,
        "H": H,
        "R": R,
        "detection_prob": 0.9,
        "gate_threshold": gate_threshold,
    }


class TestLikelihoodMatrixIllConditioned:
    """Genuinely ill-conditioned S, unlike the main 20-scenario sweep above
    (`_make_likelihood_matrix_scenario` tops out at cond(S)=6.31). dim in
    {2, 3} specifically targets `mahalanobis_distance`'s closed-form
    Cramer's-rule/adjugate dispatch branch (gating.py), the numerically
    distinct route from this restructure's `np.linalg.inv` most likely to
    disagree at high condition numbers.

    Own tolerance, NOT shared with the well-conditioned sweep above:
    rtol=1e-5, following test_gating_mahalanobis_dispatch.py's
    `TestMahalanobisDispatchNearSingular` precedent (rtol=1e-5 there for
    its own measured 2.79e-7 worst-case relative error at cond(S) up to
    ~1.17e10 -- a ~35x margin). Measured here (max relative error per
    (dim, eigval_floor) config, `compute_likelihood_matrix` vs
    `_reference_likelihood_matrix`): dim=2, cond~1.4e9-5.2e9: 2.38e-9;
    dim=3, cond~2.5e9-1.0e10: 1.50e-8; dim=2, cond~2.5e5-5.2e6: 1.10e-11;
    dim=3, cond~2.5e6-9.8e6: 1.38e-11 -- all comfortably inside rtol=1e-5
    (>=660x margin on the worst case observed here), and all much tighter
    than the C2 precedent's 2.79e-7. That's expected, not a discrepancy:
    `mahalanobis_batch` (the batched path used here) and
    `mahalanobis_distance`'s own 4<=dim<=10 branch both already use
    `np.linalg.inv` -- only dim in {2, 3} exercises genuinely different
    algorithms (closed-form here vs `np.linalg.inv`), and this scenario's
    eigenvalue floor keeps every eigenvalue but one at O(1) rather than
    clipping several down together as the C2 precedent's recipe does,
    which is the likely reason these errors run smaller.
    """

    @pytest.mark.parametrize(
        "dim,seed,eigval_floor,label",
        [
            (2, 3001, 1e-9, "dim2_cond_1e9"),
            (3, 3002, 1e-9, "dim3_cond_1e10"),
            (2, 3003, 1e-6, "dim2_cond_1e6"),
            (3, 3004, 1e-6, "dim3_cond_1e6"),
        ],
    )
    def test_matches_reference_ill_conditioned(self, dim, seed, eigval_floor, label):
        scenario = _make_ill_conditioned_scenario(dim, seed, eigval_floor)

        expected_likelihood, expected_gated = _reference_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )
        actual_likelihood, actual_gated = compute_likelihood_matrix(
            scenario["track_states"],
            scenario["track_covariances"],
            scenario["measurements"],
            scenario["H"],
            scenario["R"],
            scenario["detection_prob"],
            scenario["gate_threshold"],
        )

        # A scenario with zero gated pairs would make the likelihood
        # comparison below vacuous (both sides all-zero) without actually
        # exercising the ill-conditioned inverse's numerics.
        assert expected_gated.any(), f"{label}: scenario produced no gated pairs"
        assert_allclose(actual_gated, expected_gated)
        _assert_likelihoods_close(
            actual_likelihood, expected_likelihood, rtol=1e-5, context=label
        )


class TestLikelihoodMatrixEmptyMeasurementsRegression:
    """The pre-restructure loop's `n_meas = len(measurements)` short-circuit
    tolerated a bare Python list `[]` for `measurements`: n_meas=0 skips
    every per-measurement operation, so a plain list was never actually
    indexed or used in arithmetic. The Task 1 restructure computed
    `innovations = measurements - z_pred` unconditionally on the per-track
    fast path, which raised for a literal `[]` (numpy's binary-op fallback
    converts it to a shape-(0,) array, not (0, m), which fails to
    broadcast against z_pred's shape (m,) for m != 1). Fixed by coercing
    `measurements` to an array and early-returning the old empty shapes at
    the top of `compute_likelihood_matrix`.
    """

    def test_bare_empty_list_measurements(self):
        n_tracks = 3
        m = 2
        track_states = [np.zeros(m) for _ in range(n_tracks)]
        track_covariances = [np.eye(m) for _ in range(n_tracks)]
        H = np.eye(m)
        R = 0.1 * np.eye(m)

        likelihood_matrix, gated = compute_likelihood_matrix(
            track_states, track_covariances, [], H, R, 0.9, None
        )

        assert likelihood_matrix.shape == (n_tracks, 0)
        assert gated.shape == (n_tracks, 0)
        assert likelihood_matrix.dtype == np.float64
        assert gated.dtype == np.bool_


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
