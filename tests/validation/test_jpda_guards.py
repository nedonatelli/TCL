"""Guard test for the JPDA determinant-based positive-definiteness check.

``compute_measurement_likelihood``'s guard was ``det(S) <= 0``, which an
even number of negative eigenvalues defeats:
``np.diag([1.0, -1.0, -1.0])`` has ``det(S) = 1 > 0`` despite being
negative-definite in two of its three axes, so it sailed through as if S
were a legitimate covariance. Measured on ``S = diag(-1e-3, -1e-3)``,
innovation ``(1e-2, 1e-2)``, ``Pd=0.9``: the old guard returned likelihood
158.30407311583267 with zero warnings -- silent nonsense, not a genuinely
improbable measurement (this reproduces the brief's cited 158.304; its
cited 0.1447 for ``S = -I`` under the same conditions does not reproduce --
measured here as 0.14325377344380522, matching the closed-form formula
exactly, so the brief's figure for that case appears to be a rounding or
transcription slip, not a target this fix needs to hit).
``compute_likelihood_matrix``'s batch path carried the same
``det_S > 0`` test at the equivalent site.

The brief's batch-path sketch passes ``S`` directly to
``compute_likelihood_matrix(...)``, but the real signature takes
``track_states, track_covariances, measurements, H, R, ...`` and forms
``S = H @ P @ H.T + R`` itself. The tests below use ``H = I`` and
``R = 0`` so that ``S`` equals ``P`` exactly, letting each ``NON_PD``
matrix be passed straight through as a track covariance.

The brief's own batch-path assertion (``not gated.any()``) only holds when
every track's S is bad; it says nothing about a mixed batch where some
tracks are fine. ``test_batch_path_gates_out_only_the_bad_track`` below
builds a two-track batch -- one bad, one well-conditioned -- and checks
that the bad track's row is fully gated out and zeroed while the good
track's row is untouched, which is the actual requirement this guard must
satisfy.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.jpda import (
    compute_likelihood_matrix,
    compute_measurement_likelihood,
)

NON_PD = [
    np.diag([-1e-3, -1e-3]),
    -np.eye(2),
    np.array([[1.0, 2.0], [2.0, 1.0]]),  # indefinite, det = -3
    np.diag([1.0, -1.0, -1.0]),  # det > 0, two negative eigenvalues
]


@pytest.mark.parametrize("S", NON_PD)
def test_non_pd_innovation_covariance_warns_and_zeroes_the_likelihood(S):
    innovation = np.full(S.shape[0], 1e-2)
    with pytest.warns(RuntimeWarning, match="positive definite"):
        got = compute_measurement_likelihood(innovation, S, detection_prob=0.9)
    assert got == 0.0


@pytest.mark.parametrize("S", NON_PD)
def test_batch_path_gates_out_only_the_bad_track(S):
    """Bad track's row is zeroed and ungated; a good track sharing the
    batch is untouched -- the brief's own ``not gated.any()`` sketch would
    pass even if a good track's row were wrongly wiped out too."""
    m = S.shape[0]
    H = np.eye(m)
    R = np.zeros((m, m))
    good_P = np.eye(m) * 0.5
    states = [np.zeros(m), np.zeros(m)]
    track_covariances = [S, good_P]
    measurements = np.full((2, m), 1e-2)

    with pytest.warns(RuntimeWarning, match="positive definite"):
        lik, gated = compute_likelihood_matrix(
            states, track_covariances, measurements, H, R, detection_prob=0.9
        )

    assert lik[0].max() == 0.0
    assert not gated[0].any()

    assert lik[1].min() > 0.0
    assert gated[1].all()


def test_compute_measurement_likelihood_warning_text_names_pd_failure():
    """Pin the exact text: three earlier rounds of this patch shipped a
    warning describing a condition the code did not actually test."""
    S = -np.eye(2)
    innovation = np.full(2, 1e-2)

    with pytest.warns(RuntimeWarning) as record:
        compute_measurement_likelihood(innovation, S, detection_prob=0.9)

    assert str(record[0].message) == (
        "compute_measurement_likelihood: innovation covariance is not "
        "positive definite; likelihood set to 0.0 (numerical failure, "
        "not evidence). Check R and the covariance conditioning."
    )


def test_compute_likelihood_matrix_warning_names_the_offending_track():
    S_bad = -np.eye(2)
    good_P = np.eye(2) * 0.5
    H = np.eye(2)
    R = np.zeros((2, 2))
    states = [np.zeros(2), np.zeros(2)]
    measurements = np.full((2, 2), 1e-2)

    with pytest.warns(RuntimeWarning) as record:
        compute_likelihood_matrix(
            states, [S_bad, good_P], measurements, H, R, detection_prob=0.9
        )

    assert str(record[0].message) == (
        "compute_likelihood_matrix: innovation covariance for track 0 is "
        "not positive definite; that track's likelihoods and gating are "
        "set to 0.0 (numerical failure, not evidence). Check R and the "
        "covariance conditioning."
    )


def test_well_conditioned_batch_path_matches_scalar_path():
    """Regression guard, not a defect test: the audit found the inv-once
    batch path algebraically equivalent to the per-pair scalar path on
    well-conditioned input, and this fix must not move that. This passes
    whether or not the determinant guard is reverted, since every S here
    is well inside the PD region either way."""
    rng = np.random.default_rng(42)
    n_tracks, n_meas, m = 6, 7, 3
    H = np.eye(m)
    R = np.eye(m) * 0.2
    states = [rng.normal(size=m) for _ in range(n_tracks)]
    covs = [np.eye(m) * (0.3 + 0.1 * i) for i in range(n_tracks)]
    measurements = rng.normal(size=(n_meas, m)) * 2

    lik_batch, _ = compute_likelihood_matrix(
        states, covs, measurements, H, R, detection_prob=0.9
    )

    lik_ref = np.zeros((n_tracks, n_meas))
    for i in range(n_tracks):
        S = H @ covs[i] @ H.T + R
        for j in range(n_meas):
            innovation = measurements[j] - H @ states[i]
            lik_ref[i, j] = compute_measurement_likelihood(innovation, S, 0.9)

    assert np.max(np.abs(lik_batch - lik_ref)) < 1e-12


ASYMMETRIC = [
    np.array([[1.0, 0.0], [100.0, 1.0]]),
    np.array([[1.0, 100.0], [0.0, 1.0]]),
]


@pytest.mark.parametrize("S", ASYMMETRIC)
def test_asymmetric_covariance_scalar_and_batch_paths_agree(S):
    """Defect test: cho_factor reads the upper triangle and np.linalg.cholesky
    the lower, so for S = [[1, 0], [100, 1]] the scalar path returned 0.0585
    while the batch path warned and returned 0."""
    innovation = np.ones(2)
    with pytest.warns(RuntimeWarning, match="not symmetric"):
        scalar = compute_measurement_likelihood(innovation, S, 1.0)
    with pytest.warns(RuntimeWarning, match="not symmetric"):
        batch, gated = compute_likelihood_matrix(
            [np.zeros(2)], [S], innovation[None, :], np.eye(2), np.zeros((2, 2))
        )
    assert scalar == 0.0
    assert batch[0, 0] == 0.0 and not gated[0, 0]


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_covariance_scalar_and_batch_paths_agree(bad):
    """Defect test: cho_factor's check_finite raised a bare ValueError from a
    function annotated -> float."""
    S = np.array([[2.0, 0.5], [0.5, bad]])
    innovation = np.array([0.3, -0.2])
    with pytest.warns(RuntimeWarning, match="non-finite"):
        assert compute_measurement_likelihood(innovation, S, 0.9) == 0.0
    with pytest.warns(RuntimeWarning, match="non-finite"):
        batch, _ = compute_likelihood_matrix(
            [np.zeros(2)], [S], innovation[None, :], np.eye(2), np.zeros((2, 2))
        )
    assert batch[0, 0] == 0.0


def test_roundoff_asymmetry_is_still_accepted():
    """S = H P H^T + R is symmetric only to roundoff; that must keep working."""
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    skewed = A.copy()
    skewed[1, 0] += 1e-14
    innovation = np.array([0.3, -0.2])
    expected = compute_measurement_likelihood(innovation, A, 0.9)
    assert expected > 0.0
    assert compute_measurement_likelihood(innovation, skewed, 0.9) == pytest.approx(
        expected, rel=1e-12
    )
