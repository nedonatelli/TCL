"""Property-based tests for Kalman filter covariance invariants.

Target: ``kf_predict`` and ``kf_update`` in
``pytcl/dynamic_estimation/kalman/linear.py``. Both return NamedTuples
(``KalmanPrediction`` / ``KalmanUpdate``) exposing the new covariance as the
``.P`` field -- confirmed by reading the class definitions there, not
assumed from the names.

Three invariants, none of which are optional for a working filter:

1. After ``kf_predict``: ``P`` is symmetric (to tolerance) and PSD
   (``min(eigvalsh(P)) >= -tol``).
2. After ``kf_update``: the same two invariants.
3. After ``kf_update``: ``trace(P_post) <= trace(P_prior) + tol`` -- a
   measurement can only reduce (or leave unchanged) total uncertainty,
   never increase it.

``kf_predict`` symmetrizes its output with ``(P_pred + P_pred.T) / 2``, and
``kf_update`` does the same after computing the Joseph-form update (both
visible in linear.py). IEEE 754 addition is commutative bit for bit
(``a + b == b + a`` always), so ``(M + M.T)[i, j]`` and ``(M + M.T)[j, i]``
are the same expression with the two operands swapped -- identical result --
and dividing by 2 is exact (no rounding) unless the result underflows to a
subnormal, which the generator's magnitude floor rules out. So the
*symmetry* invariant is expected to hold at essentially machine-epsilon
tolerance; it is still checked with a (generous, scale-relative) tolerance
rather than bit-exact equality, because that argument describes the code as
read today, not a guarantee this test should stop checking under.

PSD is *not* explicitly re-clipped by either function. It holds
mathematically: ``P_pred = F @ P @ F.T + Q`` is a congruence transform of a
PSD matrix (``F @ P @ F.T``) plus a PSD matrix (``Q``), always PSD; the
Joseph-form update ``P_upd = (I - KH) @ P @ (I - KH).T + K @ R @ K.T`` has
the same shape -- a congruence transform plus a PSD term -- and is PSD
*regardless of whether K is exactly the optimal gain* (this is the whole
reason Joseph form exists: the naive ``P_upd = (I - KH) @ P`` update does
NOT have this guarantee and is the textbook "Kalman covariance degradation"
failure mode under roundoff). So the only thing that can make an observed
eigenvalue negative here is floating-point roundoff in forming the
products, and a violation large enough to look like a real defect would be
exactly that classic degradation bug, not a test artifact -- triaged per
the campaign's narrowing rule, never papered over with a generator change.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, note
from hypothesis import strategies as st

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

MAX_DIM = 5
MAX_ABS_ENTRY = 10.0
_EPS = np.finfo(np.float64).eps

# Additive diagonal shift used to build every PSD matrix below as
# A @ A.T + _PSD_FLOOR * I. A @ A.T is exactly PSD (min eigenvalue >= 0 in
# exact arithmetic) but routinely *exactly* singular for square A drawn
# from a bounded-entry generator at dim <= 5 (e.g. two identical rows, or a
# zero row -- not a rare event with a small finite float pool). Adding
# _PSD_FLOOR * I shifts every eigenvalue of A @ A.T up by exactly
# _PSD_FLOOR in exact arithmetic (eps*I commutes with any matrix), so every
# generated P/Q/R is strictly positive definite with a known eigenvalue
# floor, regardless of A's rank. That in turn guarantees kf_update's
# S = H @ P @ H.T + R is always invertible -- R alone already has
# eigenvalues >= _PSD_FLOOR, so S does too even when H is degenerate (e.g.
# the zero matrix) -- so cho_factor always succeeds and the LinAlgError
# fallback branch in kf_update is never exercised by these tests.
# 1e-6 is small relative to the matrices' typical scale (entries up to
# MAX_ABS_ENTRY=10, so A @ A.T entries can reach dim * 100) so it doesn't
# mask genuine near-singularity, but is far above float64's ~1e-16
# roundoff floor for these magnitudes, so it is not itself a source of
# numerical noise in the eigenvalue floor it establishes.
_PSD_FLOOR = 1e-6


def _matrix_entries(draw, rows: int, cols: int) -> np.ndarray:
    """Bounded float64 entries, shape (rows, cols).

    The |entry| <= MAX_ABS_ENTRY bound is a *conditioning* bound, not a way
    to dodge failures: it exists so that A @ A.T (below) and products like
    F @ P @ F.T stay within a range where float64 arithmetic is well
    behaved -- entries up to the low thousands, condition numbers up to a
    few orders of magnitude, nothing pushing into overflow or a regime
    where roundoff swamps the signal being tested. Without it, an
    unbounded generator would eventually draw e.g. an A with entries near
    1e150, making A @ A.T overflow to +inf (a fabricated "PSD violation"
    that is a generator artifact, not anything kf_predict/kf_update did
    wrong), or entries spanning enough orders of magnitude that P's
    condition number exceeds float64's ~1e16 dynamic range, at which point
    *no* algorithm computing with it in float64 can be expected to
    preserve PSD to any non-vacuous tolerance. The bound keeps every
    generated problem inside the regime where a violation is attributable
    to kf_predict/kf_update, not to float64 running out of precision to
    give it.
    """
    values = draw(
        st.lists(
            st.floats(
                min_value=-MAX_ABS_ENTRY,
                max_value=MAX_ABS_ENTRY,
                allow_nan=False,
                allow_infinity=False,
                width=64,
            ),
            min_size=rows * cols,
            max_size=rows * cols,
        )
    )
    return np.array(values, dtype=np.float64).reshape(rows, cols)


@st.composite
def psd_matrices(draw, dim: int) -> np.ndarray:
    """A (dim, dim) strictly-PSD float64 matrix, A @ A.T + _PSD_FLOOR * I."""
    a = _matrix_entries(draw, dim, dim)
    return a @ a.T + _PSD_FLOOR * np.eye(dim)


@st.composite
def predict_inputs(draw):
    """(x, P, F, Q) for kf_predict, state dimension in [1, MAX_DIM]."""
    n = draw(st.integers(min_value=1, max_value=MAX_DIM))
    x = _matrix_entries(draw, n, 1).flatten()
    P = draw(psd_matrices(n))
    F = _matrix_entries(draw, n, n)
    Q = draw(psd_matrices(n))
    return x, P, F, Q


@st.composite
def update_inputs(draw):
    """(x, P, z, H, R) for kf_update.

    State dim n and measurement dim m are drawn independently in
    [1, MAX_DIM] -- H is (m, n) and need not be square, so nothing here
    assumes n == m.
    """
    n = draw(st.integers(min_value=1, max_value=MAX_DIM))
    m = draw(st.integers(min_value=1, max_value=MAX_DIM))
    x = _matrix_entries(draw, n, 1).flatten()
    P = draw(psd_matrices(n))
    z = _matrix_entries(draw, m, 1).flatten()
    H = _matrix_entries(draw, m, n)
    R = draw(psd_matrices(m))
    return x, P, z, H, R


def _symmetry_tol(P: np.ndarray) -> float:
    """Scale-relative symmetry tolerance.

    kf_predict/kf_update both symmetrize their output as (M + M.T) / 2,
    which is exact to the bit per the module docstring's IEEE-754
    commutativity argument -- so any observed asymmetry should be at the
    scale of a handful of ULPs of P's own magnitude, not a fixed absolute
    number. P's magnitude genuinely varies by orders across the generator:
    ~1e-6 at dim=1 with near-cancelling terms, up to ~1e7 at dim=5 with
    near-max-magnitude entries (see the F @ P @ F.T norm bound in
    _psd_tol's docstring) -- a fixed atol calibrated for one end would be
    either meaninglessly loose or spuriously tight at the other.
    """
    scale = np.max(np.abs(P)) if P.size else 0.0
    return max(scale, 1.0) * 10 * _EPS


def _psd_tol(P: np.ndarray, eigs: np.ndarray) -> float:
    """Scale-relative PSD tolerance for min(eigs) >= -tol.

    P is mathematically PSD (see module docstring), so any negative
    eigenvalue observed is pure floating-point roundoff from two sources:
    forming P itself (matrix products/sums, each introducing relative
    error on the order of a few eps per operation, compounded across the
    O(dim) operations in F @ P @ F.T + Q or the Joseph-form update), and
    computing its eigenvalues (np.linalg.eigvalsh -> LAPACK's symmetric
    eigensolver, which has a standard backward-error guarantee that the
    computed eigenvalues are the exact eigenvalues of P + E for a
    perturbation ||E||_2 <= c * dim * eps * ||P||_2, with c a modest,
    single-digit, library-dependent constant). Both sources scale with
    ||P||_2 (well approximated by the largest-magnitude eigenvalue already
    being computed by the caller), not a fixed absolute epsilon: entries
    bounded by MAX_ABS_ENTRY=10 at dim<=5 give ||F||_2 <= ||F||_F <=
    sqrt(5*5)*10 = 50 and ||P_in||_2 <= ||A||_F**2 = (sqrt(25)*10)**2 =
    2500, so ||F @ P @ F.T||_2 can reach 50**2 * 2500 = 6.25e6 -- an
    absolute atol calibrated for a dim=1 case near _PSD_FLOOR would be
    meaningless (either too tight there or 12+ orders of magnitude too
    loose here). The constant (50 * dim) is a generous multiple of the
    single-digit `c` above, covering both roundoff sources without being
    so loose it would hide a genuine degradation-style defect -- which,
    per the classic failure mode, tends to produce violations orders of
    magnitude past any reasonable roundoff floor, not a narrow miss.
    """
    scale = np.max(np.abs(eigs))
    return max(scale, 1.0) * 50 * P.shape[0] * _EPS


class TestPredictCovarianceInvariants:
    """P_pred = F @ P @ F.T + Q (then symmetrized) is symmetric and PSD."""

    @given(predict_inputs())
    def test_symmetric(self, inputs):
        x, P, F, Q = inputs
        pred = kf_predict(x, P, F, Q)
        note(f"dim={P.shape[0]} P_pred={pred.P}")
        tol = _symmetry_tol(pred.P)
        np.testing.assert_allclose(pred.P, pred.P.T, atol=tol, rtol=0)

    @given(predict_inputs())
    def test_psd(self, inputs):
        x, P, F, Q = inputs
        pred = kf_predict(x, P, F, Q)
        eigs = np.linalg.eigvalsh(pred.P)
        tol = _psd_tol(pred.P, eigs)
        note(f"dim={P.shape[0]} eigs={eigs} tol={tol}")
        assert eigs.min() >= -tol


class TestUpdateCovarianceInvariants:
    """P_upd (Joseph form, then symmetrized) is symmetric, PSD, and never
    increases total uncertainty relative to the prior."""

    @given(update_inputs())
    def test_symmetric(self, inputs):
        x, P, z, H, R = inputs
        upd = kf_update(x, P, z, H, R)
        note(f"n={P.shape[0]} m={H.shape[0]} P_upd={upd.P}")
        tol = _symmetry_tol(upd.P)
        np.testing.assert_allclose(upd.P, upd.P.T, atol=tol, rtol=0)

    @given(update_inputs())
    def test_psd(self, inputs):
        x, P, z, H, R = inputs
        upd = kf_update(x, P, z, H, R)
        eigs = np.linalg.eigvalsh(upd.P)
        tol = _psd_tol(upd.P, eigs)
        note(f"n={P.shape[0]} m={H.shape[0]} eigs={eigs} tol={tol}")
        assert eigs.min() >= -tol

    @given(update_inputs())
    def test_trace_does_not_increase(self, inputs):
        x, P, z, H, R = inputs
        upd = kf_update(x, P, z, H, R)
        trace_prior = np.trace(P)
        trace_post = np.trace(upd.P)
        note(
            f"n={P.shape[0]} m={H.shape[0]} "
            f"trace_prior={trace_prior} trace_post={trace_post}"
        )
        # trace is a sum of <= MAX_DIM diagonal entries, each carrying
        # roundoff from the products that built P_upd; scale the tolerance
        # off the larger of the two traces (both nonnegative, since both
        # are PSD) rather than a fixed number, for the same reason
        # _psd_tol is scale-relative -- trace magnitude varies by orders
        # across the dim and entry-magnitude draws (see _psd_tol's
        # docstring for the concrete ||P||_2 bound this mirrors).
        scale = max(abs(trace_prior), abs(trace_post), 1.0)
        tol = scale * 50 * P.shape[0] * _EPS
        assert trace_post <= trace_prior + tol
