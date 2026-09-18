"""Guard test for the information filter's singular-F prediction defect.

``information_filter``'s singular-Y branch (unknown/partially unknown
initial state) predicts via ``M = F^-T Y F^-1``, which requires F
invertible. Before this fix, a singular F raised ``LinAlgError`` internally
and the exception was swallowed with ``pass``, leaving ``y``/``Y``
unchanged -- the filter silently skipped prediction and reported the prior
as if it had propagated, with zero warning.

Reproduced exactly as measured: with ``F = [[1, 1], [0, 0]]``, an unknown
initial state (``Y0 = 0``), and three position-only measurements, ``Y``
went ``diag(1, 0) -> diag(2, 0) -> diag(3, 0)`` -- indistinguishable from
*skipping prediction entirely*, not from running with ``F = I``: the
same scenario with ``F = I`` (and the same ``Q``) actually damps to
``1.0, 1.909, 2.603`` (measured), since the Q update still runs there.

The brief's own Step 1 sketch calls ``information_filter_predict`` directly
with a *non-singular* ``Y = eye(2)``; that call takes a different branch
(``pytcl.dynamic_estimation.kalman.linear.information_filter_predict``,
used only when Y is already full rank) which never inverts F at all and,
for that combination of F/Y/Q, returns a legitimate, non-garbage answer
today (measured: no exception, ``Y_pred = diag(1/3, 1)``) -- there is
nothing to fix on that path, and forcing it to raise would reject a
scenario that currently produces a correct result. The actual defect only
manifests through ``information_filter``'s singular-Y branch, so the tests
below drive it through that function instead, using the brief's own F and
measurement scenario.
"""

import numpy as np
import pytest

from pytcl.dynamic_estimation.information_filter import information_filter


class TestInformationFilterSingularF:
    """information_filter must not silently skip prediction on a singular F."""

    def test_singular_f_with_singular_y_raises(self):
        """F=[[1,1],[0,0]] starting from an unknown state must raise, not
        silently return the prior as if it had been predicted."""
        n = 2
        y0 = np.zeros(n)
        Y0 = np.zeros((n, n))
        F = np.array([[1.0, 1.0], [0.0, 0.0]])
        Q = np.eye(n) * 0.1
        H = np.array([[1.0, 0.0]])
        R = np.array([[1.0]])
        measurements = [np.array([1.0]), np.array([2.0]), np.array([3.0])]

        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            information_filter(y0, Y0, measurements, F, Q, H, R)

    def test_regular_f_with_singular_y_still_propagates_information(self):
        """The defect's fix must not disturb the already-correct path: a
        regular F must still let velocity information appear even when Y0
        starts singular (unknown state)."""
        y0 = np.zeros(2)
        Y0 = np.diag([1.0, 0.0])
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        Q = np.zeros((2, 2))
        H = np.array([[1.0, 0.0]])
        R = np.array([[1.0]])

        result = information_filter(y0, Y0, [None], F, Q, H, R)

        assert result.Y_filt[0][1, 1] > 0.0, "velocity information was never created"


class TestSRIFPredictSingularCovariance:
    """srif_predict's SVD fallback must not return nan/inf as an ordinary
    result when the predicted covariance is genuinely singular.

    Before this fix, a singular ``P_pred`` reaching the SVD fallback
    (Cholesky already failed) divided by a zero singular value with only
    numpy's own generic ``RuntimeWarning: divide by zero`` as a signal,
    and returned ``r_pred``/``R_pred`` containing ``nan``/``inf`` as an
    ordinary result. ``np.linalg.qr`` tolerates ``nan``/``inf`` without
    raising, so ``srif_update`` ran to completion on that garbage;
    ``srif_filter`` then failed several lines later, in its own
    state-form conversion (``np.linalg.matrix_rank(Y)``, an SVD -- not a
    QR call), with the misleading ``LinAlgError: SVD did not converge``
    -- misleading because the SVD in ``srif_predict`` itself converged
    fine; it was the *result* that was degenerate.
    """

    def _singular_case(self):
        n = 2
        P0 = np.eye(n)
        R0 = np.linalg.cholesky(np.linalg.inv(P0)).T
        x0 = np.array([1.0, 0.5])
        r0 = R0 @ x0
        F = np.array([[1.0, 1.0], [0.0, 0.0]])
        Q = np.zeros((2, 2))
        return r0, R0, F, Q

    def test_singular_predicted_covariance_raises(self):
        from pytcl.dynamic_estimation.information_filter import srif_predict

        r0, R0, F, Q = self._singular_case()

        with pytest.raises(np.linalg.LinAlgError, match="singular"):
            srif_predict(r0, R0, F, Q)

    def test_srif_filter_raises_at_the_predict_step_not_later(self):
        """Confirms the fix moves the failure to its true cause instead of
        letting srif_filter fail downstream with a misleading QR error."""
        from pytcl.dynamic_estimation.information_filter import srif_filter

        r0, R0, F, Q = self._singular_case()
        H = np.array([[1.0, 0.0]])
        R_meas = np.array([[1.0]])

        with pytest.raises(np.linalg.LinAlgError, match="srif_predict"):
            srif_filter(r0, R0, [np.array([1.0])], F, Q, H, R_meas)
