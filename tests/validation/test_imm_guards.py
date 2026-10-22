"""Guard tests for IMM's all-zero weighted-likelihood fallback.

``imm_update`` keeps the prior mode probabilities when every mode's
*weighted* likelihood (``mode_probs * likelihood``, not the raw
per-mode likelihood alone) underflows to zero -- the one case where the
filter has learned nothing from the measurement. Since v2.11 made
``kf_update`` warn and report ``likelihood=0.0`` on a non-PD innovation
covariance (rather than raising), an all-zero weighted-likelihood vector
is now the expected symptom of a numerical failure upstream, and IMM
swallowed it silently either way.

The condition is on the *weighted* sum, so a zero prior probability on
the mode(s) that do have a real likelihood also triggers it -- not just
a non-PD innovation covariance. The warning message says "weighted
likelihood" and names both possible causes accordingly; it does not
claim every raw mode likelihood is zero, only that their prior-weighted
sum is.

``imm_update`` has no ``likelihoods=`` parameter -- likelihoods are
computed internally, one ``kf_update`` call per mode -- so these tests
drive real calls that make every mode's computed likelihood underflow to
exactly 0.0 (all three modes far from a tight-covariance prediction) or
make only some of them underflow (one mode's covariance loose enough to
leave a tiny but nonzero float), rather than injecting a likelihoods
array directly.
"""

import warnings

import numpy as np
import pytest

from pytcl.dynamic_estimation.imm import imm_update


class TestIMMZeroLikelihoodGuard:
    """The all-zero fallback must become audible without changing outcome."""

    def test_all_zero_mode_likelihoods_warn(self):
        x = np.array([0.0, 0.0])
        P_tight = np.eye(2) * 0.01
        R_tight = np.eye(2) * 0.01
        H = np.eye(2)
        z_far = np.array([1000.0, 1000.0])
        prior = np.array([1 / 3, 1 / 3, 1 / 3])

        with pytest.warns(RuntimeWarning, match="weighted likelihood"):
            out = imm_update(
                [x, x, x],
                [P_tight, P_tight, P_tight],
                prior,
                z_far,
                [H, H, H],
                [R_tight, R_tight, R_tight],
            )

        assert np.array_equal(out.mode_likelihoods, np.zeros(3))
        np.testing.assert_allclose(out.mode_probs, prior)

    def test_zero_prior_on_the_only_likely_mode_also_warns(self):
        """The condition is the *weighted* sum: a zero prior on the mode(s)
        with nonzero likelihood zeros the weighted sum even though most
        raw likelihoods are nonzero -- confirming the message may not
        claim "every mode likelihood is zero"."""
        x_near = np.array([0.0, 0.0])
        x_far = np.array([1000.0, 1000.0])
        P = np.eye(2)
        H = np.eye(2)
        z = np.array([0.5, 0.0])
        R = np.eye(2)
        prior = np.array([0.0, 0.0, 1.0])

        with pytest.warns(RuntimeWarning, match="weighted likelihood"):
            out = imm_update(
                [x_near, x_near, x_far],
                [P, P, P],
                prior,
                z,
                [H, H, H],
                [R, R, R],
            )

        assert out.mode_likelihoods[0] > 0.0
        assert out.mode_likelihoods[1] > 0.0
        assert out.mode_likelihoods[2] == 0.0
        np.testing.assert_allclose(out.mode_probs, prior)

    def test_partial_underflow_does_not_warn(self):
        x = np.array([0.0, 0.0])
        P_tight = np.eye(2) * 0.01
        P_loose = np.eye(2) * 1.0
        R_tight = np.eye(2) * 0.01
        R_loose = np.eye(2) * 1.0
        H = np.eye(2)
        z = np.array([10.0, 0.0])
        prior = np.array([1 / 3, 1 / 3, 1 / 3])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = imm_update(
                [x, x, x],
                [P_tight, P_tight, P_loose],
                prior,
                z,
                [H, H, H],
                [R_tight, R_tight, R_loose],
            )

        assert out.mode_likelihoods[0] == 0.0
        assert out.mode_likelihoods[1] == 0.0
        assert out.mode_likelihoods[2] > 0.0
