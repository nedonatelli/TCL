"""The signal and statistics APIs that used to describe themselves wrongly.

gh-20 grouped eight items under one heading: *the code is defensible, the API
lies about it*. They share a failure mode rather than a subsystem -- in each
case the implementation did something reasonable and the signature, annotation
or docstring claimed something else, so a caller reading the API got a false
picture and no test disagreed.

That is why these are tested together. The assertions here are mostly about
agreement between what is promised and what is delivered, which is a different
question from "is the computation right", and it is the question nothing was
asking.

Covered here:

- ``detection_probability`` no longer takes a ``swerling_case`` that selected
  nothing;
- ``rician_cdf`` replaces the misnamed ``nuttall_q`` (deprecated alias
  removed in v2.8.0);
- ``snr_loss`` computes the derived CA-CFAR loss rather than a heuristic, and
  refuses the three methods it has no expression for;
- ``optimal_filter`` correlates linearly instead of circularly;
- ``matched_filter`` reports a processing gain that accounts for the template's
  shape;
- the ambiguity functions are annotated real, which is what they return.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms.two_dimensional.assignment import auction, hungarian
from pytcl.mathematical_functions.signal_processing.detection import (
    detection_probability,
    snr_loss,
    threshold_factor,
)
from pytcl.mathematical_functions.signal_processing.matched_filter import (
    ambiguity_function,
    cross_ambiguity,
    matched_filter,
    matched_filter_frequency,
    optimal_filter,
)
from pytcl.mathematical_functions.special_functions import (
    marcum_q,
    rician_cdf,
)


class TestDetectionProbabilityDeadParameter:
    """``swerling_case`` accepted 0-4 and every branch was the same code."""

    def test_the_dead_parameter_is_gone(self):
        with pytest.raises(TypeError):
            detection_probability(snr=10.0, pfa=1e-6, n_ref=32, swerling_case=3)

    def test_the_result_is_the_swerling_1_closed_form(self):
        """What the function actually computes, stated as an equation.

        The old docstring said cases 0-4 were selectable and the Notes admitted
        the formula was Swerling 1. Pinning the expression means the two cannot
        drift apart again.
        """
        snr, pfa, n_ref = 10.0, 1e-6, 32
        alpha = threshold_factor(pfa, n_ref, method="ca")
        expected = (1 + alpha / (n_ref * (1 + snr))) ** (-n_ref)

        assert detection_probability(snr, pfa, n_ref) == pytest.approx(
            expected, rel=1e-12
        )

    def test_a_fluctuating_target_detects_worse_than_a_steady_one(self):
        """Why the dead parameter mattered.

        A caller asking for Swerling 0 got the Swerling 1 answer, which is
        materially lower. If the two agreed, the argument doing nothing would
        not have been worth fixing.
        """
        from pytcl.mathematical_functions.special_functions import (
            swerling_detection_probability,
        )

        fluctuating = detection_probability(snr=10.0, pfa=1e-6, n_ref=32)
        steady = float(
            np.atleast_1d(
                swerling_detection_probability(10.0, 1e-6, n_pulses=1, swerling_case=0)
            )[0]
        )
        assert steady > fluctuating


class TestRicianCdfRename:
    """``nuttall_q`` computed the Rician CDF, correctly, under a wrong name."""

    @pytest.mark.parametrize("a", [0.0, 0.5, 2.0, 5.0])
    @pytest.mark.parametrize("b", [0.0, 1.0, 3.0, 7.0])
    def test_it_is_the_complement_of_the_marcum_q(self, a, b):
        assert rician_cdf(a, b) == pytest.approx(1.0 - marcum_q(a, b, m=1), rel=1e-13)

    def test_it_is_a_valid_cdf(self):
        """Bounded in [0, 1] and non-decreasing in the threshold."""
        thresholds = np.linspace(0.0, 10.0, 60)
        values = np.array([float(rician_cdf(2.0, b)) for b in thresholds])

        assert np.all((values >= 0.0) & (values <= 1.0))
        assert np.all(np.diff(values) >= -1e-12)
        assert values[0] == pytest.approx(0.0, abs=1e-12)
        assert values[-1] == pytest.approx(1.0, abs=1e-6)

    def test_the_old_name_is_gone(self):
        """The deprecated alias warned through v2.7.x and was removed in v2.8.0."""
        import pytcl.mathematical_functions.special_functions as sf

        assert not hasattr(sf, "nuttall_q")


class TestSnrLoss:
    """A heuristic replaced by the derived CA-CFAR loss."""

    def test_it_matches_the_closed_form(self):
        """Swerling 1 SNR difference between CFAR and a known-noise detector."""
        n_ref, pfa, pd = 16, 1e-6, 0.5
        snr_ideal = np.log(pfa) / np.log(pd) - 1.0
        snr_cfar = (pfa ** (-1 / n_ref) - 1) / (pd ** (-1 / n_ref) - 1) - 1.0
        expected = 10 * np.log10((1 + snr_cfar) / (1 + snr_ideal))

        assert snr_loss(n_ref, pfa=pfa, pd=pd) == pytest.approx(expected, rel=1e-12)

    def test_it_vanishes_with_unlimited_reference_cells(self):
        """The defining limit: an exact noise estimate costs nothing."""
        assert snr_loss(10**8, pfa=1e-6) == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.parametrize("pfa", [1e-3, 1e-6, 1e-9])
    def test_it_decreases_monotonically_with_reference_cells(self, pfa):
        losses = [snr_loss(n, pfa=pfa) for n in (4, 8, 16, 32, 64, 128)]
        assert all(b < a for a, b in zip(losses, losses[1:])), losses
        assert all(value > 0.0 for value in losses)

    def test_it_responds_to_the_operating_point(self):
        """The old form took neither pfa nor pd, so it could not.

        This is the assertion that fails for any function of ``n_ref`` alone --
        which is what shipped.
        """
        reference = snr_loss(16, pfa=1e-6, pd=0.5)
        assert snr_loss(16, pfa=1e-3, pd=0.5) != pytest.approx(reference, rel=1e-6)
        assert snr_loss(16, pfa=1e-6, pd=0.9) != pytest.approx(reference, rel=1e-6)

    @pytest.mark.parametrize("method", ["go", "so", "os"])
    def test_methods_without_a_derivation_refuse_rather_than_guess(self, method):
        with pytest.raises(NotImplementedError, match="only implemented"):
            snr_loss(32, pfa=1e-6, method=method)

    @pytest.mark.parametrize(
        "pfa,pd", [(0.0, 0.5), (1.0, 0.5), (1e-6, 0.0), (1e-6, 1.0)]
    )
    def test_probabilities_outside_the_open_interval_raise(self, pfa, pd):
        with pytest.raises(ValueError):
            snr_loss(16, pfa=pfa, pd=pd)


class TestOptimalFilterIsLinear:
    """FFT correlation wraps unless the transform is padded."""

    N, M = 256, 16

    def _target_at_start(self):
        template = np.ones(self.M)
        signal = np.zeros(self.N)
        signal[: self.M] = template
        return signal, template

    def test_a_target_at_the_start_produces_nothing_at_the_end(self):
        """The defect, directly.

        Circular correlation wrapped 94% of the peak into the tail -- samples
        whose correct value is exactly zero. A detector thresholding the output
        would have reported a second target there.
        """
        signal, template = self._target_at_start()
        output = optimal_filter(signal, template, np.ones(self.N))

        tail = output[self.N - self.M + 1 :]
        assert np.max(np.abs(tail)) < 1e-6 * np.max(np.abs(output)), (
            f"tail carries {100 * np.max(np.abs(tail)) / np.max(np.abs(output)):.1f}% "
            f"of the peak; the correlation is wrapping around (gh-20)"
        )

    def test_white_noise_reduces_to_plain_linear_correlation(self):
        """With a flat PSD the whitening filter is the identity.

        So the output has to equal ``np.correlate``, which is an independent
        implementation of the thing being claimed.
        """
        signal, template = self._target_at_start()
        output = optimal_filter(signal, template, np.ones(self.N))

        padded = np.concatenate([signal, np.zeros(self.M - 1)])
        reference = np.correlate(padded, template, mode="valid")[: self.N]

        np.testing.assert_allclose(output, reference, atol=1e-6)

    def test_colored_noise_still_localizes_the_target(self):
        """Whitening rings beyond the template, so the tail is not exactly zero.

        The residual is the filter's own impulse response and is physical --
        it converges to about 0.4% however far the transform is padded. What
        matters is that it is small, not that it vanishes.
        """
        signal, template = self._target_at_start()
        psd = 1.0 + 2.0 * np.exp(-5 * np.linspace(0.0, 1.0, self.N))
        output = optimal_filter(signal, template, psd)

        tail_fraction = np.max(np.abs(output[self.N - self.M + 1 :])) / np.max(
            np.abs(output)
        )
        assert tail_fraction < 0.05, f"tail carries {100 * tail_fraction:.1f}% of peak"
        assert int(np.argmax(np.abs(output))) < self.M

    def test_the_output_is_still_the_length_of_the_signal(self):
        signal, template = self._target_at_start()
        assert len(optimal_filter(signal, template, np.ones(self.N))) == self.N

    def test_the_psd_is_resampled_by_frequency_not_by_position(self):
        """Padding changes the bin spacing, so the PSD must be re-evaluated.

        Checked on the resampler directly rather than through the filter,
        because the filter output blends this with everything else. Each value
        must land back on the frequency it came from -- indexing positionally
        would shift the whole PSD, and interpolating in FFT bin order rather
        than by frequency would smear the highest and lowest frequencies
        together across the Nyquist wrap.
        """
        from pytcl.mathematical_functions.signal_processing.matched_filter import (
            _resample_psd,
        )

        source_length, target_length = 64, 271
        frequencies = np.fft.fftfreq(source_length)
        # A PSD that is a known function of frequency, so the resampled values
        # can be checked against that function rather than against each other.
        psd = 1.0 + np.cos(2 * np.pi * frequencies)

        resampled = _resample_psd(psd, target_length)

        assert len(resampled) == target_length
        expected = 1.0 + np.cos(2 * np.pi * np.fft.fftfreq(target_length))
        np.testing.assert_allclose(resampled, expected, atol=0.02)

        # DC is bin 0 in both grids and needs no interpolation at all.
        assert resampled[0] == pytest.approx(psd[0], rel=1e-12)

    def test_an_unchanged_length_is_passed_through_untouched(self):
        from pytcl.mathematical_functions.signal_processing.matched_filter import (
            _resample_psd,
        )

        psd = 1.0 + np.abs(np.fft.fftfreq(128))
        np.testing.assert_array_equal(_resample_psd(psd, 128), psd)


class TestMatchedFilterProcessingGain:
    """``10*log10(len(template))`` is only right for a flat template."""

    def _gain(self, template):
        signal = np.zeros(4 * len(template))
        signal[len(template) : 2 * len(template)] = template
        return matched_filter(signal, template).snr_gain

    def test_a_rectangular_template_still_gives_ten_log_n(self):
        """The case the old expression was right for, kept as a check."""
        n = 64
        assert self._gain(np.ones(n)) == pytest.approx(10 * np.log10(n), rel=1e-12)

    def test_a_tapered_template_gives_less_than_ten_log_n(self):
        """A Hann window has about 24 effective samples out of 64, not 64.

        Reporting the sample count overstated the gain by 4.3 dB, which is the
        difference between a detector meeting its specification and missing it.
        """
        n = 64
        gain = self._gain(np.hanning(n))
        assert gain < 10 * np.log10(n) - 4.0
        assert gain == pytest.approx(
            10 * np.log10(np.sum(np.hanning(n) ** 2) / np.max(np.hanning(n) ** 2)),
            rel=1e-12,
        )

    def test_gain_is_invariant_to_template_amplitude(self):
        """It is a ratio, so scaling the template must not change it."""
        template = np.hanning(32)
        assert self._gain(template) == pytest.approx(self._gain(1000.0 * template))

    def test_the_frequency_domain_path_agrees(self):
        """Two implementations of the same quantity must not disagree."""
        template = np.hanning(64)
        signal = np.zeros(256)
        signal[64:128] = template

        assert matched_filter(signal, template).snr_gain == pytest.approx(
            matched_filter_frequency(signal, template).snr_gain, rel=1e-12
        )


class TestAmbiguityFunctionsAreReal:
    """Annotated ``complexfloating``; always returned magnitudes."""

    @staticmethod
    def _signal():
        return np.sin(2 * np.pi * 0.05 * np.arange(128))

    def test_ambiguity_function_returns_real_values(self):
        _, _, surface = ambiguity_function(
            self._signal(), fs=1000.0, n_delay=16, n_doppler=16
        )
        assert not np.iscomplexobj(surface)
        assert np.all(surface >= 0.0)

    def test_cross_ambiguity_returns_real_values(self):
        signal = self._signal()
        _, _, surface = cross_ambiguity(
            signal, signal, fs=1000.0, n_delay=16, n_doppler=16
        )
        assert not np.iscomplexobj(surface)
        assert np.all(surface >= 0.0)

    def test_the_ambiguity_surface_is_normalized_to_its_peak(self):
        _, _, surface = ambiguity_function(
            self._signal(), fs=1000.0, n_delay=32, n_doppler=32
        )
        assert np.max(surface) == pytest.approx(1.0, rel=1e-12)


class TestAuctionIsEpsilonOptimal:
    """The docstring claimed optimality; the algorithm gives a bounded gap."""

    def test_the_gap_to_the_exact_optimum_is_within_n_epsilon(self):
        """The bound the corrected docstring states.

        Not that the gap is zero -- it usually is not -- but that it never
        exceeds what the algorithm guarantees.
        """
        rng = np.random.default_rng(7)
        n = 6
        epsilon = 1.0 / (n + 1)

        for _ in range(200):
            cost = rng.uniform(0.0, 10.0, (n, n))
            _, _, auction_cost = auction(cost)
            rows, cols = hungarian(cost)[:2]
            exact = cost[rows, cols].sum()

            gap = auction_cost - exact
            assert -1e-9 <= gap <= n * epsilon + 1e-9, (
                f"gap {gap:.4f} exceeds the n*epsilon bound {n * epsilon:.4f}"
            )

    def test_integer_costs_with_a_small_epsilon_are_exactly_optimal(self):
        """The guarantee the docstring now names.

        With integer costs and epsilon < 1/n the bound is under one unit, so
        the epsilon-optimal assignment is the optimal one.
        """
        rng = np.random.default_rng(11)
        n = 5

        for _ in range(50):
            cost = rng.integers(0, 20, (n, n)).astype(float)
            _, _, auction_cost = auction(cost, epsilon=0.5 / n)
            rows, cols = hungarian(cost)[:2]
            assert auction_cost == pytest.approx(cost[rows, cols].sum(), abs=1e-9)


class TestMultivariateGaussianFisherInformation:
    """``mle_gaussian`` returned placeholders for the multivariate case.

    ``fisher = np.eye(n_params) * n`` and ``cov = np.eye(n_params) / n``, which
    do not depend on the data at all -- a code comment admitted as much
    (gh-20). Both are now the exact expressions, checked here against Monte
    Carlo because the covariance one is easy to get subtly wrong.
    """

    DIMENSION = 3
    SIGMA = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, -0.3], [0.1, -0.3, 1.5]])
    MU = np.array([1.0, -2.0, 0.5])

    def _fit(self, n_samples, seed=0):
        from pytcl.static_estimation.maximum_likelihood import mle_gaussian

        rng = np.random.default_rng(seed)
        data = (
            self.MU
            + rng.standard_normal((n_samples, self.DIMENSION))
            @ np.linalg.cholesky(self.SIGMA).T
        )
        return mle_gaussian(data)

    def test_the_fisher_information_depends_on_the_data(self):
        """The assertion the placeholder fails.

        ``np.eye(n_params) * n`` is the same matrix for every dataset of a
        given size, so two different covariances must give two different
        Fisher matrices.
        """
        result = self._fit(4000)
        n_params = len(result.theta)

        assert not np.allclose(result.fisher_info, np.eye(n_params) * 4000)
        assert not np.allclose(result.covariance, np.eye(n_params) / 4000)

    def test_the_mean_block_is_n_times_the_inverse_covariance(self):
        n_samples = 20000
        result = self._fit(n_samples)
        d = self.DIMENSION

        expected = n_samples * np.linalg.inv(self.SIGMA)
        np.testing.assert_allclose(result.fisher_info[:d, :d], expected, rtol=0.05)

    def test_the_covariance_mean_block_matches_monte_carlo(self):
        """``Cov(mu_hat) = Sigma / n``, checked by resampling."""
        from pytcl.static_estimation.maximum_likelihood import mle_gaussian

        n_samples, trials = 500, 3000
        rng = np.random.default_rng(5)
        chol = np.linalg.cholesky(self.SIGMA)
        estimates = np.array(
            [
                mle_gaussian(
                    self.MU + rng.standard_normal((n_samples, self.DIMENSION)) @ chol.T
                ).theta[: self.DIMENSION]
                for _ in range(trials)
            ]
        )

        empirical = np.cov(estimates.T, ddof=1)
        reported = self._fit(n_samples, seed=99).covariance[
            : self.DIMENSION, : self.DIMENSION
        ]
        # Judged against the largest entry rather than element-wise: the
        # near-zero off-diagonals carry the same absolute sampling noise as
        # the diagonal, so a relative tolerance on them measures noise, not
        # agreement.
        scale = np.abs(reported).max()
        assert np.abs(empirical - reported).max() < 0.25 * scale, (
            f"empirical\n{empirical}\nreported\n{reported}"
        )

    def test_the_covariance_of_vec_sigma_is_rank_deficient(self):
        """The subtlety that makes this not simply ``inv(fisher)``.

        ``vec(Sigma)`` holds ``d^2`` entries for a symmetric matrix with only
        ``d(d+1)/2`` free ones, so the sampling covariance of the estimator has
        rank ``d(d+1)/2``. Inverting the Fisher block would give a full-rank
        matrix describing an unconstrained estimator that does not exist here.
        """
        d = self.DIMENSION
        result = self._fit(2000)

        vec_block = result.covariance[d:, d:]
        assert np.linalg.matrix_rank(vec_block) == d * (d + 1) // 2
        assert np.linalg.matrix_rank(result.fisher_info) == len(result.theta)

    def test_the_covariance_block_matches_monte_carlo(self):
        """``Cov(vec(Sigma_hat)) = (I + K)(Sigma kron Sigma) / n``."""
        from pytcl.static_estimation.maximum_likelihood import mle_gaussian

        n_samples, trials = 500, 3000
        rng = np.random.default_rng(6)
        chol = np.linalg.cholesky(self.SIGMA)
        estimates = np.array(
            [
                mle_gaussian(
                    self.MU + rng.standard_normal((n_samples, self.DIMENSION)) @ chol.T
                ).theta[self.DIMENSION :]
                for _ in range(trials)
            ]
        )

        empirical = np.cov(estimates.T, ddof=1)
        reported = self._fit(n_samples, seed=99).covariance[
            self.DIMENSION :, self.DIMENSION :
        ]
        scale = np.abs(reported).max()
        assert np.abs(empirical - reported).max() < 0.25 * scale

    def test_the_univariate_branch_is_unaffected(self):
        """Only the multivariate case held placeholders."""
        from pytcl.static_estimation.maximum_likelihood import mle_gaussian

        rng = np.random.default_rng(3)
        data = 2.0 + 1.5 * rng.standard_normal(5000)
        result = mle_gaussian(data)

        assert result.theta[0] == pytest.approx(2.0, abs=0.1)
        assert result.theta[1] == pytest.approx(1.5**2, rel=0.1)
        assert result.fisher_info[0, 0] == pytest.approx(
            5000 / result.theta[1], rel=1e-9
        )
