"""Reference coverage of the robust-estimation module.

Oracles: the textbook closed forms of the Huber, Tukey and Cauchy functions
evaluated at their breakpoints, ``statsmodels.robust.RLM`` for the IRLS
M-estimators, a noiseless line for RANSAC, and the Fischler-Bolles trial-count
formula.

The statsmodels constants below were produced with statsmodels 0.15.0::

    rng = np.random.default_rng(20261028)
    t = np.linspace(0, 10, 40); A = np.column_stack([np.ones(40), t])
    b = 1.5 + 0.8 * t + rng.normal(0, 0.5, 40); b[[5, 17, 31]] += [12, -9, 15]
    sm.RLM(b, A, M=sm.robust.norms.HuberT(1.345)).fit(maxiter=500, tol=1e-14)
    sm.RLM(b, A, M=sm.robust.norms.TukeyBiweight(4.685)).fit(maxiter=500, tol=1e-14)

``RLM``'s built-in ``"mad"`` scale is centred at zero
(``median(|r|) / 0.6745``); pytcl's ``mad`` is centred at the residual median,
so the zero-centred pair is compared through a matching ``scale_func`` and the
median-centred pair through ``statsmodels.robust.scale.mad`` passed to ``RLM``.
"""

import numpy as np
import pytest

from pytcl.static_estimation import robust as rb

GAUSSIAN_3_4 = 0.6744897501960817


def _contaminated_line():
    rng = np.random.default_rng(20261028)
    t = np.linspace(0, 10, 40)
    A = np.column_stack([np.ones(40), t])
    b = 1.5 + 0.8 * t + rng.normal(0, 0.5, 40)
    b[[5, 17, 31]] += [12, -9, 15]
    return A, b


# statsmodels 0.15.0 RLM, scale_est="mad" (zero-centred), tol=1e-14.
SM_HUBER_ZERO_CENTRED = (1.5084033101030552, 0.7992570424286283, 0.47705486291221244)
SM_TUKEY_ZERO_CENTRED = (1.4897591677311568, 0.8000210382408394, 0.47306990161578344)
# statsmodels 0.15.0 RLM, scale_est=lambda m, r: scale.mad(r, center=np.median).
SM_HUBER_MEDIAN_CENTRED = (1.5067767232798466, 0.799613179333079)
SM_TUKEY_MEDIAN_CENTRED = (1.4900551579473964, 0.8000447762482659)


def _zero_centred_mad(e):
    return float(np.median(np.abs(e)) / GAUSSIAN_3_4)


class TestInfluenceFunctionsAtBreakpoints:
    C = 1.345

    def test_huber_weight_is_one_up_to_and_including_c(self):
        c = self.C
        w = rb.huber_weight([-c, c, 0.0, np.nextafter(c, 0)], c)
        assert w.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_huber_weight_beyond_c_is_c_over_abs_r(self):
        c = self.C
        r = np.array([np.nextafter(c, 10), -2.0, 2.0, 50.0, -50.0])
        assert rb.huber_weight(r, c) == pytest.approx(c / np.abs(r), rel=1e-14, abs=0)

    def test_huber_rho_branches_meet_at_c(self):
        c = self.C
        at = rb.huber_rho([c, -c], c)
        assert at == pytest.approx([c**2 / 2, c**2 / 2], rel=1e-14, abs=0)
        just_past = rb.huber_rho([np.nextafter(c, 10)], c)
        assert just_past == pytest.approx([c**2 / 2], rel=1e-12, abs=0)
        r = np.array([3.0, -3.0])
        assert rb.huber_rho(r, c) == pytest.approx(
            c * np.abs(r) - c**2 / 2, rel=1e-14, abs=0
        )

    def test_huber_weight_is_psi_over_r(self):
        # w(r) = psi(r)/r with psi = rho'; central difference of rho.
        c = self.C
        r = np.array([0.4, 1.0, 2.5, 7.0])
        h = 1e-6
        psi = (rb.huber_rho(r + h, c) - rb.huber_rho(r - h, c)) / (2 * h)
        assert rb.huber_weight(r, c) == pytest.approx(psi / r, rel=1e-6, abs=0)

    def test_tukey_weight_vanishes_at_and_beyond_c(self):
        c = 4.685
        w = rb.tukey_weight([c, -c, 10.0, -100.0], c)
        assert w.tolist() == [0.0, 0.0, 0.0, 0.0]

    def test_tukey_weight_inside_is_biweight(self):
        c = 4.685
        r = np.array([0.0, 1.0, -2.5, np.nextafter(c, 0)])
        assert rb.tukey_weight(r, c) == pytest.approx(
            (1 - (r / c) ** 2) ** 2, rel=1e-14, abs=0
        )
        assert rb.tukey_weight([0.0], c)[0] == 1.0

    def test_tukey_rho_saturates_at_c_squared_over_6(self):
        c = 4.685
        sat = c**2 / 6
        out = rb.tukey_rho([c, -c, 9.0, -1e6], c)
        assert out == pytest.approx([sat] * 4, rel=1e-14, abs=0)
        assert rb.tukey_rho([0.0], c)[0] == 0.0
        r = np.array([1.0, -3.0])
        assert rb.tukey_rho(r, c) == pytest.approx(
            sat * (1 - (1 - (r / c) ** 2) ** 3), rel=1e-14, abs=0
        )

    def test_tukey_weight_is_psi_over_r(self):
        c = 4.685
        r = np.array([0.5, 2.0, 4.0])
        h = 1e-6
        psi = (rb.tukey_rho(r + h, c) - rb.tukey_rho(r - h, c)) / (2 * h)
        assert rb.tukey_weight(r, c) == pytest.approx(psi / r, rel=1e-6, abs=0)

    def test_cauchy_weight_closed_form(self):
        c = 2.385
        r = np.array([0.0, c, -c, 10.0])
        assert rb.cauchy_weight(r, c) == pytest.approx(
            [1.0, 0.5, 0.5, 1 / (1 + (10.0 / c) ** 2)], rel=1e-14, abs=0
        )

    def test_mad_matches_definition_and_ignores_outlier(self):
        r = np.array([1.0, 1.1, 0.9, 1.0, 100.0])
        med = np.median(r)
        assert rb.mad(r) == pytest.approx(
            1.4826 * np.median(np.abs(r - med)), rel=1e-14, abs=0
        )
        assert rb.mad(r, c=1.0) == pytest.approx(0.1, rel=1e-12, abs=0)


class TestIrlsAgainstStatsmodels:
    def test_huber_zero_centred_scale_matches_vendored_rlm(self):
        A, b = _contaminated_line()
        res = rb.irls(
            A,
            b,
            weight_func=lambda z: rb.huber_weight(z, 1.345),
            scale_func=_zero_centred_mad,
            max_iter=500,
            tol=1e-14,
        )
        assert res.converged
        assert res.x == pytest.approx(SM_HUBER_ZERO_CENTRED[:2], rel=1e-9, abs=0)
        assert res.scale == pytest.approx(SM_HUBER_ZERO_CENTRED[2], rel=1e-9, abs=0)

    def test_tukey_zero_centred_scale_matches_vendored_rlm(self):
        A, b = _contaminated_line()
        res = rb.irls(
            A,
            b,
            weight_func=lambda z: rb.tukey_weight(z, 4.685),
            scale_func=_zero_centred_mad,
            max_iter=500,
            tol=1e-14,
        )
        assert res.x == pytest.approx(SM_TUKEY_ZERO_CENTRED[:2], rel=1e-9, abs=0)
        assert res.scale == pytest.approx(SM_TUKEY_ZERO_CENTRED[2], rel=1e-9, abs=0)

    def test_huber_regression_default_scale_matches_vendored_rlm(self):
        # pytcl's mad uses 1.4826, statsmodels 1/Phi^-1(0.75) = 1.482602...:
        # a 1.5e-6 relative scale difference, hence rel 1e-6 here and the
        # exact-constant variants below at 1e-9.
        A, b = _contaminated_line()
        res = rb.huber_regression(A, b, max_iter=500, tol=1e-14)
        assert res.x == pytest.approx(SM_HUBER_MEDIAN_CENTRED, rel=1e-6, abs=0)

    def test_tukey_regression_default_scale_matches_vendored_rlm(self):
        A, b = _contaminated_line()
        res = rb.tukey_regression(A, b, max_iter=500, tol=1e-14)
        assert res.x == pytest.approx(SM_TUKEY_MEDIAN_CENTRED, rel=1e-6, abs=0)

    @pytest.mark.parametrize(
        "name, weight, vendored",
        [
            ("huber", lambda z: rb.huber_weight(z, 1.345), SM_HUBER_MEDIAN_CENTRED),
            ("tukey", lambda z: rb.tukey_weight(z, 4.685), SM_TUKEY_MEDIAN_CENTRED),
        ],
    )
    def test_median_centred_with_exact_constant_matches_rlm(
        self, name, weight, vendored
    ):
        from scipy.stats import norm

        A, b = _contaminated_line()
        res = rb.irls(
            A,
            b,
            weight_func=weight,
            scale_func=lambda e: float(
                np.median(np.abs(e - np.median(e))) / norm.ppf(0.75)
            ),
            max_iter=500,
            tol=1e-14,
        )
        assert res.x == pytest.approx(vendored, rel=1e-9, abs=0)

    def test_live_statsmodels_agreement(self):
        sm = pytest.importorskip("statsmodels.api")
        from statsmodels.robust import scale as sm_scale

        A, b = _contaminated_line()
        for norm, ours in (
            (sm.robust.norms.HuberT(1.345), rb.huber_regression),
            (sm.robust.norms.TukeyBiweight(4.685), rb.tukey_regression),
        ):
            theirs = sm.RLM(b, A, M=norm).fit(
                maxiter=500,
                tol=1e-14,
                scale_est=lambda model, e: sm_scale.mad(e, center=np.median),
            )
            assert ours(A, b, max_iter=500, tol=1e-14).x == pytest.approx(
                theirs.params, rel=1e-6, abs=0
            )

    def test_outlier_barely_moves_the_robust_fit_but_ruins_ols(self):
        A, b = _contaminated_line()
        ols = np.linalg.lstsq(A, b, rcond=None)[0]
        for fit in (rb.huber_regression(A, b), rb.tukey_regression(A, b)):
            assert np.max(np.abs(fit.x - [1.5, 0.8])) < 0.1
        assert np.max(np.abs(ols - [1.5, 0.8])) > 0.3

    def test_exact_fit_has_unit_weights_and_exact_solution(self):
        # Zero MAD must not divide by zero: the scale falls back to 1.
        t = np.arange(10.0)
        A = np.column_stack([np.ones(10), t])
        b = 2.0 + 3.0 * t
        res = rb.irls(A, b)
        assert res.x == pytest.approx([2.0, 3.0], rel=1e-12, abs=0)
        assert res.weights.tolist() == [1.0] * 10

    def test_max_iter_one_reports_non_convergence(self):
        A, b = _contaminated_line()
        res = rb.irls(A, b, max_iter=1, tol=1e-300)
        assert (res.n_iter, res.converged) == (1, False)

    def test_singular_normal_equations_fall_back_to_minimum_norm(self):
        # Duplicate columns make A'WA exactly singular; the minimum-norm
        # solution splits the one-column answer equally between them.
        rng = np.random.default_rng(5)
        b = 3.0 + rng.normal(0, 0.3, 25)
        b[3] += 40.0
        one = rb.irls(np.ones((25, 1)), b)
        two = rb.irls(np.ones((25, 2)), b)
        assert two.x == pytest.approx([one.x[0] / 2] * 2, rel=1e-8, abs=0)

    def test_max_iter_validation(self):
        with pytest.raises(ValueError, match="max_iter must be at least one, got 0"):
            rb.irls(np.eye(2), np.ones(2), max_iter=0)


def _line_with_outliers(n=100, frac=0.3, seed=11):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    A = np.column_stack([np.ones(n), t])
    b = 2.0 + 3.0 * t
    out = rng.choice(n, size=int(frac * n), replace=False)
    b[out] += rng.uniform(200, 900, out.size) * rng.choice([-1, 1], out.size)
    return A, b, out


class TestRansac:
    def test_recovers_exact_inlier_line_with_30_percent_gross_outliers(self):
        A, b, out = _line_with_outliers()
        res = rb.ransac(A, b, residual_threshold=1e-6, random_state=3)
        assert res.x == pytest.approx([2.0, 3.0], rel=1e-10, abs=0)
        assert res.n_inliers == 70
        assert np.flatnonzero(~res.inliers).tolist() == sorted(out.tolist())

    def test_fixed_seed_is_reproducible(self):
        A, b, _ = _line_with_outliers()
        a1 = rb.ransac(A, b, residual_threshold=1e-6, random_state=9, max_trials=7)
        a2 = rb.ransac(A, b, residual_threshold=1e-6, random_state=9, max_trials=7)
        assert a1.x.tolist() == a2.x.tolist()
        assert a1.inliers.tolist() == a2.inliers.tolist()

    def test_stop_n_inliers_ends_the_search_early(self):
        A, b, _ = _line_with_outliers()
        res = rb.ransac(
            A, b, residual_threshold=1e-6, stop_n_inliers=70, random_state=3
        )
        control = rb.ransac(A, b, residual_threshold=1e-6, random_state=3)
        assert control.n_iter == 100
        assert res.n_inliers == 70 and res.n_iter < control.n_iter

    def test_stop_score_ends_the_search_early(self):
        A, b, _ = _line_with_outliers()
        res = rb.ransac(A, b, residual_threshold=1e-6, stop_score=70.0, random_state=3)
        control = rb.ransac(A, b, residual_threshold=1e-6, random_state=3)
        assert control.n_iter == 100
        assert res.best_score == 70.0 and res.n_iter < control.n_iter

    def test_default_threshold_on_noiseless_line_falls_back_to_one(self):
        # OLS residuals ~1e-15 give a vanishing MAD threshold, replaced by 1.0.
        t = np.arange(30.0)
        A = np.column_stack([np.ones(30), t])
        res = rb.ransac(A, 1.0 + 0.5 * t, random_state=1)
        assert res.n_inliers == 30 and res.inliers.all()
        assert res.x == pytest.approx([1.0, 0.5], rel=1e-10, abs=0)

    # Residual magnitudes in units of median|e| = 1 (so MAD = 1.4826): each is
    # a pair placed symmetrically (t and m-1-t), signed so that e is orthogonal
    # to [1, t]. The OLS fit is then exactly the line and its residuals are e.
    # Thresholds at 2.5 MAD = 3.7065: 1.5 -> 2.22, 2.0 -> 2.97, 3.0 -> 4.45.
    PAIRS = [0.1, -0.1, -0.3, -0.4, -0.5, -0.6, 0.7, 0.8, 0.9, -1.0]
    PAIRS += [1.0, 1.5, 2.0, -2.6, 3.0, -3.5, -4.0, 4.0, 5.0, -6.0]

    def test_default_threshold_is_2p5_mad_of_ols_residuals(self):
        pairs = np.array(self.PAIRS)
        e = np.empty(40)
        order = np.random.default_rng(0).permutation(20)
        for slot, k in enumerate(order):
            e[slot] = e[39 - slot] = pairs[k]
        t = np.arange(40.0)
        A = np.column_stack([np.ones(40), t])
        b = 1.0 + 0.5 * t + e
        assert abs(e.sum()) < 1e-12 and abs(e @ t) < 1e-10
        assert np.median(e) == 0.0 and np.median(np.abs(e)) == 1.0
        # min_samples = all rows: every trial is the OLS fit, so the inlier set
        # is exactly {|e| < threshold}. 2.5 MAD keeps magnitudes <= 3.5 (32 of
        # 40); 1.5, 2.0 and 3.0 MAD would keep 26, 28 and 36.
        res = rb.ransac(A, b, min_samples=40, random_state=0, max_trials=1)
        assert res.n_inliers == 32
        assert np.all(np.abs(e[res.inliers]) <= 3.5)
        assert np.all(np.abs(e[~res.inliers]) >= 4.0)

    def test_zero_threshold_admits_no_inliers_and_skips_the_refit(self):
        A, b, _ = _line_with_outliers()
        res = rb.ransac(A, b, residual_threshold=0.0, random_state=3, max_trials=5)
        assert res.n_inliers == 0 and res.best_score == 0.0
        # The model is a two-point sample fit (a refit on zero rows would give
        # the zero vector): finite, non-zero, passing through >= 2 data points.
        assert np.all(np.isfinite(res.x)) and np.any(res.x != 0.0)
        assert np.count_nonzero(np.abs(b - A @ res.x) < 1e-8) >= 2

    @pytest.mark.parametrize("bad", [0, -3])
    def test_max_trials_validation(self, bad):
        with pytest.raises(ValueError, match="max_trials must be at least one"):
            rb.ransac(np.eye(2), np.ones(2), max_trials=bad)

    def test_min_samples_exceeding_rows(self):
        with pytest.raises(ValueError, match=r"min_samples \(5\) > n_samples \(3\)"):
            rb.ransac(np.ones((3, 2)), np.ones(3), min_samples=5)

    @staticmethod
    def _failing_lstsq(monkeypatch, fail_if):
        real = np.linalg.lstsq

        def fake(a, b, rcond=None):
            if fail_if(np.asarray(a)):
                raise np.linalg.LinAlgError("forced failure")
            return real(a, b, rcond=rcond)

        monkeypatch.setattr(np.linalg, "lstsq", fake)

    def test_every_sample_fit_failing_raises(self, monkeypatch):
        self._failing_lstsq(monkeypatch, lambda a: True)
        A, b, _ = _line_with_outliers(n=20, frac=0.0)
        with pytest.raises(np.linalg.LinAlgError, match="every RANSAC sample fit"):
            rb.ransac(A, b, residual_threshold=0.1, random_state=0)

    def test_failed_sample_fits_are_skipped(self, monkeypatch):
        # Row 0 is poisoned: any sample containing it fails to fit and is
        # skipped, the other trials still find the exact line.
        A, b, _ = _line_with_outliers(n=40, frac=0.0)
        A[0, 0] = np.nan
        self._failing_lstsq(monkeypatch, lambda a: np.isnan(a).any())
        res = rb.ransac(A, b, residual_threshold=1e-6, random_state=4)
        assert not res.inliers[0] and res.n_inliers == 39
        assert res.x == pytest.approx([2.0, 3.0], rel=1e-9, abs=0)


class TestRansacTrialCount:
    @pytest.mark.parametrize(
        "n, k, s, p", [(100, 30, 2, 0.99), (100, 50, 4, 0.999), (500, 100, 3, 0.95)]
    )
    def test_fischler_bolles_formula(self, n, k, s, p):
        want = int(np.ceil(np.log(1 - p) / np.log(1 - (1 - k / n) ** s)))
        assert rb.ransac_n_trials(n, k, s, p) == want

    def test_documented_example(self):
        assert rb.ransac_n_trials(100, 30, 2) == 7

    def test_all_points_outliers_needs_one_trial(self):
        assert rb.ransac_n_trials(10, 10, 2) == 1
        assert rb.ransac_n_trials(10, 25, 2) == 1

    def test_no_outliers_needs_one_trial(self):
        assert rb.ransac_n_trials(100, 0, 3) == 1

    def test_hopeless_inlier_rate_is_capped(self):
        assert rb.ransac_n_trials(100, 99, 12) == 10000
