"""Independent mpmath oracle for generalized_hypergeometric."""

import math
import warnings

import mpmath
import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    generalized_hypergeometric,
    hyp1f1,
    hyp1f1_regularized,
)

CASES = [
    ([], [], -40.0),
    ([], [], -100.0),
    ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -10.0),
    ([1.0], [2.0], -50.0),
    ([1.0], [2.0], 5.0),
]


@pytest.mark.parametrize("p,q,z", CASES)
def test_generalized_hypergeometric_matches_mpmath(p, q, z):
    want = float(mpmath.hyper(p, q, z))
    got = generalized_hypergeometric(p, q, z)
    assert got == pytest.approx(want, rel=1e-10, abs=0)


def test_total_cancellation_warns_and_returns_nan():
    # Largest term ~1e40 against a sum of 0.199: 1/eps is exceeded.
    p, q, z = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -100.0
    assert float(mpmath.hyper(p, q, z)) == pytest.approx(0.199294, rel=1e-5, abs=0)
    with pytest.warns(RuntimeWarning, match="catastrophic cancellation"):
        got = generalized_hypergeometric(p, q, z)
    assert math.isnan(got)


def test_partial_cancellation_warns_and_keeps_value():
    p, q, z = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -50.0
    want = float(mpmath.hyper(p, q, z))
    with pytest.warns(RuntimeWarning, match="half of the available digits"):
        got = generalized_hypergeometric(p, q, z)
    assert got == pytest.approx(want, rel=1e-4, abs=0)


def test_terminating_exact_root_is_zero_not_nan(recwarn):
    # 3F2([-1,1,1],[1,1],z) = 1 - z, exactly 0 at z = 1.
    got = generalized_hypergeometric([-1.0, 1.0, 1.0], [1.0, 1.0], [0.5, 1.0, 2.0])
    assert list(got) == [0.5, 0.0, -1.0]
    assert generalized_hypergeometric([-1.0, 1.0, 1.0], [1.0, 1.0], 1.0) == 0.0
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_terminating_total_cancellation_recovered_exactly():
    # Laguerre L_80(20): summed in floats it is 4.2e12 (largest term 3.7e28);
    # the exact polynomial value is -1986.4165.
    p, q, z = [-80.0, 1.0], [1.0, 1.0], 20.0
    want = float(mpmath.hyper(p, q, z))
    assert generalized_hypergeometric(p, q, z) == pytest.approx(want, rel=1e-12, abs=0)


def test_terminating_nan_z_returns_nan_without_cancellation_warning(recwarn):
    got = generalized_hypergeometric([-1.0, 1.0, 1.0], [1.0, 1.0], math.nan)
    assert math.isnan(got)
    assert not [w for w in recwarn if "cancellation" in str(w.message)]


def test_nonterminating_nan_z_is_nan_without_cancellation_warning(recwarn):
    p, q = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
    got = generalized_hypergeometric(p, q, [math.nan, 1.0])
    assert math.isnan(got[0])
    assert got[1] == pytest.approx(float(mpmath.hyper(p, q, 1.0)), rel=1e-13, abs=0)
    assert not [w for w in recwarn if issubclass(w.category, RuntimeWarning)]


def test_nan_parameter_is_nan_without_cancellation_warning(recwarn):
    got = generalized_hypergeometric([math.nan, 1.0, 1.0], [1.0, 1.0], 0.5)
    assert math.isnan(got)
    assert not [w for w in recwarn if "cancellation" in str(w.message)]


def test_terminating_over_bound_cancellation_returns_nan_with_warning():
    # 5000 terms is past the exact-arithmetic bound (it took ~64 s unbounded).
    with pytest.warns(RuntimeWarning):
        got = generalized_hypergeometric([-5000.0, 1.0, 1.0], [1.0, 1.0], 0.37)
    assert math.isnan(got)


def _sinhc_sqrt(z):
    """0F1(; 3/2; z) = sinh(2 sqrt z) / (2 sqrt z); sin(...)/... for z < 0."""
    if z < 0:
        r = 2.0 * math.sqrt(-z)
        return math.sin(r) / r
    r = 2.0 * math.sqrt(z)
    return np.sinh(r) / r


def _limit(f, z):
    """Read the limit of a closed form off its value at 1e30 in z's direction."""
    with np.errstate(over="ignore"):
        v = float(f(math.copysign(1e30, z)))
    if abs(v) > 1e12:
        return math.copysign(math.inf, v)
    assert abs(v) < 1e-12, f"closed form has no limit here ({v})"
    return 0.0


class TestInfiniteArgument:
    """pFq at z = +-inf against closed forms and DLMF 16.11 asymptotics."""

    CLOSED = [
        (([], []), np.exp),
        (([0.5], [0.5]), np.exp),
        (([-1.0], [2.0]), lambda z: 1 - z / 2),
        (([-2.0], [3.0]), lambda z: 1 - 2 * z / 3 + z**2 / 12),
        (([-1.0], [-3.0]), lambda z: 1 + z / 3),
        (([-1.0, 1.0, 1.0], [1.0, 1.0]), lambda z: 1 - z),
        (([], [1.5]), _sinhc_sqrt),
    ]

    @pytest.mark.parametrize("params, f", CLOSED)
    @pytest.mark.parametrize("z", [math.inf, -math.inf])
    def test_infinite_z_matches_the_closed_form_limit(self, params, f, z):
        assert generalized_hypergeometric(*params, z) == _limit(f, z)

    @pytest.mark.parametrize("a, b", [([], [0.5]), ([], [0.25]), ([], [1.0, 2.0])])
    def test_oscillating_limit_is_nan(self, a, b):
        # 0F1(; 1/2; -x) = cos(2 sqrt x); 0F1(; 1/4; -x) grows as x^(1/8);
        # 0F2 (p <= q - 2) grows oscillating.
        big = [float(mpmath.hyper(a, b, z)) for z in (-1e2, -1e4)]
        assert abs(big[1]) > 0.1 and abs(big[1]) > 0.5 * abs(big[0])
        assert np.isnan(generalized_hypergeometric(a, b, -np.inf))

    @pytest.mark.parametrize(
        "a, b, plus, minus",
        [
            ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], math.inf, 0.0),
            ([1.0, 2.0], [3.0], math.nan, 0.0),
            ([-2.0, 1.0, 1.0], [2.0, 3.0], math.inf, math.inf),
            ([-1.0, 1.0, 1.0], [2.0, 3.0], -math.inf, math.inf),
            ([1.0], [2.0, 3.0], math.inf, 0.0),
            ([2.0], [1.0, 2.0], math.inf, 0.0),
            ([], [1.0, 2.0], math.inf, math.nan),
        ],
    )
    def test_limits_without_a_closed_form(self, a, b, plus, minus):
        if minus == 0.0:
            trend = [
                abs(complex(mpmath.hyper(a, b, z, maxterms=10**6)))
                for z in (-1e2, -1e4)
            ]
            assert trend[1] < 0.1 * trend[0]
        got = [generalized_hypergeometric(a, b, z) for z in (math.inf, -math.inf)]
        np.testing.assert_array_equal(got, [plus, minus])

    def test_zero_numerator_parameter_is_one(self):
        got = [
            generalized_hypergeometric([0.0, 1.0, 1.0], [2.0, 3.0], z)
            for z in (math.inf, -math.inf)
        ]
        assert got == [1.0, 1.0]

    @pytest.mark.parametrize(
        "a, b",
        [
            ([-2.0, 1.0, 1.0], [-1.0, 3.0]),
            ([1.0, 1.0], [-0.5]),
            ([-0.5, 1.0], [2.0, 3.0]),
        ],
    )
    def test_negative_or_pole_parameter_beyond_1f1_is_nan(self, a, b):
        for z in (math.inf, -math.inf):
            assert np.isnan(generalized_hypergeometric(a, b, z))

    @pytest.mark.parametrize(
        "a, b, plus, minus",
        [
            # DLMF 13.7.2: e^z z^(a-b) Gamma(b)/Gamma(a) dominates at +inf;
            # Gamma(b)/Gamma(b-a) (-z)^(-a) at -inf.
            (-0.5, 1.0, -math.inf, math.inf),
            (1.0, -1.5, math.inf, 0.0),
            (0.5, -0.5, -math.inf, 0.0),
            (-2.0000001, 3.0, -math.inf, math.inf),
        ],
    )
    def test_1f1_limits_from_dlmf_13_7_2(self, a, b, plus, minus):
        got = [generalized_hypergeometric([a], [b], z) for z in (math.inf, -math.inf)]
        assert got == [plus, minus]

    def test_1f1_agrees_with_hyp1f1_at_infinity(self):
        for a in (-0.5, 1.0, 0.5, -2.0000001, 2.5, -1.0, 0.0, -3.0):
            for b in (1.0, -1.5, -0.5, 3.0, 0.5, -2.0000001, -2.0):
                for z in (math.inf, -math.inf):
                    np.testing.assert_array_equal(
                        generalized_hypergeometric([a], [b], z), hyp1f1(a, b, z)
                    )
        z = np.array([math.inf, 1e15, -math.inf])
        np.testing.assert_array_equal(
            generalized_hypergeometric([-0.5], [1.0], z),
            [-math.inf, -math.inf, math.inf],
        )

    @pytest.mark.parametrize(
        "a, b", [([1.0, 1.0, 1.0], [2.0]), ([1.0] * 3, [2.0, 3.0])]
    )
    def test_divergent_series_still_raises(self, a, b):
        for z in (math.inf, -math.inf):
            with pytest.raises(ValueError, match="diverges"):
                generalized_hypergeometric(a, b, z)

    def test_array_mixes_limits_with_finite_and_nan_entries(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]
        got = generalized_hypergeometric(a, b, [math.inf, -math.inf, math.nan, 1.0])
        np.testing.assert_array_equal(got[:3], [math.inf, 0.0, math.nan])
        want = float(mpmath.hyper(a, b, 1.0))
        assert got[3] == pytest.approx(want, rel=1e-13, abs=0)


def _mp_double(v):
    """mpmath value rounded to a double, with overflow kept as a signed inf."""
    if abs(v) > mpmath.mpf(np.finfo(np.float64).max):
        return math.copysign(math.inf, float(mpmath.sign(v)))
    return float(v)


def _series_sum(a, b, z, dps):
    """The 1F1 series summed term by term in mpmath, with exact parameters.

    mpmath's own hyp1f1 stops once early terms are negligible, which misses
    the terms past b + k = 0 that grow again when b is a large negative
    number (it returns 53.36 for the first case below, whose sum is
    -5.5e522)."""
    with mpmath.workdps(dps):
        a, b, z = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(z)
        term = total = mpmath.mpf(1)
        tiny = mpmath.mpf(10) ** -dps
        k = 0
        while k < 2 * abs(b) + 2 * abs(z) or abs(term) > abs(total) * tiny:
            term *= (a + k) * z / ((b + k) * (k + 1))
            total += term
            k += 1
        return total


@pytest.mark.timeout(60, method="thread")
class TestLargeFiniteArgument:
    """scipy's hyp1f1 takes time proportional to abs(z) (14 s at 1e13, past
    60 s from 1e14) for non-terminating 1F1 at z > 0, and at z < 0 when
    b - a is a non-positive integer; large abs(z) goes through DLMF 13.7.2
    instead. The class timeout is the termination check; values are
    checked against mpmath."""

    HUNG_IN_SCIPY = [  # killed after 60 s
        (-0.5, 1.0, 1e15),
        (1.0, -1.5, 1e15),
        (0.5, 1.5, 1e15),
        (0.5, 1.5, 1e300),
        (2.5, -1.5, -1e16),
        (2.5, 1.5, -1e15),
    ]
    SLOW_IN_SCIPY = [(1.0, 1.0, 1e13), (0.5, 0.5, -1e13)]  # 14 s and 36 s
    EXPANSION_REGION = [  # scipy returns at once; the first is 0.0 there
        (0.5, 1.5, -1e300),
        (-10.5, 3.0, -1e20),
        (25.5, 40.25, -1e13),
        (3.7, -2.2, -1e100),
    ]

    @pytest.mark.parametrize(
        "a, b, z", HUNG_IN_SCIPY + SLOW_IN_SCIPY + EXPANSION_REGION
    )
    def test_returns_and_matches_mpmath(self, a, b, z, recwarn):
        want = _mp_double(mpmath.hyp1f1(a, b, z))
        for got in (hyp1f1(a, b, z), generalized_hypergeometric([a], [b], z)):
            if math.isfinite(want) and want != 0.0:
                assert got == pytest.approx(want, rel=5e-14, abs=0)
            else:
                assert got == want
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, z, dps",
        [
            (-14.5, -8129.0587257028, 2564.876787104456, 600),
            (-14.5, -8129.0587257028, 3000.0, 1350),
        ],
    )
    def test_overflow_past_a_large_negative_b(self, a, b, z, dps):
        # scipy returns +inf; the series sums to -5.5e522 and -5.4e1264.
        want = _series_sum(a, b, z, dps)
        assert want < 0 and abs(want) > mpmath.mpf(np.finfo(np.float64).max)
        assert hyp1f1(a, b, z) == -math.inf

    @pytest.mark.parametrize(
        "a, b, z",
        [
            # b < 0 beyond 1e9: scipy was off by 1.4e-12 and 2.4e-12.
            (0.5, -3000.25, -1e9),
            (-0.5, -3000.25, -1e9),
            # b - a just past a pole of Gamma; stays with scipy below 1e9.
            (16.5, -20.00941712517169, -9347.412609871704),
            (-27.6795, -5430.66, -2.86e9),
            (44.5, -17.49997441131839, -1e10),
        ],
    )
    def test_negative_b_matches_mpmath(self, a, b, z):
        with mpmath.workdps(100):
            want = float(mpmath.hyp1f1(a, b, z))
        assert hyp1f1(a, b, z) == pytest.approx(want, rel=5e-14, abs=0)

    def test_underflow_without_a_coefficient_is_zero(self):
        # Gamma(1)/Gamma(-9998.5) (1e9)^(-9999.5) is far below the double
        # range; this used to emit numpy's "invalid value" warning.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert hyp1f1(9999.5, 1.0, -1e9) == 0.0

    @pytest.mark.parametrize(
        "a, b, z",
        [(-9999.5, 1.5, 1e6), (9999.5, -9999.5, 1e6), (1e4, -9999.5, 1e6)],
    )
    def test_unproven_scipy_overflow_is_nan_with_warning(self, a, b, z):
        with pytest.warns(RuntimeWarning, match="could not be proven"):
            assert np.isnan(hyp1f1(a, b, z))

    @pytest.mark.parametrize(
        "a, b, z",
        [
            (-0.5, 1.0, 800.0),
            (-2.5, 0.5, 800.0),
            (-10.5, 1e5, 1e6),
            (1e4, -1.5, 100.0),
            (1.0, -1.5, 800.0),
        ],
    )
    def test_overflow_has_the_true_sign(self, a, b, z):
        # scipy returns +inf for every overflow; the first three are -inf.
        want = _mp_double(mpmath.hyp1f1(a, b, z))
        assert math.isinf(want)
        assert hyp1f1(a, b, z) == want

    @pytest.mark.parametrize(
        "a, b", [(0.5, 1.5), (2.5, 1.0), (-0.5, 3.0), (10.0, 0.5), (3.7, -2.2)]
    )
    @pytest.mark.parametrize("edge", [-1e2, -1e9])
    def test_continuous_across_the_switch(self, a, b, edge):
        for z in (np.nextafter(edge, 0.0), edge):
            want = float(mpmath.hyp1f1(a, b, z))
            assert hyp1f1(a, b, z) == pytest.approx(want, rel=5e-14, abs=0)

    def test_array_routes_elementwise(self):
        z = np.array([-1e15, -10.0, 1e15, np.nan, -np.inf, np.inf])
        got = hyp1f1(0.5, 1.5, z)
        want = [_mp_double(mpmath.hyp1f1(0.5, 1.5, x)) for x in z[:3]]
        np.testing.assert_allclose(got[:3], want, rtol=5e-14, atol=0)
        np.testing.assert_array_equal(got[3:], [np.nan, 0.0, np.inf])

    def test_infinite_z_in_hyp1f1_is_the_limit(self):
        # 1F1(-0.5; 1; z) ~ e^z z^(-3/2) / Gamma(-1/2); 1F1(-1; 2; z) = 1 - z/2.
        assert hyp1f1(-0.5, 1.0, np.inf) == -np.inf
        assert hyp1f1(-1.0, 2.0, np.inf) == -np.inf
        assert hyp1f1(-1.0, 2.0, -np.inf) == np.inf

    def test_regularized_returns_at_large_z(self):
        want = float(mpmath.hyp1f1(0.5, 1.5, -1e15) / mpmath.gamma(1.5))
        got = hyp1f1_regularized(0.5, 1.5, -1e15)
        assert got == pytest.approx(want, rel=5e-14, abs=0)

    def test_expansion_out_of_range_is_nan_with_warning(self):
        # Neither series of DLMF 13.7.2 settles for |a| ~ 1e4, b ~ 1e5 at 1e9.
        with pytest.warns(RuntimeWarning, match="asymptotic expansion"):
            assert np.isnan(hyp1f1(-9999.5, 1e5, 1e9))

    @pytest.mark.parametrize(
        "a, b, z, want",
        [
            ([-5.0], [], 1e100, -math.inf),  # (1 - z)^5
            ([-5.0], [], -1e100, math.inf),
            ([-80.0, 1.0], [1.0, 1.0], 1e20, math.inf),  # L_80, leading 1/80!
        ],
    )
    def test_terminating_overflow_is_signed_inf(self, a, b, z, want, recwarn):
        assert generalized_hypergeometric(a, b, z) == want
