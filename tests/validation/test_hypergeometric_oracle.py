"""Independent mpmath oracle for generalized_hypergeometric."""

import math
import subprocess
import sys
import warnings
from fractions import Fraction

import mpmath
import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    _CERTIFIED_SUM_MAX_TERMS,
    _certified_overflow,
    _pole_distance,
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
    instead. Values are checked against mpmath.

    Termination: the thread-method timeout ends the process and fails
    loudly on a Python-level loop, but it cannot interrupt scipy's C loop,
    which holds the GIL (neither can the signal method). The cases that
    hung in scipy therefore also run in a subprocess with a 60 s timeout
    (``test_hung_cases_return_in_a_subprocess``)."""

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

    @pytest.mark.parametrize("a, b, z", HUNG_IN_SCIPY)
    def test_hung_cases_return_in_a_subprocess(self, a, b, z):
        code = (
            "from pytcl.mathematical_functions.special_functions import hyp1f1;"
            f"print(float(hyp1f1({a!r}, {b!r}, {z!r})))"
        )
        done = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
        )
        assert done.returncode == 0, done.stderr

    @pytest.mark.parametrize(
        "a, b, z",
        [(1000.0, 1025.0, -1e5), (2000.0, 2030.0, -1e6), (9999.5, -22.99, -1e9)],
    )
    def test_value_below_the_double_range_is_zero(self, a, b, z):
        # 52f4cec raised ValueError here (a Stirling mantissa underflowed);
        # 1F1(1000; 1025; -1e5) is 6.86e-2385.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = hyp1f1(a, b, z)
        assert got == 0.0 and not math.copysign(1.0, got) < 0

    def test_array_with_an_underflowing_element(self):
        got = hyp1f1([1.0, 9999.5], [2.0, -22.99], -1e9)
        want = float(mpmath.hyp1f1(1, 2, -1e9))
        assert got[0] == pytest.approx(want, rel=5e-14, abs=0)
        assert got[1] == 0.0

    @pytest.mark.parametrize(
        "a, b, z, dps",
        [
            (-0.5, -1000.5, 900.0, 900),
            (4.8699, -264.582, 1307.64, 1000),
            (-390.574, 109.530, -1724.15, 1200),
        ],
    )
    def test_scipy_overflow_proven_by_the_series(self, a, b, z, dps):
        # 52f4cec returned NaN for all three; v2.11.1's +inf had the wrong
        # sign for the second.
        want = _series_sum(a, b, z, dps)
        assert abs(want) > mpmath.mpf(np.finfo(np.float64).max)
        assert hyp1f1(a, b, z) == math.copysign(math.inf, float(mpmath.sign(want)))

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


def _kummer_sum(a, b, z, dps):
    """e^z 1F1(b - a; b; -z) summed term by term with exact parameters.

    ``b - a`` is formed in mpmath, so a difference that rounds to an
    integer in double precision keeps its fractional part here."""
    with mpmath.workdps(dps):
        a, b, z = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(z)
        c, x = b - a, -z
        term = total = mpmath.mpf(1)
        tiny = mpmath.mpf(10) ** -dps
        k = 0
        while k < 2 * abs(b) + 2 * abs(x) or abs(term) > abs(total) * tiny:
            term *= (c + k) * x / ((b + k) * (k + 1))
            total += term
            k += 1
        return mpmath.exp(z) * total


def _mp_exact_params(a, b, z, dps=50):
    """mpmath's hyp1f1 with the doubles taken exactly (b - a never rounded)."""
    with mpmath.workdps(dps):
        return mpmath.hyp1f1(mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(z))


@pytest.mark.timeout(60, method="thread")
class TestExactIntegerTests:
    """The integer tests on b - a use the exact difference. 7e1e8c9 rounded
    it first and returned 0.0 for every value below: fl(-400.3 - -100.3)
    is -300 while the exact difference is -300 - 1.4e-14, so the algebraic
    term Gamma(b)/Gamma(b - a) z^(-a), which is the whole value, was
    dropped."""

    def test_pole_distance_matches_exact_rational_arithmetic(self):
        rng = np.random.default_rng(0)
        for _ in range(3000):
            a = rng.choice([-1.0, 1.0]) * 10 ** rng.uniform(-3, 8)
            step = rng.choice([0.0, 1e-16, 1e-14, 1e-12, 1e-9, 0.3])
            b = a + rng.integers(-500, 500) + rng.choice([-1.0, 1.0]) * step
            exact = Fraction(b) - Fraction(a)
            n = min(0, round(exact))
            want = abs(exact - n)
            got = _pole_distance(b, a)
            assert (got == 0.0) == (want == 0)
            assert got == pytest.approx(float(want), rel=1e-15, abs=0)

    @pytest.mark.parametrize(
        "a, b, z",
        [
            (-100.3, -400.3, -1e10),
            (-100.3, -400.3, -1e12),
            (-60.7, -210.7, -1e12),
            (-100.3, -400.3 + 1e-12, -1e10),  # the other side of the pole
            (-100.3, -400.3 - 1e-12, -1e10),
        ],
    )
    def test_near_integer_b_minus_a_overflow_beyond_1e9(self, a, b, z, recwarn):
        assert _pole_distance(b, a) != 0.0 and Fraction(b) - Fraction(a) != round(b - a)
        want = _mp_exact_params(a, b, z)
        assert abs(want) > mpmath.mpf(np.finfo(np.float64).max)
        for got in (hyp1f1(a, b, z), generalized_hypergeometric([a], [b], z)):
            assert got == math.copysign(math.inf, float(mpmath.sign(want)))
        assert not recwarn

    def test_near_integer_b_minus_a_finite_beyond_1e9(self, recwarn):
        a, b, z = 0.5000000000000001, -4.5, -1e10
        want = float(_mp_exact_params(a, b, z))  # -7.996e-21
        assert want != 0.0
        for got in (hyp1f1(a, b, z), generalized_hypergeometric([a], [b], z)):
            assert got == pytest.approx(want, rel=5e-14, abs=0)
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, want",
        [
            (-100.3, -400.3, math.inf),
            (-100.3, -400.3 + 1e-12, -math.inf),
            (-60.7, -210.7, -math.inf),
            (0.5000000000000001, -4.5, 0.0),  # a > 0: z^(-a) -> 0
            (2.5, -0.5, 0.0),  # exact b - a = -3: the term vanishes
        ],
    )
    def test_near_integer_b_minus_a_at_minus_infinity(self, a, b, want, recwarn):
        assert hyp1f1(a, b, -np.inf) == want
        assert generalized_hypergeometric([a], [b], -np.inf) == want
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, z, dps",
        [
            (-60.7, -210.7, -2e4, 60),  # -5.25e110; scipy gives 0.0
            (0.5000000000000001, -4.5, -5000.0, 60),  # -1.13e-17; scipy 0.0
        ],
    )
    def test_near_pole_in_the_scipy_band_uses_the_expansion(self, a, b, z, dps):
        assert 0.0 < _pole_distance(b, a) < 1e-6
        want = float(_kummer_sum(a, b, z, dps))
        assert want != 0.0
        assert hyp1f1(a, b, z) == pytest.approx(want, rel=5e-14, abs=0)

    @pytest.mark.parametrize(
        "a, b, z, dps",
        [
            (15.7, -84.3, -150.0, 60),  # -574373.13432876; b - a = -100 - 3.6e-15
            (117.17471070022194, 0.17471117939940314, -183.93047506716982, 250),
        ],
    )
    def test_near_pole_without_the_expansion_keeps_scipy(self, a, b, z, dps, recwarn):
        # 1fbe3f5 returned NaN here; scipy is accurate at these points.
        assert 0.0 < _pole_distance(b, a) < 1e-6
        want = float(_kummer_sum(a, b, z, dps))
        assert hyp1f1(a, b, z) == pytest.approx(want, rel=1e-13, abs=0)
        assert not recwarn

    def test_parameters_at_infinity_warn_nothing(self):
        # 1fbe3f5 ran the TwoSum over infinite parameters (inf - inf).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for f in (hyp1f1, lambda a, b, z: generalized_hypergeometric([a], [b], z)):
                assert np.isnan(f(np.inf, 1.0, 1.0))
                assert f(1.0, np.inf, 1.0) == 1.0
                assert np.isnan(f(-np.inf, 2.0, -200.0))

    def test_exact_pole_in_the_scipy_band_stays_with_scipy(self, recwarn):
        # b - a = -1 exactly: the algebraic term vanishes and scipy is right.
        want = float(_mp_exact_params(2.5, 1.5, -100.0))
        assert hyp1f1(2.5, 1.5, -100.0) == pytest.approx(want, rel=5e-14, abs=0)
        assert hyp1f1(-0.5, -0.5, -100.0) == pytest.approx(math.exp(-100.0), rel=5e-14)
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, z",
        [
            (1.0000000000000002, -3.0, -1e10),  # -5.33e-25; was 0.0
            (1.0000000000000002, -3.0, -60.0),  # -9.69e-17; was 1.1e-19
            (1.0000000000000002, -3.0, -25.0),  # v2.11.1 off by 5e-11
            (1.0000001, -3.0, -500.0),  # v2.11.1 off by 2.2e-9
        ],
    )
    def test_regularized_at_a_pole_with_a_rounded_shift_is_nan(self, a, b, z):
        # a + m + 1 rounds (5.0000000000000002 to 5) and 1 - a is within
        # 1e-6 of a pole of Gamma(1 - a); the sin(pi a) part is lost.
        with pytest.warns(RuntimeWarning, match="sin\\(pi a\\)"):
            assert np.isnan(hyp1f1_regularized(a, b, z))

    @pytest.mark.parametrize(
        "a, b, z",
        [
            (1.0000000000000002, -3.0, 1.0),  # e
            (1.0000000000000002, -3.0, -1.0),
            (1.0000000000000002, -3.0, -10.0),
            (1.0000001, -3.0, 0.5),
            (2.0000001, -1.0, 2.0),
            (0.9999999, -2.0, 1.0),
            (1.0000001, 0.0, 1.0),
            (1.0000001, -3.0, 50.0),
        ],
    )
    def test_regularized_at_a_pole_with_a_rounded_shift_at_small_z(
        self, a, b, z, recwarn
    ):
        # 1fbe3f5 returned NaN for all of these; the rounding of a + m + 1
        # is harmless where the sin(pi a) part is negligible.
        m = int(-b)
        with mpmath.workdps(60):
            want = float(
                mpmath.rf(mpmath.mpf(a), m + 1)
                * mpmath.mpf(z) ** (m + 1)
                / mpmath.factorial(m + 1)
                * mpmath.hyp1f1(mpmath.mpf(a) + m + 1, m + 2, mpmath.mpf(z))
            )
        assert hyp1f1_regularized(a, b, z) == pytest.approx(want, rel=1e-14, abs=0)
        assert not recwarn

    def test_regularized_at_a_pole_with_an_exact_shift(self, recwarn):
        a, m, z = 2.5, 3, -1e10
        with mpmath.workdps(50):
            want = float(
                mpmath.rf(a, m + 1)
                * mpmath.mpf(z) ** (m + 1)
                / mpmath.factorial(m + 1)
                * mpmath.hyp1f1(a + m + 1, m + 2, mpmath.mpf(z))
            )
        assert hyp1f1_regularized(a, -float(m), z) == pytest.approx(want, rel=5e-14)
        assert not recwarn


@pytest.mark.timeout(60, method="thread")
class TestCertifiedOverflow:
    CAP_CASE = (-80.725, -2418.98)  # expansion and tail proof both fail here

    @staticmethod
    def _terms(a, b, z):
        return math.ceil(2.0 * max(abs(a), abs(b)) + 6.0 * abs(z) + 100.0)

    def test_sum_at_the_term_cap_is_proven(self, recwarn):
        a, b = self.CAP_CASE
        z = 165843.0
        assert self._terms(a, b, z) <= _CERTIFIED_SUM_MAX_TERMS
        # Past k = 2419 every term has the sign of Gamma(b) Gamma(a) > 0
        # (both arguments lie in an interval (-odd, -even)).
        assert hyp1f1(a, b, z) == math.inf
        assert not recwarn

    def test_sum_past_the_term_cap_is_nan_with_warning(self):
        a, b = self.CAP_CASE
        z = 165844.0
        assert self._terms(a, b, z) > _CERTIFIED_SUM_MAX_TERMS
        with pytest.warns(RuntimeWarning, match="could not be proven"):
            assert np.isnan(hyp1f1(a, b, z))

    def test_seed_43_past_cap_point_is_nan_with_warning(self):
        with pytest.warns(RuntimeWarning, match="could not be proven"):
            assert np.isnan(hyp1f1(-80.725, -2418.98, 178743.4))

    @pytest.mark.parametrize(
        "a, z",
        [
            (-1375.3019969147488, 1932.0649875745569),  # 0.99999998 DBL_MAX
            (-1375.3019969067489, 1932.064987578126),  # 0.99999999 DBL_MAX
        ],
    )
    def test_finite_value_just_below_the_double_range_is_not_proven(self, a, z):
        # The certified sum is within a factor of about 1.5 of the value;
        # 7e1e8c9 decided on the sum and returned inf for these.
        assert np.isnan(_certified_overflow(a, -3092.1991021034255, z))


@pytest.mark.timeout(60, method="thread")
class TestExpansionAndProofPaths:
    """Each branch of the DLMF 13.7.2 evaluation and of the overflow proofs,
    against mpmath with exact parameters, the exact series, or the
    documented NaN with its warning."""

    def test_log_gamma_of_the_difference(self):
        from pytcl.mathematical_functions.special_functions.hypergeometric import (
            _lgamma_diff,
        )

        assert _lgamma_diff(-0.5, 2.5) == (math.inf, 0.0)  # Gamma(-3): a pole
        log_abs, sign = _lgamma_diff(-0.5, 2.0)  # Gamma(-2.5) = -0.9453
        assert sign == -1.0 and log_abs == pytest.approx(math.lgamma(-2.5), rel=1e-14)
        assert _lgamma_diff(2.5, -0.5) == (pytest.approx(math.lgamma(3.0)), 1.0)

    def test_coefficient_by_reflection_with_negative_b(self, recwarn):
        # Gamma(-0.5)/Gamma(2.8): the b < 0, b - a > 0 branch of _gamma_ratio.
        want = float(_mp_exact_params(-3.3, -0.5, -1e10))
        assert hyp1f1(-3.3, -0.5, -1e10) == pytest.approx(want, rel=5e-14, abs=0)
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, z, rel",
        [
            # Gamma(200)/Gamma(5): no shift fits, the logarithms place the
            # value inside the double range and scipy is used (1.1e-14).
            (195.0, 200.0, -1000.0, 5e-14),
            # scipy's value where the exponential term is not negligible
            # against the algebraic one at -z = 1e2 (scipy is 2.9e-12 off).
            (40.0, 41.0, -100.0, 1e-11),
            # z > 0, 1e2 <= z < 1e9: the expansion settles but does not
            # overflow, so scipy's value is returned.
            (0.5, 1.5, 200.0, 5e-14),
            # z > 0 with Gamma(a) = 1e28: the algebraic term is within eps of
            # the exponential one, so the expansion declines and scipy is used.
            (1e-28, 0.5, 100.0, 5e-14),
        ],
    )
    def test_values_where_the_expansion_hands_over_to_scipy(
        self, a, b, z, rel, recwarn
    ):
        want = float(_mp_exact_params(a, b, z))
        assert hyp1f1(a, b, z) == pytest.approx(want, rel=rel, abs=0)
        assert not recwarn

    def test_underflow_without_a_coefficient_shift(self, recwarn):
        # Gamma(10200)/Gamma(10): the shift would need 10,030 steps; the
        # logarithms prove the value (1e-34879) is below the double range.
        with mpmath.workdps(50):
            assert (
                mpmath.log10(abs(mpmath.hyp1f1(10190, 10200, mpmath.mpf(-1e7)))) < -3e4
            )
        assert hyp1f1(10190.0, 10200.0, -1e7) == 0.0
        assert not recwarn

    @pytest.mark.parametrize(
        "a, b, z",
        [
            (-195.5, 5.0, -1e5),  # Gamma(5)/Gamma(200.5): no shift fits
            (-10190.5, 10.0, -2e8),  # more than 10,000 shift steps
            (-190.3, -200.5, -1e5),  # reflected: Gamma(11.2)/Gamma(201.5)
            (-190.3, -200.5, -1e10),
        ],
    )
    def test_overflow_without_a_coefficient_shift(self, a, b, z, recwarn):
        # The leading algebraic term Gamma(b)/Gamma(b - a) (-z)^-a with exact
        # parameters is far beyond the double range; its sign is the result.
        with mpmath.workdps(50):
            A, B, x = mpmath.mpf(a), mpmath.mpf(b), mpmath.mpf(-z)
            lead = mpmath.gamma(B) * mpmath.rgamma(B - A) * x ** (-A)
        assert abs(lead) > mpmath.mpf(np.finfo(np.float64).max) * 1e100
        assert hyp1f1(a, b, z) == math.copysign(math.inf, float(mpmath.sign(lead)))
        assert not recwarn

    def test_overflow_proven_from_the_tail_with_no_head(self, recwarn):
        # a, b > 0: every term has one sign; the expansion diverges (ratio
        # 125) so the largest term alone proves the overflow (1.75e598).
        with mpmath.workdps(30):
            assert mpmath.hyp1f1(500.5, 1000.5, 2000) > mpmath.mpf(
                np.finfo(np.float64).max
            )
        assert hyp1f1(500.5, 1000.5, 2000.0) == math.inf
        assert not recwarn

    def test_overflow_too_marginal_for_the_tail_is_proven_by_the_series(self, recwarn):
        # scipy overflows from z = 1134.9284963691885 on; there the largest
        # term is below DBL_MAX and the certified sum decides.
        z = 1135.0284963691884
        with mpmath.workdps(30):
            value = mpmath.hyp1f1(500.5, 1000.5, z) / mpmath.mpf(
                np.finfo(np.float64).max
            )
        assert float(value) == pytest.approx(1.0752834, rel=1e-6)
        assert hyp1f1(500.5, 1000.5, z) == math.inf
        assert not recwarn

    def test_overflow_by_one_ulp_is_nan_with_warning(self):
        # 1.0000000000002 DBL_MAX: within the factor the certified sum
        # leaves undecided, so neither inf nor a finite value is returned.
        z = 1134.9284963691885
        with mpmath.workdps(30):
            ratio = mpmath.hyp1f1(500.5, 1000.5, z) / mpmath.mpf(
                np.finfo(np.float64).max
            )
            assert 1.0 < ratio < 1.0 + 1e-12
        with pytest.warns(RuntimeWarning, match="could not be proven"):
            assert np.isnan(hyp1f1(500.5, 1000.5, z))

    def test_overflow_dominated_by_the_head_is_proven_by_the_series(self, recwarn):
        # k0 = 501: the terms before it carry the value, so the tail proof
        # declines and the certified sum gives the sign (-1.68e741).
        want = _series_sum(-500.5, 0.5, 3000.0, 600)
        assert want < 0 and abs(want) > mpmath.mpf(np.finfo(np.float64).max)
        assert hyp1f1(-500.5, 0.5, 3000.0) == -math.inf
        assert not recwarn

    def test_overflow_past_the_tail_cap_is_nan_with_warning(self):
        # scipy returns inf; the exponential series of the expansion decays
        # too slowly to settle in 200 terms, the tail proof needs k0 > 1e6
        # and the certified sum more than 1e6 terms.
        with pytest.warns(RuntimeWarning, match="could not be proven"):
            assert np.isnan(hyp1f1(0.5, -1000000.5, 1e6))

    def test_expansion_whose_exponential_series_diverges_is_nan_with_warning(self):
        # 1F1(1e-17; 5e9; -1e9) = 1 - 2e-18, but the exponential series has
        # ratio 5 at |z| = 1e9 where scipy is not called.
        with pytest.warns(RuntimeWarning, match="asymptotic expansion"):
            assert np.isnan(hyp1f1(1e-17, 5e9, -1e9))

    def test_terminating_overflow_in_hyp1f1_is_summed_exactly(self, recwarn):
        # (-5)_5 z^5 / (5! 5!) = -z^5 / 120 leads: scipy gives +inf.
        assert hyp1f1(-5.0, 1.0, 1e100) == -math.inf
        assert generalized_hypergeometric([-5.0], [1.0], 1e100) == -math.inf
        assert not recwarn

    def test_limit_at_infinity_with_a_non_finite_parameter_is_nan(self, recwarn):
        assert np.isnan(generalized_hypergeometric([np.inf], [2.0, 3.0], np.inf))
        assert np.isnan(generalized_hypergeometric([1.0], [np.nan, 3.0], -np.inf))
        assert not recwarn

    def test_certified_series_declines_at_a_zero_crossing(self):
        from pytcl.mathematical_functions.special_functions.hypergeometric import (
            _certified_series,
        )

        a, b, z0 = -1401.3019969167487, -3092.1991021034255, 1950.0466930916314
        assert _certified_series(a, b, z0) is None  # |sum| below its error bound
        assert np.isnan(_certified_overflow(a, b, z0))
        # 1e-4 away the value is +-e^(709.8 + 24.6) with signs -/+.
        for dz, sign in ((1e-4, -1.0), (-1e-4, 1.0)):
            want = _series_sum(a, b, z0 + dz, 120)
            assert mpmath.sign(want) == sign
            got_sign, log_lower = _certified_series(a, b, z0 + dz)
            assert got_sign == sign
            assert log_lower < float(mpmath.log(abs(want))) < log_lower + 1e-6
            assert _certified_overflow(a, b, z0 + dz) == sign * math.inf

    def test_certified_series_with_a_rounded_parameter(self):
        from pytcl.mathematical_functions.special_functions.hypergeometric import (
            _certified_series,
        )

        b, x = -500.3, 2e4
        # a rounded onto a non-positive integer: the exact and the rounded
        # series differ in kind, so any rounding error declines.
        assert _certified_series(-300.0, b, x, 1e-14) is None
        # a within 1e-10 of it: the perturbation bound P = err / 1e-10 is
        # 0.1 for err = 1e-11 (declined) and 1e-3 for 1e-13 (certified).
        a = -299.9999999999
        assert _certified_series(a, b, x, 1e-11) is None
        want = _series_sum(a, b, x, 80)  # -1.787e9016
        for err in (1e-13, 0.0):
            sign, log_lower = _certified_series(a, b, x, err)
            assert sign == -1.0 == mpmath.sign(want)
            assert log_lower < float(mpmath.log(abs(want))) < log_lower + 1e-2
