"""Independent mpmath oracle for generalized_hypergeometric."""

import math

import mpmath
import numpy as np
import pytest

from pytcl.mathematical_functions.special_functions.hypergeometric import (
    generalized_hypergeometric,
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
            ([-0.5], [1.0]),
            ([1.0], [-1.5]),
            ([-2.0, 1.0, 1.0], [-1.0, 3.0]),
            ([1.0, 1.0], [-0.5]),
        ],
    )
    def test_negative_non_integer_a_or_pole_b_is_nan(self, a, b):
        for z in (math.inf, -math.inf):
            assert np.isnan(generalized_hypergeometric(a, b, z))

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
