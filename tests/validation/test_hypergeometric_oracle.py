"""Independent mpmath oracle for generalized_hypergeometric."""

import math

import mpmath
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
    assert got == pytest.approx(want, rel=1e-10)


def test_total_cancellation_warns_and_returns_nan():
    # Largest term ~1e40 against a sum of 0.199: 1/eps is exceeded.
    p, q, z = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -100.0
    assert float(mpmath.hyper(p, q, z)) == pytest.approx(0.199294, rel=1e-5)
    with pytest.warns(RuntimeWarning, match="catastrophic cancellation"):
        got = generalized_hypergeometric(p, q, z)
    assert math.isnan(got)


def test_partial_cancellation_warns_and_keeps_value():
    p, q, z = [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], -50.0
    want = float(mpmath.hyper(p, q, z))
    with pytest.warns(RuntimeWarning, match="half of the available digits"):
        got = generalized_hypergeometric(p, q, z)
    assert got == pytest.approx(want, rel=1e-4)


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
    assert generalized_hypergeometric(p, q, z) == pytest.approx(want, rel=1e-12)
