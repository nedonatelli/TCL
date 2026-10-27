"""``tau_scale`` against the Maronna-Zamar tau scale as statsmodels implements it.

Vendored values come from statsmodels 0.15.0:

    uv run --no-project --python 3.12 --with statsmodels==0.15.0 python -c \\
        "import numpy as np; from statsmodels.robust.scale import scale_tau; \\
         a = np.random.default_rng(0).normal(size=200); b = a.copy(); \\
         b[:20] = 50.0; print(scale_tau(a)[1], scale_tau(b)[1], \\
         scale_tau(np.array([1., 2, 3, 4, 100]))[1])"

All three printed values equal the constants below.
"""

import numpy as np
import pytest

from pytcl.static_estimation import tau_scale


def _normal(seed, n):
    return np.random.default_rng(seed).normal(size=n)


def _normal_with_tail(seed, n, n_out, value):
    r = _normal(seed, n)
    r[:n_out] = value
    return r


VENDORED = [
    (_normal(seed=0, n=200), 0.9766651091871259),
    (_normal_with_tail(seed=0, n=200, n_out=20, value=50.0), 1.2407140659984943),
    (np.array([1.0, 2.0, 3.0, 4.0, 100.0]), 1.744097372590729),
]


@pytest.mark.parametrize("r, want", VENDORED)
def test_tau_scale_matches_statsmodels(r, want):
    assert tau_scale(r) == pytest.approx(want, rel=1e-12, abs=0)


def test_tau_scale_matches_statsmodels_live():
    sm = pytest.importorskip("statsmodels.robust.scale")
    for seed in range(20):
        r = np.random.default_rng(seed).standard_t(3, size=500)
        assert tau_scale(r) == pytest.approx(sm.scale_tau(r)[1], rel=1e-12, abs=0)


def test_tau_scale_zero_mad_returns_zero():
    assert tau_scale(np.array([2.0, 2.0, 2.0, 5.0, -7.0])) == 0.0
