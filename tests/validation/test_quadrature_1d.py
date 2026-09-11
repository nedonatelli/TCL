"""Validation of the 1-D quadrature builders against MATLAB fixtures.

Fixtures come from ``scripts/matlab_capture/capture_quadrature_1d.m``
(TCL commit a9acd8f); tests skip when a CSV is absent unless
``PYTCL_REQUIRE_MATLAB_FIXTURES=1``. Property tests (exactness on
polynomials of the advertised order, weight sums) run regardless.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pytcl.mathematical_functions.numerical_integration import (
    clenshaw_curtis_points_1d,
    conform_map_quad_pts_1d,
    fejer_points_1d,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "matlab"
REQUIRE = os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1"


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"required MATLAB fixture missing: {name}")
        pytest.skip(f"MATLAB fixture not captured: {name}")
    data = np.loadtxt(path, delimiter=",")
    return data[0], data[1]  # xi row, w row


class TestClenshawCurtisMatlab:
    @pytest.mark.parametrize("n", [2, 5, 8, 16, 33, 64])
    def test_matches_matlab(self, n):
        # n = 64 exercises MATLAB's pretabulated path; the table and
        # the recurrence agree to ~1e-14 (float noise between the two
        # evaluation orders), the computed n to 1e-15.
        xi_ref, w_ref = _load(f"quad1d_cc_n{n}.csv")
        xi, w = clenshaw_curtis_points_1d(n)
        np.testing.assert_allclose(xi, xi_ref, atol=5e-14)
        np.testing.assert_allclose(w, w_ref, atol=5e-14)


class TestFejerMatlab:
    @pytest.mark.parametrize("n", [5, 16, 31])
    @pytest.mark.parametrize("rule", [1, 2])
    def test_matches_matlab(self, n, rule):
        xi_ref, w_ref = _load(f"quad1d_fejer_n{n}_r{rule}.csv")
        xi, w = fejer_points_1d(n, rule)
        np.testing.assert_allclose(xi, xi_ref, atol=1e-15)
        np.testing.assert_allclose(w, w_ref, atol=1e-15)


class TestConformMapMatlab:
    CASES = [
        (0, 0, None),
        (0, 1, None),
        (0, 0, 7),
        (1, 0, None),
        (1, 1, None),
        (1, 0, 2.0),
        (2, 0, None),
        (2, 0, 1.7),
        (3, 0, None),
        (3, 1, None),
        (3, 0, 2.0),
    ]

    @pytest.mark.parametrize("mapping,point_type,param", CASES)
    def test_matches_matlab(self, mapping, point_type, param):
        tag = "def" if param is None else f"{param:g}".replace(".", "p")
        xi_ref, w_ref = _load(f"quad1d_conform_m{mapping}_p{point_type}_{tag}.csv")
        xi, w = conform_map_quad_pts_1d(20, mapping, point_type, param)

        if mapping == 3 and point_type == 0:
            # MATLAB's positional endpoint substitution corrupts the
            # weights of its first and last stored nodes (one extreme,
            # one INTERIOR, an artifact of its Gauss-Legendre storage
            # order); this port fixed that loudly, so those two nodes
            # are excluded from the oracle comparison.
            keep_ref = np.ones(len(xi_ref), dtype=bool)
            keep_ref[[0, -1]] = False
            keep = ~np.isin(
                np.arange(len(xi)),
                [np.argmin(np.abs(xi - xi_ref[0])), np.argmin(np.abs(xi - xi_ref[-1]))],
            )
            xi_ref, w_ref = xi_ref[keep_ref], w_ref[keep_ref]
            xi, w = xi[keep], w[keep]

        # Base Gauss-Legendre orderings differ between MATLAB and
        # numpy; compare as (xi, w) pairs sorted by xi.
        order = np.argsort(xi)
        order_ref = np.argsort(xi_ref)
        np.testing.assert_allclose(xi[order], xi_ref[order_ref], atol=1e-12)
        np.testing.assert_allclose(w[order], w_ref[order_ref], atol=1e-12)


class TestQuadratureProperties:
    def test_clenshaw_curtis_exactness(self):
        # Order-n rule integrates x^k exactly for k <= n.
        xi, w = clenshaw_curtis_points_1d(10)
        for k in range(0, 11, 2):
            np.testing.assert_allclose(np.sum(w * xi**k), 2.0 / (k + 1), rtol=1e-12)

    def test_fejer_first_rule_exactness(self):
        xi, w = fejer_points_1d(12)  # order 11
        for k in range(0, 12, 2):
            np.testing.assert_allclose(np.sum(w * xi**k), 2.0 / (k + 1), rtol=1e-12)

    def test_fejer_second_rule_counts(self):
        xi, w = fejer_points_1d(12, rule=2)
        assert len(xi) == 11 and len(w) == 11
        np.testing.assert_allclose(np.sum(w), 2.0, rtol=1e-12)

    def test_conformal_maps_integrate_analytic_functions(self):
        exact = np.e - np.exp(-1.0)
        for mapping in (0, 1, 2):
            xi, w = conform_map_quad_pts_1d(24, mapping=mapping)
            np.testing.assert_allclose(np.sum(w * np.exp(xi)), exact, rtol=1e-5)

    def test_jacobi_elliptic_identity(self):
        from pytcl.mathematical_functions.special_functions import (
            jacobi_elliptic,
        )

        sn, cn, dn = jacobi_elliptic(np.array([0.3, 0.9]), 0.4)
        np.testing.assert_allclose(sn**2 + cn**2, 1.0, rtol=1e-12)
        np.testing.assert_allclose(dn**2 + 0.4 * sn**2, 1.0, rtol=1e-12)

    def test_strip_map_rejects_clenshaw_curtis_base(self):
        with pytest.raises(ValueError, match="strip"):
            conform_map_quad_pts_1d(10, mapping=2, point_type=1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
