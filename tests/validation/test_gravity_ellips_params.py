"""Validation of the gravity ellipsoidal-parameter conversions.

MATLAB fixtures from ``scripts/matlab_capture/capture_gravity_ellips.m``
(TCL commit a9acd8f), plus closed-form cross-checks that need no
fixture. Fixture tests skip when the CSV is absent unless
``PYTCL_REQUIRE_MATLAB_FIXTURES=1``.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from pytcl.gravity import (
    WGS84,
    alt_ellips_param_to_flattening,
    ellips_grav_coeffs,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "matlab"
REQUIRE = os.environ.get("PYTCL_REQUIRE_MATLAB_FIXTURES") == "1"


def _load(name):
    path = FIXTURES / name
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"required MATLAB fixture missing: {name}")
        pytest.skip(f"MATLAB fixture not captured: {name}")
    return np.loadtxt(path, delimiter=",", ndmin=2)


class TestAltEllipsParamMatlab:
    def test_matches_matlab(self):
        rows = _load("grav_alt_ellips_flattening.csv")
        for omega, a, c20_bar, gm, f_ref in rows:
            f = alt_ellips_param_to_flattening(omega, a, c20_bar, gm)
            # The fixed-point iteration settles into an ulp-scale
            # 2-cycle; MATLAB and this port can stop on different
            # phases of it, so agreement is a few parts in 1e12.
            np.testing.assert_allclose(f, f_ref, rtol=1e-11)


class TestEllipsGravCoeffsMatlab:
    def test_matches_matlab(self):
        rows = _load("grav_ellips_coeffs.csv")
        cache = {}
        for max_order, is_norm, n, m, c_ref in rows:
            key = (int(max_order), bool(is_norm))
            if key not in cache:
                cache[key] = ellips_grav_coeffs(key[0], key[1])[0]
            np.testing.assert_allclose(cache[key][int(n), int(m)], c_ref, rtol=1e-13)


class TestClosedFormProperties:
    def test_egm2008_constants_recover_reference_flattening(self):
        # EGM2008's defining (omega, a, C20bar, GM) yield 1/f = 298.2576...
        f = alt_ellips_param_to_flattening(
            7292115e-11, 6378136.3, -484.1654767e-6, 3986004.415e8
        )
        np.testing.assert_allclose(1.0 / f, 298.2576, atol=2e-4)

    def test_roundtrip_through_coefficients(self):
        # ellips_grav_coeffs(WGS84) -> C20bar; feeding that C20bar back
        # through alt_ellips_param_to_flattening must recover WGS84's f.
        C, _, _, _ = ellips_grav_coeffs()
        f = alt_ellips_param_to_flattening(WGS84.omega, WGS84.a, C[2, 0], WGS84.GM)
        np.testing.assert_allclose(f, WGS84.f, rtol=1e-9)

    def test_unnormalized_j2_matches_wgs84(self):
        C, _, _, _ = ellips_grav_coeffs(is_normalized=False)
        np.testing.assert_allclose(-C[2, 0], 1.082629821e-3, rtol=1e-8)

    def test_normalization_factor(self):
        Cn, _, _, _ = ellips_grav_coeffs(max_order=4, is_normalized=True)
        Cu, _, _, _ = ellips_grav_coeffs(max_order=4, is_normalized=False)
        for n in (0, 2, 4):
            np.testing.assert_allclose(
                Cn[n, 0], Cu[n, 0] / np.sqrt(2 * n + 1), rtol=1e-14
            )

    def test_sine_coefficients_are_zero(self):
        _, S, _, _ = ellips_grav_coeffs(max_order=8)
        assert np.all(S == 0)


def test_nonconvergent_input_raises_loudly():
    from pytcl.core.exceptions import ConvergenceError

    # A positive C20bar (prolate "ellipsoid") drives the iterate to a
    # square root of a negative number; MATLAB silently exits its loop
    # on the resulting NaN comparison and returns NaN.
    with pytest.raises(ConvergenceError, match="did not converge"):
        alt_ellips_param_to_flattening(7292115e-11, 6378137.0, +484.0e-6, 3.986e14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
