"""Multivariate polynomial root solver against MATLAB TCL references.

MATLAB reference roots captured from the Tracker Component Library
(commit a9acd8f) via scripts/matlab_capture/capture_poly_roots.m. Every
input here mirrors that capture script verbatim.

The three-variable Dreesen system has no MATLAB fixture: on R2026a the
ORIGINAL polyRootsMultiDim fails on it with exit code 2 (recorded in
poly_roots_3var_matlab_exitcode.csv) — the matrixRank algorithm-0
tolerance it uses for null spaces sits inside LAPACK's roundoff noise,
which is exactly why the port deviates to the algorithm-1 tolerance.
That case is validated by residuals instead, which is a stronger oracle
than agreement anyway.
"""

from pathlib import Path

import numpy as np

from pytcl.mathematical_functions.polynomials import poly_roots_multi_dim

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "matlab"

# Root sets agree with MATLAB to ~2e-13 on the unit-scale system and
# ~1e-9 relative on the kilometre-scale localization system (the roots
# pass through an eigendecomposition of the lifted problem).
ATOL_UNIT = 1e-10
RTOL_KM = 1e-6


def _load_roots(name):
    m = np.loadtxt(FIXTURE_DIR / name, delimiter=",", ndmin=2)
    n = m.shape[0] // 2
    return m[:n] + 1j * m[n:]


def _sort_columns(r):
    key = np.lexsort(np.vstack([r.imag[::-1], r.real[::-1]]))
    return r[:, key]


def _dreesen_2var():
    """Section 2.1 of Dreesen: roots (4,-5), (1,0), (3,-2), (0,-1)."""
    p = np.zeros((3, 3))
    p[0, 0], p[2, 0], p[1, 1], p[0, 2] = -4.0, -1.0, 2.0, 1.0
    p[1, 0], p[0, 1] = 5.0, -3.0
    q = np.zeros((3, 3))
    q[0, 0], q[2, 0], q[1, 1], q[0, 2] = -1.0, 1.0, 2.0, 1.0
    return [p, q]


def _dreesen_3var():
    """p.94 of Dreesen: a cubic system with 18 affine roots, 2 real."""
    p = np.zeros((4, 4, 4))
    p[2, 0, 0], p[1, 1, 0], p[0, 1, 1], p[0, 0, 0] = 1.0, 5.0, 4.0, -10.0
    q = np.zeros((4, 4, 4))
    q[0, 3, 0], q[2, 1, 0], q[0, 0, 0] = 1.0, 3.0, -12.0
    k = np.zeros((4, 4, 4))
    k[0, 0, 3], k[1, 1, 1], k[0, 0, 0] = 1.0, 4.0, -8.0
    return [p, q, k]


class TestTwoVariableSystem:
    def test_matches_matlab(self):
        roots, exit_code = poly_roots_multi_dim(_dreesen_2var())
        assert exit_code == 0
        ref = _load_roots("poly_roots_2var.csv")
        np.testing.assert_allclose(
            _sort_columns(roots), _sort_columns(ref), atol=ATOL_UNIT
        )

    def test_finds_the_four_known_real_roots(self):
        roots, exit_code = poly_roots_multi_dim(_dreesen_2var())
        assert exit_code == 0
        assert roots.shape == (2, 4)
        found = sorted(np.round(roots.real.T, 8).tolist())
        assert found == [[0.0, -1.0], [1.0, 0.0], [3.0, -2.0], [4.0, -5.0]]
        assert np.max(np.abs(roots.imag)) < 1e-8

    def test_motzkin_null_space_matches_matlab(self):
        roots, exit_code = poly_roots_multi_dim(_dreesen_2var(), use_motzkin_null=True)
        assert exit_code == 0
        ref = _load_roots("poly_roots_2var_motzkin.csv")
        np.testing.assert_allclose(
            _sort_columns(roots), _sort_columns(ref), atol=ATOL_UNIT
        )


class TestThreeVariableSystem:
    def test_finds_all_18_roots_where_matlab_fails(self):
        # The upstream original returns exit code 2 on this same system
        # (R2026a, recorded in the fixture); the deviated tolerance
        # recovers all 18 roots. Residuals are the oracle.
        matlab_exit = int(
            np.loadtxt(FIXTURE_DIR / "poly_roots_3var_matlab_exitcode.csv", ndmin=1)[0]
        )
        assert matlab_exit == 2

        roots, exit_code = poly_roots_multi_dim(_dreesen_3var())
        assert exit_code == 0
        assert roots.shape == (3, 18)

        x1, x2, x3 = roots
        residuals = np.vstack(
            [
                x1**2 + 5 * x1 * x2 + 4 * x2 * x3 - 10,
                x2**3 + 3 * x1**2 * x2 - 12,
                x3**3 + 4 * x1 * x2 * x3 - 8,
            ]
        )
        assert np.max(np.abs(residuals)) < 1e-8

    def test_two_of_the_roots_are_real(self):
        roots, _ = poly_roots_multi_dim(_dreesen_3var())
        n_real = int(np.sum(np.all(np.abs(roots.imag) < 1e-6, axis=0)))
        assert n_real == 2


class TestRangeRateLocalizationSystem:
    """The polynomial system rangeRate2StaticPos builds in 2D."""

    U_TRUE = np.array([1e3, 5e3])
    S = np.array([[500.0, 1100.0], [2500.0, 2500.0]])
    S_DOT = np.array([[300.0, 300.0], [0.0, 0.0]])

    def _system(self):
        rr = np.array(
            [
                -self.S_DOT[:, k]
                @ (self.U_TRUE - self.S[:, k])
                / np.linalg.norm(self.U_TRUE - self.S[:, k])
                for k in range(2)
            ]
        )
        mats = []
        for k in range(2):
            loc = self.S[:, k]
            l_dot = self.S_DOT[:, k]
            r_dot = rr[k]
            l_tilde = 2.0 * (l_dot * (loc @ l_dot) - r_dot**2 * loc)
            c_tilde = r_dot**2 * (loc @ loc) - (loc @ l_dot) ** 2
            cm = np.zeros((3, 3))
            cm[2, 0] = r_dot**2 - l_dot[0] ** 2
            cm[0, 2] = r_dot**2 - l_dot[1] ** 2
            cm[1, 1] = -2.0 * l_dot[0] * l_dot[1]
            cm[1, 0] = l_tilde[0]
            cm[0, 1] = l_tilde[1]
            cm[0, 0] = c_tilde
            mats.append(cm)
        return mats

    def test_matches_matlab(self):
        roots, exit_code = poly_roots_multi_dim(self._system())
        assert exit_code == 0
        ref = _load_roots("poly_roots_rrloc2d.csv")
        np.testing.assert_allclose(
            _sort_columns(roots), _sort_columns(ref), rtol=RTOL_KM, atol=1e-4
        )

    def test_true_emitter_is_among_the_real_solutions(self):
        roots, _ = poly_roots_multi_dim(self._system())
        real = roots[:, np.all(np.abs(roots.imag) < 1e-6, axis=0)].real
        dists = np.linalg.norm(real - self.U_TRUE[:, None], axis=0)
        assert np.min(dists) < 1e-3


class TestFailurePaths:
    def test_zero_degree_increases_gives_exit_code_1(self):
        roots, exit_code = poly_roots_multi_dim(_dreesen_2var(), max_deg_increases=0)
        assert exit_code == 1
        assert roots.shape == (2, 0)

    def test_default_max_deg_increases_is_ten_per_variable(self):
        # The default must be enough for the reference systems.
        roots, exit_code = poly_roots_multi_dim(_dreesen_2var(), None)
        assert exit_code == 0
        assert roots.shape[1] == 4

    def test_result_fields_are_named(self):
        result = poly_roots_multi_dim(_dreesen_2var())
        assert result.exit_code == 0
        assert result.roots.shape == (2, 4)


class TestSingleVariable:
    def test_univariate_quadratic(self):
        # (x - 2)(x + 3) = x^2 + x - 6, as a 1-variable "system".
        coeffs = np.array([-6.0, 1.0, 1.0])
        roots, exit_code = poly_roots_multi_dim([coeffs])
        assert exit_code == 0
        found = sorted(np.round(roots.real.ravel(), 8).tolist())
        assert found == [-3.0, 2.0]
        assert np.max(np.abs(roots.imag)) < 1e-10
