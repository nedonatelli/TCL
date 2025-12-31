"""
Unit tests for core module (constants, validation, array utilities).
"""

import math

import numpy as np
import pytest

from pytcl.core.array_utils import (
    block_diag,
    column_vector,
    is_positive_definite,
    normalize_vector,
    row_vector,
    skew_symmetric,
    unskew,
    unvec,
    vec,
    wrap_to_2pi,
    wrap_to_pi,
    wrap_to_range,
)
from pytcl.core.constants import (
    DEG_TO_RAD,
    EARTH_FLATTENING,
    EARTH_SEMI_MAJOR_AXIS,
    PI,
    RAD_TO_DEG,
    SPEED_OF_LIGHT,
    TWO_PI,
    WGS84,
    PhysicalConstants,
)
from pytcl.core.validation import (
    ValidationError,
    ensure_2d,
    ensure_column_vector,
    ensure_row_vector,
    ensure_square_matrix,
    ensure_symmetric,
    validate_array,
)


class TestConstants:
    """Tests for physical and mathematical constants."""

    def test_speed_of_light(self):
        """Speed of light should be the standard value."""
        assert SPEED_OF_LIGHT == 299_792_458.0

    def test_earth_parameters(self):
        """Earth parameters should match WGS84."""
        assert EARTH_SEMI_MAJOR_AXIS == 6_378_137.0
        assert abs(EARTH_FLATTENING - 1 / 298.257223563) < 1e-15

    def test_wgs84_ellipsoid(self):
        """WGS84 ellipsoid should have correct derived parameters."""
        assert WGS84.a == 6_378_137.0
        assert WGS84.name == "WGS84"

        # Check derived parameters
        expected_b = WGS84.a * (1 - WGS84.f)
        assert abs(WGS84.b - expected_b) < 1e-6

        # Check eccentricity
        expected_e2 = 2 * WGS84.f - WGS84.f**2
        assert abs(WGS84.e2 - expected_e2) < 1e-15

    def test_mathematical_constants(self):
        """Mathematical constants should be accurate."""
        assert PI == math.pi
        assert TWO_PI == 2 * math.pi
        assert abs(DEG_TO_RAD - math.pi / 180) < 1e-15
        assert abs(RAD_TO_DEG - 180 / math.pi) < 1e-15

    def test_physical_constants_class(self):
        """PhysicalConstants dataclass should work correctly."""
        pc = PhysicalConstants()
        assert pc.c == SPEED_OF_LIGHT
        assert pc.g_0 == 9.80665


class TestValidation:
    """Tests for input validation functions."""

    def test_validate_array_basic(self):
        """Basic array validation should work."""
        result = validate_array([1, 2, 3], "test")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_validate_array_ndim(self):
        """Array dimension validation should work."""
        # Should pass
        validate_array([[1, 2], [3, 4]], "matrix", ndim=2)

        # Should fail
        with pytest.raises(ValidationError):
            validate_array([1, 2, 3], "vector", ndim=2)

    def test_validate_array_shape(self):
        """Array shape validation should work."""
        # Should pass
        validate_array([[1, 2, 3]], "row", shape=(1, 3))

        # Should fail
        with pytest.raises(ValidationError):
            validate_array([[1, 2], [3, 4]], "matrix", shape=(3, 2))

    def test_validate_array_finite(self):
        """Finite value validation should work."""
        # Should pass
        validate_array([1, 2, 3], "test", finite=True)

        # Should fail with inf
        with pytest.raises(ValidationError):
            validate_array([1, np.inf, 3], "test", finite=True)

        # Should fail with nan
        with pytest.raises(ValidationError):
            validate_array([1, np.nan, 3], "test", finite=True)

    def test_validate_array_non_negative(self):
        """Non-negative validation should work."""
        # Should pass
        validate_array([0, 1, 2], "test", non_negative=True)

        # Should fail
        with pytest.raises(ValidationError):
            validate_array([-1, 0, 1], "test", non_negative=True)

    def test_ensure_2d(self):
        """Ensure 2D should promote 1D arrays correctly."""
        # Column vector (default)
        result = ensure_2d([1, 2, 3])
        assert result.shape == (3, 1)

        # Row vector
        result = ensure_2d([1, 2, 3], axis="row")
        assert result.shape == (1, 3)

        # Already 2D should stay unchanged
        result = ensure_2d([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

    def test_ensure_column_vector(self):
        """Ensure column vector should work correctly."""
        result = ensure_column_vector([1, 2, 3])
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3])

    def test_ensure_row_vector(self):
        """Ensure row vector should work correctly."""
        result = ensure_row_vector([1, 2, 3])
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3])

    def test_ensure_square_matrix(self):
        """Ensure square matrix should work correctly."""
        # Should pass
        result = ensure_square_matrix([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

        # Should fail
        with pytest.raises(ValidationError):
            ensure_square_matrix([[1, 2, 3], [4, 5, 6]])

    def test_ensure_symmetric(self):
        """Ensure symmetric should work correctly."""
        # Should pass and symmetrize
        A = np.array([[1.0, 2.0], [2.0 + 1e-12, 3.0]])
        result = ensure_symmetric(A)
        np.testing.assert_allclose(result, result.T)

        # Should fail for non-symmetric
        with pytest.raises(ValidationError):
            ensure_symmetric([[1, 2], [3, 4]])


class TestArrayUtils:
    """Tests for array utility functions."""

    def test_wrap_to_pi(self):
        """wrap_to_pi should wrap angles correctly."""
        # Basic cases
        assert abs(wrap_to_pi(0) - 0) < 1e-10
        assert (
            abs(wrap_to_pi(3 * np.pi) - (-np.pi)) < 1e-10
            or abs(wrap_to_pi(3 * np.pi) - np.pi) < 1e-10
        )

        # Array input
        angles = np.array([-4, -2, 0, 2, 4])
        wrapped = wrap_to_pi(angles)
        assert np.all(wrapped >= -np.pi)
        assert np.all(wrapped < np.pi)

    def test_wrap_to_2pi(self):
        """wrap_to_2pi should wrap angles correctly."""
        # Basic cases
        assert abs(wrap_to_2pi(0) - 0) < 1e-10
        assert abs(wrap_to_2pi(-np.pi / 2) - 3 * np.pi / 2) < 1e-10
        assert abs(wrap_to_2pi(3 * np.pi) - np.pi) < 1e-10

        # Array input
        angles = np.array([-4, -2, 0, 2, 4, 8])
        wrapped = wrap_to_2pi(angles)
        assert np.all(wrapped >= 0)
        assert np.all(wrapped < 2 * np.pi)

    def test_wrap_to_range(self):
        """wrap_to_range should wrap values to specified interval."""
        assert abs(wrap_to_range(370, 0, 360) - 10) < 1e-10
        assert abs(wrap_to_range(-10, 0, 360) - 350) < 1e-10
        assert abs(wrap_to_range(5, -1, 1) - (-1)) < 1e-10

    def test_column_vector(self):
        """column_vector should create (n, 1) arrays."""
        result = column_vector([1, 2, 3])
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3])

        # From row vector
        result = column_vector([[1, 2, 3]])
        assert result.shape == (3, 1)

    def test_row_vector(self):
        """row_vector should create (1, n) arrays."""
        result = row_vector([1, 2, 3])
        assert result.shape == (1, 3)
        np.testing.assert_array_equal(result.flatten(), [1, 2, 3])

        # From column vector
        result = row_vector([[1], [2], [3]])
        assert result.shape == (1, 3)

    def test_skew_symmetric(self):
        """skew_symmetric should create correct matrix."""
        v = [1, 2, 3]
        S = skew_symmetric(v)

        # Should be skew-symmetric
        np.testing.assert_allclose(S, -S.T)

        # Cross product test: S @ u = v × u
        u = np.array([4, 5, 6])
        np.testing.assert_allclose(S @ u, np.cross(v, u))

    def test_unskew(self):
        """unskew should recover vector from skew-symmetric matrix."""
        v_original = np.array([1.5, -2.3, 4.7])
        S = skew_symmetric(v_original)
        v_recovered = unskew(S)
        np.testing.assert_allclose(v_recovered, v_original)

    def test_normalize_vector(self):
        """normalize_vector should create unit vectors."""
        v = np.array([3.0, 4.0])
        v_unit = normalize_vector(v)

        assert abs(np.linalg.norm(v_unit) - 1.0) < 1e-10
        np.testing.assert_allclose(v_unit, [0.6, 0.8])

        # With norm return
        v_unit, norm = normalize_vector([3, 4], return_norm=True)
        assert abs(norm - 5.0) < 1e-10

    def test_normalize_vector_zero(self):
        """normalize_vector should handle zero vectors."""
        v = np.array([0.0, 0.0, 0.0])
        v_unit = normalize_vector(v)
        np.testing.assert_array_equal(v_unit, [0, 0, 0])

    def test_vec_unvec(self):
        """vec and unvec should be inverses."""
        A = np.array([[1, 2], [3, 4], [5, 6]])

        # Column-major (MATLAB style)
        v = vec(A, order="F")
        assert v.shape == (6, 1)
        A_recovered = unvec(v, (3, 2), order="F")
        np.testing.assert_array_equal(A_recovered, A)

        # Row-major
        v = vec(A, order="C")
        A_recovered = unvec(v, (3, 2), order="C")
        np.testing.assert_array_equal(A_recovered, A)

    def test_block_diag(self):
        """block_diag should create block diagonal matrices."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5]])
        C = np.array([[6, 7, 8]])

        result = block_diag(A, B, C)

        expected = np.array(
            [
                [1, 2, 0, 0, 0, 0],
                [3, 4, 0, 0, 0, 0],
                [0, 0, 5, 0, 0, 0],
                [0, 0, 0, 6, 7, 8],
            ]
        )
        np.testing.assert_array_equal(result, expected)

    def test_is_positive_definite(self):
        """is_positive_definite should correctly identify PD matrices."""
        # Positive definite
        A = np.array([[4, 2], [2, 5]])
        assert is_positive_definite(A) is True

        # Not positive definite (negative eigenvalue)
        B = np.array([[1, 2], [2, 1]])
        assert is_positive_definite(B) is False

        # Not symmetric
        C = np.array([[1, 2], [3, 4]])
        assert is_positive_definite(C) is False

        # Identity is PD
        assert is_positive_definite(np.eye(5)) is True


class TestWrapConsistency:
    """Test consistency of wrap functions."""

    def test_wrap_round_trip(self):
        """Wrapping should be idempotent."""
        angles = np.linspace(-10, 10, 100)

        # wrap_to_pi should be idempotent
        wrapped_once = wrap_to_pi(angles)
        wrapped_twice = wrap_to_pi(wrapped_once)
        np.testing.assert_allclose(wrapped_once, wrapped_twice)

        # wrap_to_2pi should be idempotent
        wrapped_once = wrap_to_2pi(angles)
        wrapped_twice = wrap_to_2pi(wrapped_once)
        np.testing.assert_allclose(wrapped_once, wrapped_twice)

    def test_wrap_preserves_differences(self):
        """Angle differences should be preserved (mod 2π)."""
        a1, a2 = 0.5, 1.5

        # Original difference
        diff_original = a2 - a1

        # After adding multiples of 2π
        a1_shifted = a1 + 4 * np.pi
        a2_shifted = a2 - 2 * np.pi

        # Wrapped difference should be equivalent
        diff_wrapped = wrap_to_pi(wrap_to_pi(a2_shifted) - wrap_to_pi(a1_shifted))
        np.testing.assert_allclose(diff_wrapped, diff_original, atol=1e-10)
