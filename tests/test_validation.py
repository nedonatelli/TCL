"""
Tests for the input validation framework.

Tests cover:
- validate_array function
- ArraySpec and ScalarSpec classes
- @validate_inputs decorator
- Matrix validation helpers (square, symmetric, positive definite)
"""

import numpy as np
import pytest

from pytcl.core.validation import (
    ArraySpec,
    ScalarSpec,
    ValidationError,
    check_compatible_shapes,
    ensure_2d,
    ensure_column_vector,
    ensure_positive_definite,
    ensure_row_vector,
    ensure_square_matrix,
    ensure_symmetric,
    validate_array,
    validate_inputs,
    validate_same_shape,
)


class TestValidateArray:
    """Tests for the validate_array function."""

    def test_basic_conversion(self):
        """Test basic list-to-array conversion."""
        result = validate_array([1, 2, 3], "test")
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_dtype_conversion(self):
        """Test dtype conversion."""
        result = validate_array([1, 2, 3], "test", dtype=np.float64)
        assert result.dtype == np.float64

    def test_ndim_check(self):
        """Test dimensionality validation."""
        # Should pass
        validate_array([1, 2, 3], "test", ndim=1)
        validate_array([[1, 2], [3, 4]], "test", ndim=2)

        # Should fail
        with pytest.raises(ValidationError, match="must have 2 dimension"):
            validate_array([1, 2, 3], "test", ndim=2)

    def test_ndim_tuple(self):
        """Test multiple valid dimensionalities."""
        # Both 1D and 2D should be valid
        validate_array([1, 2, 3], "test", ndim=(1, 2))
        validate_array([[1, 2]], "test", ndim=(1, 2))

    def test_min_max_ndim(self):
        """Test min/max dimensionality constraints."""
        validate_array([[1, 2]], "test", min_ndim=2)
        validate_array([[1, 2]], "test", max_ndim=2)

        with pytest.raises(ValidationError, match="at least 2"):
            validate_array([1, 2], "test", min_ndim=2)

        with pytest.raises(ValidationError, match="at most 2"):
            validate_array([[[1]]], "test", max_ndim=2)

    def test_shape_check(self):
        """Test shape validation."""
        validate_array([[1, 2, 3], [4, 5, 6]], "test", shape=(2, 3))
        validate_array([[1, 2], [3, 4]], "test", shape=(2, None))

        with pytest.raises(ValidationError, match="dimension 0 must be 3"):
            validate_array([[1, 2]], "test", shape=(3, 2))

    def test_finite_check(self):
        """Test finite values check."""
        validate_array([1, 2, 3], "test", finite=True)

        with pytest.raises(ValidationError, match="only finite"):
            validate_array([1, np.inf, 3], "test", finite=True)

        with pytest.raises(ValidationError, match="only finite"):
            validate_array([1, np.nan, 3], "test", finite=True)

    def test_non_negative_check(self):
        """Test non-negative check."""
        validate_array([0, 1, 2], "test", non_negative=True)

        with pytest.raises(ValidationError, match="non-negative"):
            validate_array([-1, 0, 1], "test", non_negative=True)

    def test_positive_check(self):
        """Test positive check."""
        validate_array([1, 2, 3], "test", positive=True)

        with pytest.raises(ValidationError, match="only positive"):
            validate_array([0, 1, 2], "test", positive=True)

    def test_empty_array(self):
        """Test empty array handling."""
        validate_array([], "test", allow_empty=True)

        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_array([], "test", allow_empty=False)


class TestEnsureFunctions:
    """Tests for ensure_* helper functions."""

    def test_ensure_2d_column(self):
        """Test ensure_2d with column axis."""
        result = ensure_2d([1, 2, 3], "test", axis="column")
        assert result.shape == (3, 1)

    def test_ensure_2d_row(self):
        """Test ensure_2d with row axis."""
        result = ensure_2d([1, 2, 3], "test", axis="row")
        assert result.shape == (1, 3)

    def test_ensure_column_vector(self):
        """Test ensure_column_vector."""
        result = ensure_column_vector([1, 2, 3])
        assert result.shape == (3, 1)

        # Already column vector
        result = ensure_column_vector([[1], [2], [3]])
        assert result.shape == (3, 1)

        # Invalid shape
        with pytest.raises(ValidationError, match="column vector"):
            ensure_column_vector([[1, 2], [3, 4]])

    def test_ensure_row_vector(self):
        """Test ensure_row_vector."""
        result = ensure_row_vector([1, 2, 3])
        assert result.shape == (1, 3)

        # Already row vector
        result = ensure_row_vector([[1, 2, 3]])
        assert result.shape == (1, 3)

        # Invalid shape
        with pytest.raises(ValidationError, match="row vector"):
            ensure_row_vector([[1, 2], [3, 4]])

    def test_ensure_square_matrix(self):
        """Test ensure_square_matrix."""
        result = ensure_square_matrix([[1, 2], [3, 4]])
        assert result.shape == (2, 2)

        with pytest.raises(ValidationError, match="square"):
            ensure_square_matrix([[1, 2, 3], [4, 5, 6]])

    def test_ensure_symmetric(self):
        """Test ensure_symmetric."""
        # Exactly symmetric
        result = ensure_symmetric([[1, 2], [2, 1]])
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, result.T)

        # Nearly symmetric (within tolerance)
        mat = [[1, 2.0000000001], [2, 1]]
        result = ensure_symmetric(mat)
        np.testing.assert_array_almost_equal(result, result.T)

        # Not symmetric
        with pytest.raises(ValidationError, match="symmetric"):
            ensure_symmetric([[1, 2], [3, 1]])

    def test_ensure_positive_definite(self):
        """Test ensure_positive_definite."""
        # Valid positive definite
        result = ensure_positive_definite([[2, 1], [1, 2]])
        assert result.shape == (2, 2)

        # Not positive definite
        with pytest.raises(ValidationError, match="positive definite"):
            ensure_positive_definite([[1, 2], [2, 1]])


class TestShapeChecks:
    """Tests for shape validation functions."""

    def test_validate_same_shape(self):
        """Test validate_same_shape."""
        # Same shapes - should pass
        validate_same_shape([1, 2, 3], [4, 5, 6], names=["a", "b"])

        # Different shapes - should fail
        with pytest.raises(ValidationError, match="same shape"):
            validate_same_shape([1, 2, 3], [1, 2], names=["a", "b"])

    def test_check_compatible_shapes(self):
        """Test check_compatible_shapes."""
        # Compatible (for matrix multiply)
        check_compatible_shapes((3, 4), (4, 5), names=["A", "B"])

        # Check specific dimension
        with pytest.raises(ValidationError, match="incompatible sizes"):
            check_compatible_shapes((3, 4), (5, 4), names=["A", "B"], dimension=0)


class TestArraySpec:
    """Tests for the ArraySpec class."""

    def test_basic_validation(self):
        """Test basic ArraySpec validation."""
        spec = ArraySpec(ndim=2, finite=True)
        result = spec.validate([[1, 2], [3, 4]], "matrix")
        assert result.shape == (2, 2)

    def test_square_spec(self):
        """Test ArraySpec with square constraint."""
        spec = ArraySpec(ndim=2, square=True)
        result = spec.validate([[1, 2], [3, 4]], "matrix")
        assert result.shape == (2, 2)

        with pytest.raises(ValidationError):
            spec.validate([[1, 2, 3]], "matrix")

    def test_symmetric_spec(self):
        """Test ArraySpec with symmetric constraint."""
        spec = ArraySpec(ndim=2, symmetric=True)
        result = spec.validate([[1, 2], [2, 1]], "matrix")
        np.testing.assert_array_equal(result, result.T)

    def test_positive_definite_spec(self):
        """Test ArraySpec with positive definite constraint."""
        spec = ArraySpec(ndim=2, positive_definite=True)
        result = spec.validate([[2, 1], [1, 2]], "matrix")
        assert np.all(np.linalg.eigvalsh(result) > 0)


class TestScalarSpec:
    """Tests for the ScalarSpec class."""

    def test_dtype_validation(self):
        """Test ScalarSpec dtype validation."""
        spec = ScalarSpec(dtype=int)
        result = spec.validate(5, "k")
        assert isinstance(result, int)

        # Should convert float to int
        result = spec.validate(5.0, "k")
        assert isinstance(result, int)

    def test_range_validation(self):
        """Test ScalarSpec range validation."""
        spec = ScalarSpec(min_value=1, max_value=10)

        result = spec.validate(5, "k")
        assert result == 5

        with pytest.raises(ValidationError, match=">= 1"):
            spec.validate(0, "k")

        with pytest.raises(ValidationError, match="<= 10"):
            spec.validate(11, "k")

    def test_positive_check(self):
        """Test ScalarSpec positive constraint."""
        spec = ScalarSpec(positive=True)
        spec.validate(1, "k")

        with pytest.raises(ValidationError, match="positive"):
            spec.validate(0, "k")

        with pytest.raises(ValidationError, match="positive"):
            spec.validate(-1, "k")

    def test_non_negative_check(self):
        """Test ScalarSpec non-negative constraint."""
        spec = ScalarSpec(non_negative=True)
        spec.validate(0, "k")
        spec.validate(1, "k")

        with pytest.raises(ValidationError, match="non-negative"):
            spec.validate(-1, "k")

    def test_finite_check(self):
        """Test ScalarSpec finite constraint."""
        spec = ScalarSpec(finite=True)
        spec.validate(1.0, "x")

        with pytest.raises(ValidationError, match="finite"):
            spec.validate(np.inf, "x")


class TestValidateInputsDecorator:
    """Tests for the @validate_inputs decorator."""

    def test_basic_usage(self):
        """Test basic decorator usage with ArraySpec."""

        @validate_inputs(x=ArraySpec(ndim=1, finite=True))
        def sum_array(x):
            return np.sum(x)

        result = sum_array([1, 2, 3])
        assert result == 6

        with pytest.raises(ValidationError, match="must have 1"):
            sum_array([[1, 2], [3, 4]])

    def test_multiple_params(self):
        """Test decorator with multiple parameters."""

        @validate_inputs(
            x=ArraySpec(ndim=1),
            k=ScalarSpec(dtype=int, min_value=1),
        )
        def get_first_k(x, k):
            return x[:k]

        result = get_first_k([1, 2, 3, 4, 5], 3)
        np.testing.assert_array_equal(result, [1, 2, 3])

        with pytest.raises(ValidationError, match=">= 1"):
            get_first_k([1, 2, 3], 0)

    def test_dict_shorthand(self):
        """Test decorator with dict shorthand."""

        @validate_inputs(x={"ndim": 1, "finite": True})
        def sum_array(x):
            return np.sum(x)

        result = sum_array([1, 2, 3])
        assert result == 6

    def test_positive_definite(self):
        """Test decorator with positive definite matrix."""

        @validate_inputs(P=ArraySpec(ndim=2, positive_definite=True))
        def get_eigenvalues(P):
            return np.linalg.eigvalsh(P)

        result = get_eigenvalues([[2, 1], [1, 2]])
        assert np.all(result > 0)

        with pytest.raises(ValidationError, match="positive definite"):
            get_eigenvalues([[1, 2], [2, 1]])

    def test_preserves_function_metadata(self):
        """Test that decorator preserves function metadata."""

        @validate_inputs(x=ArraySpec(ndim=1))
        def my_function(x):
            """My docstring."""
            return x

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_with_default_args(self):
        """Test decorator with default arguments."""

        @validate_inputs(
            x=ArraySpec(ndim=1),
            k=ScalarSpec(dtype=int, min_value=1),
        )
        def get_first_k(x, k=5):
            return x[:k]

        # Using default k
        result = get_first_k([1, 2, 3, 4, 5, 6, 7])
        np.testing.assert_array_equal(result, [1, 2, 3, 4, 5])

        # Overriding k
        result = get_first_k([1, 2, 3, 4, 5, 6, 7], k=3)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_kwargs_only(self):
        """Test decorator with keyword-only arguments."""

        @validate_inputs(data=ArraySpec(ndim=2, finite=True))
        def process(data):
            return data.sum()

        result = process(data=[[1, 2], [3, 4]])
        assert result == 10


class TestKalmanFilterValidation:
    """Integration tests for Kalman filter-style validation."""

    def test_kalman_predict_validation(self):
        """Test validation pattern for Kalman filter predict."""

        @validate_inputs(
            x=ArraySpec(ndim=1, finite=True),
            P=ArraySpec(ndim=2, positive_definite=True),
            F=ArraySpec(ndim=2, square=True),
            Q=ArraySpec(ndim=2, symmetric=True, non_negative=False),
        )
        def kf_predict(x, P, F, Q):
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            return x_pred, P_pred

        # Valid inputs
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        F = np.eye(2)
        Q = np.eye(2) * 0.01

        x_pred, P_pred = kf_predict(x, P, F, Q)
        assert x_pred.shape == (2,)
        assert P_pred.shape == (2, 2)

        # Invalid covariance (not positive definite)
        with pytest.raises(ValidationError, match="positive definite"):
            kf_predict(x, np.array([[1, 2], [2, 1]]), F, Q)

    def test_kalman_update_validation(self):
        """Test validation pattern for Kalman filter update."""

        @validate_inputs(
            x=ArraySpec(ndim=1, finite=True),
            P=ArraySpec(ndim=2, positive_definite=True),
            z=ArraySpec(ndim=1, finite=True),
            H=ArraySpec(ndim=2, finite=True),
            R=ArraySpec(ndim=2, positive_definite=True),
        )
        def kf_update(x, P, z, H, R):
            y = z - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x_upd = x + K @ y
            P_upd = (np.eye(len(x)) - K @ H) @ P
            return x_upd, P_upd

        # Valid inputs
        x = np.array([1.0, 2.0])
        P = np.eye(2) * 0.1
        z = np.array([1.1])
        H = np.array([[1, 0]])
        R = np.array([[0.1]])

        x_upd, P_upd = kf_update(x, P, z, H, R)
        assert x_upd.shape == (2,)
        assert P_upd.shape == (2, 2)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_spec_type(self):
        """Test error handling for invalid spec type."""

        @validate_inputs(x="invalid")  # type: ignore
        def bad_func(x):
            return x

        with pytest.raises(TypeError, match="Invalid spec type"):
            bad_func([1, 2, 3])

    def test_none_value(self):
        """Test handling of None values."""

        @validate_inputs(x=ArraySpec(ndim=1))
        def process(x, y=None):
            return x

        # Should work - y is not validated
        result = process([1, 2, 3], y=None)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_empty_specs(self):
        """Test decorator with no specs."""

        @validate_inputs()
        def identity(x):
            return x

        result = identity([1, 2, 3])
        assert result == [1, 2, 3]
