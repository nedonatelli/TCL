"""
Tests for matrix decomposition functions.

Tests cover:
- chol_semi_def (semi-definite Cholesky)
- tria (triangular square root)
- tria_sqrt (triangular square root of matrix products)
- pinv_truncated (truncated pseudo-inverse)
- matrix_sqrt (principal matrix square root)
- rank_revealing_qr
- null_space
- range_space
"""

import numpy as np
import pytest

from pytcl.mathematical_functions.basic_matrix.decompositions import (
    chol_semi_def,
    matrix_sqrt,
    null_space,
    pinv_truncated,
    range_space,
    rank_revealing_qr,
    tria,
    tria_sqrt,
)


class TestCholSemiDef:
    """Tests for semi-definite Cholesky decomposition."""

    def test_positive_definite(self):
        """Test with positive definite matrix."""
        A = np.array([[4, 2], [2, 3]])
        L = chol_semi_def(A)
        np.testing.assert_allclose(L @ L.T, A, rtol=1e-10)

    def test_positive_semidefinite(self):
        """Test with positive semi-definite (nearly singular) matrix."""
        A = np.array([[4, 2, 0], [2, 1.0001, 0], [0, 0, 1]])
        L = chol_semi_def(A)
        reconstructed = L @ L.T
        np.testing.assert_allclose(reconstructed, A, rtol=1e-3)

    def test_upper_triangular(self):
        """Test upper triangular output."""
        A = np.array([[4, 2], [2, 3]])
        R = chol_semi_def(A, upper=True)
        np.testing.assert_allclose(R.T @ R, A, rtol=1e-10)

    def test_non_square_raises(self):
        """Test non-square matrix raises error."""
        with pytest.raises(ValueError):
            chol_semi_def(np.array([[1, 2, 3], [4, 5, 6]]))


class TestTria:
    """Tests for triangular square root."""

    def test_tria_basic(self):
        """Test basic triangular factor."""
        A = np.array([[4, 2], [2, 3]])
        S = tria(A)
        np.testing.assert_allclose(S @ S.T, A, rtol=1e-10)
        assert np.allclose(S, np.tril(S))


class TestTriaSqrt:
    """Tests for triangular square root of matrix products."""

    def test_single_matrix(self):
        """Test with single matrix."""
        A = np.random.randn(3, 4)
        S = tria_sqrt(A)
        np.testing.assert_allclose(S @ S.T, A @ A.T, rtol=1e-10)

    def test_two_matrices(self):
        """Test with two matrices."""
        A = np.random.randn(3, 4)
        B = np.random.randn(3, 2)
        S = tria_sqrt(A, B)
        expected = A @ A.T + B @ B.T
        np.testing.assert_allclose(S @ S.T, expected, rtol=1e-10)

    def test_shape_mismatch_raises(self):
        """Test row count mismatch raises error."""
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)
        with pytest.raises(ValueError):
            tria_sqrt(A, B)


class TestPinvTruncated:
    """Tests for truncated pseudo-inverse."""

    def test_full_rank(self):
        """Test with full rank matrix."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        A_pinv = pinv_truncated(A)
        np.testing.assert_allclose(A @ A_pinv @ A, A, rtol=1e-10)

    def test_rank_truncation(self):
        """Test with explicit rank truncation."""
        A = np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9]])
        A_pinv = pinv_truncated(A, rank=1)
        assert A_pinv.shape == (3, 3)

    def test_tolerance_truncation(self):
        """Test with tolerance-based truncation."""
        A = np.diag([1, 0.1, 0.001])
        A_pinv = pinv_truncated(A, tol=0.01)
        assert A_pinv.shape == (3, 3)


class TestMatrixSqrt:
    """Tests for principal matrix square root."""

    def test_diagonal_matrix(self):
        """Test with diagonal matrix."""
        A = np.diag([4, 9, 16])
        S = matrix_sqrt(A, method="schur")
        np.testing.assert_allclose(S @ S, A, rtol=1e-10)
        np.testing.assert_allclose(np.diag(S), [2, 3, 4], rtol=1e-10)

    def test_eigenvalue_method(self):
        """Test eigenvalue-based method."""
        A = np.array([[4, 0], [0, 9]])
        S = matrix_sqrt(A, method="eigenvalue")
        np.testing.assert_allclose(S @ S, A, rtol=1e-10)

    def test_denman_beavers_method(self):
        """Test Denman-Beavers iterative method."""
        A = np.array([[4, 0], [0, 9]])
        S = matrix_sqrt(A, method="denman_beavers")
        np.testing.assert_allclose(S @ S, A, rtol=1e-6)

    def test_invalid_method_raises(self):
        """Test invalid method raises error."""
        with pytest.raises(ValueError):
            matrix_sqrt(np.eye(2), method="invalid")

    def test_non_square_raises(self):
        """Test non-square matrix raises error."""
        with pytest.raises(ValueError):
            matrix_sqrt(np.array([[1, 2, 3], [4, 5, 6]]))


class TestRankRevealingQR:
    """Tests for rank-revealing QR decomposition."""

    def test_full_rank(self):
        """Test with full rank matrix."""
        A = np.random.randn(4, 3)
        Q, R, P, rank = rank_revealing_qr(A)
        assert rank == 3
        np.testing.assert_allclose(A[:, P], Q @ R, rtol=1e-10)

    def test_rank_deficient(self):
        """Test with rank-deficient matrix."""
        A = np.array([[1, 2, 3], [2, 4, 6], [1, 1, 1], [2, 2, 2]])
        Q, R, P, rank = rank_revealing_qr(A)
        assert rank == 2


class TestNullSpace:
    """Tests for null space computation."""

    def test_null_space_basic(self):
        """Test basic null space."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        N = null_space(A)
        assert N.shape[1] == 1
        np.testing.assert_allclose(A @ N, 0, atol=1e-10)

    def test_full_rank_empty_nullspace(self):
        """Test full rank matrix has empty null space."""
        A = np.eye(3)
        N = null_space(A)
        assert N.shape[1] == 0


class TestRangeSpace:
    """Tests for range space computation."""

    def test_range_space_basic(self):
        """Test basic range space."""
        A = np.array([[1, 2], [2, 4], [3, 6]])
        R = range_space(A)
        assert R.shape == (3, 1)
        np.testing.assert_allclose(R.T @ R, np.eye(1), rtol=1e-10)
