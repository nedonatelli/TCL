"""
Tests for special matrix constructors.

Tests cover:
- vec/unvec operations
- Kronecker product
- Block diagonal matrix
- Vandermonde, Toeplitz, Hankel, Circulant matrices
- Hilbert, Hadamard matrices
- Commutation, Duplication, Elimination matrices
"""

import numpy as np

from pytcl.mathematical_functions import (
    block_diag,
    kron,
    unvec,
    vec,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    circulant,
    commutation_matrix,
    duplication_matrix,
    elimination_matrix,
    hadamard,
    hankel,
    hilbert,
    toeplitz,
    vandermonde,
)


class TestVecUnvec:
    """Tests for vec and unvec operations."""

    def test_vec_basic(self):
        """Test basic vec operation."""
        A = np.array([[1, 2], [3, 4]])
        v = vec(A)
        # Column-major order: columns stacked
        expected = np.array([1, 3, 2, 4])
        np.testing.assert_array_equal(v, expected)

    def test_unvec_basic(self):
        """Test basic unvec operation."""
        v = np.array([1, 3, 2, 4])
        A = unvec(v, 2, 2)
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(A, expected)

    def test_vec_unvec_roundtrip(self):
        """Test vec and unvec are inverses."""
        A = np.random.randn(3, 4)
        v = vec(A)
        A_recovered = unvec(v, 3, 4)
        np.testing.assert_allclose(A_recovered, A)

    def test_unvec_vec_roundtrip(self):
        """Test unvec and vec are inverses."""
        v = np.random.randn(12)
        A = unvec(v, 3, 4)
        v_recovered = vec(A)
        np.testing.assert_allclose(v_recovered, v)


class TestKronecker:
    """Tests for Kronecker product."""

    def test_kron_basic(self):
        """Test basic Kronecker product."""
        A = np.array([[1, 2], [3, 4]])
        B = np.eye(2)
        K = kron(A, B)

        expected = np.array(
            [
                [1, 0, 2, 0],
                [0, 1, 0, 2],
                [3, 0, 4, 0],
                [0, 3, 0, 4],
            ]
        )
        np.testing.assert_allclose(K, expected)

    def test_kron_shape(self):
        """Test Kronecker product shape."""
        A = np.random.randn(2, 3)
        B = np.random.randn(4, 5)
        K = kron(A, B)
        assert K.shape == (8, 15)

    def test_kron_associativity(self):
        """Test Kronecker product is associative."""
        A = np.random.randn(2, 2)
        B = np.random.randn(2, 2)
        C = np.random.randn(2, 2)

        K1 = kron(kron(A, B), C)
        K2 = kron(A, kron(B, C))
        np.testing.assert_allclose(K1, K2)


class TestBlockDiag:
    """Tests for block diagonal matrix."""

    def test_block_diag_basic(self):
        """Test basic block diagonal construction."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5]])
        D = block_diag(A, B)

        expected = np.array(
            [
                [1, 2, 0],
                [3, 4, 0],
                [0, 0, 5],
            ]
        )
        np.testing.assert_allclose(D, expected)

    def test_block_diag_shape(self):
        """Test block diagonal shape."""
        A = np.random.randn(2, 3)
        B = np.random.randn(4, 5)
        D = block_diag(A, B)
        assert D.shape == (6, 8)


class TestVandermonde:
    """Tests for Vandermonde matrix."""

    def test_vandermonde_basic(self):
        """Test basic Vandermonde matrix."""
        V = vandermonde([1, 2, 3], 3)
        expected = np.array(
            [
                [1, 1, 1],
                [4, 2, 1],
                [9, 3, 1],
            ]
        )
        np.testing.assert_allclose(V, expected)

    def test_vandermonde_increasing(self):
        """Test Vandermonde with increasing powers."""
        V = vandermonde([1, 2, 3], 3, increasing=True)
        expected = np.array(
            [
                [1, 1, 1],
                [1, 2, 4],
                [1, 3, 9],
            ]
        )
        np.testing.assert_allclose(V, expected)


class TestToeplitz:
    """Tests for Toeplitz matrix."""

    def test_toeplitz_basic(self):
        """Test basic Toeplitz matrix."""
        T = toeplitz([1, 2, 3], [1, 4, 5])
        expected = np.array(
            [
                [1, 4, 5],
                [2, 1, 4],
                [3, 2, 1],
            ]
        )
        np.testing.assert_allclose(T, expected)


class TestHankel:
    """Tests for Hankel matrix."""

    def test_hankel_basic(self):
        """Test basic Hankel matrix."""
        H = hankel([1, 2, 3], [3, 4, 5])
        expected = np.array(
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
            ]
        )
        np.testing.assert_allclose(H, expected)


class TestCirculant:
    """Tests for circulant matrix."""

    def test_circulant_basic(self):
        """Test basic circulant matrix."""
        C = circulant([1, 2, 3])
        expected = np.array(
            [
                [1, 3, 2],
                [2, 1, 3],
                [3, 2, 1],
            ]
        )
        np.testing.assert_allclose(C, expected)


class TestHilbert:
    """Tests for Hilbert matrix."""

    def test_hilbert_basic(self):
        """Test basic Hilbert matrix."""
        H = hilbert(3)
        expected = np.array(
            [
                [1, 1 / 2, 1 / 3],
                [1 / 2, 1 / 3, 1 / 4],
                [1 / 3, 1 / 4, 1 / 5],
            ]
        )
        np.testing.assert_allclose(H, expected)

    def test_hilbert_ill_conditioned(self):
        """Test that Hilbert matrix is ill-conditioned."""
        H = hilbert(10)
        cond = np.linalg.cond(H)
        assert cond > 1e10  # Very ill-conditioned


class TestHadamard:
    """Tests for Hadamard matrix."""

    def test_hadamard_basic(self):
        """Test basic Hadamard matrix."""
        H = hadamard(4)
        assert H.shape == (4, 4)
        # All entries should be +1 or -1
        assert np.all(np.abs(H) == 1)

    def test_hadamard_orthogonality(self):
        """Test Hadamard orthogonality property."""
        n = 8
        H = hadamard(n)
        # H @ H.T = n * I
        np.testing.assert_allclose(H @ H.T, n * np.eye(n))


class TestCommutationMatrix:
    """Tests for commutation matrix."""

    def test_commutation_property(self):
        """Test K @ vec(A) = vec(A.T)."""
        m, n = 2, 3
        K = commutation_matrix(m, n)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(K @ vec(A), vec(A.T))


class TestDuplicationMatrix:
    """Tests for duplication matrix."""

    def test_duplication_shape(self):
        """Test duplication matrix shape."""
        n = 3
        D = duplication_matrix(n)
        assert D.shape == (n * n, n * (n + 1) // 2)


class TestEliminationMatrix:
    """Tests for elimination matrix."""

    def test_elimination_shape(self):
        """Test elimination matrix shape."""
        n = 3
        L = elimination_matrix(n)
        assert L.shape == (n * (n + 1) // 2, n * n)
