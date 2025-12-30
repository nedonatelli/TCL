"""
Basic matrix operations and constructions.

This module provides:
- Matrix decompositions (Cholesky, SVD-based, QR)
- Special matrix constructions (Vandermonde, Toeplitz, Hankel, etc.)
- Matrix vectorization operations (vec, unvec, Kronecker products)
"""

from pytcl.mathematical_functions.basic_matrix.decompositions import (  # noqa: E501
    chol_semi_def,
    tria,
    tria_sqrt,
    pinv_truncated,
    matrix_sqrt,
    rank_revealing_qr,
    null_space,
    range_space,
)

from pytcl.mathematical_functions.basic_matrix.special_matrices import (  # noqa: E501
    vandermonde,
    toeplitz,
    hankel,
    circulant,
    block_diag,
    companion,
    hilbert,
    invhilbert,
    hadamard,
    dft_matrix,
    kron,
    vec,
    unvec,
    commutation_matrix,
    duplication_matrix,
    elimination_matrix,
)

__all__ = [
    # Decompositions
    "chol_semi_def",
    "tria",
    "tria_sqrt",
    "pinv_truncated",
    "matrix_sqrt",
    "rank_revealing_qr",
    "null_space",
    "range_space",
    # Special matrices
    "vandermonde",
    "toeplitz",
    "hankel",
    "circulant",
    "block_diag",
    "companion",
    "hilbert",
    "invhilbert",
    "hadamard",
    "dft_matrix",
    "kron",
    "vec",
    "unvec",
    "commutation_matrix",
    "duplication_matrix",
    "elimination_matrix",
]
