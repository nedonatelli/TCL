"""
Basic matrix operations and constructions.

This module provides:
- Matrix decompositions (Cholesky, SVD-based, QR)
- Special matrix constructions (Vandermonde, Toeplitz, Hankel, etc.)
- Matrix vectorization operations (vec, unvec, Kronecker products)
"""

from pytcl.mathematical_functions.basic_matrix.decompositions import (  # noqa: E501
    chol_semi_def,
)
from pytcl.mathematical_functions.basic_matrix.decompositions import matrix_sqrt
from pytcl.mathematical_functions.basic_matrix.decompositions import null_space
from pytcl.mathematical_functions.basic_matrix.decompositions import pinv_truncated
from pytcl.mathematical_functions.basic_matrix.decompositions import range_space
from pytcl.mathematical_functions.basic_matrix.decompositions import rank_revealing_qr
from pytcl.mathematical_functions.basic_matrix.decompositions import tria
from pytcl.mathematical_functions.basic_matrix.decompositions import tria_sqrt
from pytcl.mathematical_functions.basic_matrix.special_matrices import (  # noqa: E501
    block_diag,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import circulant
from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    commutation_matrix,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import companion
from pytcl.mathematical_functions.basic_matrix.special_matrices import dft_matrix
from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    duplication_matrix,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import (
    elimination_matrix,
)
from pytcl.mathematical_functions.basic_matrix.special_matrices import hadamard
from pytcl.mathematical_functions.basic_matrix.special_matrices import hankel
from pytcl.mathematical_functions.basic_matrix.special_matrices import hilbert
from pytcl.mathematical_functions.basic_matrix.special_matrices import invhilbert
from pytcl.mathematical_functions.basic_matrix.special_matrices import kron
from pytcl.mathematical_functions.basic_matrix.special_matrices import toeplitz
from pytcl.mathematical_functions.basic_matrix.special_matrices import unvec
from pytcl.mathematical_functions.basic_matrix.special_matrices import vandermonde
from pytcl.mathematical_functions.basic_matrix.special_matrices import vec

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
