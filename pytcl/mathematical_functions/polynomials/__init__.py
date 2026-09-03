"""
Polynomial algorithms.

Simultaneous multivariate polynomial root finding via the Macaulay
null-space method.
"""

from pytcl.mathematical_functions.polynomials.multivariate import (
    PolyRootsResult,
    poly_roots_multi_dim,
)

__all__ = [
    "PolyRootsResult",
    "poly_roots_multi_dim",
]
