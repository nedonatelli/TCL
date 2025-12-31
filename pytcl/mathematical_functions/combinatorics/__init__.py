"""
Combinatorics utilities.

This module provides:
- Permutation and combination generation
- Permutation ranking/unranking
- Integer partitions
- Combinatorial numbers (Stirling, Bell, Catalan)
"""

from pytcl.mathematical_functions.combinatorics.combinatorics import (  # noqa: E501
    bell_number,
    catalan_number,
    combinations,
    combinations_with_replacement,
    derangements_count,
    factorial,
    multinomial_coefficient,
    n_choose_k,
    n_permute_k,
    next_permutation,
    partition_count,
    partitions,
    permutation_rank,
    permutation_unrank,
    permutations,
    stirling_second,
    subfactorial,
)

__all__ = [
    "factorial",
    "n_choose_k",
    "n_permute_k",
    "permutations",
    "combinations",
    "combinations_with_replacement",
    "permutation_rank",
    "permutation_unrank",
    "next_permutation",
    "partition_count",
    "partitions",
    "multinomial_coefficient",
    "stirling_second",
    "bell_number",
    "catalan_number",
    "derangements_count",
    "subfactorial",
]
