"""
Combinatorics utilities.

This module provides:
- Permutation and combination generation
- Permutation ranking/unranking
- Integer partitions
- Combinatorial numbers (Stirling, Bell, Catalan)
"""

from pytcl.mathematical_functions.combinatorics.combinatorics import (  # noqa: E501
    factorial,
    n_choose_k,
    n_permute_k,
    permutations,
    combinations,
    combinations_with_replacement,
    permutation_rank,
    permutation_unrank,
    next_permutation,
    partition_count,
    partitions,
    multinomial_coefficient,
    stirling_second,
    bell_number,
    catalan_number,
    derangements_count,
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
