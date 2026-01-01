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
)
from pytcl.mathematical_functions.combinatorics.combinatorics import catalan_number
from pytcl.mathematical_functions.combinatorics.combinatorics import combinations
from pytcl.mathematical_functions.combinatorics.combinatorics import (
    combinations_with_replacement,
)
from pytcl.mathematical_functions.combinatorics.combinatorics import derangements_count
from pytcl.mathematical_functions.combinatorics.combinatorics import factorial
from pytcl.mathematical_functions.combinatorics.combinatorics import (
    multinomial_coefficient,
)
from pytcl.mathematical_functions.combinatorics.combinatorics import n_choose_k
from pytcl.mathematical_functions.combinatorics.combinatorics import n_permute_k
from pytcl.mathematical_functions.combinatorics.combinatorics import next_permutation
from pytcl.mathematical_functions.combinatorics.combinatorics import partition_count
from pytcl.mathematical_functions.combinatorics.combinatorics import partitions
from pytcl.mathematical_functions.combinatorics.combinatorics import permutation_rank
from pytcl.mathematical_functions.combinatorics.combinatorics import permutation_unrank
from pytcl.mathematical_functions.combinatorics.combinatorics import permutations
from pytcl.mathematical_functions.combinatorics.combinatorics import stirling_second
from pytcl.mathematical_functions.combinatorics.combinatorics import subfactorial

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
