"""
Combinatorics utilities.

This module provides functions for permutations, combinations, and
related operations commonly used in assignment problems and data association.
"""

import itertools
from functools import lru_cache
from typing import Any, Iterator, List, Optional, Tuple

from numpy.typing import ArrayLike


def factorial(n: int) -> int:
    """
    Compute factorial of n.

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    n! : int
        Factorial of n.

    Examples
    --------
    >>> factorial(5)
    120
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def n_choose_k(n: int, k: int) -> int:
    """
    Compute binomial coefficient C(n, k).

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items to choose.

    Returns
    -------
    C(n, k) : int
        Number of ways to choose k items from n.

    Examples
    --------
    >>> n_choose_k(5, 2)
    10
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1

    # Use symmetry to minimize iterations
    k = min(k, n - k)

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def n_permute_k(n: int, k: int) -> int:
    """
    Compute number of k-permutations of n items.

    Parameters
    ----------
    n : int
        Total number of items.
    k : int
        Number of items in each permutation.

    Returns
    -------
    P(n, k) : int
        Number of k-permutations: n! / (n-k)!

    Examples
    --------
    >>> n_permute_k(5, 2)
    20
    """
    if k < 0 or k > n:
        return 0
    result = 1
    for i in range(k):
        result *= n - i
    return result


def permutations(
    items: ArrayLike,
    k: Optional[int] = None,
) -> Iterator[tuple[Any, ...]]:
    """
    Generate all k-permutations of items.

    Parameters
    ----------
    items : array_like
        Items to permute.
    k : int, optional
        Length of permutations. Default is len(items).

    Yields
    ------
    perm : tuple
        Each k-permutation of items.

    Examples
    --------
    >>> list(permutations([1, 2, 3], 2))
    [(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
    """
    items = list(items)
    return itertools.permutations(items, k)


def combinations(
    items: ArrayLike,
    k: int,
) -> Iterator[tuple[Any, ...]]:
    """
    Generate all k-combinations of items.

    Parameters
    ----------
    items : array_like
        Items to combine.
    k : int
        Size of each combination.

    Yields
    ------
    comb : tuple
        Each k-combination of items.

    Examples
    --------
    >>> list(combinations([1, 2, 3, 4], 2))
    [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]
    """
    items = list(items)
    return itertools.combinations(items, k)


def combinations_with_replacement(
    items: ArrayLike,
    k: int,
) -> Iterator[tuple[Any, ...]]:
    """
    Generate all k-combinations with replacement.

    Parameters
    ----------
    items : array_like
        Items to combine.
    k : int
        Size of each combination.

    Yields
    ------
    comb : tuple
        Each k-combination with replacement.

    Examples
    --------
    >>> list(combinations_with_replacement([1, 2], 2))
    [(1, 1), (1, 2), (2, 2)]
    """
    items = list(items)
    return itertools.combinations_with_replacement(items, k)


def permutation_rank(perm: ArrayLike) -> int:
    """
    Compute the lexicographic rank of a permutation.

    The rank is the zero-based index of the permutation in the
    lexicographically sorted list of all permutations.

    Parameters
    ----------
    perm : array_like
        Permutation of integers 0, 1, ..., n-1.

    Returns
    -------
    rank : int
        Lexicographic rank (0-indexed).

    Examples
    --------
    >>> permutation_rank([0, 1, 2])  # First permutation
    0
    >>> permutation_rank([2, 1, 0])  # Last permutation
    5
    """
    perm = list(perm)
    n = len(perm)
    rank = 0
    available = list(range(n))

    for i in range(n):
        pos = available.index(perm[i])
        rank += pos * factorial(n - 1 - i)
        available.remove(perm[i])

    return rank


def permutation_unrank(rank: int, n: int) -> List[int]:
    """
    Compute the permutation with a given lexicographic rank.

    Parameters
    ----------
    rank : int
        Lexicographic rank (0-indexed).
    n : int
        Length of the permutation.

    Returns
    -------
    perm : list
        Permutation of [0, 1, ..., n-1] with the given rank.

    Examples
    --------
    >>> permutation_unrank(0, 3)
    [0, 1, 2]
    >>> permutation_unrank(5, 3)
    [2, 1, 0]
    """
    if rank < 0 or rank >= factorial(n):
        raise ValueError(f"Rank must be in [0, {factorial(n) - 1}]")

    available = list(range(n))
    perm = []

    for i in range(n):
        divisor = factorial(n - 1 - i)
        idx, rank = divmod(rank, divisor)
        perm.append(available.pop(idx))

    return perm


def next_permutation(perm: ArrayLike) -> Optional[List[Any]]:
    """
    Generate the next permutation in lexicographic order.

    Parameters
    ----------
    perm : array_like
        Current permutation.

    Returns
    -------
    next_perm : list or None
        Next permutation, or None if perm is the last permutation.

    Examples
    --------
    >>> next_permutation([1, 2, 3])
    [1, 3, 2]
    >>> next_permutation([3, 2, 1])  # Last permutation
    None
    """
    perm = list(perm)
    n = len(perm)

    # Find largest i such that perm[i] < perm[i+1]
    i = n - 2
    while i >= 0 and perm[i] >= perm[i + 1]:
        i -= 1

    if i < 0:
        return None  # Last permutation

    # Find largest j such that perm[i] < perm[j]
    j = n - 1
    while perm[i] >= perm[j]:
        j -= 1

    # Swap and reverse
    perm[i], perm[j] = perm[j], perm[i]
    perm[i + 1 :] = reversed(perm[i + 1 :])

    return perm


def partition_count(n: int, k: Optional[int] = None) -> int:
    """
    Count the number of integer partitions of n.

    A partition of n is a way of writing n as a sum of positive integers,
    where order doesn't matter.

    Parameters
    ----------
    n : int
        Number to partition.
    k : int, optional
        If specified, count only partitions with exactly k parts.

    Returns
    -------
    count : int
        Number of partitions.

    Examples
    --------
    >>> partition_count(5)  # 5 = 5 = 4+1 = 3+2 = 3+1+1 = 2+2+1 = 2+1+1+1 = 1+1+1+1+1
    7
    >>> partition_count(5, 2)  # 5 = 4+1 = 3+2
    2
    """

    @lru_cache(maxsize=None)
    def p(n: int, max_val: int) -> int:
        if n == 0:
            return 1
        if n < 0 or max_val == 0:
            return 0
        return p(n - max_val, max_val) + p(n, max_val - 1)

    @lru_cache(maxsize=None)
    def pk(n: int, k: int, max_val: int) -> int:
        if k == 0:
            return 1 if n == 0 else 0
        if n <= 0 or max_val == 0:
            return 0
        return pk(n - max_val, k - 1, max_val) + pk(n, k, max_val - 1)

    if k is None:
        return p(n, n)
    else:
        return pk(n, k, n)


def partitions(n: int, k: Optional[int] = None) -> Iterator[Tuple[int, ...]]:
    """
    Generate all integer partitions of n.

    Parameters
    ----------
    n : int
        Number to partition.
    k : int, optional
        If specified, generate only partitions with exactly k parts.

    Yields
    ------
    partition : tuple
        Each partition as a tuple of integers in descending order.

    Examples
    --------
    >>> list(partitions(4))
    [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
    """

    def gen_partitions(
        n: int, max_val: int, prefix: Tuple[int, ...]
    ) -> Iterator[Tuple[int, ...]]:
        if n == 0:
            yield prefix
            return
        for i in range(min(n, max_val), 0, -1):
            yield from gen_partitions(n - i, i, prefix + (i,))

    def gen_partitions_k(
        n: int, k: int, max_val: int, prefix: Tuple[int, ...]
    ) -> Iterator[Tuple[int, ...]]:
        if k == 0:
            if n == 0:
                yield prefix
            return
        if n <= 0 or max_val == 0:
            return
        for i in range(min(n, max_val), 0, -1):
            yield from gen_partitions_k(n - i, k - 1, i, prefix + (i,))

    if k is None:
        yield from gen_partitions(n, n, ())
    else:
        yield from gen_partitions_k(n, k, n, ())


def multinomial_coefficient(*args: int) -> int:
    """
    Compute multinomial coefficient.

    multinomial(n1, n2, ..., nk) = (n1 + n2 + ... + nk)! / (n1! * n2! * ... * nk!)

    Parameters
    ----------
    *args : int
        Non-negative integers.

    Returns
    -------
    coeff : int
        Multinomial coefficient.

    Examples
    --------
    >>> multinomial_coefficient(2, 3, 1)  # 6! / (2! * 3! * 1!)
    60
    """
    n = sum(args)
    result = factorial(n)
    for k in args:
        result //= factorial(k)
    return result


def stirling_second(n: int, k: int) -> int:
    """
    Stirling number of the second kind.

    S(n, k) is the number of ways to partition n elements into k
    non-empty subsets.

    Parameters
    ----------
    n : int
        Number of elements.
    k : int
        Number of subsets.

    Returns
    -------
    S(n, k) : int
        Stirling number of the second kind.

    Examples
    --------
    >>> stirling_second(4, 2)  # {{1,2,3},{4}}, {{1,2,4},{3}}, ...
    7
    """
    if n == 0 and k == 0:
        return 1
    if n == 0 or k == 0:
        return 0
    if k > n:
        return 0

    @lru_cache(maxsize=None)
    def S(n: int, k: int) -> int:
        if n == k:
            return 1
        if k == 1:
            return 1
        return k * S(n - 1, k) + S(n - 1, k - 1)

    return S(n, k)


def bell_number(n: int) -> int:
    """
    Bell number B_n.

    B_n is the number of ways to partition a set of n elements.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    B_n : int
        n-th Bell number.

    Examples
    --------
    >>> bell_number(4)
    15
    """
    return sum(stirling_second(n, k) for k in range(n + 1))


def catalan_number(n: int) -> int:
    """
    Catalan number C_n.

    Catalan numbers count many combinatorial structures including:
    - Valid parenthesizations
    - Full binary trees with n+1 leaves
    - Triangulations of a polygon with n+2 sides

    Parameters
    ----------
    n : int
        Non-negative integer.

    Returns
    -------
    C_n : int
        n-th Catalan number.

    Examples
    --------
    >>> catalan_number(5)
    42
    """
    return n_choose_k(2 * n, n) // (n + 1)


def derangements_count(n: int) -> int:
    """
    Count the number of derangements.

    A derangement is a permutation with no fixed points.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    D_n : int
        Number of derangements.

    Examples
    --------
    >>> derangements_count(4)  # {2,1,4,3}, {2,3,4,1}, ...
    9
    """
    if n == 0:
        return 1
    if n == 1:
        return 0

    # Use recurrence: D(n) = (n-1) * (D(n-1) + D(n-2))
    d_prev2 = 1  # D(0)
    d_prev1 = 0  # D(1)

    for i in range(2, n + 1):
        d_curr = (i - 1) * (d_prev1 + d_prev2)
        d_prev2 = d_prev1
        d_prev1 = d_curr

    return d_prev1


def subfactorial(n: int) -> int:
    """
    Subfactorial (number of derangements).

    Alias for derangements_count.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    !n : int
        Subfactorial of n.
    """
    return derangements_count(n)


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
