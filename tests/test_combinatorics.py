"""
Tests for combinatorics utilities.

Tests cover:
- Factorial and binomial coefficients
- Permutations and combinations
- Permutation ranking/unranking
- Integer partitions
- Multinomial coefficient
- Stirling and Bell numbers
- Catalan numbers
- Derangements
"""

import pytest

from pytcl.mathematical_functions.combinatorics.combinatorics import (
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

# =============================================================================
# Tests for factorial
# =============================================================================


class TestFactorial:
    """Tests for factorial function."""

    def test_factorial_zero(self):
        """Test 0! = 1."""
        assert factorial(0) == 1

    def test_factorial_one(self):
        """Test 1! = 1."""
        assert factorial(1) == 1

    def test_factorial_five(self):
        """Test 5! = 120."""
        assert factorial(5) == 120

    def test_factorial_ten(self):
        """Test 10! = 3628800."""
        assert factorial(10) == 3628800

    def test_factorial_negative_raises(self):
        """Test negative n raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            factorial(-1)


# =============================================================================
# Tests for binomial coefficient
# =============================================================================


class TestNChooseK:
    """Tests for binomial coefficient."""

    def test_n_choose_0(self):
        """Test C(n, 0) = 1."""
        for n in range(10):
            assert n_choose_k(n, 0) == 1

    def test_n_choose_n(self):
        """Test C(n, n) = 1."""
        for n in range(10):
            assert n_choose_k(n, n) == 1

    def test_5_choose_2(self):
        """Test C(5, 2) = 10."""
        assert n_choose_k(5, 2) == 10

    def test_10_choose_3(self):
        """Test C(10, 3) = 120."""
        assert n_choose_k(10, 3) == 120

    def test_symmetry(self):
        """Test C(n, k) = C(n, n-k)."""
        for n in range(10):
            for k in range(n + 1):
                assert n_choose_k(n, k) == n_choose_k(n, n - k)

    def test_k_greater_than_n(self):
        """Test C(n, k) = 0 when k > n."""
        assert n_choose_k(5, 6) == 0

    def test_negative_k(self):
        """Test C(n, k) = 0 when k < 0."""
        assert n_choose_k(5, -1) == 0


# =============================================================================
# Tests for k-permutations count
# =============================================================================


class TestNPermuteK:
    """Tests for k-permutations count."""

    def test_n_permute_n(self):
        """Test P(n, n) = n!."""
        for n in range(1, 8):
            assert n_permute_k(n, n) == factorial(n)

    def test_5_permute_2(self):
        """Test P(5, 2) = 20."""
        assert n_permute_k(5, 2) == 20

    def test_n_permute_0(self):
        """Test P(n, 0) = 1."""
        for n in range(10):
            assert n_permute_k(n, 0) == 1

    def test_k_greater_than_n(self):
        """Test P(n, k) = 0 when k > n."""
        assert n_permute_k(3, 5) == 0

    def test_negative_k(self):
        """Test P(n, k) = 0 when k < 0."""
        assert n_permute_k(5, -1) == 0


# =============================================================================
# Tests for permutations generator
# =============================================================================


class TestPermutations:
    """Tests for permutations generator."""

    def test_permutations_all(self):
        """Test all permutations of [1, 2, 3]."""
        result = list(permutations([1, 2, 3]))
        assert len(result) == 6
        assert (1, 2, 3) in result
        assert (3, 2, 1) in result

    def test_permutations_k(self):
        """Test 2-permutations of [1, 2, 3]."""
        result = list(permutations([1, 2, 3], 2))
        assert len(result) == 6
        assert (1, 2) in result
        assert (2, 1) in result

    def test_permutations_empty(self):
        """Test permutations of empty list."""
        result = list(permutations([]))
        assert result == [()]


# =============================================================================
# Tests for combinations generator
# =============================================================================


class TestCombinations:
    """Tests for combinations generator."""

    def test_combinations_basic(self):
        """Test 2-combinations of [1, 2, 3, 4]."""
        result = list(combinations([1, 2, 3, 4], 2))
        assert len(result) == 6
        assert (1, 2) in result
        assert (3, 4) in result
        # Order matters - should be in order
        assert (2, 1) not in result

    def test_combinations_all(self):
        """Test n-combinations of n items."""
        result = list(combinations([1, 2, 3], 3))
        assert result == [(1, 2, 3)]

    def test_combinations_one(self):
        """Test 1-combinations."""
        result = list(combinations([1, 2, 3], 1))
        assert len(result) == 3


class TestCombinationsWithReplacement:
    """Tests for combinations with replacement."""

    def test_combinations_with_replacement_basic(self):
        """Test 2-combinations with replacement of [1, 2]."""
        result = list(combinations_with_replacement([1, 2], 2))
        assert len(result) == 3
        assert (1, 1) in result
        assert (1, 2) in result
        assert (2, 2) in result

    def test_combinations_with_replacement_count(self):
        """Test count of combinations with replacement."""
        # C(n+k-1, k) = C(3+2-1, 2) = C(4, 2) = 6
        result = list(combinations_with_replacement([1, 2, 3], 2))
        assert len(result) == 6


# =============================================================================
# Tests for permutation ranking
# =============================================================================


class TestPermutationRank:
    """Tests for permutation ranking."""

    def test_first_permutation(self):
        """Test first permutation has rank 0."""
        assert permutation_rank([0, 1, 2]) == 0

    def test_last_permutation(self):
        """Test last permutation has rank n!-1."""
        assert permutation_rank([2, 1, 0]) == 5

    def test_middle_permutation(self):
        """Test middle permutation."""
        assert permutation_rank([1, 0, 2]) == 2

    def test_larger_permutation(self):
        """Test with larger permutation."""
        # [3, 2, 1, 0] is the last permutation of 4 elements
        assert permutation_rank([3, 2, 1, 0]) == 23


class TestPermutationUnrank:
    """Tests for permutation unranking."""

    def test_unrank_zero(self):
        """Test rank 0 gives first permutation."""
        assert permutation_unrank(0, 3) == [0, 1, 2]

    def test_unrank_last(self):
        """Test last rank gives last permutation."""
        assert permutation_unrank(5, 3) == [2, 1, 0]

    def test_unrank_middle(self):
        """Test middle rank."""
        assert permutation_unrank(2, 3) == [1, 0, 2]

    def test_roundtrip(self):
        """Test rank and unrank are inverses."""
        for n in range(1, 5):
            for rank in range(factorial(n)):
                perm = permutation_unrank(rank, n)
                assert permutation_rank(perm) == rank

    def test_invalid_rank_raises(self):
        """Test invalid rank raises ValueError."""
        with pytest.raises(ValueError):
            permutation_unrank(10, 3)  # Only 6 permutations

        with pytest.raises(ValueError):
            permutation_unrank(-1, 3)


# =============================================================================
# Tests for next permutation
# =============================================================================


class TestNextPermutation:
    """Tests for next permutation in lexicographic order."""

    def test_next_permutation_basic(self):
        """Test basic next permutation."""
        assert next_permutation([1, 2, 3]) == [1, 3, 2]

    def test_next_permutation_sequence(self):
        """Test sequence of next permutations."""
        perm = [1, 2, 3]
        expected = [[1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for exp in expected:
            perm = next_permutation(perm)
            assert perm == exp

    def test_last_permutation_returns_none(self):
        """Test last permutation returns None."""
        assert next_permutation([3, 2, 1]) is None


# =============================================================================
# Tests for partition count
# =============================================================================


class TestPartitionCount:
    """Tests for partition counting."""

    def test_partition_count_5(self):
        """Test partition count of 5."""
        assert partition_count(5) == 7

    def test_partition_count_10(self):
        """Test partition count of 10."""
        assert partition_count(10) == 42

    def test_partition_count_with_parts(self):
        """Test partition count with specific number of parts."""
        assert partition_count(5, 2) == 2  # 4+1, 3+2

    def test_partition_count_one(self):
        """Test partition count of 1."""
        assert partition_count(1) == 1


# =============================================================================
# Tests for partitions generator
# =============================================================================


class TestPartitions:
    """Tests for partitions generator."""

    def test_partitions_4(self):
        """Test partitions of 4."""
        result = list(partitions(4))
        expected = [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)]
        assert result == expected

    def test_partitions_with_k(self):
        """Test partitions with k parts."""
        result = list(partitions(5, 2))
        assert (4, 1) in result
        assert (3, 2) in result
        assert len(result) == 2

    def test_partitions_count_matches(self):
        """Test generator count matches partition_count."""
        for n in range(1, 10):
            assert len(list(partitions(n))) == partition_count(n)


# =============================================================================
# Tests for multinomial coefficient
# =============================================================================


class TestMultinomialCoefficient:
    """Tests for multinomial coefficient."""

    def test_multinomial_basic(self):
        """Test multinomial(2, 3, 1) = 60."""
        assert multinomial_coefficient(2, 3, 1) == 60

    def test_multinomial_equals_binomial(self):
        """Test multinomial with 2 args equals binomial."""
        assert multinomial_coefficient(3, 2) == n_choose_k(5, 3)

    def test_multinomial_single_arg(self):
        """Test multinomial with single arg is 1."""
        assert multinomial_coefficient(5) == 1


# =============================================================================
# Tests for Stirling numbers
# =============================================================================


class TestStirlingSecond:
    """Tests for Stirling numbers of the second kind."""

    def test_stirling_diagonal(self):
        """Test S(n, n) = 1."""
        for n in range(1, 8):
            assert stirling_second(n, n) == 1

    def test_stirling_4_2(self):
        """Test S(4, 2) = 7."""
        assert stirling_second(4, 2) == 7

    def test_stirling_n_1(self):
        """Test S(n, 1) = 1."""
        for n in range(1, 8):
            assert stirling_second(n, 1) == 1

    def test_stirling_boundary_cases(self):
        """Test boundary cases."""
        assert stirling_second(0, 0) == 1
        assert stirling_second(0, 1) == 0
        assert stirling_second(5, 0) == 0
        assert stirling_second(3, 5) == 0


# =============================================================================
# Tests for Bell numbers
# =============================================================================


class TestBellNumber:
    """Tests for Bell numbers."""

    def test_bell_sequence(self):
        """Test first Bell numbers."""
        expected = [1, 1, 2, 5, 15, 52, 203]
        for n, exp in enumerate(expected):
            assert bell_number(n) == exp


# =============================================================================
# Tests for Catalan numbers
# =============================================================================


class TestCatalanNumber:
    """Tests for Catalan numbers."""

    def test_catalan_sequence(self):
        """Test first Catalan numbers."""
        expected = [1, 1, 2, 5, 14, 42, 132]
        for n, exp in enumerate(expected):
            assert catalan_number(n) == exp


# =============================================================================
# Tests for derangements
# =============================================================================


class TestDerangements:
    """Tests for derangements count."""

    def test_derangements_sequence(self):
        """Test first derangement numbers."""
        expected = [1, 0, 1, 2, 9, 44, 265]
        for n, exp in enumerate(expected):
            assert derangements_count(n) == exp

    def test_subfactorial_alias(self):
        """Test subfactorial is alias for derangements_count."""
        for n in range(10):
            assert subfactorial(n) == derangements_count(n)


# =============================================================================
# Integration tests
# =============================================================================


class TestCombinatoricsIntegration:
    """Integration tests for combinatorics functions."""

    def test_pascals_triangle(self):
        """Test Pascal's triangle relation."""
        for n in range(1, 10):
            for k in range(1, n):
                assert n_choose_k(n, k) == n_choose_k(n - 1, k - 1) + n_choose_k(
                    n - 1, k
                )

    def test_permutation_combination_relation(self):
        """Test P(n,k) = k! * C(n,k)."""
        for n in range(1, 8):
            for k in range(n + 1):
                assert n_permute_k(n, k) == factorial(k) * n_choose_k(n, k)

    def test_bell_stirling_relation(self):
        """Test B(n) = sum of S(n, k) for k = 0 to n."""
        for n in range(8):
            assert bell_number(n) == sum(stirling_second(n, k) for k in range(n + 1))

    def test_all_permutations_count(self):
        """Test permutations generator gives correct count."""
        for n in range(1, 6):
            items = list(range(n))
            count = len(list(permutations(items)))
            assert count == factorial(n)

    def test_all_combinations_count(self):
        """Test combinations generator gives correct count."""
        for n in range(1, 8):
            for k in range(n + 1):
                items = list(range(n))
                count = len(list(combinations(items, k)))
                assert count == n_choose_k(n, k)
