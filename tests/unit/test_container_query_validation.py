"""``validate_query_input``, the shared entry check for every spatial container.

The gh-49 allowlist described this one as "an internal helper, exported
unnecessarily", with unexporting offered as the resolution. It is not internal:
KD-tree, R-tree, VP-tree and cover-tree all route their ``query`` and
``query_radius`` calls through it, eleven call sites in total, and it is
published from ``pytcl.containers``. So it gets tested rather than removed.

Its job is small and entirely about what a caller is allowed to pass. A single
point may be given as a flat ``(n_features,)`` array and is promoted to one row;
anything whose feature count disagrees with the tree is rejected. Getting the
promotion wrong is the dangerous case, because a 3-element vector read as three
1-D queries returns three confident, meaningless answers instead of raising.
"""

import numpy as np
import pytest

from pytcl.containers import validate_query_input


class TestSinglePointPromotion:
    """A flat array is one query, not many."""

    @pytest.mark.parametrize("n_features", [1, 2, 3, 10])
    def test_a_one_dimensional_query_becomes_a_single_row(self, n_features):
        point = np.arange(n_features, dtype=float)
        result = validate_query_input(point, n_features)

        assert result.shape == (1, n_features), (
            f"a flat {n_features}-element query became {result.shape}; read as "
            f"several queries it would return several answers to a question "
            f"that was asked once"
        )
        np.testing.assert_array_equal(result[0], point)

    def test_a_two_dimensional_query_is_left_alone(self):
        points = np.arange(12, dtype=float).reshape(4, 3)
        result = validate_query_input(points, 3)
        assert result.shape == (4, 3)
        np.testing.assert_array_equal(result, points)

    def test_a_single_row_two_dimensional_query_is_not_flattened(self):
        """``(1, n)`` in must stay ``(1, n)`` out, not collapse to ``(n,)``."""
        points = np.array([[1.0, 2.0, 3.0]])
        assert validate_query_input(points, 3).shape == (1, 3)


class TestFeatureCountEnforcement:
    """The check that stops a query being silently answered against the wrong
    number of dimensions."""

    @pytest.mark.parametrize(
        "given,expected",
        [(2, 3), (3, 2), (1, 5), (10, 3)],
        ids=["2v3", "3v2", "1v5", "10v3"],
    )
    def test_a_feature_count_mismatch_raises(self, given, expected):
        with pytest.raises(ValueError, match=f"{given} features, expected {expected}"):
            validate_query_input(np.zeros(given), expected)

    def test_the_message_names_both_counts(self):
        """A caller has to be able to tell which side is wrong."""
        with pytest.raises(ValueError) as excinfo:
            validate_query_input(np.zeros((5, 7)), 3)
        message = str(excinfo.value)
        assert "7" in message and "3" in message

    def test_a_matching_feature_count_is_accepted(self):
        validate_query_input(np.zeros((5, 3)), 3)

    def test_a_mismatch_raises_for_batched_queries_too(self):
        """The 2-D path must be checked as well as the promoted 1-D one."""
        with pytest.raises(ValueError):
            validate_query_input(np.zeros((5, 4)), 3)


class TestInputConversion:
    """What a caller may pass, beyond a float64 array."""

    @pytest.mark.parametrize(
        "query",
        [
            [1.0, 2.0, 3.0],
            (1.0, 2.0, 3.0),
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            np.array([1, 2, 3], dtype=np.int32),
        ],
        ids=["list", "tuple", "nested-list", "int-array"],
    )
    def test_array_likes_are_accepted_and_converted(self, query):
        result = validate_query_input(query, 3)
        assert result.dtype == np.float64, (
            "downstream distance arithmetic assumes float64; an integer array "
            "left unconverted would do integer division"
        )
        assert result.ndim == 2

    def test_integer_input_is_converted_without_changing_values(self):
        result = validate_query_input(np.array([[3, 4, 5]], dtype=np.int64), 3)
        np.testing.assert_array_equal(result, [[3.0, 4.0, 5.0]])

    def test_an_empty_batch_of_the_right_width_is_allowed(self):
        """Zero queries is a legitimate request, not an error.

        A caller filtering a candidate list can legitimately end up with none,
        and should get an empty result rather than an exception.
        """
        result = validate_query_input(np.zeros((0, 3)), 3)
        assert result.shape == (0, 3)


def test_validation_does_not_modify_the_callers_array():
    """A caller's array must come back out of a query unchanged.

    Worth stating explicitly, because validation passes the array straight
    through when no conversion is needed -- ``np.asarray`` on an array that is
    already float64 and 2-D returns that same object, so the containers index
    the caller's own memory. That is safe as things stand: every container only
    reads rows out of it and none retains it past the call. This test is what
    would fail if either of those changed.
    """
    original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    before = original.copy()

    validate_query_input(original, 3)

    np.testing.assert_array_equal(original, before, "validation mutated its input")


def test_a_flat_query_is_promoted_without_disturbing_the_original():
    """The reshape path must not reshape the caller's array in place."""
    original = np.array([1.0, 2.0, 3.0])
    result = validate_query_input(original, 3)

    assert original.shape == (3,), (
        f"the caller's array was reshaped to {original.shape} underneath them"
    )
    assert result.shape == (1, 3)
