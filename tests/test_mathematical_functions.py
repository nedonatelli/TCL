"""
Tests for mathematical_functions module.

Tests cover:
- Basic matrix operations (vec, unvec, kron, block_diag)
- Special matrices (Vandermonde, Toeplitz, Hankel, etc.)
- Geometry functions (point in polygon, convex hull, etc.)
- Combinatorics (factorial, combinations, permutations)
"""

import numpy as np
import pytest

from pytcl.mathematical_functions import (  # Basic matrix; Geometry; Combinatorics
    block_diag,
    bounding_box,
    combinations,
    convex_hull,
    factorial,
    kron,
    line_intersection,
    n_choose_k,
    permutations,
    point_in_polygon,
    polygon_area,
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
from pytcl.mathematical_functions.geometry.geometry import (
    barycentric_coordinates,
    convex_hull_area,
    point_to_line_distance,
    point_to_line_segment_distance,
    points_in_polygon,
    polygon_centroid,
    triangle_area,
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


class TestPointInPolygon:
    """Tests for point in polygon."""

    @pytest.fixture
    def square(self):
        """Unit square polygon."""
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

    def test_point_inside(self, square):
        """Test point inside polygon."""
        assert point_in_polygon([0.5, 0.5], square)

    def test_point_outside(self, square):
        """Test point outside polygon."""
        assert not point_in_polygon([2, 2], square)

    def test_point_on_edge(self, square):
        """Test point on edge (boundary case)."""
        # Edge behavior can vary; just ensure no crash
        result = point_in_polygon([0.5, 0], square)
        assert isinstance(result, bool)

    def test_points_in_polygon_multiple(self, square):
        """Test multiple points."""
        points = np.array([[0.5, 0.5], [2, 2], [0.1, 0.1]])
        results = points_in_polygon(points, square)
        np.testing.assert_array_equal(results, [True, False, True])


class TestConvexHull:
    """Tests for convex hull."""

    def test_convex_hull_square(self):
        """Test convex hull of square."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5]])
        vertices, indices = convex_hull(points)
        # Interior point should not be in hull
        assert len(indices) == 4

    def test_convex_hull_area(self):
        """Test convex hull area."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        area = convex_hull_area(points)
        np.testing.assert_allclose(area, 1.0)


class TestPolygonArea:
    """Tests for polygon area."""

    def test_square_area(self):
        """Test area of unit square."""
        square = [[0, 0], [1, 0], [1, 1], [0, 1]]
        area = polygon_area(square)
        np.testing.assert_allclose(area, 1.0)

    def test_triangle_area_shoelace(self):
        """Test area of triangle."""
        triangle = [[0, 0], [2, 0], [1, 1]]
        area = polygon_area(triangle)
        np.testing.assert_allclose(area, 1.0)


class TestPolygonCentroid:
    """Tests for polygon centroid."""

    def test_square_centroid(self):
        """Test centroid of unit square."""
        square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        centroid = polygon_centroid(square)
        np.testing.assert_allclose(centroid, [0.5, 0.5])


class TestLineIntersection:
    """Tests for line intersection."""

    def test_crossing_lines(self):
        """Test intersection of crossing lines."""
        result = line_intersection([0, 0], [1, 1], [0, 1], [1, 0])
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_parallel_lines(self):
        """Test parallel lines don't intersect."""
        result = line_intersection([0, 0], [1, 0], [0, 1], [1, 1])
        assert result is None

    def test_non_intersecting_segments(self):
        """Test non-intersecting segments."""
        result = line_intersection([0, 0], [0.4, 0.4], [0.6, 0.6], [1, 1])
        assert result is None


class TestBoundingBox:
    """Tests for bounding box."""

    def test_bounding_box_basic(self):
        """Test basic bounding box."""
        points = np.array([[0, 1], [2, 3], [1, 2]])
        min_c, max_c = bounding_box(points)
        np.testing.assert_allclose(min_c, [0, 1])
        np.testing.assert_allclose(max_c, [2, 3])


class TestPointToLineDistance:
    """Tests for point to line distance."""

    def test_perpendicular_distance(self):
        """Test perpendicular distance."""
        dist = point_to_line_distance([0, 1], [0, 0], [1, 0])
        np.testing.assert_allclose(dist, 1.0)

    def test_zero_distance(self):
        """Test point on line."""
        dist = point_to_line_distance([0.5, 0], [0, 0], [1, 0])
        np.testing.assert_allclose(dist, 0.0)


class TestPointToLineSegmentDistance:
    """Tests for point to line segment distance."""

    def test_perpendicular_projection(self):
        """Test when projection falls on segment."""
        dist = point_to_line_segment_distance([0.5, 1], [0, 0], [1, 0])
        np.testing.assert_allclose(dist, 1.0)

    def test_endpoint_distance(self):
        """Test when closest point is endpoint."""
        dist = point_to_line_segment_distance([2, 0], [0, 0], [1, 0])
        np.testing.assert_allclose(dist, 1.0)


class TestTriangleArea:
    """Tests for triangle area."""

    def test_triangle_area_2d(self):
        """Test 2D triangle area."""
        area = triangle_area([0, 0], [1, 0], [0, 1])
        np.testing.assert_allclose(area, 0.5)

    def test_triangle_area_3d(self):
        """Test 3D triangle area."""
        area = triangle_area([0, 0, 0], [1, 0, 0], [0, 1, 0])
        np.testing.assert_allclose(area, 0.5)


class TestBarycentricCoordinates:
    """Tests for barycentric coordinates."""

    def test_vertex_coords(self):
        """Test coordinates at vertices."""
        p1, p2, p3 = [0, 0], [1, 0], [0, 1]

        coords = barycentric_coordinates(p1, p1, p2, p3)
        np.testing.assert_allclose(coords, [1, 0, 0])

        coords = barycentric_coordinates(p2, p1, p2, p3)
        np.testing.assert_allclose(coords, [0, 1, 0])

        coords = barycentric_coordinates(p3, p1, p2, p3)
        np.testing.assert_allclose(coords, [0, 0, 1])

    def test_centroid_coords(self):
        """Test coordinates at centroid."""
        p1, p2, p3 = [0, 0], [1, 0], [0, 1]
        centroid = np.array([1 / 3, 1 / 3])
        coords = barycentric_coordinates(centroid, p1, p2, p3)
        np.testing.assert_allclose(coords, [1 / 3, 1 / 3, 1 / 3], rtol=1e-10)


class TestFactorial:
    """Tests for factorial."""

    def test_factorial_basic(self):
        """Test basic factorial values."""
        assert factorial(0) == 1
        assert factorial(1) == 1
        assert factorial(5) == 120
        assert factorial(10) == 3628800


class TestNChooseK:
    """Tests for binomial coefficient."""

    def test_n_choose_k_basic(self):
        """Test basic binomial coefficients."""
        assert n_choose_k(5, 0) == 1
        assert n_choose_k(5, 5) == 1
        assert n_choose_k(5, 2) == 10
        assert n_choose_k(10, 3) == 120

    def test_n_choose_k_symmetry(self):
        """Test C(n,k) = C(n,n-k)."""
        n = 10
        for k in range(n + 1):
            assert n_choose_k(n, k) == n_choose_k(n, n - k)


class TestPermutations:
    """Tests for permutations."""

    def test_permutations_count(self):
        """Test number of permutations."""
        perms = list(permutations([1, 2, 3]))
        assert len(perms) == 6  # 3!

    def test_permutations_uniqueness(self):
        """Test all permutations are unique."""
        perms = list(permutations([1, 2, 3, 4]))
        assert len(perms) == len(set(perms))


class TestCombinations:
    """Tests for combinations."""

    def test_combinations_count(self):
        """Test number of combinations."""
        combs = list(combinations([1, 2, 3, 4, 5], 2))
        assert len(combs) == 10  # C(5,2)

    def test_combinations_uniqueness(self):
        """Test all combinations are unique."""
        combs = list(combinations([1, 2, 3, 4, 5], 3))
        comb_tuples = [tuple(sorted(c)) for c in combs]
        assert len(comb_tuples) == len(set(comb_tuples))
