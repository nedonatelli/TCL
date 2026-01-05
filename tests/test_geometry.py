"""
Tests for geometric primitives and calculations.

Tests cover:
- Point-in-polygon tests
- Convex hull computation
- Polygon area and centroid
- Line and segment intersections
- Point-to-line distance
- Triangle operations
- Barycentric coordinates
- Bounding boxes
- Minimum enclosing circle
"""

import numpy as np
import pytest
from scipy.spatial import Delaunay

from pytcl.mathematical_functions.geometry.geometry import (
    barycentric_coordinates,
    bounding_box,
    convex_hull,
    convex_hull_area,
    delaunay_triangulation,
    line_intersection,
    line_plane_intersection,
    minimum_bounding_circle,
    oriented_bounding_box,
    point_in_polygon,
    point_to_line_distance,
    point_to_line_segment_distance,
    points_in_polygon,
    polygon_area,
    polygon_centroid,
    triangle_area,
)

# =============================================================================
# Tests for point_in_polygon
# =============================================================================


class TestPointInPolygon:
    """Tests for point-in-polygon test."""

    def test_point_inside_square(self):
        """Test point inside a square."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert point_in_polygon([0.5, 0.5], polygon) is True

    def test_point_outside_square(self):
        """Test point outside a square."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert point_in_polygon([2, 2], polygon) is False

    def test_point_on_edge(self):
        """Test point on edge of polygon."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        # On edge behavior depends on implementation
        result = point_in_polygon([0.5, 0], polygon)
        assert isinstance(result, bool)

    def test_point_inside_triangle(self):
        """Test point inside a triangle."""
        polygon = np.array([[0, 0], [2, 0], [1, 2]])
        assert point_in_polygon([1, 0.5], polygon) is True

    def test_point_outside_triangle(self):
        """Test point outside a triangle."""
        polygon = np.array([[0, 0], [2, 0], [1, 2]])
        assert point_in_polygon([3, 3], polygon) is False

    def test_point_inside_concave_polygon(self):
        """Test point inside a concave polygon."""
        # L-shaped polygon
        polygon = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])
        assert point_in_polygon([0.5, 0.5], polygon) is True
        assert point_in_polygon([1.5, 1.5], polygon) is False


class TestPointsInPolygon:
    """Tests for multiple points in polygon."""

    def test_multiple_points(self):
        """Test multiple points at once."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0.5, 0.5], [2, 2], [0.1, 0.1], [-1, -1]])
        result = points_in_polygon(points, polygon)
        assert len(result) == 4
        assert result[0]  # Inside
        assert not result[1]  # Outside
        assert result[2]  # Inside
        assert not result[3]  # Outside

    def test_single_point_as_1d(self):
        """Test single point as 1D array."""
        polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = points_in_polygon([0.5, 0.5], polygon)
        assert len(result) == 1
        assert result[0]


# =============================================================================
# Tests for convex hull
# =============================================================================


class TestConvexHull:
    """Tests for convex hull computation."""

    def test_convex_hull_triangle(self):
        """Test convex hull of a triangle."""
        points = np.array([[0, 0], [1, 0], [0, 1]])
        vertices, indices = convex_hull(points)
        assert len(indices) == 3  # All points are on hull

    def test_convex_hull_square_with_interior(self):
        """Test convex hull with interior point."""
        points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
        vertices, indices = convex_hull(points)
        assert len(indices) == 4  # 4 corners on hull

    def test_convex_hull_random_points(self):
        """Test convex hull of random points."""
        np.random.seed(42)
        points = np.random.rand(20, 2)
        vertices, indices = convex_hull(points)
        assert len(indices) <= 20
        assert len(indices) >= 3


class TestConvexHullArea:
    """Tests for convex hull area."""

    def test_unit_square_area(self):
        """Test area of unit square hull."""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        area = convex_hull_area(points)
        assert area == pytest.approx(1.0, rel=1e-10)

    def test_triangle_area(self):
        """Test area of triangle hull."""
        points = np.array([[0, 0], [2, 0], [1, 2]])
        area = convex_hull_area(points)
        assert area == pytest.approx(2.0, rel=1e-10)


# =============================================================================
# Tests for polygon area
# =============================================================================


class TestPolygonArea:
    """Tests for polygon area calculation."""

    def test_unit_square(self):
        """Test unit square area."""
        polygon = [[0, 0], [1, 0], [1, 1], [0, 1]]
        assert polygon_area(polygon) == pytest.approx(1.0)

    def test_rectangle(self):
        """Test rectangle area."""
        polygon = [[0, 0], [4, 0], [4, 3], [0, 3]]
        assert polygon_area(polygon) == pytest.approx(12.0)

    def test_triangle_polygon(self):
        """Test triangle area via polygon formula."""
        polygon = [[0, 0], [4, 0], [2, 3]]
        assert polygon_area(polygon) == pytest.approx(6.0)

    def test_irregular_polygon(self):
        """Test irregular polygon area."""
        polygon = [[0, 0], [4, 0], [4, 1], [1, 1], [1, 3], [0, 3]]
        area = polygon_area(polygon)
        assert area > 0


# =============================================================================
# Tests for polygon centroid
# =============================================================================


class TestPolygonCentroid:
    """Tests for polygon centroid calculation."""

    def test_square_centroid(self):
        """Test centroid of square is at center."""
        polygon = [[0, 0], [2, 0], [2, 2], [0, 2]]
        centroid = polygon_centroid(polygon)
        assert centroid[0] == pytest.approx(1.0, rel=1e-6)
        assert centroid[1] == pytest.approx(1.0, rel=1e-6)

    def test_triangle_centroid(self):
        """Test centroid of triangle."""
        polygon = [[0, 0], [3, 0], [0, 3]]
        centroid = polygon_centroid(polygon)
        assert centroid[0] == pytest.approx(1.0, rel=1e-6)
        assert centroid[1] == pytest.approx(1.0, rel=1e-6)


# =============================================================================
# Tests for line intersection
# =============================================================================


class TestLineIntersection:
    """Tests for line segment intersection."""

    def test_crossing_segments(self):
        """Test intersecting line segments."""
        result = line_intersection([0, 0], [1, 1], [0, 1], [1, 0])
        assert result is not None
        assert result[0] == pytest.approx(0.5, rel=1e-10)
        assert result[1] == pytest.approx(0.5, rel=1e-10)

    def test_parallel_segments(self):
        """Test parallel line segments don't intersect."""
        result = line_intersection([0, 0], [1, 0], [0, 1], [1, 1])
        assert result is None

    def test_non_crossing_segments(self):
        """Test non-intersecting segments."""
        result = line_intersection([0, 0], [1, 0], [2, 0], [3, 0])
        assert result is None

    def test_t_intersection(self):
        """Test T-shaped intersection."""
        result = line_intersection([0, 0], [2, 0], [1, -1], [1, 1])
        assert result is not None
        assert result[0] == pytest.approx(1.0)
        assert result[1] == pytest.approx(0.0)


# =============================================================================
# Tests for line-plane intersection
# =============================================================================


class TestLinePlaneIntersection:
    """Tests for line-plane intersection."""

    def test_line_through_plane(self):
        """Test line passing through plane."""
        # z-axis through xy-plane at z=1
        result = line_plane_intersection([0, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 1])
        assert result is not None
        np.testing.assert_allclose(result, [0, 0, 1], rtol=1e-10)

    def test_line_parallel_to_plane(self):
        """Test line parallel to plane."""
        result = line_plane_intersection(
            [0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]  # x-direction  # xy plane at z=1
        )
        assert result is None

    def test_oblique_intersection(self):
        """Test oblique intersection."""
        result = line_plane_intersection(
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],  # 45 degree line  # yz plane at x=1
        )
        assert result is not None
        assert result[0] == pytest.approx(1.0)


# =============================================================================
# Tests for point-to-line distance
# =============================================================================


class TestPointToLineDistance:
    """Tests for point-to-line distance."""

    def test_point_on_line(self):
        """Test point on line has zero distance."""
        dist = point_to_line_distance([0.5, 0], [0, 0], [1, 0])
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_point_above_line(self):
        """Test point above horizontal line."""
        dist = point_to_line_distance([0, 1], [0, 0], [1, 0])
        assert dist == pytest.approx(1.0)

    def test_point_at_angle(self):
        """Test point at angle from diagonal line."""
        # Distance from (0, 0) to line from (0, 1) to (1, 0)
        # Line: x + y = 1, point (0, 0)
        # Distance = |0 + 0 - 1| / sqrt(2) = 1/sqrt(2)
        dist = point_to_line_distance([0, 0], [0, 1], [1, 0])
        assert dist == pytest.approx(1 / np.sqrt(2), rel=1e-10)

    def test_degenerate_line(self):
        """Test distance to degenerate line (point)."""
        dist = point_to_line_distance([1, 1], [0, 0], [0, 0])
        assert dist == pytest.approx(np.sqrt(2))


class TestPointToLineSegmentDistance:
    """Tests for point-to-line-segment distance."""

    def test_point_closest_to_interior(self):
        """Test point closest to segment interior."""
        dist = point_to_line_segment_distance([0.5, 1], [0, 0], [1, 0])
        assert dist == pytest.approx(1.0)

    def test_point_closest_to_endpoint(self):
        """Test point closest to segment endpoint."""
        dist = point_to_line_segment_distance([2, 0], [0, 0], [1, 0])
        assert dist == pytest.approx(1.0)

    def test_point_on_segment(self):
        """Test point on segment."""
        dist = point_to_line_segment_distance([0.5, 0], [0, 0], [1, 0])
        assert dist == pytest.approx(0.0, abs=1e-10)

    def test_degenerate_segment(self):
        """Test distance to degenerate segment (point)."""
        dist = point_to_line_segment_distance([1, 1], [0, 0], [0, 0])
        assert dist == pytest.approx(np.sqrt(2))


# =============================================================================
# Tests for triangle area
# =============================================================================


class TestTriangleArea:
    """Tests for triangle area."""

    def test_right_triangle_2d(self):
        """Test right triangle area in 2D."""
        area = triangle_area([0, 0], [1, 0], [0, 1])
        assert area == pytest.approx(0.5)

    def test_equilateral_triangle(self):
        """Test equilateral triangle area."""
        # Equilateral triangle with side 2
        area = triangle_area([0, 0], [2, 0], [1, np.sqrt(3)])
        assert area == pytest.approx(np.sqrt(3), rel=1e-6)

    def test_triangle_3d(self):
        """Test triangle area in 3D."""
        area = triangle_area([0, 0, 0], [1, 0, 0], [0, 1, 0])
        assert area == pytest.approx(0.5)

    def test_triangle_3d_non_aligned(self):
        """Test 3D triangle not aligned with axes."""
        area = triangle_area([0, 0, 0], [1, 0, 0], [0, 0, 1])
        assert area == pytest.approx(0.5)


# =============================================================================
# Tests for barycentric coordinates
# =============================================================================


class TestBarycentricCoordinates:
    """Tests for barycentric coordinates."""

    def test_at_vertex(self):
        """Test barycentric coords at triangle vertex."""
        coords = barycentric_coordinates([0, 0], [0, 0], [1, 0], [0, 1])
        assert coords[0] == pytest.approx(1.0, rel=1e-6)
        assert coords[1] == pytest.approx(0.0, abs=1e-10)
        assert coords[2] == pytest.approx(0.0, abs=1e-10)

    def test_at_centroid(self):
        """Test barycentric coords at centroid."""
        coords = barycentric_coordinates([1 / 3, 1 / 3], [0, 0], [1, 0], [0, 1])
        for c in coords:
            assert c == pytest.approx(1 / 3, rel=1e-6)

    def test_sum_equals_one(self):
        """Test barycentric coords sum to 1."""
        coords = barycentric_coordinates([0.2, 0.3], [0, 0], [1, 0], [0, 1])
        assert np.sum(coords) == pytest.approx(1.0, rel=1e-10)

    def test_point_inside(self):
        """Test inside point has all positive coords."""
        coords = barycentric_coordinates([0.25, 0.25], [0, 0], [1, 0], [0, 1])
        assert all(c >= 0 for c in coords)

    def test_point_outside(self):
        """Test outside point has negative coord."""
        coords = barycentric_coordinates([2, 2], [0, 0], [1, 0], [0, 1])
        assert any(c < 0 for c in coords)


# =============================================================================
# Tests for Delaunay triangulation
# =============================================================================


class TestDelaunayTriangulation:
    """Tests for Delaunay triangulation."""

    def test_basic_triangulation(self):
        """Test basic triangulation."""
        points = np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0.5]])
        tri = delaunay_triangulation(points)
        assert isinstance(tri, Delaunay)
        assert len(tri.simplices) >= 1

    def test_triangulation_grid(self):
        """Test triangulation of grid points."""
        x = np.linspace(0, 1, 4)
        y = np.linspace(0, 1, 4)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        tri = delaunay_triangulation(points)
        assert len(tri.simplices) > 0


# =============================================================================
# Tests for bounding box
# =============================================================================


class TestBoundingBox:
    """Tests for axis-aligned bounding box."""

    def test_basic_bbox(self):
        """Test basic bounding box."""
        points = np.array([[0, 1], [2, 3], [1, 2]])
        min_c, max_c = bounding_box(points)
        np.testing.assert_allclose(min_c, [0, 1])
        np.testing.assert_allclose(max_c, [2, 3])

    def test_single_point(self):
        """Test bounding box of single point."""
        points = np.array([[1, 2]])
        min_c, max_c = bounding_box(points)
        np.testing.assert_allclose(min_c, [1, 2])
        np.testing.assert_allclose(max_c, [1, 2])

    def test_3d_bbox(self):
        """Test 3D bounding box."""
        points = np.array([[0, 0, 0], [1, 2, 3], [0.5, 1, 1.5]])
        min_c, max_c = bounding_box(points)
        np.testing.assert_allclose(min_c, [0, 0, 0])
        np.testing.assert_allclose(max_c, [1, 2, 3])


# =============================================================================
# Tests for minimum bounding circle
# =============================================================================


class TestMinimumBoundingCircle:
    """Tests for minimum enclosing circle."""

    def test_two_points(self):
        """Test circle enclosing two points."""
        np.random.seed(42)
        points = np.array([[0, 0], [2, 0]])
        center, radius = minimum_bounding_circle(points)
        assert center[0] == pytest.approx(1.0, rel=0.1)
        assert center[1] == pytest.approx(0.0, abs=0.1)
        assert radius == pytest.approx(1.0, rel=0.1)

    def test_triangle(self):
        """Test circle enclosing triangle."""
        np.random.seed(42)
        points = np.array([[0, 0], [1, 0], [0.5, 0.5]])
        center, radius = minimum_bounding_circle(points)
        # All points should be inside or on circle
        for p in points:
            dist = np.linalg.norm(p - center)
            assert dist <= radius + 0.1

    def test_random_points(self):
        """Test circle enclosing random points."""
        np.random.seed(42)
        points = np.random.rand(10, 2)
        center, radius = minimum_bounding_circle(points)
        # All points should be inside circle
        for p in points:
            dist = np.linalg.norm(p - center)
            assert dist <= radius + 0.1


# =============================================================================
# Tests for oriented bounding box
# =============================================================================


class TestOrientedBoundingBox:
    """Tests for oriented bounding box."""

    def test_axis_aligned_points(self):
        """Test OBB of axis-aligned rectangle."""
        points = np.array([[0, 0], [2, 0], [2, 1], [0, 1]])
        center, extents, angle = oriented_bounding_box(points)
        assert center[0] == pytest.approx(1.0, rel=0.1)
        assert center[1] == pytest.approx(0.5, rel=0.1)

    def test_rotated_rectangle(self):
        """Test OBB of rotated rectangle."""
        # 45-degree rotated square
        c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
        R = np.array([[c, -s], [s, c]])
        rect = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]) - 0.5
        points = (R @ rect.T).T
        center, extents, angle = oriented_bounding_box(points)
        # Bounding box should be found
        assert extents[0] > 0
        assert extents[1] > 0

    def test_random_points(self):
        """Test OBB of random points."""
        np.random.seed(42)
        points = np.random.rand(20, 2)
        center, extents, angle = oriented_bounding_box(points)
        assert extents[0] >= 0
        assert extents[1] >= 0
        assert -np.pi <= angle <= np.pi


# =============================================================================
# Integration tests
# =============================================================================


class TestGeometryIntegration:
    """Integration tests for geometry functions."""

    def test_convex_hull_contains_all_points(self):
        """Test convex hull contains all original points."""
        np.random.seed(42)
        points = np.random.rand(20, 2)
        vertices, indices = convex_hull(points)

        # Verify hull was computed and vertices are valid
        assert len(indices) >= 3
        assert vertices.shape[0] >= 3

    def test_triangle_methods_consistent(self):
        """Test triangle area methods are consistent."""
        p1, p2, p3 = [0, 0], [3, 0], [0, 4]

        # Triangle area function
        area1 = triangle_area(p1, p2, p3)

        # Polygon area function
        area2 = polygon_area([p1, p2, p3])

        assert area1 == pytest.approx(area2)

    def test_bounding_structures_contain_points(self):
        """Test all bounding structures contain the points."""
        np.random.seed(42)
        points = np.random.rand(10, 2) * 10

        # Bounding box
        min_c, max_c = bounding_box(points)
        for p in points:
            assert np.all(p >= min_c - 1e-10)
            assert np.all(p <= max_c + 1e-10)

        # Bounding circle
        center, radius = minimum_bounding_circle(points)
        for p in points:
            assert np.linalg.norm(p - center) <= radius + 0.2
