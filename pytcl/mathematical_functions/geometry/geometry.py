"""
Geometric primitives and calculations.

This module provides geometric functions for points, lines, planes,
polygons, and related operations used in tracking applications.
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import ConvexHull, Delaunay


def point_in_polygon(
    point: ArrayLike,
    polygon: ArrayLike,
) -> bool:
    """
    Test if a point is inside a polygon.

    Uses the ray casting algorithm.

    Parameters
    ----------
    point : array_like
        Point coordinates (x, y).
    polygon : array_like
        Polygon vertices of shape (n, 2), ordered.

    Returns
    -------
    inside : bool
        True if point is inside the polygon.

    Examples
    --------
    >>> polygon = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    >>> point_in_polygon([0.5, 0.5], polygon)
    True
    >>> point_in_polygon([2, 2], polygon)
    False
    """
    point = np.asarray(point, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)

    x, y = point[0], point[1]
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def points_in_polygon(
    points: ArrayLike,
    polygon: ArrayLike,
) -> NDArray[np.bool_]:
    """
    Test if multiple points are inside a polygon.

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, 2).
    polygon : array_like
        Polygon vertices of shape (m, 2).

    Returns
    -------
    inside : ndarray
        Boolean array of shape (n,).
    """
    points = np.asarray(points, dtype=np.float64)
    polygon = np.asarray(polygon, dtype=np.float64)

    if points.ndim == 1:
        return np.array([point_in_polygon(points, polygon)])

    return np.array([point_in_polygon(p, polygon) for p in points])


def convex_hull(points: ArrayLike) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """
    Compute the convex hull of a set of points.

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, d).

    Returns
    -------
    vertices : ndarray
        Vertices of the convex hull.
    indices : ndarray
        Indices into points of the hull vertices.

    Examples
    --------
    >>> points = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0.5]])
    >>> vertices, indices = convex_hull(points)
    >>> len(indices)
    3
    """
    points = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(points)
    return points[hull.vertices], hull.vertices


def convex_hull_area(points: ArrayLike) -> float:
    """
    Compute the area (or volume) of the convex hull.

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, d).

    Returns
    -------
    area : float
        Area (2D) or volume (3D) of the convex hull.
    """
    points = np.asarray(points, dtype=np.float64)
    hull = ConvexHull(points)
    return hull.volume


def polygon_area(vertices: ArrayLike) -> float:
    """
    Compute the area of a polygon using the shoelace formula.

    Parameters
    ----------
    vertices : array_like
        Polygon vertices of shape (n, 2), ordered.

    Returns
    -------
    area : float
        Signed area (positive if counterclockwise).

    Examples
    --------
    >>> polygon_area([[0, 0], [1, 0], [1, 1], [0, 1]])
    1.0
    """
    vertices = np.asarray(vertices, dtype=np.float64)

    x = vertices[:, 0]
    y = vertices[:, 1]

    # Shoelace formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area


def polygon_centroid(vertices: ArrayLike) -> NDArray[np.floating]:
    """
    Compute the centroid of a polygon.

    Parameters
    ----------
    vertices : array_like
        Polygon vertices of shape (n, 2), ordered.

    Returns
    -------
    centroid : ndarray
        Centroid coordinates (x, y).
    """
    vertices = np.asarray(vertices, dtype=np.float64)

    x = vertices[:, 0]
    y = vertices[:, 1]

    # Signed area
    a = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))

    # Centroid
    factor = 1.0 / (3.0 * a)
    cx = factor * np.sum(
        (x + np.roll(x, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
    )
    cy = factor * np.sum(
        (y + np.roll(y, -1)) * (x * np.roll(y, -1) - np.roll(x, -1) * y)
    )

    return np.array([cx, cy], dtype=np.float64)


def line_intersection(
    p1: ArrayLike,
    p2: ArrayLike,
    p3: ArrayLike,
    p4: ArrayLike,
) -> Optional[NDArray[np.floating]]:
    """
    Find the intersection point of two line segments.

    Parameters
    ----------
    p1, p2 : array_like
        Endpoints of first line segment.
    p3, p4 : array_like
        Endpoints of second line segment.

    Returns
    -------
    intersection : ndarray or None
        Intersection point, or None if segments don't intersect.

    Examples
    --------
    >>> line_intersection([0, 0], [1, 1], [0, 1], [1, 0])
    array([0.5, 0.5])
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    p4 = np.asarray(p4, dtype=np.float64)

    d1 = p2 - p1
    d2 = p4 - p3

    cross = d1[0] * d2[1] - d1[1] * d2[0]

    if np.abs(cross) < 1e-12:
        return None  # Parallel

    d3 = p3 - p1
    t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
    u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * d1

    return None


def line_plane_intersection(
    line_point: ArrayLike,
    line_dir: ArrayLike,
    plane_point: ArrayLike,
    plane_normal: ArrayLike,
) -> Optional[NDArray[np.floating]]:
    """
    Find the intersection of a line and a plane.

    Parameters
    ----------
    line_point : array_like
        A point on the line.
    line_dir : array_like
        Direction vector of the line.
    plane_point : array_like
        A point on the plane.
    plane_normal : array_like
        Normal vector of the plane.

    Returns
    -------
    intersection : ndarray or None
        Intersection point, or None if line is parallel to plane.
    """
    line_point = np.asarray(line_point, dtype=np.float64)
    line_dir = np.asarray(line_dir, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)

    denom = np.dot(line_dir, plane_normal)

    if np.abs(denom) < 1e-12:
        return None  # Line parallel to plane

    t = np.dot(plane_point - line_point, plane_normal) / denom
    return line_point + t * line_dir


def point_to_line_distance(
    point: ArrayLike,
    line_p1: ArrayLike,
    line_p2: ArrayLike,
) -> float:
    """
    Compute the distance from a point to a line.

    Parameters
    ----------
    point : array_like
        Point coordinates.
    line_p1, line_p2 : array_like
        Two points defining the line.

    Returns
    -------
    distance : float
        Perpendicular distance from point to line.

    Examples
    --------
    >>> point_to_line_distance([0, 1], [0, 0], [1, 0])
    1.0
    """
    point = np.asarray(point, dtype=np.float64)
    line_p1 = np.asarray(line_p1, dtype=np.float64)
    line_p2 = np.asarray(line_p2, dtype=np.float64)

    line_vec = line_p2 - line_p1
    point_vec = point - line_p1

    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return np.linalg.norm(point_vec)

    cross = np.abs(np.cross(line_vec, point_vec))
    return cross / line_len


def point_to_line_segment_distance(
    point: ArrayLike,
    seg_p1: ArrayLike,
    seg_p2: ArrayLike,
) -> float:
    """
    Compute the distance from a point to a line segment.

    Parameters
    ----------
    point : array_like
        Point coordinates.
    seg_p1, seg_p2 : array_like
        Endpoints of the line segment.

    Returns
    -------
    distance : float
        Distance from point to nearest point on segment.
    """
    point = np.asarray(point, dtype=np.float64)
    seg_p1 = np.asarray(seg_p1, dtype=np.float64)
    seg_p2 = np.asarray(seg_p2, dtype=np.float64)

    seg_vec = seg_p2 - seg_p1
    point_vec = point - seg_p1

    seg_len_sq = np.dot(seg_vec, seg_vec)

    if seg_len_sq < 1e-12:
        return np.linalg.norm(point_vec)

    t = max(0, min(1, np.dot(point_vec, seg_vec) / seg_len_sq))
    projection = seg_p1 + t * seg_vec

    return np.linalg.norm(point - projection)


def triangle_area(
    p1: ArrayLike,
    p2: ArrayLike,
    p3: ArrayLike,
) -> float:
    """
    Compute the area of a triangle.

    Parameters
    ----------
    p1, p2, p3 : array_like
        Vertices of the triangle.

    Returns
    -------
    area : float
        Area of the triangle.

    Examples
    --------
    >>> triangle_area([0, 0], [1, 0], [0, 1])
    0.5
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    v1 = p2 - p1
    v2 = p3 - p1

    if len(p1) == 2:
        return 0.5 * np.abs(v1[0] * v2[1] - v1[1] * v2[0])
    else:
        return 0.5 * np.linalg.norm(np.cross(v1, v2))


def barycentric_coordinates(
    point: ArrayLike,
    p1: ArrayLike,
    p2: ArrayLike,
    p3: ArrayLike,
) -> NDArray[np.floating]:
    """
    Compute barycentric coordinates of a point in a triangle.

    Parameters
    ----------
    point : array_like
        Point coordinates.
    p1, p2, p3 : array_like
        Triangle vertices.

    Returns
    -------
    coords : ndarray
        Barycentric coordinates (λ1, λ2, λ3) where point = λ1*p1 + λ2*p2 + λ3*p3.

    Notes
    -----
    If all coordinates are in [0, 1], the point is inside the triangle.
    """
    point = np.asarray(point, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)

    v0 = p3 - p1
    v1 = p2 - p1
    v2 = point - p1

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom

    return np.array([1 - u - v, v, u], dtype=np.float64)


def delaunay_triangulation(
    points: ArrayLike,
) -> Delaunay:
    """
    Compute Delaunay triangulation.

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, 2) or (n, 3).

    Returns
    -------
    tri : Delaunay
        Delaunay triangulation object.
        - tri.simplices: Indices of triangle vertices
        - tri.neighbors: Indices of neighboring triangles
    """
    points = np.asarray(points, dtype=np.float64)
    return Delaunay(points)


def bounding_box(
    points: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute axis-aligned bounding box.

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, d).

    Returns
    -------
    min_corner : ndarray
        Minimum coordinates.
    max_corner : ndarray
        Maximum coordinates.

    Examples
    --------
    >>> points = np.array([[0, 1], [2, 3], [1, 2]])
    >>> min_c, max_c = bounding_box(points)
    >>> min_c
    array([0., 1.])
    >>> max_c
    array([2., 3.])
    """
    points = np.asarray(points, dtype=np.float64)
    return points.min(axis=0), points.max(axis=0)


def minimum_bounding_circle(
    points: ArrayLike,
) -> Tuple[NDArray[np.floating], float]:
    """
    Compute minimum enclosing circle (2D).

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, 2).

    Returns
    -------
    center : ndarray
        Center of the enclosing circle.
    radius : float
        Radius of the enclosing circle.

    Notes
    -----
    Uses Welzl's algorithm with expected O(n) time complexity.
    """
    points = np.asarray(points, dtype=np.float64)

    def circle_from_two_points(p1: Any, p2: Any) -> tuple[Any, Any]:
        center = (p1 + p2) / 2
        radius = np.linalg.norm(p1 - center)
        return center, radius

    def circle_from_three_points(p1: Any, p2: Any, p3: Any) -> tuple[Any, Any]:
        ax, ay = p1
        bx, by = p2
        cx, cy = p3

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if np.abs(d) < 1e-12:
            # Collinear points
            d1 = np.linalg.norm(p1 - p2)
            d2 = np.linalg.norm(p2 - p3)
            d3 = np.linalg.norm(p1 - p3)
            if d1 >= d2 and d1 >= d3:
                return circle_from_two_points(p1, p2)
            elif d2 >= d3:
                return circle_from_two_points(p2, p3)
            else:
                return circle_from_two_points(p1, p3)

        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d

        center = np.array([ux, uy])
        radius = np.linalg.norm(p1 - center)
        return center, radius

    def is_inside(c: Any, r: Any, p: Any) -> Any:
        return np.linalg.norm(p - c) <= r + 1e-10

    def welzl(P: Any, R: Any) -> tuple[Any, Any]:
        if len(P) == 0 or len(R) == 3:
            if len(R) == 0:
                return np.array([0.0, 0.0]), 0.0
            elif len(R) == 1:
                return R[0].copy(), 0.0
            elif len(R) == 2:
                return circle_from_two_points(R[0], R[1])
            else:
                return circle_from_three_points(R[0], R[1], R[2])

        idx = np.random.randint(len(P))
        p = P[idx]
        P_new = np.delete(P, idx, axis=0)

        c, r = welzl(P_new, R)

        if is_inside(c, r, p):
            return c, r

        return welzl(P_new, R + [p])

    # Shuffle for randomized algorithm
    points_shuffled = points.copy()
    np.random.shuffle(points_shuffled)

    return welzl(points_shuffled, [])


def oriented_bounding_box(
    points: ArrayLike,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], float]:
    """
    Compute minimum-area oriented bounding box (2D).

    Parameters
    ----------
    points : array_like
        Point coordinates of shape (n, 2).

    Returns
    -------
    center : ndarray
        Center of the bounding box.
    extents : ndarray
        Half-widths along each principal direction.
    angle : float
        Rotation angle in radians.
    """
    points = np.asarray(points, dtype=np.float64)

    # Get convex hull
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]

    min_area = np.inf
    best_box = None

    n = len(hull_points)
    for i in range(n):
        edge = hull_points[(i + 1) % n] - hull_points[i]
        angle = np.arctan2(edge[1], edge[0])

        # Rotation matrix
        c, s = np.cos(-angle), np.sin(-angle)
        R = np.array([[c, -s], [s, c]])

        # Rotate points
        rotated = (R @ hull_points.T).T

        # Axis-aligned bounding box of rotated points
        min_pt = rotated.min(axis=0)
        max_pt = rotated.max(axis=0)

        area = (max_pt[0] - min_pt[0]) * (max_pt[1] - min_pt[1])

        if area < min_area:
            min_area = area
            center_rotated = (min_pt + max_pt) / 2
            extents = (max_pt - min_pt) / 2

            # Rotate center back
            R_inv = np.array([[c, s], [-s, c]])
            center = R_inv @ center_rotated
            best_box = (center, extents, angle)

    return best_box


__all__ = [
    "point_in_polygon",
    "points_in_polygon",
    "convex_hull",
    "convex_hull_area",
    "polygon_area",
    "polygon_centroid",
    "line_intersection",
    "line_plane_intersection",
    "point_to_line_distance",
    "point_to_line_segment_distance",
    "triangle_area",
    "barycentric_coordinates",
    "delaunay_triangulation",
    "bounding_box",
    "minimum_bounding_circle",
    "oriented_bounding_box",
]
