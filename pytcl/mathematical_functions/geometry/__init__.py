"""
Geometric primitives and calculations.

This module provides:
- Point-in-polygon tests
- Convex hull computation
- Line and plane intersections
- Triangle operations
- Bounding box computation
"""

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
