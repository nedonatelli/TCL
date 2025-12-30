"""
Geometric primitives and calculations.

This module provides:
- Point-in-polygon tests
- Convex hull computation
- Line and plane intersections
- Triangle operations
- Bounding box computation
"""

from tcl.mathematical_functions.geometry.geometry import (
    point_in_polygon,
    points_in_polygon,
    convex_hull,
    convex_hull_area,
    polygon_area,
    polygon_centroid,
    line_intersection,
    line_plane_intersection,
    point_to_line_distance,
    point_to_line_segment_distance,
    triangle_area,
    barycentric_coordinates,
    delaunay_triangulation,
    bounding_box,
    minimum_bounding_circle,
    oriented_bounding_box,
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
