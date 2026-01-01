"""
Geometric primitives and calculations.

This module provides:
- Point-in-polygon tests
- Convex hull computation
- Line and plane intersections
- Triangle operations
- Bounding box computation
"""

from pytcl.mathematical_functions.geometry.geometry import barycentric_coordinates
from pytcl.mathematical_functions.geometry.geometry import bounding_box
from pytcl.mathematical_functions.geometry.geometry import convex_hull
from pytcl.mathematical_functions.geometry.geometry import convex_hull_area
from pytcl.mathematical_functions.geometry.geometry import delaunay_triangulation
from pytcl.mathematical_functions.geometry.geometry import line_intersection
from pytcl.mathematical_functions.geometry.geometry import line_plane_intersection
from pytcl.mathematical_functions.geometry.geometry import minimum_bounding_circle
from pytcl.mathematical_functions.geometry.geometry import oriented_bounding_box
from pytcl.mathematical_functions.geometry.geometry import point_in_polygon
from pytcl.mathematical_functions.geometry.geometry import point_to_line_distance
from pytcl.mathematical_functions.geometry.geometry import (
    point_to_line_segment_distance,
)
from pytcl.mathematical_functions.geometry.geometry import points_in_polygon
from pytcl.mathematical_functions.geometry.geometry import polygon_area
from pytcl.mathematical_functions.geometry.geometry import polygon_centroid
from pytcl.mathematical_functions.geometry.geometry import triangle_area

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
