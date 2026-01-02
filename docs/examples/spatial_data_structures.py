"""
Spatial Data Structures Example
===============================

This example demonstrates spatial data structures in PyTCL for
efficient nearest neighbor queries and spatial indexing:

K-D Tree:
- Construction and querying
- K-nearest neighbor search
- Radius/range queries

Ball Tree:
- Alternative to K-D tree for high dimensions
- Similar query interface

R-Tree:
- Spatial indexing for bounding boxes
- Rectangle intersection queries

VP-Tree (Vantage Point Tree):
- Metric space indexing
- Works with any distance metric

Cover Tree:
- Approximate nearest neighbor search
- O(c^12 log n) query complexity

These data structures are essential for efficient data association
in multi-target tracking and spatial analysis applications.
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

# Output directory for generated plots
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "_static" / "images" / "examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.containers import (  # K-D Tree; Ball Tree; R-Tree; VP-Tree; Cover Tree
    BallTree,
    BoundingBox,
    CoverTree,
    KDTree,
    NearestNeighborResult,
    RTree,
    VPTree,
    box_from_point,
    box_from_points,
    merge_boxes,
)


def demo_kdtree_basics():
    """Demonstrate K-D tree construction and basic queries."""
    print("=" * 70)
    print("K-D Tree Basics Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate 2D point cloud
    n_points = 100
    points = np.random.randn(n_points, 2) * 10

    print(f"\nBuilding K-D tree with {n_points} 2D points")

    # Build tree
    tree = KDTree(points)

    print(f"Tree built successfully")
    print(f"Point cloud bounds:")
    print(f"  X: [{points[:, 0].min():.2f}, {points[:, 0].max():.2f}]")
    print(f"  Y: [{points[:, 1].min():.2f}, {points[:, 1].max():.2f}]")

    # Query point
    query = np.array([0.0, 0.0])
    print(f"\nQuery point: {query}")

    # K-nearest neighbors
    k = 5
    result = tree.query(query, k=k)

    print(f"\n{k} nearest neighbors:")
    # indices/distances are 2D arrays, extract the first row for single query
    for i, (idx, dist) in enumerate(zip(result.indices[0], result.distances[0])):
        print(f"  {i+1}. Point {idx}: {points[idx]} (distance={dist:.4f})")

    # Plot KDTree result
    if SHOW_PLOTS:
        fig = go.Figure()

        # All points
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="markers",
                marker=dict(color="lightblue", size=8, opacity=0.6),
                name="Points",
            )
        )

        # K nearest neighbors
        nn_indices = result.indices[0]
        fig.add_trace(
            go.Scatter(
                x=points[nn_indices, 0],
                y=points[nn_indices, 1],
                mode="markers",
                marker=dict(color="green", size=12, opacity=0.8),
                name=f"{k} nearest neighbors",
            )
        )

        # Query point
        fig.add_trace(
            go.Scatter(
                x=[query[0]],
                y=[query[1]],
                mode="markers",
                marker=dict(color="red", size=15, symbol="star"),
                name="Query",
            )
        )

        # Draw circle for max distance
        max_dist = result.distances[0, -1]
        theta = np.linspace(0, 2 * np.pi, 100)
        circle_x = query[0] + max_dist * np.cos(theta)
        circle_y = query[1] + max_dist * np.sin(theta)
        fig.add_trace(
            go.Scatter(
                x=circle_x,
                y=circle_y,
                mode="lines",
                line=dict(color="green", dash="dash", width=2),
                name="Search radius",
                showlegend=True,
            )
        )

        fig.update_layout(
            title="K-D Tree: K-Nearest Neighbor Query",
            xaxis_title="x",
            yaxis_title="y",
            height=600,
            width=600,
            showlegend=True,
            xaxis=dict(scaleanchor="y", scaleratio=1),
        )
        fig.write_html(str(OUTPUT_DIR / "spatial_kdtree.html"))
        print("\n  [Plot saved to spatial_kdtree.html]")


def demo_kdtree_queries():
    """Demonstrate K-D tree query types."""
    print("\n" + "=" * 70)
    print("K-D Tree Query Types Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create structured point cloud
    # Grid + noise
    x = np.linspace(-10, 10, 11)
    y = np.linspace(-10, 10, 11)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    points += np.random.randn(*points.shape) * 0.3

    tree = KDTree(points)

    print(f"\nGrid-based point cloud: {len(points)} points")

    # Different query types
    query = np.array([0.0, 0.0])

    # K-NN query
    k_values = [1, 5, 10, 20]
    print("\n--- K-Nearest Neighbors ---")
    for k in k_values:
        result = tree.query(query, k=k)
        max_dist = result.distances[0, -1]  # 2D array: [query_idx, neighbor_idx]
        print(f"  k={k:>2}: max distance = {max_dist:.4f}")

    # Radius query
    print("\n--- Radius Queries ---")
    radii = [1.0, 2.0, 5.0, 10.0]
    for r in radii:
        result = tree.query_radius(query, r)
        # query_radius returns list of index arrays (one per query point)
        print(f"  radius={r:.1f}: {len(result[0])} points found")


def demo_balltree():
    """Demonstrate Ball Tree for higher dimensions."""
    print("\n" + "=" * 70)
    print("Ball Tree Demo")
    print("=" * 70)

    np.random.seed(42)

    # Higher dimensional data
    n_points = 500
    n_dims = 10  # 10-dimensional space

    points = np.random.randn(n_points, n_dims)

    print(f"\nBuilding Ball Tree with {n_points} points in {n_dims}D space")

    tree = BallTree(points)

    # Query
    query = np.zeros(n_dims)
    k = 5

    result = tree.query(query, k=k)

    print(f"\nQuery: origin in {n_dims}D")
    print(f"{k} nearest neighbors:")
    for i, (idx, dist) in enumerate(zip(result.indices[0], result.distances[0])):
        print(f"  {i+1}. Point {idx}: distance = {dist:.4f}")

    print("\nNote: Ball Tree is often more efficient than K-D Tree")
    print("for higher dimensional data (curse of dimensionality).")


def demo_rtree():
    """Demonstrate R-Tree for bounding box indexing."""
    print("\n" + "=" * 70)
    print("R-Tree Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create bounding boxes (e.g., for spatial objects)
    n_boxes = 50
    boxes = []

    for i in range(n_boxes):
        # Random center and size
        center = np.random.uniform(-50, 50, 2)
        size = np.random.uniform(2, 10, 2)

        min_coords = center - size / 2
        max_coords = center + size / 2

        box = BoundingBox(min_coords=min_coords, max_coords=max_coords)
        boxes.append(box)

    print(f"\nCreated {n_boxes} bounding boxes")

    # Build R-Tree
    tree = RTree()
    for i, box in enumerate(boxes):
        tree.insert(box, i)

    print("R-Tree built successfully")

    # Query: find boxes intersecting a search region
    search_min = np.array([-10, -10])
    search_max = np.array([10, 10])
    search_box = BoundingBox(min_coords=search_min, max_coords=search_max)

    print(f"\nSearch region: ({search_min} to {search_max})")

    result = tree.query_intersect(search_box)

    print(f"Found {len(result.indices)} intersecting boxes")

    # Show some results
    if len(result.indices) > 0:
        print("\nFirst 5 intersecting boxes:")
        for idx in result.indices[:5]:
            box = boxes[idx]
            print(f"  Box {idx}: ({box.min_coords} to {box.max_coords})")

    # Plot R-Tree result
    if SHOW_PLOTS:
        fig = go.Figure()

        # Draw all boxes
        for i, box in enumerate(boxes):
            is_intersecting = i in result.indices
            color = "green" if is_intersecting else "lightblue"
            opacity = 0.6 if is_intersecting else 0.3

            # Create rectangle as a filled shape
            x0, y0 = box.min_coords
            x1, y1 = box.max_coords

            fig.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor=color,
                line=dict(color="black", width=1),
                opacity=opacity,
            )

        # Draw search region
        fig.add_shape(
            type="rect",
            x0=search_min[0],
            y0=search_min[1],
            x1=search_max[0],
            y1=search_max[1],
            fillcolor="rgba(0,0,0,0)",
            line=dict(color="red", width=3, dash="dash"),
        )

        # Add legend traces (invisible points for legend)
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=15, color="green", opacity=0.6),
                name="Intersecting",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=15, color="lightblue", opacity=0.3),
                name="Non-intersecting",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Search region",
            )
        )

        fig.update_layout(
            title=f"R-Tree: {len(result.indices)} boxes intersecting search region",
            xaxis_title="x",
            yaxis_title="y",
            height=700,
            width=700,
            showlegend=True,
            xaxis=dict(range=[-60, 60], scaleanchor="y", scaleratio=1),
            yaxis=dict(range=[-60, 60]),
        )
        fig.write_html(str(OUTPUT_DIR / "spatial_rtree.html"))
        print("\n  [Plot saved to spatial_rtree.html]")


def demo_bounding_box_operations():
    """Demonstrate bounding box utility functions."""
    print("\n" + "=" * 70)
    print("Bounding Box Operations Demo")
    print("=" * 70)

    # Create boxes
    box1 = BoundingBox(min_coords=np.array([0, 0]), max_coords=np.array([5, 5]))

    box2 = BoundingBox(min_coords=np.array([3, 3]), max_coords=np.array([8, 8]))

    box3 = BoundingBox(min_coords=np.array([10, 10]), max_coords=np.array([12, 12]))

    print("\nBox 1: (0,0) to (5,5)")
    print(f"  Center: {box1.center}")
    print(f"  Dimensions: {box1.dimensions}")
    print(f"  Volume: {box1.volume}")

    print("\nBox 2: (3,3) to (8,8)")
    print(f"  Center: {box2.center}")

    print("\nBox 3: (10,10) to (12,12)")
    print(f"  Center: {box3.center}")

    # Intersection tests
    print("\n--- Intersection Tests ---")
    print(f"  Box1 intersects Box2: {box1.intersects(box2)}")
    print(f"  Box1 intersects Box3: {box1.intersects(box3)}")
    print(f"  Box2 intersects Box3: {box2.intersects(box3)}")

    # Point containment
    test_points = [
        np.array([2.5, 2.5]),
        np.array([4.0, 4.0]),
        np.array([7.0, 7.0]),
    ]

    print("\n--- Point Containment Tests ---")
    for p in test_points:
        print(f"  Point {p}:")
        print(f"    In Box1: {box1.contains_point(p)}")
        print(f"    In Box2: {box2.contains_point(p)}")

    # Merge boxes
    merged = merge_boxes([box1, box2])  # Takes a list of boxes
    print("\n--- Merged Box (Box1 + Box2) ---")
    print(f"  Min: {merged.min_coords}")
    print(f"  Max: {merged.max_coords}")

    # Create box from points
    points = np.array([[1, 2], [5, 3], [2, 8], [7, 4]])
    bbox = box_from_points(points)
    print("\n--- Bounding Box of Points ---")
    print(f"  Points:\n{points}")
    print(f"  Bounding box: ({bbox.min_coords} to {bbox.max_coords})")


def demo_vptree():
    """Demonstrate VP-Tree for metric space indexing."""
    print("\n" + "=" * 70)
    print("VP-Tree Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate points
    n_points = 200
    points = np.random.randn(n_points, 3) * 5

    print(f"\nBuilding VP-Tree with {n_points} 3D points")

    tree = VPTree(points)

    # Query
    query = np.array([1.0, 1.0, 1.0])
    k = 5

    result = tree.query(query, k=k)

    print(f"\nQuery point: {query}")
    print(f"{k} nearest neighbors:")
    for i, (idx, dist) in enumerate(zip(result.indices[0], result.distances[0])):
        print(f"  {i+1}. Point {idx}: distance = {dist:.4f}")

    print("\nNote: VP-Tree works with any distance metric,")
    print("not just Euclidean distance.")


def demo_covertree():
    """Demonstrate Cover Tree for approximate nearest neighbor."""
    print("\n" + "=" * 70)
    print("Cover Tree Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate points
    n_points = 300
    points = np.random.randn(n_points, 4) * 3  # 4D

    print(f"\nBuilding Cover Tree with {n_points} 4D points")

    tree = CoverTree(points)

    # Query
    query = np.zeros(4)
    k = 5

    result = tree.query(query, k=k)

    print(f"\nQuery: origin in 4D")
    print(f"{k} nearest neighbors:")
    for i, (idx, dist) in enumerate(zip(result.indices[0], result.distances[0])):
        print(f"  {i+1}. Point {idx}: distance = {dist:.4f}")

    print("\nNote: Cover Tree provides O(c^12 log n) query complexity")
    print("where c is the expansion constant of the data.")


def demo_performance_comparison():
    """Compare performance of different spatial data structures."""
    print("\n" + "=" * 70)
    print("Performance Comparison Demo")
    print("=" * 70)

    import time

    np.random.seed(42)

    # Test data
    n_points = 5000
    n_queries = 100
    dims = 3
    k = 10

    points = np.random.randn(n_points, dims) * 10
    queries = np.random.randn(n_queries, dims) * 10

    print(f"\nDataset: {n_points} points in {dims}D")
    print(f"Queries: {n_queries} k-NN queries (k={k})")

    results = {}

    # K-D Tree
    t0 = time.time()
    kdtree = KDTree(points)
    build_time = time.time() - t0

    t0 = time.time()
    for q in queries:
        kdtree.query(q, k=k)
    query_time = time.time() - t0

    results["K-D Tree"] = (build_time, query_time)

    # Ball Tree
    t0 = time.time()
    balltree = BallTree(points)
    build_time = time.time() - t0

    t0 = time.time()
    for q in queries:
        balltree.query(q, k=k)
    query_time = time.time() - t0

    results["Ball Tree"] = (build_time, query_time)

    # VP Tree
    t0 = time.time()
    vptree = VPTree(points)
    build_time = time.time() - t0

    t0 = time.time()
    for q in queries:
        vptree.query(q, k=k)
    query_time = time.time() - t0

    results["VP-Tree"] = (build_time, query_time)

    # Cover Tree
    t0 = time.time()
    covertree = CoverTree(points)
    build_time = time.time() - t0

    t0 = time.time()
    for q in queries:
        covertree.query(q, k=k)
    query_time = time.time() - t0

    results["Cover Tree"] = (build_time, query_time)

    # Print results
    print("\n" + "-" * 50)
    print(f"{'Structure':<15} {'Build (ms)':>12} {'Query (ms)':>12}")
    print("-" * 50)
    for name, (build, query) in results.items():
        print(f"{name:<15} {build*1000:>12.2f} {query*1000:>12.2f}")

    print("\nNote: Performance depends on data distribution and dimensionality.")


def demo_tracking_application():
    """Demonstrate spatial indexing in tracking context."""
    print("\n" + "=" * 70)
    print("Tracking Application Demo")
    print("=" * 70)

    np.random.seed(42)

    # Simulated scenario: sensor provides measurements,
    # need to associate with predicted track positions

    # Track predictions
    n_tracks = 20
    track_positions = np.random.uniform(-100, 100, (n_tracks, 2))

    # Measurements (some from tracks, some false alarms)
    n_measurements = 30
    # First n_tracks measurements near track positions
    measurements = np.zeros((n_measurements, 2))
    for i in range(min(n_tracks, n_measurements)):
        measurements[i] = track_positions[i] + np.random.randn(2) * 2.0

    # Remaining are false alarms
    for i in range(n_tracks, n_measurements):
        measurements[i] = np.random.uniform(-100, 100, 2)

    print(f"\n{n_tracks} track predictions")
    print(
        f"{n_measurements} measurements ({n_tracks} true + "
        f"{n_measurements - n_tracks} false alarms)"
    )

    # Build spatial index on track predictions
    tree = KDTree(track_positions)

    # For each measurement, find nearest track
    print("\nMeasurement-to-track association using K-D tree:")
    print("-" * 50)

    gating_threshold = 5.0  # meters
    associations = []

    for m_idx, meas in enumerate(measurements):
        result = tree.query(meas, k=1)
        nearest_track = result.indices[0, 0]  # 2D array [query_idx, neighbor_idx]
        distance = result.distances[0, 0]

        if distance < gating_threshold:
            associations.append((m_idx, nearest_track, distance))

    print(f"Gating threshold: {gating_threshold} m")
    print(f"Measurements passing gate: {len(associations)}/{n_measurements}")

    # Show some associations
    print("\nFirst 5 associations:")
    for m_idx, t_idx, dist in associations[:5]:
        true_assoc = m_idx == t_idx  # Simplified ground truth
        status = "+" if true_assoc else "?"
        print(f"  Meas {m_idx:>2} -> Track {t_idx:>2} " f"(dist={dist:.2f}) {status}")

    # Radius query for gating
    print("\n--- Using Radius Query for Gating ---")
    meas_test = measurements[0]
    result = tree.query_radius(meas_test, gating_threshold)
    # query_radius returns list of index arrays (one per query point)
    print(f"Measurement 0: {len(result[0])} tracks within gate")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Spatial Data Structures Example")
    print("#" * 70)

    demo_kdtree_basics()
    demo_kdtree_queries()
    demo_balltree()
    demo_rtree()
    demo_bounding_box_operations()
    demo_vptree()
    demo_covertree()
    demo_performance_comparison()
    demo_tracking_application()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: spatial_kdtree.html, spatial_rtree.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
