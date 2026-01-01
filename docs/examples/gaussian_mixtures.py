"""
Gaussian Mixtures Example
=========================

This example demonstrates Gaussian mixture operations and clustering
algorithms in PyTCL:

Gaussian Mixture Operations:
- Component representation and manipulation
- Moment matching (computing mean and covariance)
- Mixture merging and reduction
- Runnalls' and West's reduction algorithms

Clustering Algorithms:
- K-means with K-means++ initialization
- DBSCAN (density-based clustering)
- Hierarchical/agglomerative clustering
- Elbow method for K selection

These algorithms are essential for multi-target tracking (PHD filters),
hypothesis reduction in MHT, and general density estimation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global flag to control plotting
SHOW_PLOTS = True


def create_ellipse_trace(mean, cov, color="blue", opacity=0.3, n_std=2, name=None):
    """Create a plotly trace for an ellipse representing a 2D Gaussian covariance."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Angle in radians
    angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

    # Semi-axes lengths
    a = n_std * np.sqrt(eigvals[0])
    b = n_std * np.sqrt(eigvals[1])

    # Parametric ellipse
    t = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotate
    x_rot = x * np.cos(angle) - y * np.sin(angle) + mean[0]
    y_rot = x * np.sin(angle) + y * np.cos(angle) + mean[1]

    return go.Scatter(
        x=x_rot,
        y=y_rot,
        mode="lines",
        fill="toself",
        fillcolor=color,
        opacity=opacity,
        line=dict(color=color, width=2),
        name=name,
        showlegend=name is not None,
    )


from pytcl.clustering import (  # Gaussian mixture operations; K-means; DBSCAN; Hierarchical clustering
    DBSCANResult,
    GaussianComponent,
    GaussianMixture,
    HierarchicalResult,
    KMeansResult,
    agglomerative_clustering,
    compute_distance_matrix,
    cut_dendrogram,
    dbscan,
    dbscan_predict,
    kmeans,
    kmeans_elbow,
    kmeans_plusplus_init,
    merge_gaussians,
    moment_match,
    prune_mixture,
    reduce_mixture_runnalls,
    reduce_mixture_west,
    runnalls_merge_cost,
    west_merge_cost,
)


def demo_gaussian_components():
    """Demonstrate Gaussian component operations."""
    print("=" * 70)
    print("Gaussian Component Operations Demo")
    print("=" * 70)

    # Create individual Gaussian components
    comp1 = GaussianComponent(
        weight=0.4,
        mean=np.array([0.0, 0.0]),
        covariance=np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    comp2 = GaussianComponent(
        weight=0.6,
        mean=np.array([3.0, 3.0]),
        covariance=np.array([[2.0, 0.5], [0.5, 2.0]]),
    )

    print("\nComponent 1:")
    print(f"  Weight: {comp1.weight}")
    print(f"  Mean: {comp1.mean}")
    print(f"  Covariance diagonal: {np.diag(comp1.covariance)}")

    print("\nComponent 2:")
    print(f"  Weight: {comp2.weight}")
    print(f"  Mean: {comp2.mean}")
    print(f"  Covariance diagonal: {np.diag(comp2.covariance)}")

    # Create a mixture
    mixture = GaussianMixture([comp1, comp2])
    print(f"\nMixture has {len(mixture)} components")
    print(f"Total weight: {sum(c.weight for c in mixture.components)}")


def demo_moment_matching():
    """Demonstrate moment matching for mixture approximation."""
    print("\n" + "=" * 70)
    print("Moment Matching Demo")
    print("=" * 70)

    # Create a mixture of 3 components
    weights = np.array([0.3, 0.5, 0.2])
    means = [np.array([0.0, 0.0]), np.array([2.0, 1.0]), np.array([1.0, 3.0])]
    covariances = [np.eye(2) * 0.5, np.eye(2) * 0.8, np.eye(2) * 0.3]

    print("\nOriginal mixture (3 components):")
    for i, (w, m) in enumerate(zip(weights, means)):
        print(f"  Component {i+1}: weight={w:.1f}, mean={m}")

    # Moment match to single Gaussian - takes (weights, means, covariances)
    mean, cov = moment_match(weights, means, covariances)

    print("\nMoment-matched single Gaussian:")
    print(f"  Mean: ({mean[0]:.3f}, {mean[1]:.3f})")
    print(f"  Covariance:\n{cov}")

    # The mean should be the weighted average
    weighted_mean = sum(w * m for w, m in zip(weights, means))
    print(f"\nVerification - weighted mean: ({weighted_mean[0]:.3f}, " f"{weighted_mean[1]:.3f})")


def demo_mixture_merging():
    """Demonstrate merging Gaussian components."""
    print("\n" + "=" * 70)
    print("Mixture Merging Demo")
    print("=" * 70)

    # Two nearby components
    comp1 = GaussianComponent(0.4, np.array([0.0, 0.0]), np.eye(2) * 1.0)
    comp2 = GaussianComponent(0.3, np.array([0.5, 0.5]), np.eye(2) * 1.0)

    # Two far apart components
    comp3 = GaussianComponent(0.3, np.array([5.0, 5.0]), np.eye(2) * 1.0)

    print("\nThree components:")
    print(f"  1: mean={comp1.mean}, weight={comp1.weight}")
    print(f"  2: mean={comp2.mean}, weight={comp2.weight}")
    print(f"  3: mean={comp3.mean}, weight={comp3.weight}")

    # Merge costs
    cost_12 = runnalls_merge_cost(comp1, comp2)
    cost_13 = runnalls_merge_cost(comp1, comp3)
    cost_23 = runnalls_merge_cost(comp2, comp3)

    print("\nRunnalls merge costs (lower = better candidates for merging):")
    print(f"  Cost(1,2): {cost_12:.4f}")
    print(f"  Cost(1,3): {cost_13:.4f}")
    print(f"  Cost(2,3): {cost_23:.4f}")

    # Merge the closest pair
    merge_result = merge_gaussians(comp1, comp2)
    merged = merge_result.component

    print("\nMerged component (1+2):")
    print(f"  Weight: {merged.weight:.2f}")
    print(f"  Mean: ({merged.mean[0]:.3f}, {merged.mean[1]:.3f})")
    print(f"  Merge cost: {merge_result.cost:.4f}")


def demo_mixture_reduction():
    """Demonstrate mixture reduction algorithms."""
    print("\n" + "=" * 70)
    print("Mixture Reduction Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create a large mixture (e.g., from PHD filter output)
    n_components = 20
    components = []

    # Cluster 1: around (0, 0)
    for _ in range(8):
        mean = np.array([0, 0]) + np.random.randn(2) * 0.5
        cov = np.eye(2) * (0.3 + np.random.rand() * 0.2)
        weight = 0.3 + np.random.rand() * 0.4
        components.append(GaussianComponent(weight, mean, cov))

    # Cluster 2: around (5, 3)
    for _ in range(7):
        mean = np.array([5, 3]) + np.random.randn(2) * 0.5
        cov = np.eye(2) * (0.3 + np.random.rand() * 0.2)
        weight = 0.2 + np.random.rand() * 0.3
        components.append(GaussianComponent(weight, mean, cov))

    # Scattered components
    for _ in range(5):
        mean = np.random.rand(2) * 10
        cov = np.eye(2) * 0.5
        weight = 0.05 + np.random.rand() * 0.1
        components.append(GaussianComponent(weight, mean, cov))

    print(f"\nOriginal mixture: {len(components)} components")
    print(f"Total weight: {sum(c.weight for c in components):.2f}")

    # Prune low-weight components
    pruned = prune_mixture(components, weight_threshold=0.1)
    print(f"\nAfter pruning (weight_threshold=0.1): {len(pruned)} components")

    # Reduce using Runnalls' algorithm
    n_target = 5
    result_runnalls = reduce_mixture_runnalls(components, n_target)
    print(f"\nRunnalls reduction to {n_target} components:")
    print(f"  Final components: {result_runnalls.n_reduced}")
    print(f"  Total merge cost: {result_runnalls.total_cost:.4f}")
    for i, c in enumerate(result_runnalls.components):
        print(f"    {i+1}: weight={c.weight:.3f}, " f"mean=({c.mean[0]:.2f}, {c.mean[1]:.2f})")

    # Reduce using West's algorithm
    result_west = reduce_mixture_west(components, n_target)
    print(f"\nWest reduction to {n_target} components:")
    print(f"  Final components: {result_west.n_reduced}")
    print(f"  Total merge cost: {result_west.total_cost:.4f}")


def demo_kmeans():
    """Demonstrate K-means clustering."""
    print("\n" + "=" * 70)
    print("K-Means Clustering Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate clustered data
    n_per_cluster = 50
    centers_true = np.array(
        [
            [0, 0],
            [5, 5],
            [0, 5],
        ]
    )

    data = []
    for center in centers_true:
        cluster = center + np.random.randn(n_per_cluster, 2) * 0.8
        data.append(cluster)
    data = np.vstack(data)

    print(f"\nGenerated {len(data)} points in 3 clusters")
    print(f"True centers:\n{centers_true}")

    # K-means clustering
    result = kmeans(data, n_clusters=3, max_iter=100)

    print(f"\nK-means result:")
    print(f"  Iterations: {result.n_iter}")
    print(f"  Inertia (within-cluster sum of squares): {result.inertia:.2f}")
    print(f"  Found centers:\n{result.centers}")

    # Cluster sizes
    unique, counts = np.unique(result.labels, return_counts=True)
    print("\n  Cluster sizes:", dict(zip(unique, counts)))

    # Plot K-means result
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Ground Truth Clusters", "K-means Clustering Result"],
        )

        # True clusters
        colors = ["blue", "green", "orange"]
        for i, center in enumerate(centers_true):
            mask = np.arange(len(data)) // n_per_cluster == i
            fig.add_trace(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode="markers",
                    marker=dict(color=colors[i], size=6, opacity=0.6),
                    name=f"True cluster {i}",
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=centers_true[:, 0],
                y=centers_true[:, 1],
                mode="markers",
                marker=dict(color="black", size=15, symbol="x", line=dict(width=3)),
                name="True centers",
            ),
            row=1,
            col=1,
        )

        # K-means result
        colors_km = ["red", "green", "blue"]
        for i in range(3):
            mask = result.labels == i
            fig.add_trace(
                go.Scatter(
                    x=data[mask, 0],
                    y=data[mask, 1],
                    mode="markers",
                    marker=dict(color=colors_km[i], size=6, opacity=0.6),
                    name=f"Cluster {i}",
                    showlegend=True,
                ),
                row=1,
                col=2,
            )

        fig.add_trace(
            go.Scatter(
                x=result.centers[:, 0],
                y=result.centers[:, 1],
                mode="markers",
                marker=dict(color="black", size=15, symbol="x", line=dict(width=3)),
                name="K-means centers",
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="y")
        fig.update_layout(height=500, width=1000, showlegend=True)
        fig.write_html("gaussian_kmeans.html")
        print("\n  [Plot saved to gaussian_kmeans.html]")


def demo_kmeans_plusplus():
    """Demonstrate K-means++ initialization."""
    print("\n" + "=" * 70)
    print("K-Means++ Initialization Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate data with 4 well-separated clusters
    data = np.vstack(
        [
            np.random.randn(30, 2) + [0, 0],
            np.random.randn(30, 2) + [10, 0],
            np.random.randn(30, 2) + [0, 10],
            np.random.randn(30, 2) + [10, 10],
        ]
    )

    n_clusters = 4

    # Random initialization
    random_centers = data[np.random.choice(len(data), n_clusters, replace=False)]

    # K-means++ initialization
    plusplus_centers = kmeans_plusplus_init(data, n_clusters)

    print(f"\nComparing initialization methods for n_clusters={n_clusters}:")

    # Run K-means with each initialization
    # For random, run multiple times (use init='random' for random init)
    results_random = []
    for _ in range(10):
        idx = np.random.choice(len(data), n_clusters, replace=False)
        result = kmeans(data, n_clusters=n_clusters, init=data[idx], n_init=1)
        results_random.append(result.inertia)

    result_plusplus = kmeans(data, n_clusters=n_clusters, init=plusplus_centers, n_init=1)

    print(f"\n  Random initialization (10 runs):")
    print(f"    Mean inertia: {np.mean(results_random):.2f}")
    print(f"    Std inertia: {np.std(results_random):.2f}")
    print(f"    Best inertia: {np.min(results_random):.2f}")

    print(f"\n  K-means++ initialization:")
    print(f"    Inertia: {result_plusplus.inertia:.2f}")

    print("\nNote: K-means++ typically provides better, more consistent results.")


def demo_elbow_method():
    """Demonstrate elbow method for K selection."""
    print("\n" + "=" * 70)
    print("Elbow Method Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate data with 3 true clusters
    data = np.vstack(
        [
            np.random.randn(40, 2) + [0, 0],
            np.random.randn(40, 2) + [4, 4],
            np.random.randn(40, 2) + [8, 0],
        ]
    )

    print(f"\nData: 120 points from 3 true clusters")
    print("\nInertia for different K values:")
    print("-" * 40)

    elbow_result = kmeans_elbow(data, k_range=range(1, 8))
    k_values = elbow_result["k_values"]
    inertias = elbow_result["inertias"]

    for k, inertia in zip(k_values, inertias):
        bar = "#" * int(inertia / max(inertias) * 30)
        print(f"  K={k}: {inertia:>8.1f} {bar}")

    print("\nThe 'elbow' should appear around K=3")
    print("(where adding more clusters gives diminishing returns)")

    # Plot elbow method
    if SHOW_PLOTS:
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=list(k_values),
                y=inertias,
                mode="lines+markers",
                line=dict(color="blue", width=2),
                marker=dict(size=10),
                name="Inertia",
            )
        )

        fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="True K=3")

        fig.update_layout(
            title="Elbow Method for K Selection",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia (Within-cluster Sum of Squares)",
            height=500,
            width=700,
            showlegend=True,
        )
        fig.write_html("gaussian_elbow.html")
        print("\n  [Plot saved to gaussian_elbow.html]")


def demo_dbscan():
    """Demonstrate DBSCAN clustering."""
    print("\n" + "=" * 70)
    print("DBSCAN Clustering Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate data: two dense clusters + noise
    cluster1 = np.random.randn(50, 2) * 0.5 + [0, 0]
    cluster2 = np.random.randn(50, 2) * 0.5 + [4, 4]
    noise = np.random.uniform(-2, 8, (20, 2))  # Scattered noise points

    data = np.vstack([cluster1, cluster2, noise])

    print(f"\nData: 100 cluster points + 20 noise points")

    # DBSCAN clustering
    result = dbscan(data, eps=0.8, min_samples=5)

    print(f"\nDBSCAN result (eps=0.8, min_samples=5):")
    print(f"  Clusters found: {result.n_clusters}")
    print(f"  Core sample indices: {len(result.core_sample_indices)}")
    print(f"  Noise points: {result.n_noise}")

    # Cluster sizes
    unique_labels = np.unique(result.labels)
    for label in unique_labels:
        count = np.sum(result.labels == label)
        if label == -1:
            print(f"  Noise points: {count}")
        else:
            print(f"  Cluster {label}: {count} points")

    print("\nNote: DBSCAN identifies noise points (label=-1)")
    print("and doesn't require specifying the number of clusters.")

    # Plot DBSCAN result
    if SHOW_PLOTS:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=[
                "Ground Truth",
                f"DBSCAN Result ({result.n_clusters} clusters)",
            ],
        )

        # Ground truth
        fig.add_trace(
            go.Scatter(
                x=cluster1[:, 0],
                y=cluster1[:, 1],
                mode="markers",
                marker=dict(color="blue", size=8, opacity=0.6),
                name="Cluster 1",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=cluster2[:, 0],
                y=cluster2[:, 1],
                mode="markers",
                marker=dict(color="green", size=8, opacity=0.6),
                name="Cluster 2",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=noise[:, 0],
                y=noise[:, 1],
                mode="markers",
                marker=dict(color="red", size=8, opacity=0.6),
                name="Noise",
            ),
            row=1,
            col=1,
        )

        # DBSCAN result
        colors = ["blue", "green", "purple", "orange"]
        for label in unique_labels:
            mask = result.labels == label
            if label == -1:
                fig.add_trace(
                    go.Scatter(
                        x=data[mask, 0],
                        y=data[mask, 1],
                        mode="markers",
                        marker=dict(color="red", size=8, opacity=0.6, symbol="x"),
                        name="Noise",
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=data[mask, 0],
                        y=data[mask, 1],
                        mode="markers",
                        marker=dict(color=colors[label % len(colors)], size=8, opacity=0.6),
                        name=f"Cluster {label}",
                        showlegend=True,
                    ),
                    row=1,
                    col=2,
                )

        # Mark core samples
        core_mask = np.zeros(len(data), dtype=bool)
        core_mask[result.core_sample_indices] = True
        fig.add_trace(
            go.Scatter(
                x=data[core_mask, 0],
                y=data[core_mask, 1],
                mode="markers",
                marker=dict(color="rgba(0,0,0,0)", size=15, line=dict(color="black", width=1)),
                name="Core samples",
                showlegend=True,
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text="x")
        fig.update_yaxes(title_text="y")
        fig.update_layout(height=500, width=1000, showlegend=True)
        fig.write_html("gaussian_dbscan.html")
        print("\n  [Plot saved to gaussian_dbscan.html]")


def demo_hierarchical():
    """Demonstrate hierarchical clustering."""
    print("\n" + "=" * 70)
    print("Hierarchical Clustering Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate small dataset for visualization
    data = np.array(
        [
            [0, 0],
            [0.5, 0.5],
            [1, 0],  # Cluster A
            [5, 5],
            [5.5, 5.5],
            [5, 6],  # Cluster B
            [2.5, 2.5],  # Between clusters
        ]
    )

    print(f"\nData points:\n{data}")

    # Compute distance matrix
    dist_matrix = compute_distance_matrix(data)
    print(f"\nDistance matrix:\n{np.round(dist_matrix, 2)}")

    # Agglomerative clustering
    result = agglomerative_clustering(data, linkage="average")

    print("\nHierarchical clustering (average linkage):")
    print(f"  Labels: {result.labels}")
    print(f"  Number of clusters: {result.n_clusters}")

    # Cut at different thresholds
    n_samples = len(data)
    for threshold in [1.0, 3.0, 5.0]:
        labels = cut_dendrogram(result.linkage_matrix, n_samples, distance_threshold=threshold)
        n_clusters = len(set(labels))
        print(f"  Threshold {threshold:.1f}: {n_clusters} clusters, labels={list(labels)}")


def demo_tracking_application():
    """Demonstrate mixture reduction in tracking context."""
    print("\n" + "=" * 70)
    print("Tracking Application Demo")
    print("=" * 70)

    np.random.seed(42)

    # Simulated PHD filter output: mixture representing target density
    # After several updates, the mixture can have many components

    # True targets at these locations
    true_targets = np.array(
        [
            [10.0, 20.0],
            [30.0, 40.0],
            [25.0, 25.0],
        ]
    )

    print(f"\nTrue target positions: {len(true_targets)} targets")
    for i, t in enumerate(true_targets):
        print(f"  Target {i+1}: ({t[0]:.1f}, {t[1]:.1f})")

    # Create mixture with components clustered around true targets
    # Plus some spurious components (false alarms, etc.)
    components = []

    # Components near true targets (higher weight)
    for target in true_targets:
        for _ in range(4):
            mean = target + np.random.randn(2) * 1.0
            cov = np.eye(2) * (0.5 + np.random.rand() * 0.5)
            weight = 0.6 + np.random.rand() * 0.4
            components.append(GaussianComponent(weight, mean, cov))

    # Spurious components (lower weight)
    for _ in range(8):
        mean = np.random.uniform(5, 45, 2)
        cov = np.eye(2) * 2.0
        weight = 0.05 + np.random.rand() * 0.1
        components.append(GaussianComponent(weight, mean, cov))

    print(f"\nPHD mixture: {len(components)} components")
    total_weight = sum(c.weight for c in components)
    print(f"  Total weight (expected target count): {total_weight:.2f}")

    # Reduce to extract target estimates
    n_expected = int(round(total_weight))
    reduced = reduce_mixture_runnalls(components, n_expected)

    print(f"\nAfter reduction to {n_expected} components:")
    for i, c in enumerate(reduced.components):
        # Find closest true target
        dists = [np.linalg.norm(c.mean - t) for t in true_targets]
        closest = np.argmin(dists)
        error = min(dists)
        print(
            f"  Estimate {i+1}: ({c.mean[0]:.1f}, {c.mean[1]:.1f}), "
            f"weight={c.weight:.2f}, error to target {closest+1}={error:.2f}"
        )


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Gaussian Mixtures and Clustering Example")
    print("#" * 70)

    # Gaussian mixture operations
    demo_gaussian_components()
    demo_moment_matching()
    demo_mixture_merging()
    demo_mixture_reduction()

    # Clustering algorithms
    demo_kmeans()
    demo_kmeans_plusplus()
    demo_elbow_method()
    demo_dbscan()
    demo_hierarchical()

    # Application
    demo_tracking_application()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: gaussian_kmeans.html, gaussian_elbow.html, gaussian_dbscan.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
