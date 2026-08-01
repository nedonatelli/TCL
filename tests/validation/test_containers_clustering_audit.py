"""Correctness audit tests for pytcl.containers and pytcl.clustering.

Ground-truth verification:
- Spatial indexes (KDTree, BallTree, VPTree, CoverTree, RTree) against
  brute-force distance computation on seeded random point sets across
  dimensions 1-6, with duplicate points and exact ties.
- k-means / DBSCAN / hierarchical against scikit-learn / scipy references
  (skipped if scikit-learn is not installed).
- Gaussian mixture reduction against analytic moment preservation.
- Container classes (TrackList, MeasurementSet, ClusterSet) against
  straightforward list-comprehension ground truth.

Includes regression tests for the CoverTree query bugs fixed in this audit
(duplicate indices in k-NN results, missed neighbors due to invalid
level-based pruning after non-invariant-preserving insertion).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from pytcl.clustering import (
    GaussianComponent,
    GaussianMixture,
    agglomerative_clustering,
    assign_clusters,
    compute_distance_matrix,
    compute_neighbors,
    cut_dendrogram,
    dbscan,
    dbscan_predict,
    fcluster,
    kmeans,
    kmeans_elbow,
    merge_gaussians,
    moment_match,
    prune_mixture,
    reduce_mixture_runnalls,
    reduce_mixture_west,
    runnalls_merge_cost,
    update_centers,
    west_merge_cost,
)
from pytcl.containers import (
    BallTree,
    BoundingBox,
    ClusterSet,
    CoverTree,
    KDTree,
    Measurement,
    MeasurementSet,
    RTree,
    TrackList,
    VPTree,
    box_from_point,
    box_from_points,
    cluster_tracks_dbscan,
    cluster_tracks_kmeans,
    compute_cluster_centroid,
    merge_boxes,
)
from pytcl.trackers.multi_target import Track, TrackStatus

# =============================================================================
# Spatial index correctness vs brute force
# =============================================================================

TREE_FACTORIES = {
    "KDTree": lambda d: KDTree(d),
    "KDTree_leaf1": lambda d: KDTree(d, leaf_size=1),
    "BallTree": lambda d: BallTree(d),
    "VPTree": lambda d: VPTree(d),
    "CoverTree": lambda d: CoverTree(d),
    "RTree": lambda d: RTree.from_points(d),
    "RTree_max4": lambda d: RTree.from_points(d, max_entries=4),
}


def _make_dataset(dim, n, kind, rng):
    if kind == "uniform":
        return rng.uniform(-5, 5, (n, dim))
    if kind == "duplicates":
        base = rng.uniform(-5, 5, (max(n // 3, 1), dim))
        return base[rng.integers(0, len(base), n)]
    # grid: exact distance ties
    side = int(np.ceil(n ** (1 / dim)))
    grids = np.meshgrid(*[np.arange(side)] * dim)
    return np.stack([g.ravel() for g in grids], axis=1)[:n].astype(float)


def _brute_knn_distances(data, q, k):
    d = np.sqrt(np.sum((data - q) ** 2, axis=1))
    return np.sort(d)[: min(k, len(data))]


def _brute_radius(data, q, r):
    d = np.sqrt(np.sum((data - q) ** 2, axis=1))
    return set(np.where(d <= r)[0].tolist())


@pytest.mark.parametrize("tree_name", sorted(TREE_FACTORIES))
@pytest.mark.parametrize(
    "dim,n,kind",
    [
        (1, 2, "uniform"),
        (1, 40, "uniform"),
        (2, 17, "uniform"),
        (2, 80, "uniform"),
        (3, 60, "uniform"),
        (4, 50, "uniform"),
        (6, 40, "uniform"),
        (2, 45, "duplicates"),
        (3, 30, "duplicates"),
        (1, 25, "grid"),
        (2, 30, "grid"),
    ],
)
def test_knn_matches_brute_force(tree_name, dim, n, kind):
    """k-NN distances must match brute force exactly (ties order-free)."""
    rng = np.random.default_rng(42 + dim * 100 + n)
    data = _make_dataset(dim, n, kind, rng)
    tree = TREE_FACTORIES[tree_name](data)
    queries = np.vstack([rng.uniform(-6, 6, (4, dim)), data[rng.integers(0, n, 2)]])

    # Clamped and deduplicated: asking for more neighbors than the index holds
    # now raises (gh-22), and the smallest dataset here has only two points.
    for k in sorted({1, min(3, n), min(n, 8), n}):
        result = tree.query(queries, k=k)
        for qi, q in enumerate(queries):
            ref = _brute_knn_distances(data, q, k)
            got_d = np.sort(result.distances[qi][: len(ref)])
            assert_allclose(
                got_d,
                ref,
                atol=1e-8,
                err_msg=(f"{tree_name} dim={dim} n={n} kind={kind} k={k}"),
            )
            # No duplicate indices among returned neighbors
            got_idx = result.indices[qi][: len(ref)]
            assert len(set(got_idx.tolist())) == len(ref)
            # Reported distance consistent with reported index
            for ii, dd in zip(got_idx, result.distances[qi][: len(ref)]):
                true_d = np.sqrt(np.sum((data[ii] - q) ** 2))
                assert abs(true_d - dd) < 1e-8


@pytest.mark.parametrize("tree_name", sorted(TREE_FACTORIES))
@pytest.mark.parametrize(
    "dim,n,kind",
    [
        (1, 40, "uniform"),
        (2, 80, "uniform"),
        (3, 60, "uniform"),
        (4, 50, "uniform"),
        (2, 45, "duplicates"),
        (1, 25, "grid"),
    ],
)
def test_radius_matches_brute_force(tree_name, dim, n, kind):
    """Radius query index sets must match brute force exactly."""
    rng = np.random.default_rng(7 + dim * 100 + n)
    data = _make_dataset(dim, n, kind, rng)
    tree = TREE_FACTORIES[tree_name](data)
    queries = np.vstack([rng.uniform(-6, 6, (4, dim)), data[rng.integers(0, n, 2)]])

    for r in (0.5, 1.5, 3.0):
        results = tree.query_radius(queries, r)
        for qi, q in enumerate(queries):
            ref = _brute_radius(data, q, r)
            got = set(int(i) for i in results[qi])
            assert got == ref, (
                f"{tree_name} dim={dim} n={n} kind={kind} r={r}: "
                f"missing={ref - got}, extra={got - ref}"
            )
        # query_ball_point is an alias for query_radius
        alias = tree.query_ball_point(queries, r)
        assert [sorted(a) for a in alias] == [sorted(b) for b in results]


def test_covertree_exact_match_regression():
    """Regression: CoverTree used to miss exact matches and duplicate indices."""
    data = np.arange(25, dtype=float).reshape(-1, 1)
    tree = CoverTree(data)

    # Query on an exact data point must find it at distance 0
    result = tree.query([[7.0]], k=3)
    assert 7 in result.indices[0]
    assert_allclose(np.sort(result.distances[0]), [0.0, 1.0, 1.0])
    # No duplicated indices among the k results
    assert len(set(result.indices[0].tolist())) == 3

    # Radius query must find the exact point
    assert 7 in tree.query_radius([[7.0]], 0.5)[0]


def test_metric_trees_custom_metric():
    """VPTree/CoverTree with Manhattan metric vs brute force."""

    def manhattan(x, y):
        return float(np.sum(np.abs(x - y)))

    rng = np.random.default_rng(99)
    data = rng.uniform(-5, 5, (50, 3))
    queries = rng.uniform(-6, 6, (5, 3))

    for cls in (VPTree, CoverTree):
        tree = cls(data, metric=manhattan)
        for q in queries:
            ref = np.sort([manhattan(q, p) for p in data])
            result = tree.query(q, k=4)
            assert_allclose(np.sort(result.distances[0]), ref[:4], atol=1e-9)
            got = set(int(i) for i in tree.query_radius(q, 3.0)[0])
            expected = {i for i, p in enumerate(data) if manhattan(q, p) <= 3.0}
            assert got == expected, cls.__name__


def test_query_k_greater_than_n_raises():
    """Asking for more neighbors than exist is a caller error (gh-22).

    This test previously asserted the opposite: that the shortfall was padded,
    with ``inf`` distances marking the padding. The indices were padded too --
    with ``0``, a *valid* index -- so a caller who read ``result.indices``
    without also checking ``result.distances`` silently used point 0 as a
    neighbor, once per overshoot. Raising matches ``sklearn.neighbors`` and
    makes that impossible.
    """
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    for tree in (
        KDTree(data),
        BallTree(data),
        VPTree(data),
        CoverTree(data),
        RTree.from_points(data),
    ):
        with pytest.raises(ValueError, match="exceeds the 2 point"):
            tree.query([[0.1, 0.1]], k=5)


def test_query_k_equal_to_n_is_still_allowed():
    """The boundary. Every point being a neighbor is a legitimate request."""
    data = np.array([[0.0, 0.0], [1.0, 1.0]])
    ref = np.sort(np.sqrt(((data - [0.1, 0.1]) ** 2).sum(axis=1)))
    for tree in (
        KDTree(data),
        BallTree(data),
        VPTree(data),
        CoverTree(data),
        RTree.from_points(data),
    ):
        result = tree.query([[0.1, 0.1]], k=2)
        assert np.all(np.isfinite(result.distances[0]))
        assert_allclose(np.sort(result.distances[0]), ref)
        assert sorted(result.indices[0].tolist()) == [0, 1]


# =============================================================================
# RTree box operations
# =============================================================================


def test_bounding_box_predicates():
    b1 = BoundingBox(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    b2 = BoundingBox(np.array([1.0, 1.0]), np.array([3.0, 3.0]))
    b3 = BoundingBox(np.array([5.0, 5.0]), np.array([6.0, 6.0]))

    assert b1.intersects(b2) and b2.intersects(b1)
    assert not b1.intersects(b3)
    assert b1.contains_point([1.0, 1.0])
    assert b1.contains_point([2.0, 2.0])  # boundary inclusive
    assert not b1.contains_point([3.0, 3.0])
    big = BoundingBox(np.array([0.0, 0.0]), np.array([4.0, 4.0]))
    assert big.contains_box(b1)
    assert not b1.contains_box(b2)

    assert_allclose(b1.center, [1.0, 1.0])
    assert_allclose(b1.dimensions, [2.0, 2.0])
    assert b1.volume == 4.0

    merged = merge_boxes([b1, b3])
    assert_allclose(merged.min_coords, [0.0, 0.0])
    assert_allclose(merged.max_coords, [6.0, 6.0])
    with pytest.raises(ValueError):
        merge_boxes([])

    bp = box_from_point([1.0, 2.0])
    assert_allclose(bp.min_coords, [1.0, 2.0])
    assert bp.volume == 0.0

    bps = box_from_points([[0.0, 5.0], [3.0, 1.0], [-1.0, 2.0]])
    assert_allclose(bps.min_coords, [-1.0, 1.0])
    assert_allclose(bps.max_coords, [3.0, 5.0])


def test_rtree_box_queries_vs_brute_force():
    """Box-entry queries (intersect/contains/point/nearest) with forced splits."""
    rng = np.random.default_rng(5)
    tree = RTree(max_entries=4)
    boxes = []
    for i in range(40):
        lo = rng.uniform(0, 10, 2)
        hi = lo + rng.uniform(0.1, 2, 2)
        boxes.append(BoundingBox(lo, hi))
        tree.insert(boxes[-1], i)
    assert len(tree) == 40

    qb = BoundingBox(np.array([2.0, 2.0]), np.array([6.0, 6.0]))
    assert set(tree.query_intersect(qb).indices) == {
        i for i, b in enumerate(boxes) if b.intersects(qb)
    }
    assert set(tree.query_contains(qb).indices) == {
        i for i, b in enumerate(boxes) if qb.contains_box(b)
    }
    pt = np.array([5.0, 5.0])
    assert set(tree.query_point(pt).indices) == {
        i for i, b in enumerate(boxes) if b.contains_point(pt)
    }

    def min_dist(p, b):
        c = np.clip(p, b.min_coords, b.max_coords)
        return float(np.sqrt(((p - c) ** 2).sum()))

    for q in rng.uniform(-2, 12, (5, 2)):
        _, dists = tree.nearest(q, k=3)
        ref = np.sort([min_dist(q, b) for b in boxes])[:3]
        assert_allclose(np.sort(dists), ref, atol=1e-9)


def test_rtree_empty_query_raises():
    empty = RTree()
    with pytest.raises(ValueError):
        empty.query([[0.0, 0.0]], k=1)
    with pytest.raises(ValueError):
        empty.query_radius([[0.0, 0.0]], 1.0)


# =============================================================================
# Clustering vs reference implementations
# =============================================================================


def _blobs(rng, n_per, centers, spread=0.3):
    return np.vstack([rng.normal(c, spread, (n_per, len(c))) for c in centers])


def test_kmeans_matches_sklearn():
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(7)
    for centers in [[(0, 0), (8, 8)], [(0, 0), (10, 0), (0, 10)]]:
        X = _blobs(rng, 40, centers)
        result = kmeans(X, n_clusters=len(centers), rng=np.random.default_rng(1))
        sk = sklearn_cluster.KMeans(
            n_clusters=len(centers), n_init=10, random_state=0
        ).fit(X)
        ari = metrics.adjusted_rand_score(sk.labels_, result.labels)
        assert ari > 0.99
        assert abs(result.inertia - sk.inertia_) / sk.inertia_ < 0.01
        assert result.converged


def test_dbscan_matches_sklearn():
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(7)
    for eps, min_samples in [(0.5, 5), (0.8, 3), (1.0, 2)]:
        X = np.vstack([_blobs(rng, 30, [(0, 0), (5, 5)]), rng.uniform(-3, 8, (8, 2))])
        result = dbscan(X, eps=eps, min_samples=min_samples)
        sk = sklearn_cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)

        # Noise sets and core sample sets must match exactly
        assert np.array_equal(result.labels == -1, sk.labels_ == -1)
        assert set(result.core_sample_indices.tolist()) == set(
            sk.core_sample_indices_.tolist()
        )
        # Cluster partition identical up to label permutation
        mask = sk.labels_ >= 0
        ari = metrics.adjusted_rand_score(sk.labels_[mask], result.labels[mask])
        assert ari > 0.9999
        assert result.n_noise == int((sk.labels_ == -1).sum())


def test_dbscan_predict():
    X_train = np.array(
        [[0.0, 0.0], [0.2, 0.0], [0.0, 0.2], [5.0, 5.0], [5.2, 5.0], [5.0, 5.2]]
    )
    result = dbscan(X_train, eps=0.5, min_samples=2)
    pred = dbscan_predict(
        np.array([[0.1, 0.1], [10.0, 10.0], [5.1, 5.1]]),
        X_train,
        result.labels,
        eps=0.5,
    )
    assert pred.tolist() == [0, -1, 1]


@pytest.mark.parametrize("linkage", ["single", "complete", "average", "ward"])
def test_hierarchical_matches_scipy(linkage):
    sch = pytest.importorskip("scipy.cluster.hierarchy")
    metrics = pytest.importorskip("sklearn.metrics")

    rng = np.random.default_rng(3)
    X = _blobs(rng, 12, [(0, 0), (4, 4), (8, 0)], spread=0.4)
    Z = sch.linkage(X, method=linkage)
    ref = sch.fcluster(Z, t=3, criterion="maxclust")

    result = agglomerative_clustering(X, n_clusters=3, linkage=linkage)
    assert result.n_clusters == 3
    assert metrics.adjusted_rand_score(ref, result.labels) > 0.9999

    # Full dendrogram: merge distances must match scipy's exactly
    full = agglomerative_clustering(X, n_clusters=1, linkage=linkage)
    assert_allclose(np.sort(full.linkage_matrix[:, 2]), np.sort(Z[:, 2]), atol=1e-8)

    # cut_dendrogram / fcluster reproduce the same partition
    labels_cut = cut_dendrogram(full.linkage_matrix, len(X), n_clusters=3)
    assert metrics.adjusted_rand_score(ref, labels_cut) > 0.9999
    labels_f = fcluster(full.linkage_matrix, len(X), t=3, criterion="maxclust")
    assert labels_f.min() == 1  # 1-indexed
    assert metrics.adjusted_rand_score(ref, labels_f) > 0.9999


def test_hierarchical_distance_threshold():
    rng = np.random.default_rng(4)
    X = _blobs(rng, 10, [(0, 0), (10, 10)], spread=0.3)
    result = agglomerative_clustering(X, distance_threshold=3.0, linkage="single")
    assert result.n_clusters == 2


def test_distance_and_neighbor_helpers():
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(11)
    X = rng.normal(0, 1, (12, 3))
    ref = cdist(X, X)
    assert_allclose(compute_distance_matrix(X), ref, atol=1e-12)

    neighbors = compute_neighbors(X, 1.0)
    for i in range(len(X)):
        assert set(neighbors[i].tolist()) == set(np.where(ref[i] <= 1.0)[0])


def test_kmeans_helpers():
    X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [11.0, 10.0]])
    centers = np.array([[0.5, 0.0], [10.5, 10.0]])
    labels, inertia = assign_clusters(X, centers)
    assert labels.tolist() == [0, 0, 1, 1]
    assert abs(inertia - 1.0) < 1e-12  # 4 points at squared distance 0.25

    updated = update_centers(X, labels, 2)
    assert_allclose(updated, centers)

    elbow = kmeans_elbow(X, k_range=range(1, 4), rng=np.random.default_rng(3))
    assert elbow["k_values"] == [1, 2, 3]
    inertias = elbow["inertias"]
    assert all(inertias[i] >= inertias[i + 1] - 1e-9 for i in range(2))


# =============================================================================
# Gaussian mixture reduction: analytic moment preservation
# =============================================================================


def _random_mixture(k, d, rng):
    w = rng.uniform(0.2, 1.0, k)
    w = w / w.sum()
    comps = []
    for i in range(k):
        A = rng.normal(0, 1, (d, d))
        comps.append(
            GaussianComponent(w[i], rng.normal(0, 3, d), A @ A.T + d * np.eye(d))
        )
    return comps


def test_moment_match_analytic():
    rng = np.random.default_rng(21)
    comps = _random_mixture(5, 3, rng)
    m, P = moment_match(
        [c.weight for c in comps],
        [c.mean for c in comps],
        [c.covariance for c in comps],
    )
    m_ref = sum(c.weight * c.mean for c in comps)
    P_ref = sum(
        c.weight * (c.covariance + np.outer(c.mean - m_ref, c.mean - m_ref))
        for c in comps
    )
    assert_allclose(m, m_ref, atol=1e-12)
    assert_allclose(P, P_ref, atol=1e-12)
    assert_allclose(P, P.T)  # symmetric


@pytest.mark.parametrize("reducer", [reduce_mixture_runnalls, reduce_mixture_west])
@pytest.mark.parametrize("k,d", [(2, 1), (5, 2), (8, 3), (10, 4)])
def test_reduction_preserves_moments(reducer, k, d):
    """Runnalls/West merging must preserve overall mixture mean and covariance."""
    rng = np.random.default_rng(31 + k + d)
    comps = _random_mixture(k, d, rng)
    m0, P0 = moment_match(
        [c.weight for c in comps],
        [c.mean for c in comps],
        [c.covariance for c in comps],
    )
    for target in {1, max(1, k // 2), k - 1}:
        result = reducer(comps, max_components=target, weight_threshold=0.0)
        assert len(result.components) == target
        weights = [c.weight for c in result.components]
        assert abs(sum(weights) - 1.0) < 1e-12
        m1, P1 = moment_match(
            weights,
            [c.mean for c in result.components],
            [c.covariance for c in result.components],
        )
        assert_allclose(m1, m0, atol=1e-10)
        assert_allclose(P1, P0, atol=1e-9)


def test_merge_gaussians_and_costs_analytic():
    c1 = GaussianComponent(0.3, np.array([0.0, 0.0]), np.eye(2) * 0.1)
    c2 = GaussianComponent(0.2, np.array([1.0, 0.0]), np.eye(2) * 0.3)

    merged = merge_gaussians(c1, c2)
    m_ref, P_ref = moment_match(
        [0.6, 0.4], [c1.mean, c2.mean], [c1.covariance, c2.covariance]
    )
    assert abs(merged.component.weight - 0.5) < 1e-15
    assert_allclose(merged.component.mean, m_ref, atol=1e-12)
    assert_allclose(merged.component.covariance, P_ref, atol=1e-12)

    # Runnalls cost: 0.5*(w_m log|P_m| - w1 log|P1| - w2 log|P2|)
    expected = 0.5 * (
        0.5 * np.linalg.slogdet(P_ref)[1]
        - 0.3 * np.linalg.slogdet(c1.covariance)[1]
        - 0.2 * np.linalg.slogdet(c2.covariance)[1]
    )
    assert abs(runnalls_merge_cost(c1, c2) - max(0.0, expected)) < 1e-12

    # West cost: (w1*w2/(w1+w2)) * Mahalanobis^2 under weighted-average cov
    P_avg = (0.3 * c1.covariance + 0.2 * c2.covariance) / 0.5
    diff = c1.mean - c2.mean
    expected_w = (0.3 * 0.2 / 0.5) * (diff @ np.linalg.inv(P_avg) @ diff)
    assert abs(west_merge_cost(c1, c2) - expected_w) < 1e-12

    # Cost ordering: nearer components must be cheaper to merge
    far = GaussianComponent(0.2, np.array([8.0, 8.0]), np.eye(2) * 0.1)
    assert runnalls_merge_cost(c1, c2) < runnalls_merge_cost(c1, far)
    assert west_merge_cost(c1, c2) < west_merge_cost(c1, far)


def test_prune_mixture_renormalizes():
    comps = [
        GaussianComponent(0.9, np.zeros(1), np.eye(1)),
        GaussianComponent(1e-6, np.ones(1), np.eye(1)),
    ]
    pruned = prune_mixture(comps, weight_threshold=1e-5)
    assert len(pruned) == 1
    assert abs(pruned[0].weight - 1.0) < 1e-15


def test_gaussian_mixture_class():
    from scipy.stats import multivariate_normal

    gm = GaussianMixture()
    gm.add_component(0.5, [0.0, 0.0], np.eye(2) * 0.1)
    gm.add_component(0.5, [2.0, 2.0], np.eye(2) * 0.1)
    assert len(gm) == 2
    assert gm.dim == 2
    assert_allclose(gm.mean, [1.0, 1.0])

    x = np.array([0.5, 0.7])
    ref_pdf = 0.5 * multivariate_normal.pdf(
        x, [0, 0], np.eye(2) * 0.1
    ) + 0.5 * multivariate_normal.pdf(x, [2, 2], np.eye(2) * 0.1)
    assert abs(gm.pdf(x) - ref_pdf) < 1e-12

    samples = gm.sample(400, rng=np.random.default_rng(5))
    assert samples.shape == (400, 2)
    assert abs(samples.mean(axis=0)[0] - 1.0) < 0.3

    reduced = gm.reduce_runnalls(1)
    assert len(reduced) == 1
    assert_allclose(reduced.components[0].mean, [1.0, 1.0])
    assert len(gm.reduce_west(1)) == 1

    clone = gm.copy()
    clone.components[0].mean[0] = 99.0
    assert gm.components[0].mean[0] != 99.0  # deep copy

    gm2 = GaussianMixture(
        [
            GaussianComponent(2.0, np.zeros(1), np.eye(1)),
            GaussianComponent(2.0, np.ones(1), np.eye(1)),
        ]
    )
    gm2.normalize_weights()
    assert_allclose(gm2.weights, [0.5, 0.5])


# =============================================================================
# TrackList behavioral contracts
# =============================================================================


def _mk_track(
    tid, x, y, status=TrackStatus.CONFIRMED, hits=5, misses=0, time=1.0, vx=1.0, vy=0.0
):
    return Track(
        id=tid,
        state=np.array([x, vx, y, vy]),
        covariance=np.eye(4),
        status=status,
        hits=hits,
        misses=misses,
        time=time,
    )


@pytest.fixture
def sample_tracks():
    return [
        _mk_track(0, 0.0, 0.0),
        _mk_track(
            1, 1.0, 1.0, status=TrackStatus.TENTATIVE, hits=2, misses=1, time=2.0
        ),
        _mk_track(2, 10.0, 10.0, time=3.0),
        _mk_track(
            3, 11.0, 10.0, status=TrackStatus.DELETED, hits=1, misses=5, time=3.0
        ),
    ]


def test_track_list_queries(sample_tracks):
    tl = TrackList(sample_tracks)
    assert len(tl) == 4
    assert tl[1] is sample_tracks[1]
    assert isinstance(tl[1:3], TrackList) and len(tl[1:3]) == 2
    assert 0 in tl and 99 not in tl
    assert tl.get_by_id(2) is sample_tracks[2]
    assert tl.get_by_id(99) is None
    assert [t.id for t in tl.get_by_ids([3, 0, 99])] == [3, 0]

    assert [t.id for t in tl.confirmed] == [0, 2]
    assert [t.id for t in tl.tentative] == [1]
    assert [t.id for t in tl.filter_by_status(TrackStatus.DELETED)] == [3]
    assert [t.id for t in tl.filter_by_time(min_time=2.0)] == [1, 2, 3]
    assert [t.id for t in tl.filter_by_time(min_time=2.0, max_time=2.0)] == [1]
    assert [t.id for t in tl.filter_by_region([0, 0], 2.0)] == [0, 1]
    assert [t.id for t in tl.filter_by_predicate(lambda t: t.hits >= 5)] == [0, 2]
    assert tl.track_ids == [0, 1, 2, 3]


def test_track_list_arrays_and_stats(sample_tracks):
    tl = TrackList(sample_tracks)
    assert_allclose(tl.states(), np.array([t.state for t in sample_tracks]))
    assert tl.covariances().shape == (4, 4, 4)
    assert_allclose(tl.positions(), [[0, 0], [1, 1], [10, 10], [11, 10]])
    assert_allclose(tl.positions(indices=(1, 3)), [[1, 0]] * 4)

    stats = tl.stats()
    assert (stats.n_tracks, stats.n_confirmed, stats.n_tentative, stats.n_deleted) == (
        4,
        2,
        1,
        1,
    )
    assert abs(stats.mean_hits - 13 / 4) < 1e-12
    assert abs(stats.mean_misses - 6 / 4) < 1e-12

    empty = TrackList()
    assert empty.stats().n_tracks == 0
    assert empty.states().shape == (0, 0)
    assert empty.positions().shape == (0, 2)


def test_track_list_immutable_ops(sample_tracks):
    tl = TrackList(sample_tracks)
    extra = _mk_track(5, 3.0, 3.0)

    tl2 = tl.add(extra)
    assert len(tl2) == 5 and len(tl) == 4
    assert tl2.get_by_id(5) is extra

    tl3 = tl.remove(1)
    assert [t.id for t in tl3] == [0, 2, 3] and len(tl) == 4

    merged = tl.merge(TrackList([extra]))
    assert [t.id for t in merged] == [0, 1, 2, 3, 5]
    assert [t.id for t in tl.copy()] == [0, 1, 2, 3]


# =============================================================================
# MeasurementSet behavioral contracts
# =============================================================================


@pytest.fixture
def sample_measurements():
    return MeasurementSet.from_arrays(
        values=[[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0]],
        times=[0.0, 0.0, 1.0, 2.0],
        sensor_ids=[0, 1, 0, 1],
    )


def test_measurement_set_filters(sample_measurements):
    ms = sample_measurements
    assert len(ms) == 4
    assert ms[0].id == 0 and ms[1].sensor_id == 1 and ms[3].time == 2.0
    assert isinstance(ms[1:3], MeasurementSet)
    assert [m.id for m in ms.at_time(0.0)] == [0, 1]
    assert [m.id for m in ms.in_time_window(1.0, 2.0)] == [2, 3]
    assert [m.id for m in ms.in_region([0.0, 0.0], 1.5)] == [0, 1]
    assert [m.id for m in ms.by_sensor(1)] == [1, 3]
    assert_allclose(ms.times, [0.0, 1.0, 2.0])
    assert sorted(ms.sensors) == [0, 1]
    assert ms.time_range == (0.0, 2.0)
    assert ms.values().shape == (4, 2)
    assert_allclose(ms.values_at_time(0.0), [[0.0, 0.0], [1.0, 0.0]])


def test_measurement_set_nearest_and_ops(sample_measurements):
    ms = sample_measurements
    q = ms.nearest_to([0.4, 0.0], k=2)
    assert q.indices == [0, 1]  # distances 0.4 and 0.6
    assert q.measurements[0] is ms[0]
    # k clamped to available measurements
    assert len(ms.nearest_to([0.0, 0.0], k=99).indices) == 4

    extra = Measurement(value=np.array([5.0, 5.0]), time=3.0, id=7)
    assert len(ms.add(extra)) == 5 and len(ms) == 4
    assert len(ms.add_batch([extra, extra])) == 6
    assert len(ms.merge(MeasurementSet([extra]))) == 5
    assert len(ms.copy()) == 4

    empty = MeasurementSet()
    assert empty.time_range == (0.0, 0.0)
    assert empty.nearest_to([0.0, 0.0]).indices == []


# =============================================================================
# ClusterSet behavioral contracts
# =============================================================================


@pytest.fixture
def clustered_tracks():
    tracks = TrackList(
        [
            _mk_track(0, 0.0, 0.0),
            _mk_track(1, 1.0, 0.0),
            _mk_track(2, 0.5, 0.5),
            _mk_track(10, 20.0, 20.0),
            _mk_track(11, 21.0, 20.0),
            _mk_track(12, 20.0, 21.0),
        ]
    )
    return tracks, cluster_tracks_dbscan(tracks, eps=3.0, min_samples=2)


def test_cluster_tracks_grouping(clustered_tracks):
    tracks, cs = clustered_tracks
    assert len(cs) == 2
    groups = sorted(sorted(c.track_ids) for c in cs)
    assert groups == [[0, 1, 2], [10, 11, 12]]

    # Centroids match per-cluster position means
    for cluster in cs:
        positions = [
            [tracks.get_by_id(tid).state[0], tracks.get_by_id(tid).state[2]]
            for tid in cluster.track_ids
        ]
        assert_allclose(cluster.centroid, np.mean(positions, axis=0))

    cs_k = cluster_tracks_kmeans(tracks, n_clusters=2, rng=np.random.default_rng(0))
    assert sorted(sorted(c.track_ids) for c in cs_k) == groups

    assert (
        len(ClusterSet.from_tracks(tracks, method="dbscan", eps=3.0, min_samples=2))
        == 2
    )
    with pytest.raises(ValueError):
        ClusterSet.from_tracks(tracks, method="bogus")

    assert len(cluster_tracks_dbscan(TrackList(), eps=1.0)) == 0
    assert len(cluster_tracks_kmeans(TrackList(), n_clusters=2)) == 0
    assert_allclose(compute_cluster_centroid([]), [0.0, 0.0])


def test_cluster_set_queries_and_stats(clustered_tracks):
    tracks, cs = clustered_tracks
    assert cs.get_cluster(0) is not None
    assert cs.get_cluster(99) is None
    found = cs.get_cluster_for_track(11)
    assert found is not None and 11 in found.track_ids
    assert cs.get_cluster_for_track(999) is None

    region = cs.clusters_in_region([0.0, 0.0], 5.0)
    assert len(region) == 1 and 0 in region[0].track_ids
    assert sorted(cs.cluster_ids) == [0, 1]
    assert cs.n_tracks_total == 6

    cluster = cs.get_cluster_for_track(0)
    stats = cs.cluster_stats(cluster.id, tracks=tracks)
    positions = np.array(
        [
            [tracks.get_by_id(tid).state[0], tracks.get_by_id(tid).state[2]]
            for tid in cluster.track_ids
        ]
    )
    seps = np.sqrt(((positions - cluster.centroid) ** 2).sum(axis=1))
    assert abs(stats.mean_separation - seps.mean()) < 1e-12
    assert abs(stats.max_separation - seps.max()) < 1e-12
    assert abs(stats.velocity_coherence - 1.0) < 1e-9  # identical velocities

    assert set(cs.all_stats(tracks=tracks)) == {0, 1}
    assert cs.cluster_stats(42) is None


def test_cluster_set_merge_split(clustered_tracks):
    tracks, cs = clustered_tracks
    merged = cs.merge_clusters(0, 1)
    assert len(merged) == 1
    assert sorted(merged.get_cluster(0).track_ids) == [0, 1, 2, 10, 11, 12]
    with pytest.raises(ValueError):
        cs.merge_clusters(0, 42)

    split = cs.split_cluster(0, [0, 1], [2], tracks=tracks)
    assert len(split) == 3
    assert sorted(split.get_cluster(0).track_ids) == [0, 1]
    assert_allclose(split.get_cluster(0).centroid, [0.5, 0.0])
    with pytest.raises(ValueError):
        cs.split_cluster(42, [0], [1])

    first = cs.get_cluster(0)
    added = cs.add_cluster(first._replace(id=5))
    assert len(added) == 3 and 5 in added
    assert len(cs.remove_cluster(0)) == 1
    assert len(cs.copy()) == 2
