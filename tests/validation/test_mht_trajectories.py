"""Reference-trajectory tests for the MHT tracker.

The v2.11.0 audit found the previous implementation stored one global
track per id, so every hypothesis shared the last association's state
and N-scan pruning never fired. These tests pin the rebuilt behavior:
hypotheses that disagree about an association carry genuinely different
branch states, N-scan pruning actually collapses stale ambiguity, and
crossing targets survive as two distinct confirmed tracks.
"""

import numpy as np
import pytest

from pytcl.trackers.mht import MHTConfig, MHTTracker


def _cv_tracker(**config_kwargs):
    """2D constant-velocity tracker: state [x, vx, y, vy], measure (x, y)."""

    def F(dt):
        return np.array(
            [
                [1.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, dt],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def Q(dt):
        return 0.01 * np.eye(4)

    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = 0.1 * np.eye(2)
    config = MHTConfig(**config_kwargs) if config_kwargs else MHTConfig()
    # A tight initial covariance keeps the detection density above the
    # miss + new-track alternative for on-trajectory measurements.
    init_P = np.diag([1.0, 4.0, 1.0, 4.0])
    return MHTTracker(4, 2, F, H, Q, R, config=config, init_covariance=init_P)


def _establish_single_track(tracker, n_scans=3, v=(1.0, 0.5)):
    """Feed an unambiguous straight-line target; return its leaf track."""
    for k in range(n_scans):
        pos = np.array([v[0] * k, v[1] * k])
        tracker.process([pos], dt=1.0)
    best = tracker.hypothesis_tree.get_best_hypothesis()
    [leaf_id] = best.track_ids
    return tracker.hypothesis_tree.tracks[leaf_id]


class TestHypothesisBranching:
    def test_ambiguous_scan_creates_distinct_branch_states(self):
        tracker = _cv_tracker(n_scan=5, min_hypothesis_prob=1e-12)
        leaf = _establish_single_track(tracker)

        # Two gated measurements straddling the prediction: hypotheses
        # assigning each must hold different Kalman updates.
        predicted = np.array([3.0, 1.5])
        m0 = predicted + np.array([0.0, 0.4])
        m1 = predicted + np.array([0.0, -0.4])
        tracker.process([m0, m1], dt=1.0)

        tree = tracker.hypothesis_tree
        children = {
            tid: tr
            for tid, tr in tree.tracks.items()
            if tr.parent_id == leaf.id and tr.history[-1] >= 0
        }
        assert len(children) == 2, "one branch per gated measurement"
        (state_a, state_b) = [tr.state for tr in children.values()]
        assert not np.allclose(state_a, state_b), (
            "branches of the same track must carry different states"
        )

        # Different hypotheses must reference different branches.
        referencing = {
            tid: [h.id for h in tree.hypotheses if tid in h.track_ids]
            for tid in children
        }
        hyp_sets = [set(v) for v in referencing.values()]
        assert all(hyp_sets), "each branch appears in some hypothesis"
        assert hyp_sets[0].isdisjoint(hyp_sets[1]), (
            "no hypothesis may contain contradictory branches"
        )

    def test_new_track_created_once_per_measurement(self):
        tracker = _cv_tracker()
        tracker.process([np.array([0.0, 0.0])], dt=1.0)
        tree = tracker.hypothesis_tree
        assert len(tree.tracks) == 1
        assert len(tree.hypotheses) == 1

    def test_certain_detection_with_empty_scan_keeps_best(self):
        # With Pd = 1 a missed detection has zero likelihood, so an
        # empty scan zeroes every association; the tracker must keep
        # the best prior hypothesis rather than ending with none.
        tracker = _cv_tracker(detection_prob=1.0)
        tracker.process([np.array([0.0, 0.0])], dt=1.0)
        result = tracker.process([], dt=1.0)
        assert tracker.n_hypotheses == 1
        assert len(result.all_tracks) == 1

    def test_identical_interpretations_merge(self):
        # With delete_threshold = 1 every miss branch dies immediately:
        # after an ambiguous scan followed by an empty scan, all
        # hypotheses collapse to the same (empty) interpretation, whose
        # probabilities must merge into a single hypothesis.
        tracker = _cv_tracker(delete_threshold=1, n_scan=10, min_hypothesis_prob=1e-15)
        _establish_single_track(tracker)
        predicted = np.array([3.0, 1.5])
        tracker.process([predicted + [0.0, 0.4], predicted + [0.0, -0.4]], dt=1.0)
        assert tracker.n_hypotheses > 1
        tracker.process([], dt=1.0)
        assert tracker.n_hypotheses == 1
        np.testing.assert_allclose(
            tracker.hypothesis_tree.hypotheses[0].probability, 1.0
        )


class TestNScanPruning:
    def test_stale_ambiguity_is_pruned(self):
        n_scan = 2
        tracker = _cv_tracker(n_scan=n_scan, min_hypothesis_prob=1e-12)
        leaf = _establish_single_track(tracker)

        predicted = np.array([3.0, 1.5])
        tracker.process([predicted + [0.0, 0.4], predicted + [0.0, -0.4]], dt=1.0)
        assert tracker.n_hypotheses > 1, "ambiguity must branch"
        tree = tracker.hypothesis_tree
        disputed = {
            tid
            for tid, tr in tree.tracks.items()
            if tr.parent_id == leaf.id and tr.history[-1] >= 0
        }
        assert len(disputed) == 2

        # Unambiguous scans (following the +0.4 branch) age the disputed
        # decision past the N-scan window: the losing branch's whole
        # subtree must be pruned, leaving only the MAP-committed child.
        for k in range(5, 5 + n_scan + 1):
            tracker.process([np.array([k * 1.0, k * 0.5 + 0.4])], dt=1.0)

        assert tree.current_scan == tracker._scan
        surviving_disputed = disputed & set(tree.tracks.keys())
        assert len(surviving_disputed) == 1, (
            "N-scan pruning must commit to exactly one side of a stale "
            f"ambiguity; {len(surviving_disputed)} of {len(disputed)} "
            "branches survive"
        )

        # Every surviving hypothesis agrees with the MAP hypothesis
        # about all decisions older than the window (prune idempotent).
        from pytcl.trackers.hypothesis import n_scan_prune

        survivors, _ = n_scan_prune(
            tree.hypotheses, tree.tracks, n_scan, tree.current_scan
        )
        assert len(survivors) == len(tree.hypotheses)


class TestCrossingTargets:
    def test_two_crossing_targets_stay_confirmed(self):
        tracker = _cv_tracker(
            n_scan=3,
            max_hypotheses=50,
            detection_prob=0.95,
            confirm_threshold=3,
            min_hypothesis_prob=1e-10,
        )

        # Two constant-velocity targets crossing at t = 10, y = 5.
        def truth(t):
            return [
                np.array([1.0 * t, 0.5 * t]),
                np.array([1.0 * t, 10.0 - 0.5 * t]),
            ]

        n_steps = 16
        for t in range(1, n_steps + 1):
            tracker.process(truth(t), dt=1.0)

        result = tracker.process(truth(n_steps + 1), dt=1.0)
        confirmed = result.confirmed_tracks
        assert len(confirmed) == 2, (
            f"expected both targets confirmed through the crossing, "
            f"got {len(confirmed)}"
        )

        true_final = truth(n_steps + 1)
        est = sorted(
            [(tr.state[0], tr.state[2]) for tr in confirmed],
            key=lambda p: p[1],
        )
        ref = sorted([(p[0], p[1]) for p in true_final], key=lambda p: p[1])
        for (ex, ey), (rx, ry) in zip(est, ref):
            assert abs(ex - rx) < 1.0 and abs(ey - ry) < 1.0, (
                f"estimate ({ex:.2f}, {ey:.2f}) far from truth ({rx:.2f}, {ry:.2f})"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
