# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nrl-tracker",
#     "rerun-sdk>=0.24",
# ]
# [tool.uv.sources]
# nrl-tracker = { path = "..", editable = true }
# ///
"""
Multi-target tracking visualized with Rerun.

Same scenario as ``multi_target_tracking.py`` (two crossing targets, missed
detections, false alarms, GNN association), but logged to a Rerun timeline so
you can scrub through the run step by step: measurements arriving, tracks
being initiated, confirmed, and deleted, and each track's 95% position
uncertainty ellipse shrinking as it converges.

Run with:
    uv run examples/multi_target_tracking_rerun.py

By default this saves a recording to ``examples/output/multi_target_tracking.rrd``
(open it with ``uv tool run rerun-sdk <file>`` or the Rerun desktop viewer).
Set ``PYTCL_RERUN_SPAWN=1`` to open a live viewer instead.
"""

import os
from pathlib import Path

import numpy as np
import rerun as rr
from scipy.stats import chi2

from pytcl.trackers import MultiTargetTracker, TrackStatus

OUTPUT_DIR = Path(__file__).parent / "output"

N_STEPS = 50
DT = 1.0

# Plotly/tab10-style palette, keyed by track id
COLORS = [
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
    (188, 189, 34),
    (23, 190, 207),
]


def simulate_targets(rng: np.random.Generator):
    """Two crossing constant-velocity targets with noisy position measurements."""
    starts = np.array([[0.0, 0.0], [100.0, 0.0]])
    velocities = np.array([[2.0, 1.0], [-2.0, 1.5]])
    R = np.eye(2) * 2.0
    pd = 0.95

    true_states = []
    measurements = []
    for k in range(N_STEPS):
        positions = starts + velocities * (k * DT)
        true_states.append(positions)

        meas = [
            p + rng.multivariate_normal([0, 0], R)
            for p in positions
            if rng.random() < pd
        ]
        if rng.random() < 0.1:
            meas.append(np.array([rng.uniform(-10, 110), rng.uniform(-10, 60)]))
        measurements.append(meas)

    return true_states, measurements


def make_tracker() -> MultiTargetTracker:
    """Constant-velocity GNN tracker, state = [x, vx, y, vy]."""

    def F(dt):
        return np.array(
            [[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]],
            dtype=np.float64,
        )

    def Q(dt):
        q = 0.5
        return (
            np.array(
                [
                    [dt**4 / 4, dt**3 / 2, 0, 0],
                    [dt**3 / 2, dt**2, 0, 0],
                    [0, 0, dt**4 / 4, dt**3 / 2],
                    [0, 0, dt**3 / 2, dt**2],
                ]
            )
            * q**2
        )

    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    return MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=np.eye(2) * 2.0,
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=np.diag([10.0, 5.0, 10.0, 5.0]),
    )


def covariance_ellipse(
    center: np.ndarray, cov: np.ndarray, confidence: float = 0.95, n_points: int = 40
) -> np.ndarray:
    """Points on the ``confidence`` ellipse of a 2x2 position covariance."""
    scale = chi2.ppf(confidence, df=2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    return center + (eigvecs @ (np.sqrt(scale * eigvals)[:, None] * circle)).T


def main():
    rr.init("pytcl_multi_target_tracking")
    if os.environ.get("PYTCL_RERUN_SPAWN") == "1":
        rr.spawn()
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        recording = OUTPUT_DIR / "multi_target_tracking.rrd"
        rr.save(str(recording))
        print(f"Saving recording to {recording}")

    rng = np.random.default_rng(42)
    true_states, measurements = simulate_targets(rng)
    tracker = make_tracker()

    # Full ground-truth trajectories, logged once as static context
    truth = np.array(true_states)  # (n_steps, 2 targets, 2)
    rr.log(
        "truth",
        rr.LineStrips2D(
            [truth[:, 0], truth[:, 1]],
            colors=[(0, 160, 0), (0, 90, 200)],
            labels=["Target 1 (truth)", "Target 2 (truth)"],
        ),
        static=True,
    )

    track_paths: dict[int, list] = {}
    for k, meas in enumerate(measurements):
        rr.set_time("step", sequence=k)

        rr.log(
            "measurements",
            rr.Points2D(meas, colors=(60, 60, 60), radii=0.6),
        )

        tracks = tracker.process(meas, DT)
        active = set()
        for track in tracks:
            active.add(track.id)
            color = COLORS[track.id % len(COLORS)]
            confirmed = track.status == TrackStatus.CONFIRMED
            position = track.state[[0, 2]]
            track_paths.setdefault(track.id, []).append(position)

            entity = f"tracks/{track.id}"
            rr.log(
                entity,
                rr.Points2D(
                    [position],
                    colors=color if confirmed else (170, 170, 170),
                    radii=1.2,
                    labels=[f"Track {track.id} ({track.status.value})"],
                ),
            )
            if len(track_paths[track.id]) > 1:
                rr.log(
                    f"{entity}/path",
                    rr.LineStrips2D([np.array(track_paths[track.id])], colors=color),
                )
            rr.log(
                f"{entity}/gate",
                rr.LineStrips2D(
                    [
                        covariance_ellipse(
                            position, track.covariance[np.ix_([0, 2], [0, 2])]
                        )
                    ],
                    colors=(*color, 120),
                ),
            )

        # Clear entities for tracks the tracker has dropped
        for track_id in list(track_paths):
            if track_id not in active:
                rr.log(f"tracks/{track_id}", rr.Clear(recursive=True))
                del track_paths[track_id]

    print(
        f"Processed {N_STEPS} steps, {sum(len(m) for m in measurements)} measurements"
    )


if __name__ == "__main__":
    main()
