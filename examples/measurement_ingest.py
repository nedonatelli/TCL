"""
Measurement Ingest Example.

Demonstrates the Results I/O pipeline end to end:

1. Synthesize a small CSV of radar-style detections (multiple targets,
   constant-velocity motion, additive noise, an explicit sensor/detection
   id column)
2. Read it back with `read_measurements_csv` (column mapping: time_column,
   measurement_columns, id_column)
3. Run a GNN multi-target tracker over the recovered scans
4. Convert the resulting track history to a long-format polars DataFrame
   with `tracks_to_polars`, print an ASCII per-track summary
5. Write the DataFrame to Parquet in examples/output/

Run with: python examples/measurement_ingest.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np  # noqa: E402

from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity  # noqa: E402
from pytcl.io import read_measurements_csv, tracks_to_polars  # noqa: E402
from pytcl.trackers import MultiTargetTracker, TrackStatus  # noqa: E402

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_TARGETS = 3
N_SCANS = 15
DT = 1.0
NOISE_STD = 3.0


def synthesize_detections_csv(path: Path) -> None:
    """Write a small CSV of noisy constant-velocity detections.

    Columns: t, x, y, det_id -- the det_id column exercises
    `read_measurements_csv`'s optional id_column mapping.
    """
    rng = np.random.default_rng(7)
    starts = [
        np.array([0.0, 2.0, 0.0, 1.0]),
        np.array([100.0, -1.5, 0.0, 1.5]),
        np.array([0.0, 1.0, 100.0, -1.0]),
    ]

    lines = ["t,x,y,det_id"]
    for k in range(N_SCANS):
        t = k * DT
        for i, s0 in enumerate(starts):
            x = s0[0] + s0[1] * t + rng.normal(0.0, NOISE_STD)
            y = s0[2] + s0[3] * t + rng.normal(0.0, NOISE_STD)
            lines.append(f"{t:.1f},{x:.4f},{y:.4f},det_{k:03d}_{i}")

    path.write_text("\n".join(lines) + "\n")


def run_tracker(history_csv: Path):
    """Read the CSV back and run a GNN multi-target tracker over it.

    Returns (ms, history, times): the parsed `MeasurementSet`, plus
    (history, times) in the shape `tracks_to_polars` expects.
    """
    ms = read_measurements_csv(
        history_csv,
        time_column="t",
        measurement_columns=["x", "y"],
        id_column="det_id",
    )

    F = f_constant_velocity(DT, num_dims=2)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    Q = q_constant_velocity(DT, sigma_a=0.5, num_dims=2)
    R = np.eye(2) * NOISE_STD**2
    P0 = np.diag([100.0, 25.0, 100.0, 25.0])

    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=R,
        gate_probability=0.99,
        confirm_hits=3,
        max_misses=5,
        init_covariance=P0,
    )

    history = []
    times = []
    prev_t = None
    for t, scan in zip(ms.times, ms.scans):
        dt = DT if prev_t is None else float(t - prev_t)
        prev_t = t
        tracks = tracker.process([row for row in scan], dt=dt)
        history.append(tracks)
        times.append(float(t))

    return ms, history, times


def print_summary(ms, df) -> None:
    """ASCII per-track summary from the long-format tracks DataFrame."""
    n_detections = sum(scan.shape[0] for scan in ms.scans)
    print(f"  Detections read:  {n_detections}")
    print(f"  Scans:            {len(ms.times)}")
    print(f"  Track-scan rows:  {df.height}")

    print(
        f"\n  {'track_id':<10} {'n_records':<10} {'first_t':<10} "
        f"{'last_t':<10} {'final_status':<14}"
    )
    print("  " + "-" * 56)
    for track_id in sorted(df["track_id"].unique().to_list()):
        rows = df.filter(df["track_id"] == track_id).sort("t")
        first_t = rows["t"][0]
        last_t = rows["t"][rows.height - 1]
        final_status = rows["status"][rows.height - 1]
        print(
            f"  {track_id:<10} {rows.height:<10} {first_t:<10.1f} "
            f"{last_t:<10.1f} {final_status:<14}"
        )

    n_confirmed_rows = df.filter(df["status"] == TrackStatus.CONFIRMED.value).height
    print(f"\n  Confirmed-status rows: {n_confirmed_rows} / {df.height}")


def main() -> None:
    print("Measurement Ingest Example")
    print("=" * 60)

    work_dir = Path(tempfile.mkdtemp(prefix="pytcl_measurement_ingest_"))
    csv_path = work_dir / "detections.csv"

    print("\nStep 1: Synthesizing detection CSV...")
    synthesize_detections_csv(csv_path)
    print(f"  Wrote {csv_path}")

    print("\nStep 2-3: Reading CSV and running GNN tracker...")
    ms, history, times = run_tracker(csv_path)

    print("\nStep 4: Converting to a long-format polars DataFrame...")
    df = tracks_to_polars(history, times)
    print_summary(ms, df)

    print("\nStep 5: Writing Parquet output...")
    out_path = OUTPUT_DIR / "measurement_ingest_tracks.parquet"
    df.write_parquet(out_path)
    print(f"  Wrote {out_path} ({out_path.stat().st_size} bytes)")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
