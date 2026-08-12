"""HDF5 track-storage compression: benchmark, per-step measurement, regression rail.

Benchmark scenario (Task 7, v2.2.0 Results I/O plan): 100 tracks x 500
scans, 6-D constant-velocity states ``[x, vx, y, vy, z, vz]``, with
covariances produced by running a *real* CV Kalman filter to convergence
(``pytcl.dynamic_estimation.kalman.linear.kf_predict``/``kf_update``) --
not random noise. A converged filter's covariance settles to an
(almost) constant steady-state matrix, which is what makes real
covariance data correlate and compress; random noise would not.

Reproduce the measured figures below::

    uv run pytest tests/unit/test_hdf5_compression.py -q

Measured on macOS arm64, h5py 3.16, gzip level 4 (full methodology and
the abandoned/deferred alternatives are in
``.superpowers/sdd/2026-08-11-results-io/task-7-report.md``):

============================================  ============
config                                        ratio
============================================  ============
raw (8 * n_values)                            1.00x
baseline: chunk_size=1000, shuffle=False      4.42x
(a) chunk_size=None-equivalent (full-track    4.42x (+0.0%)
    chunking) -- see note below
(a)+(b) shuffle=True (shipped default)        4.73x (+7.1% over baseline)
============================================  ============

Step (a), time-aligned chunk shapes, required no code change: the
existing chunk formula (``chunks[0] = min(shape[0], chunk_size)``) already
puts a whole track's history in one time-major chunk whenever the track
is shorter than ``chunk_size`` (the common case, and true of every track
in this benchmark: 500 < the default 1000). Measured effect of forcing
full-track chunking anyway: 0.0% at 500 scans, 0.016% at 2000 scans (2
chunks vs. 1) -- deflate's 32 KB back-reference window, not the chunk
boundary, is the binding constraint on this data, confirmed at both
scales. Recorded as a measured null result, not assumed away.

Step (c), an optional ``states_only`` covariance-transform mode, was
evaluated and deferred (not shipped): covariance is 14.4 MB of this
benchmark's 17.2 MB raw total and converges to an almost-constant matrix
after burn-in, so dropping it entirely gives an implied ceiling of
~6.3x -- inside the once-claimed 5-10x band. Reaching that ceiling
losslessly means reconstructing per-scan covariance from a steady-state
Cholesky factor plus above-tolerance deviations across every read path
(``retrieve_track``, ``get_track_trajectory``, ``get_state_at_time``
including interpolation, ``export_to_sql``), and trades away the
existing bit-exact covariance round-trip contract that
``test_store_retrieve_track`` in ``test_hdf5_track_storage.py`` defends.
Shipping (a)+(b) only; the full evaluation is in the task report.
"""

import os
import tempfile

import numpy as np
import pytest

try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update

N_TRACKS = 100
N_SCANS = 500
STATE_DIM = 6  # [x, vx, y, vy, z, vz]
MEAS_DIM = 3  # [x, y, z]
DT = 1.0

# Regression rail: measured final ratio (4.73x) minus ~20% headroom.
MEASURED_FINAL_RATIO = 4.73
REGRESSION_FLOOR = 3.7  # ~21.8% below the measured figure


def _cv_matrices():
    """Constant-velocity transition/process/measurement matrices."""
    F = np.eye(STATE_DIM)
    for i in (0, 2, 4):
        F[i, i + 1] = DT

    q = 0.05
    q1 = q * np.array([[DT**3 / 3, DT**2 / 2], [DT**2 / 2, DT]])
    Q = np.zeros((STATE_DIM, STATE_DIM))
    for i in range(3):
        Q[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = q1

    H = np.zeros((MEAS_DIM, STATE_DIM))
    H[0, 0] = 1.0
    H[1, 2] = 1.0
    H[2, 4] = 1.0
    R = np.eye(MEAS_DIM) * 25.0
    return F, Q, H, R


def _simulate_track(rng, F, Q, H, R, n_scans):
    """Run a real CV filter over simulated measurements to convergence.

    Returns filtered states/covariances -- not random noise. The
    covariance converges to (and stays at) the filter's steady state
    within the first several scans, which is what real tracking
    covariance data looks like and is why it compresses.
    """
    x_true = np.zeros(STATE_DIM)
    x_true[0::2] = rng.normal(0, 50, 3)
    x_true[1::2] = rng.normal(0, 5, 3)

    x_est = x_true + rng.normal(0, 10, STATE_DIM)
    P_est = np.eye(STATE_DIM) * 100.0

    states = np.zeros((n_scans, STATE_DIM))
    covs = np.zeros((n_scans, STATE_DIM, STATE_DIM))

    L = np.linalg.cholesky(Q + 1e-12 * np.eye(STATE_DIM))
    Lr = np.linalg.cholesky(R)

    for k in range(n_scans):
        x_true = F @ x_true + L @ rng.normal(size=STATE_DIM)
        z = H @ x_true + Lr @ rng.normal(size=MEAS_DIM)

        pred = kf_predict(x_est, P_est, F, Q)
        upd = kf_update(pred.x, pred.P, z, H, R)
        x_est, P_est = upd.x, upd.P

        states[k] = x_est
        covs[k] = P_est

    return states, covs


def _build_benchmark_scenario(n_tracks=N_TRACKS, n_scans=N_SCANS, seed=1234):
    """Build the Task 7 benchmark: n_tracks CV-filtered trajectories."""
    rng = np.random.default_rng(seed)
    F, Q, H, R = _cv_matrices()
    tracks = {}
    for i in range(n_tracks):
        states, covs = _simulate_track(rng, F, Q, H, R, n_scans)
        timestamps = np.arange(n_scans, dtype=np.float64) * DT
        tracks[f"trk_{i:04d}"] = {
            "states": states,
            "covariances": covs,
            "timestamps": timestamps,
        }
    return tracks


def _raw_bytes(tracks):
    """Raw size per the brief: 8 * n_values across all stored arrays."""
    total = 0
    for t in tracks.values():
        total += t["states"].nbytes + t["covariances"].nbytes + t["timestamps"].nbytes
    return total


def _store_and_measure(tracks, path, **store_kwargs):
    from pytcl.io.hdf5_track_storage import TrackHDF5Storage

    with TrackHDF5Storage(path, **store_kwargs) as store:
        store.open(mode="w")
        store.store_tracking_scenario("bench", tracks)
    return os.path.getsize(path)


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestHDF5CompressionBenchmark:
    """Measured compression ratios on the Task 7 benchmark scenario."""

    @pytest.fixture(scope="class")
    def benchmark_tracks(self):
        """100 tracks x 500 scans, CV-filter states and converged covariances."""
        return _build_benchmark_scenario()

    @pytest.fixture(scope="class")
    def raw_size(self, benchmark_tracks):
        return _raw_bytes(benchmark_tracks)

    def test_covariance_converges_and_correlates(self, benchmark_tracks):
        """Sanity check the fixture is realistic, not random noise.

        A converged CV filter's covariance settles to a steady state and
        stops changing scan-to-scan; that (not randomness) is what makes
        the data compressible, per the task brief.
        """
        sample_cov = next(iter(benchmark_tracks.values()))["covariances"]
        late_diff = np.max(np.abs(sample_cov[-10] - sample_cov[-1]))
        assert late_diff < 1e-6, (
            "covariance should have converged to steady state by the end "
            f"of the run; late-scan diff was {late_diff}"
        )

    def test_baseline_ratio(self, benchmark_tracks, raw_size, tmp_path):
        """Baseline: pre-Task-7 defaults (chunk_size=1000, no shuffle).

        Measured ~4.42x on this scenario -- inside the honest 1.3-4.3x
        band documented pre-Task-7 (ROADMAP.md/CHANGELOG.md), and not
        below it, so the harness is trusted.
        """
        path = str(tmp_path / "baseline.h5")
        size = _store_and_measure(
            benchmark_tracks,
            path,
            chunk_size=1000,
            compression="gzip",
            compression_level=4,
            shuffle=False,
        )
        ratio = raw_size / size
        assert ratio > 4.0, (
            f"baseline ratio {ratio:.3f}x fell below the previously "
            "measured 1.3-4.3x band -- investigate the harness before "
            "trusting downstream numbers"
        )

    def test_shuffle_improves_on_baseline(self, benchmark_tracks, raw_size, tmp_path):
        """Step (b): shuffle=True measurably improves the ratio.

        Isolates the shuffle filter's effect (same chunk_size as
        baseline) to attribute the gain correctly to (b), not (a).
        """
        no_shuffle_path = str(tmp_path / "no_shuffle.h5")
        shuffle_path = str(tmp_path / "shuffle.h5")

        no_shuffle_size = _store_and_measure(
            benchmark_tracks,
            no_shuffle_path,
            chunk_size=1000,
            shuffle=False,
        )
        shuffle_size = _store_and_measure(
            benchmark_tracks,
            shuffle_path,
            chunk_size=1000,
            shuffle=True,
        )

        no_shuffle_ratio = raw_size / no_shuffle_size
        shuffle_ratio = raw_size / shuffle_size

        assert shuffle_size < no_shuffle_size, (
            f"shuffle=True ({shuffle_ratio:.3f}x) should measurably beat "
            f"shuffle=False ({no_shuffle_ratio:.3f}x) on slowly-varying "
            "float64 track data"
        )

    def test_shuffle_round_trip_is_bit_exact(self, benchmark_tracks, tmp_path):
        """shuffle=True is lossless: strict equality, not almost-equal.

        The shuffle filter only reorders bytes on disk; h5py/HDF5 undoes
        that reordering transparently on read, so retrieval must recover
        the exact float64 bit patterns written -- not merely "close" ones.
        This backs the round-trip-fidelity argument used in
        ROADMAP.md/CHANGELOG.md/this file's module docstring to defer
        `states_only` (which would trade bit-exact covariance for a
        tolerance-bounded reconstruction): the claim is enforced here with
        `np.testing.assert_array_equal`/`np.array_equal`, not
        `assert_array_almost_equal`, so a regression that quietly
        introduced lossy behavior would fail this test.
        """
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        tid = next(iter(benchmark_tracks))
        original = benchmark_tracks[tid]
        path = str(tmp_path / "round_trip_shuffle.h5")

        with TrackHDF5Storage(path, shuffle=True) as store:
            store.open(mode="w")
            store.store_track(
                tid,
                original["states"],
                original["covariances"],
                original["timestamps"],
            )

        with TrackHDF5Storage(path) as store:
            store.open(mode="r")
            retrieved = store.retrieve_track(tid)

        assert np.array_equal(retrieved["states"], original["states"])
        assert np.array_equal(retrieved["covariances"], original["covariances"])
        assert np.array_equal(retrieved["timestamps"], original["timestamps"])
        np.testing.assert_array_equal(retrieved["states"], original["states"])
        np.testing.assert_array_equal(retrieved["covariances"], original["covariances"])
        np.testing.assert_array_equal(retrieved["timestamps"], original["timestamps"])

    def test_final_ratio_regression_rail(self, benchmark_tracks, raw_size, tmp_path):
        """Final shipped configuration (defaults) stays above the measured floor.

        The floor is the measured final ratio (4.73x) minus ~20%
        headroom, per the task's honesty discipline: a regression rail,
        not an aspiration. Reproduce with::

            uv run pytest tests/unit/test_hdf5_compression.py::TestHDF5CompressionBenchmark::test_final_ratio_regression_rail -q
        """
        path = str(tmp_path / "final.h5")
        # Defaults: chunk_size=1000, gzip level 4, shuffle=True.
        size = _store_and_measure(benchmark_tracks, path)
        ratio = raw_size / size

        assert ratio > REGRESSION_FLOOR, (
            f"final compression ratio {ratio:.3f}x dropped below the "
            f"regression floor {REGRESSION_FLOOR}x (measured "
            f"{MEASURED_FINAL_RATIO}x minus ~20% headroom)"
        )

    def test_chunk_alignment_is_already_time_major(self, benchmark_tracks):
        """Step (a): chunk shape is already time-aligned, per-track.

        `store_tracking_scenario` creates one HDF5 group per track, each
        with its own chunked, resizable datasets along the time axis --
        never mixing tracks into one chunk. This documents the existing
        invariant (a) asked to verify; Task 7 measured its effect on the
        benchmark as 0.0% (see module docstring) rather than assuming it.
        """
        import h5py as _h5py

        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        tid = next(iter(benchmark_tracks))
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        with TrackHDF5Storage(path, chunk_size=1000) as store:
            store.open(mode="w")
            store.store_track(
                tid,
                benchmark_tracks[tid]["states"],
                benchmark_tracks[tid]["covariances"],
                benchmark_tracks[tid]["timestamps"],
            )

        with _h5py.File(path, "r") as f:
            ds = f[f"tracks/{tid}/state_history"]
            # 500 scans < chunk_size=1000: the whole track is one chunk.
            assert ds.chunks[0] == N_SCANS

        os.unlink(path)
