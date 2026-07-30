"""A complete tracking pipeline, exercised end to end.

Every other test in this suite checks one function against a reference. Nothing
checked that the subsystems *compose*: that a polar detection can be converted
to Cartesian with a consistent covariance, gated, associated, filtered, carried
through track confirmation, persisted, reloaded, and scored. That gap is how a
library reaches 4000 passing tests while its own examples call an API it does
not have.

The pipeline is:

    truth -> polar measurement -> Cartesian conversion (with covariance)
          -> gating + association + filtering + track management
          -> HDF5 persistence -> round trip -> OSPA / NEES scoring

Assertions are on properties that only hold if the whole chain is correct:

* the converted measurement covariance has the cross-range spread the geometry
  demands (``r * sigma_bearing``), which fails if the Jacobian is wrong;
* the filter's matched position error is *below* the measurement error, which
  fails if the filter is not actually integrating information;
* NEES sits inside a chi-square interval, which fails if the covariance is
  mis-scaled in either direction -- a filter can track well and still be
  badly over-confident;
* clutter does not make the track count run away;
* a persisted track reloads bit-for-bit.

State layout throughout is ``[x, vx, y, vy]``: ``f_constant_velocity`` builds a
block-diagonal F with a (position, velocity) pair per spatial dimension, not
``[x, y, vx, vy]``.
"""

import numpy as np
import pytest

from pytcl.assignment_algorithms import assign2d
from pytcl.coordinate_systems import cart2pol, pol2cart, polar_jacobian_inv
from pytcl.dynamic_estimation import kf_predict, kf_update
from pytcl.dynamic_models import f_constant_velocity, q_constant_velocity
from pytcl.io import TrackHDF5Storage
from pytcl.performance_evaluation import consistency_test, nees_sequence, ospa
from pytcl.trackers import MultiTargetTracker, TrackStatus

IX, IVX, IY, IVY = 0, 1, 2, 3

DT = 1.0
N_SCANS = 40
SIGMA_RANGE = 5.0
SIGMA_BEARING = np.radians(0.5)
P_DETECT = 0.95
CLUTTER_RATE = 1.0

# [x, vx, y, vy]; well separated in bearing so association is unambiguous
TARGETS = np.array(
    [
        [1000.0, 12.0, 1200.0, -4.0],
        [-900.0, 9.0, 1500.0, 3.0],
        [200.0, -6.0, 2200.0, 8.0],
    ]
)


def _polar_covariance():
    return np.diag([SIGMA_RANGE**2, SIGMA_BEARING**2])


def _convert(r, theta):
    """Polar detection -> Cartesian position and covariance."""
    xy = np.asarray(pol2cart(float(r), float(theta)), dtype=float).ravel()[:2]
    J = polar_jacobian_inv(float(r), float(theta))
    return xy, J @ _polar_covariance() @ J.T


def _truth_track():
    F = f_constant_velocity(DT, num_dims=2)
    truth = np.zeros((N_SCANS, len(TARGETS), 4))
    state = TARGETS.copy()
    for k in range(N_SCANS):
        truth[k] = state
        state = state @ F.T
    return F, truth


class PipelineResult:
    """Everything one run of the pipeline produced."""

    def __init__(self):
        self.confirmed_per_scan = []
        self.ospa_per_scan = []
        self.position_errors = []
        self.position_nees = []
        self.total_tracks = 0
        self.first_confirmed_scan = None
        self.track_histories = {}


def _run_pipeline(seed, p_detect=P_DETECT, clutter_rate=CLUTTER_RATE):
    rng = np.random.default_rng(seed)
    F, truth = _truth_track()
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    Q = q_constant_velocity(DT, sigma_a=0.5, num_dims=2)
    # A converted polar detection has an anisotropic, range-dependent
    # covariance: sigma_range (5 m) down-range against r * sigma_bearing
    # (14-19 m here) cross-range. The tracker is given each detection's own
    # covariance, so this fixed R is only the fallback used for clutter, whose
    # true covariance is unknown by definition.
    R = np.diag([16.0**2, 16.0**2])

    tracker = MultiTargetTracker(
        state_dim=4,
        meas_dim=2,
        F=F,
        H=H,
        Q=Q,
        R=R,
        gate_probability=0.99,
        confirm_hits=3,
        confirm_window=5,
        max_misses=3,
    )

    out = PipelineResult()
    for k in range(N_SCANS):
        detections, covariances = [], []
        for t in range(len(TARGETS)):
            if rng.random() >= p_detect:
                continue
            r, theta = cart2pol(np.array([truth[k, t, IX], truth[k, t, IY]]))
            r_n = float(r) + rng.normal(0.0, SIGMA_RANGE)
            th_n = float(theta) + rng.normal(0.0, SIGMA_BEARING)
            xy, R_converted = _convert(r_n, th_n)
            detections.append(xy)
            covariances.append(R_converted)
        for _ in range(rng.poisson(clutter_rate)):
            detections.append(rng.uniform(-3000.0, 3000.0, size=2))
            covariances.append(R)  # clutter has no meaningful covariance

        tracks = tracker.process(detections, DT, measurement_covariances=covariances)
        confirmed = [t for t in tracks if t.status is TrackStatus.CONFIRMED]
        out.confirmed_per_scan.append(len(confirmed))
        if confirmed and out.first_confirmed_scan is None:
            out.first_confirmed_scan = k

        for tr in confirmed:
            out.track_histories.setdefault(tr.id, []).append(
                (
                    float(k) * DT,
                    np.asarray(tr.state, float),
                    np.asarray(tr.covariance, float),
                )
            )

        estimates = [np.array([t.state[IX], t.state[IY]]) for t in confirmed]
        actual = [
            np.array([truth[k, i, IX], truth[k, i, IY]]) for i in range(len(TARGETS))
        ]
        out.ospa_per_scan.append(ospa(actual, estimates, c=200.0, p=2.0).ospa)

        # score only the settled half, matching tracks to truth by position
        if k >= N_SCANS // 2 and confirmed:
            cost = np.array(
                [
                    [
                        np.linalg.norm(a - np.array([t.state[IX], t.state[IY]]))
                        for t in confirmed
                    ]
                    for a in actual
                ]
            )
            assignment = assign2d(cost)
            for i, j in enumerate(assignment.col_indices):
                if not 0 <= j < len(confirmed):
                    continue
                tr = confirmed[j]
                err = actual[i] - np.array([tr.state[IX], tr.state[IY]])
                out.position_errors.append(float(np.linalg.norm(err)))
                S = np.asarray(tr.covariance, float)[np.ix_([IX, IY], [IX, IY])]
                out.position_nees.append(float(err @ np.linalg.solve(S, err)))

    out.total_tracks = len(tracker.tracks)
    return out


@pytest.fixture(scope="module")
def pipeline():
    return _run_pipeline(seed=7)


@pytest.fixture(scope="module")
def clean_pipeline():
    """No clutter, every target detected -- the easy case."""
    return _run_pipeline(seed=11, p_detect=1.0, clutter_rate=0.0)


# --- the conversion stage --------------------------------------------------


@pytest.mark.parametrize("bearing_deg", [0.0, 37.0, 90.0, 175.0, -120.0])
@pytest.mark.parametrize("range_m", [500.0, 5000.0, 40000.0])
def test_converted_covariance_matches_geometry(range_m, bearing_deg):
    """Cross-range spread must be r * sigma_bearing, down-range sigma_range.

    This is the check that a wrong Jacobian cannot survive: the converted
    covariance's principal axes are fixed by the geometry, independent of
    where the detection sits.
    """
    theta = np.radians(bearing_deg)
    _, R_cart = _convert(range_m, theta)

    eigenvalues = np.sort(np.linalg.eigvalsh(R_cart))
    down_range, cross_range = np.sqrt(eigenvalues[0]), np.sqrt(eigenvalues[1])
    expected_cross = range_m * SIGMA_BEARING
    expected_down = SIGMA_RANGE

    # at short range the two are comparable, so compare the sorted pair
    expected = np.sort([expected_down, expected_cross])
    got = np.sort([down_range, cross_range])
    np.testing.assert_allclose(got, expected, rtol=1e-9)


def test_polar_round_trip_is_exact():
    """cart2pol and pol2cart must invert each other."""
    rng = np.random.default_rng(3)
    points = rng.uniform(-5000.0, 5000.0, size=(200, 2))
    for xy in points:
        r, theta = cart2pol(xy)
        back = np.asarray(pol2cart(float(r), float(theta)), dtype=float).ravel()[:2]
        np.testing.assert_allclose(back, xy, rtol=1e-9, atol=1e-9)


# --- the tracking stages --------------------------------------------------


def test_all_targets_are_confirmed(pipeline):
    """Three targets in, three confirmed tracks held."""
    settled = pipeline.confirmed_per_scan[N_SCANS // 4 :]
    correct = sum(1 for n in settled if n == len(TARGETS))
    assert correct / len(settled) >= 0.7, (
        f"held {len(TARGETS)} confirmed tracks on only {correct}/{len(settled)} "
        f"settled scans: {pipeline.confirmed_per_scan}"
    )


def test_confirmation_is_prompt(pipeline):
    """confirm_hits=3 means confirmation cannot take many more than 3 scans."""
    assert pipeline.first_confirmed_scan is not None, "nothing was ever confirmed"
    assert pipeline.first_confirmed_scan <= 6, (
        f"first confirmation at scan {pipeline.first_confirmed_scan}"
    )


def test_filter_beats_the_raw_measurements(pipeline):
    """The whole point of filtering: less error than a single detection.

    If association or the update step were broken the estimate would be no
    better than the measurement -- or worse.
    """
    mean_range = float(np.mean(np.linalg.norm(TARGETS[:, [IX, IY]], axis=1)))
    measurement_sigma = float(np.hypot(SIGMA_RANGE, mean_range * SIGMA_BEARING))
    mean_error = float(np.mean(pipeline.position_errors))
    assert mean_error < 0.9 * measurement_sigma, (
        f"mean matched error {mean_error:.2f} is not better than a raw "
        f"detection ({measurement_sigma:.2f})"
    )


def test_covariance_is_not_overconfident(pipeline):
    """The pipeline must not claim more accuracy than it has.

    Clutter and missed detections keep this from being a clean chi-square
    sample, so the band is wide. What it rules out is the dangerous direction:
    NEES well above 2 means the filter reports a tight covariance around a
    wrong state. The sharp calibration checks are
    test_clean_pipeline_covariance_is_calibrated and
    test_wellspecified_filter_is_statistically_consistent.
    """
    mean_nees = float(np.mean(pipeline.position_nees))
    assert 0.2 <= mean_nees <= 4.0, (
        f"position NEES {mean_nees:.2f} outside the plausible band for df=2; "
        f"above 4 means the covariance is too small for the actual error"
    )


def test_clean_pipeline_covariance_is_calibrated(clean_pipeline):
    """With per-detection covariances the pipeline itself is consistent.

    This is what passing each detection's own covariance buys. Handing the
    tracker a single fixed R forces a choice: size it to the down-range term
    and the 99% gate is too tight -- true detections fall outside it and 3
    targets produce 4-5 confirmed tracks even with no clutter -- or size it to
    the cross-range term and cardinality is right but the covariance is
    inflated, NEES falling to about 1.0 and failing this test. Measured across
    sigma in {10, 15, 20, 25, 30, 40}, no fixed value satisfies both. With the
    converted covariance supplied per detection, both hold at once.
    """
    values = np.asarray(clean_pipeline.position_nees)
    result = consistency_test(values, df=2, confidence=0.95)
    assert result.is_consistent, (
        f"position NEES {result.statistic:.2f} outside "
        f"[{result.lower_bound:.2f}, {result.upper_bound:.2f}] for df=2"
    )


def test_wellspecified_filter_is_statistically_consistent():
    """Give the filter the exact generating model; NEES must equal state_dim.

    This is the sharp check on the predict/update covariance arithmetic. It is
    separate from the pipeline because it removes every modelling compromise:
    linear dynamics, isotropic measurement noise, and an R that is exactly
    right. If the covariance recursion were wrong, no amount of tuning would
    put NEES at 4.
    """
    n_steps, sigma = 300, 8.0
    F = f_constant_velocity(DT, num_dims=2)
    Q = q_constant_velocity(DT, sigma_a=0.6, num_dims=2)
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    R = np.eye(2) * sigma**2

    rng = np.random.default_rng(23)
    noise_chol = np.linalg.cholesky(Q + np.eye(4) * 1e-12)

    x_true = np.array([100.0, 5.0, -50.0, 3.0])
    x = x_true + rng.normal(0.0, 10.0, 4)
    P = np.eye(4) * 100.0

    truths, estimates, covariances = [], [], []
    for step in range(n_steps):
        x_true = F @ x_true + noise_chol @ rng.normal(size=4)
        z = H @ x_true + rng.normal(0.0, sigma, 2)
        prediction = kf_predict(x, P, F, Q)
        update = kf_update(prediction.x, prediction.P, z, H, R)
        x, P = update.x, update.P
        if step > 30:  # let the initial covariance settle
            truths.append(x_true.copy())
            estimates.append(x.copy())
            covariances.append(P.copy())

    values = nees_sequence(np.array(truths), np.array(estimates), np.array(covariances))
    result = consistency_test(values, df=4, confidence=0.95)
    assert result.is_consistent, (
        f"NEES {result.statistic:.3f} outside "
        f"[{result.lower_bound:.2f}, {result.upper_bound:.2f}] for a correctly "
        f"specified 4-state filter (ideal 4.0)"
    )


def test_clutter_does_not_multiply_tracks(pipeline):
    """A false alarm per scan must not spawn a track per scan."""
    # An unbounded tracker would create roughly one track per false alarm,
    # so ~N_SCANS of them. Three per target leaves room for the handful of
    # short-lived tracks clutter legitimately starts before they are deleted.
    assert pipeline.total_tracks <= 3 * len(TARGETS), (
        f"{pipeline.total_tracks} tracks created for {len(TARGETS)} targets "
        f"over {N_SCANS} scans with {CLUTTER_RATE} false alarms per scan"
    )


def test_clean_scenario_is_exact_on_cardinality(clean_pipeline):
    """With no clutter and no missed detections there is nothing to excuse."""
    settled = clean_pipeline.confirmed_per_scan[N_SCANS // 4 :]
    assert set(settled) == {len(TARGETS)}, (
        f"expected exactly {len(TARGETS)} tracks every settled scan, saw "
        f"{sorted(set(settled))}"
    )
    assert clean_pipeline.total_tracks == len(TARGETS), (
        f"{clean_pipeline.total_tracks} tracks created with no clutter"
    )


def test_ospa_reflects_a_working_tracker(pipeline):
    """OSPA over the settled half stays well inside the cutoff."""
    settled = np.asarray(pipeline.ospa_per_scan[N_SCANS // 2 :])
    assert settled.mean() < 60.0, (
        f"mean OSPA {settled.mean():.1f} over the settled half"
    )


# --- the persistence stage ------------------------------------------------


def test_tracks_round_trip_through_hdf5(pipeline, tmp_path):
    """A persisted track must reload bit-for-bit, metadata included."""
    assert pipeline.track_histories, "no confirmed track history to persist"
    path = tmp_path / "pipeline_tracks.h5"

    written = {}
    storage = TrackHDF5Storage(str(path))
    storage.open("w")
    for track_id, history in pipeline.track_histories.items():
        timestamps = np.array([h[0] for h in history])
        states = np.stack([h[1] for h in history])
        covariances = np.stack([h[2] for h in history])
        written[str(track_id)] = (states, covariances, timestamps)
        storage.store_track(
            str(track_id),
            states,
            covariances,
            timestamps,
            metadata={"status": "confirmed", "scans": len(history)},
        )
    storage.close()

    storage = TrackHDF5Storage(str(path))
    storage.open("r")
    try:
        assert sorted(storage.list_tracks()) == sorted(written)
        for track_id, (states, covariances, timestamps) in written.items():
            got = storage.retrieve_track(track_id)
            np.testing.assert_array_equal(np.asarray(got["states"]), states)
            np.testing.assert_array_equal(np.asarray(got["covariances"]), covariances)
            np.testing.assert_array_equal(np.asarray(got["timestamps"]), timestamps)
            assert got["metadata"]["status"] == "confirmed"
            assert int(got["metadata"]["scans"]) == len(timestamps)
    finally:
        storage.close()


def test_reloaded_tracks_still_score_the_same(pipeline, tmp_path):
    """Scoring must not depend on whether the data came from memory or disk."""
    path = tmp_path / "score_tracks.h5"
    track_id, history = next(iter(pipeline.track_histories.items()))
    states = np.stack([h[1] for h in history])
    covariances = np.stack([h[2] for h in history])
    timestamps = np.array([h[0] for h in history])

    storage = TrackHDF5Storage(str(path))
    storage.open("w")
    storage.store_track(str(track_id), states, covariances, timestamps)
    storage.close()

    storage = TrackHDF5Storage(str(path))
    storage.open("r")
    try:
        trajectory = storage.get_track_trajectory(str(track_id))
    finally:
        storage.close()

    reloaded = np.asarray(trajectory["states"])
    in_memory = [np.array([s[IX], s[IY]]) for s in states]
    from_disk = [np.array([s[IX], s[IY]]) for s in reloaded]
    assert ospa(in_memory, from_disk, c=200.0, p=2.0).ospa == 0.0
