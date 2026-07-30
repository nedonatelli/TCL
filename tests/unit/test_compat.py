"""Tests for pytcl.io.compat backward compatibility adapters."""

import numpy as np
import pytest

from pytcl.dynamic_estimation.kalman.linear import kf_predict, kf_update
from pytcl.io import TrackDatabaseManager, TrackDatabaseStatus
from pytcl.io.compat import (
    EKFTrackAdapter,
    IMMTrackAdapter,
    KalmanTrackAdapter,
    ParticleFilterTrackAdapter,
    TrackerDatabaseAdapter,
    UKFTrackAdapter,
    store_filter_result,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture()
def db(tmp_path):
    """Fresh SQL database."""
    path = str(tmp_path / "compat_test.db")
    db = TrackDatabaseManager(path)
    db.open(mode="w")
    yield db
    db.close()


@pytest.fixture()
def cv_model():
    """Constant-velocity model matrices."""
    dt = 1.0
    F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = 0.1 * np.eye(4)
    R = np.eye(2) * 4.0
    return F, H, Q, R


# =============================================================================
# KalmanTrackAdapter
# =============================================================================


class TestKalmanTrackAdapter:
    """Tests for KalmanTrackAdapter."""

    def test_initialize(self, db, cv_model):
        F, H, Q, R = cv_model
        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)

        assert adapter.state is not None
        assert adapter.track_id == "trk_kf"
        info = db.get_track("trk_kf")
        assert info["status"] == TrackDatabaseStatus.TENTATIVE.value

    def test_predict(self, db, cv_model):
        F, H, Q, R = cv_model
        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        adapter.initialize(np.array([1.0, 1.0, 0.0, 0.0]), np.eye(4), 0.0)
        adapter.predict(1.0)

        # State should have advanced (x += vx*dt)
        assert adapter.state[0] > 1.0

    def test_update(self, db, cv_model):
        F, H, Q, R = cv_model
        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)
        adapter.predict(1.0)
        adapter.update(np.array([5.0, 3.0]), 1.0, sensor_id="radar")

        # State should move toward measurement
        assert adapter.state[0] > 0.0

        # Detection should be stored and associated
        dets = db.retrieve_detections(sensor_id="radar")
        assert len(dets) >= 1

    def test_predict_update_cycle(self, db, cv_model):
        """Run 10 predict-update cycles."""
        F, H, Q, R = cv_model
        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)

        rng = np.random.default_rng(42)
        for k in range(1, 11):
            z = np.array([k * 2.0, k * 1.0]) + rng.normal(0, 1, 2)
            adapter.predict_update(z, float(k))

        history = db.get_track_history("trk_kf")
        assert len(history["timestamps"]) >= 10

    def test_uninit_raises(self, db, cv_model):
        """Predict/update before initialize raises RuntimeError."""
        F, H, Q, R = cv_model
        adapter = KalmanTrackAdapter(db, "trk_kf", F, H, Q, R)
        with pytest.raises(RuntimeError):
            adapter.predict(1.0)


# =============================================================================
# EKFTrackAdapter
# =============================================================================


class TestEKFTrackAdapter:
    """Tests for EKFTrackAdapter."""

    def test_ekf_cycle(self, db):
        dt = 1.0
        Q = 0.1 * np.eye(4)
        R = np.diag([1.0, 0.01])

        def f(x):
            return np.array([x[0] + x[1] * dt, x[1], x[2] + x[3] * dt, x[3]])

        def F_jac(x):
            return np.array(
                [
                    [1, dt, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, dt],
                    [0, 0, 0, 1],
                ]
            )

        def h(x):
            r = np.sqrt(x[0] ** 2 + x[2] ** 2)
            theta = np.arctan2(x[2], x[0])
            return np.array([r, theta])

        def H_jac(x):
            r = max(np.sqrt(x[0] ** 2 + x[2] ** 2), 1e-10)
            return np.array(
                [
                    [x[0] / r, 0, x[2] / r, 0],
                    [-x[2] / r**2, 0, x[0] / r**2, 0],
                ]
            )

        adapter = EKFTrackAdapter(db, "trk_ekf", f, F_jac, h, H_jac, Q, R)
        adapter.initialize(np.array([10.0, 1.0, 10.0, 0.5]), np.eye(4) * 10.0, 0.0)

        for k in range(1, 6):
            adapter.predict(float(k))
            z = h(adapter.state) + np.array([0.1, 0.01])
            adapter.update(z, float(k))

        assert adapter.state is not None
        history = db.get_track_history("trk_ekf")
        assert len(history["timestamps"]) >= 5


# =============================================================================
# UKFTrackAdapter
# =============================================================================


class TestUKFTrackAdapter:
    """Tests for UKFTrackAdapter."""

    def test_ukf_cycle(self, db, cv_model):
        F, H, Q, R = cv_model

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        adapter = UKFTrackAdapter(db, "trk_ukf", f, h, Q, R)
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)

        rng = np.random.default_rng(42)
        for k in range(1, 6):
            adapter.predict(float(k))
            z = np.array([k * 1.0, k * 0.5]) + rng.normal(0, 0.5, 2)
            adapter.update(z, float(k))

        assert adapter.state is not None
        history = db.get_track_history("trk_ukf")
        assert len(history["timestamps"]) >= 5


# =============================================================================
# TrackerDatabaseAdapter
# =============================================================================


class TestTrackerDatabaseAdapter:
    """Tests for TrackerDatabaseAdapter."""

    def test_process_scan(self, db):
        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]),
            H=np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
            Q=0.1 * np.eye(4),
            R=np.eye(2) * 4.0,
        )

        adapter = TrackerDatabaseAdapter(db, tracker)

        # Process a few scans with measurements near origin
        rng = np.random.default_rng(42)
        for k in range(10):
            meas = [rng.normal(0, 2, 2) for _ in range(3)]
            adapter.process_scan(meas, dt=1.0, timestamp=float(k))

        # Should have some tracks in the database
        all_tracks = db.retrieve_all_tracks()
        assert len(all_tracks) > 0

    def test_get_track_list(self, db):
        from pytcl.trackers import MultiTargetTracker

        tracker = MultiTargetTracker(
            state_dim=4,
            meas_dim=2,
            F=np.eye(4),
            H=np.array([[1, 0, 0, 0], [0, 0, 1, 0]]),
            Q=0.1 * np.eye(4),
            R=np.eye(2),
        )

        adapter = TrackerDatabaseAdapter(db, tracker)
        rng = np.random.default_rng(42)
        for k in range(5):
            meas = [rng.normal(0, 1, 2)]
            adapter.process_scan(meas, dt=1.0, timestamp=float(k))

        track_list = adapter.get_track_list()
        assert track_list is not None


# =============================================================================
# IMMTrackAdapter
# =============================================================================


class TestIMMTrackAdapter:
    """Tests for IMMTrackAdapter."""

    def test_imm_cycle(self, db):
        from pytcl.dynamic_estimation.imm import IMMEstimator

        dt = 1.0
        F_cv = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        F_ca = F_cv.copy()  # simplified: same model for test
        Q = 0.1 * np.eye(4)
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        R = np.eye(2) * 4.0

        trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        imm = IMMEstimator(n_modes=2, state_dim=4, transition_matrix=trans)
        imm.set_mode_model(0, F_cv, Q)
        imm.set_mode_model(1, F_ca, Q * 2)
        imm.set_measurement_model(H, R)

        adapter = IMMTrackAdapter(db, "trk_imm", imm)
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)

        rng = np.random.default_rng(42)
        for k in range(1, 6):
            adapter.predict(float(k))
            z = np.array([k * 1.0, k * 0.5]) + rng.normal(0, 1, 2)
            adapter.update(z, float(k))

        assert adapter.mode_probs is not None
        assert len(adapter.mode_probs) == 2
        history = db.get_track_history("trk_imm")
        assert len(history["timestamps"]) >= 5


# =============================================================================
# ParticleFilterTrackAdapter
# =============================================================================


class TestParticleFilterTrackAdapter:
    """Tests for ParticleFilterTrackAdapter."""

    def test_pf_cycle(self, db, cv_model):
        F, H, Q, R = cv_model

        def f(x):
            return F @ x

        def h(x):
            return H @ x

        adapter = ParticleFilterTrackAdapter(
            db,
            "trk_pf",
            f,
            h,
            Q,
            R,
            n_particles=100,
        )
        adapter.initialize(np.zeros(4), np.eye(4) * 10.0, 0.0)

        rng = np.random.default_rng(42)
        for k in range(1, 6):
            z = np.array([k * 1.0, k * 0.5]) + rng.normal(0, 1, 2)
            adapter.predict_update(z, float(k))

        assert adapter.particles is not None
        assert adapter.particles.shape == (100, 4)
        history = db.get_track_history("trk_pf")
        assert len(history["timestamps"]) >= 5


# =============================================================================
# store_filter_result
# =============================================================================


class TestStoreFilterResult:
    """Tests for the store_filter_result helper."""

    def test_store_kalman_update(self, db, cv_model):
        F, H, Q, R = cv_model
        x0 = np.zeros(4)
        P0 = np.eye(4) * 10.0
        db.initiate_track("trk_sfr", x0, P0, 0.0)

        pred = kf_predict(x0, P0, F, Q)
        z = np.array([1.0, 0.5])
        upd = kf_update(pred.x, pred.P, z, H, R)

        store_filter_result(db, "trk_sfr", upd, 1.0)

        state = db.get_track_state("trk_sfr")
        np.testing.assert_allclose(state["state"], upd.x, atol=1e-10)

    def test_store_prediction(self, db, cv_model):
        F, H, Q, R = cv_model
        x0 = np.zeros(4)
        P0 = np.eye(4)
        db.initiate_track("trk_sfr2", x0, P0, 0.0)

        pred = kf_predict(x0, P0, F, Q)
        store_filter_result(db, "trk_sfr2", pred, 1.0, update_type="prediction")

        state = db.get_track_state("trk_sfr2")
        np.testing.assert_allclose(state["state"], pred.x, atol=1e-10)

    def test_store_sr_result(self, db):
        """Square-root result (with .S instead of .P) is handled."""
        from pytcl.dynamic_estimation.kalman.square_root import srkf_predict

        F = np.eye(4)
        S = np.eye(4)
        S_Q = np.eye(4) * 0.1

        db.initiate_track("trk_sr", np.zeros(4), np.eye(4), 0.0)

        pred = srkf_predict(np.zeros(4), S, F, S_Q)
        store_filter_result(db, "trk_sr", pred, 1.0, update_type="prediction")

        state = db.get_track_state("trk_sr")
        assert state["covariance"].shape == (4, 4)
        # P = S @ S.T should be PD
        eigvals = np.linalg.eigvalsh(state["covariance"])
        assert np.all(eigvals > 0)
