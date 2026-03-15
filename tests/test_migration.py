"""Tests for pytcl.io.migration module."""

import numpy as np
import pytest

from pytcl.io import TrackDatabaseManager, TrackHDF5Storage
from pytcl.io.migration import AnalysisResult, MigrationHelper


@pytest.fixture()
def helper():
    return MigrationHelper()


# =============================================================================
# Code Analysis
# =============================================================================


class TestAnalyzeV1Code:
    """Tests for MigrationHelper.analyze_v1_code."""

    def test_detect_kf(self, helper):
        code = """
from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
pred = kf_predict(x, P, F, Q)
upd = kf_update(pred.x, pred.P, z, H, R)
"""
        result = helper.analyze_v1_code(code)
        assert "kf" in result.filter_types

    def test_detect_ekf(self, helper):
        code = "result = ekf_predict(x, P, f, F, Q)"
        result = helper.analyze_v1_code(code)
        assert "ekf" in result.filter_types

    def test_detect_ukf(self, helper):
        code = "result = ukf_predict(x, P, f, Q)"
        result = helper.analyze_v1_code(code)
        assert "ukf" in result.filter_types

    def test_detect_imm(self, helper):
        code = "imm = IMMEstimator(n_modes=3, state_dim=6)"
        result = helper.analyze_v1_code(code)
        assert "imm" in result.filter_types

    def test_detect_particle(self, helper):
        code = "particles, weights = initialize_particles(x0, P0, 500)"
        result = helper.analyze_v1_code(code)
        assert "particle" in result.filter_types

    def test_detect_mht(self, helper):
        code = "tracker = MHTTracker(config=MHTConfig(n_scan=3))"
        result = helper.analyze_v1_code(code)
        assert "mht" in result.filter_types

    def test_detect_multi_target(self, helper):
        code = "tracker = MultiTargetTracker(state_dim=4, meas_dim=2)"
        result = helper.analyze_v1_code(code)
        assert "multi_target" in result.filter_types

    def test_detect_pickle_storage(self, helper):
        code = "pickle.dump(tracks, open('tracks.pkl', 'wb'))"
        result = helper.analyze_v1_code(code)
        assert "pickle" in result.storage_patterns

    def test_detect_numpy_storage(self, helper):
        code = "np.save('states.npy', track_states)"
        result = helper.analyze_v1_code(code)
        assert "numpy" in result.storage_patterns

    def test_detect_imports(self, helper):
        code = "from pytcl.dynamic_estimation.kalman import kf_predict"
        result = helper.analyze_v1_code(code)
        assert len(result.detected_imports) >= 1

    def test_complexity_low(self, helper):
        code = "kf_predict(x, P, F, Q)"
        result = helper.analyze_v1_code(code)
        assert result.estimated_complexity == "low"

    def test_complexity_medium(self, helper):
        code = "tracker = MHTTracker(config); kf_predict(x, P, F, Q)"
        result = helper.analyze_v1_code(code)
        assert result.estimated_complexity == "medium"

    def test_recommendations_nonempty(self, helper):
        code = "kf_predict(x, P, F, Q)"
        result = helper.analyze_v1_code(code)
        assert len(result.recommendations) > 0

    def test_analyze_from_file(self, helper, tmp_path):
        """analyze_v1_code can read from a file path."""
        code = "from pytcl.dynamic_estimation.kalman import kf_predict\nkf_predict(x, P, F, Q)\n"
        file_path = str(tmp_path / "legacy.py")
        with open(file_path, "w") as f:
            f.write(code)

        result = helper.analyze_v1_code(file_path)
        assert "kf" in result.filter_types


class TestAnalysisResult:
    """Tests for AnalysisResult."""

    def test_summary_format(self):
        r = AnalysisResult()
        r.filter_types = ["kf", "ekf"]
        r.storage_patterns = ["pickle"]
        r.estimated_complexity = "medium"
        r.recommendations = ["Use KalmanTrackAdapter", "Replace pickle"]

        text = r.summary()
        assert "kf" in text
        assert "pickle" in text
        assert "medium" in text

    def test_repr(self):
        r = AnalysisResult()
        r.filter_types = ["kf"]
        assert "kf" in repr(r)


# =============================================================================
# Data Conversion
# =============================================================================


class TestConvertLegacyTracks:
    """Tests for data format conversion."""

    def test_convert_to_sql(self, helper, tmp_path):
        rng = np.random.default_rng(42)
        legacy = {
            "trk_0": {
                "states": rng.normal(0, 1, (20, 4)),
                "covariances": np.array([np.eye(4) for _ in range(20)]),
                "timestamps": np.arange(20, dtype=float),
            },
            "trk_1": {
                "states": rng.normal(0, 1, (15, 4)),
                "covariances": None,  # Should default to identity
                "timestamps": None,  # Should default to 0..14
            },
        }

        db_path = str(tmp_path / "converted.db")
        count = helper.convert_legacy_tracks_to_sql(legacy, db_path)
        assert count == 2

        # Verify data
        db = TrackDatabaseManager(db_path)
        db.open(mode="r")
        tracks = db.retrieve_all_tracks()
        assert len(tracks) == 2
        for t in tracks:
            assert t["status"] == "confirmed"

        h = db.get_track_history("trk_0")
        assert h["states"].shape[1] == 4
        assert len(h["timestamps"]) >= 19  # 20 states stored

        h1 = db.get_track_history("trk_1")
        assert h1["states"].shape[1] == 4
        db.close()

    def test_convert_to_hdf5(self, helper, tmp_path):
        rng = np.random.default_rng(42)
        legacy = {
            "trk_0": {
                "states": rng.normal(0, 1, (30, 4)),
                "covariances": np.array([np.eye(4) for _ in range(30)]),
                "timestamps": np.arange(30, dtype=float),
            },
        }

        h5_path = str(tmp_path / "converted.h5")
        count = helper.convert_legacy_tracks_to_hdf5(
            legacy,
            h5_path,
            scenario_id="legacy_import",
        )
        assert count == 1

        store = TrackHDF5Storage(h5_path)
        store.open(mode="r")
        tracks = store.list_tracks("legacy_import")
        assert "trk_0" in tracks
        store.close()

    def test_convert_minimal_track(self, helper, tmp_path):
        """Track with only states (no covariances, no timestamps)."""
        legacy = {
            "trk_min": {
                "states": np.ones((5, 3)),
            },
        }
        db_path = str(tmp_path / "minimal.db")
        count = helper.convert_legacy_tracks_to_sql(legacy, db_path)
        assert count == 1


# =============================================================================
# Template Generation
# =============================================================================


class TestGenerateTemplate:
    """Tests for v2.0.0 template generation."""

    def test_sql_kf_template(self, helper):
        template = helper.generate_v2_template(backend="sql", filter_type="kf")
        assert "KalmanTrackAdapter" in template
        assert "TrackDatabaseManager" in template

    def test_sql_ekf_template(self, helper):
        template = helper.generate_v2_template(backend="sql", filter_type="ekf")
        assert "EKFTrackAdapter" in template

    def test_hdf5_template(self, helper):
        template = helper.generate_v2_template(backend="hdf5", filter_type="kf")
        assert "TrackHDF5Storage" in template

    def test_both_template(self, helper):
        template = helper.generate_v2_template(backend="both", filter_type="kf")
        assert "TrackDatabaseManager" in template
        assert "TrackHDF5Storage" in template
        assert "import_from_sql" in template

    def test_custom_n_targets(self, helper):
        template = helper.generate_v2_template(n_targets=7)
        assert "7" in template

    def test_fallback_template(self, helper):
        """Unknown filter_type falls back gracefully."""
        template = helper.generate_v2_template(
            backend="sql",
            filter_type="unknown_filter",
        )
        assert len(template) > 0


# =============================================================================
# Migration Checklist
# =============================================================================


class TestMigrationChecklist:
    """Tests for the migration checklist generator."""

    def test_checklist_format(self):
        checklist = MigrationHelper.generate_migration_checklist()
        assert "Pre-Migration" in checklist
        assert "Data Migration" in checklist
        assert "Code Migration" in checklist
        assert "Validation" in checklist
        assert "Post-Migration" in checklist
        assert "[ ]" in checklist
