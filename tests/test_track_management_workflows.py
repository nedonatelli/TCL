"""Smoke tests for track management workflow examples."""

import sys
from pathlib import Path

import pytest

# Ensure examples directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

try:
    import h5py  # noqa: F401

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


class TestWorkflowSmoke:
    """Verify each workflow runs without errors."""

    def test_workflow_radar_tracking(self, tmp_path):
        """Workflow 1: RADAR pipeline completes."""
        from track_management_workflows import workflow_radar_tracking

        workflow_radar_tracking(output_dir=tmp_path, n_steps=10)

    def test_workflow_multi_sensor_fusion(self, tmp_path):
        """Workflow 2: Multi-sensor fusion completes."""
        from track_management_workflows import workflow_multi_sensor_fusion

        workflow_multi_sensor_fusion(output_dir=tmp_path, n_steps=10)

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_workflow_realtime_to_archive(self, tmp_path):
        """Workflow 3: SQL to HDF5 archive completes."""
        from track_management_workflows import workflow_realtime_to_archive

        workflow_realtime_to_archive(output_dir=tmp_path, n_steps=15)

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_workflow_track_comparison(self, tmp_path):
        """Workflow 4: GNN vs MHT comparison completes."""
        from track_management_workflows import workflow_track_comparison

        workflow_track_comparison(output_dir=tmp_path, n_steps=8)

    @pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
    def test_workflow_scenario_replay(self, tmp_path):
        """Workflow 5: Scenario replay completes."""
        from track_management_workflows import workflow_scenario_replay

        workflow_scenario_replay(output_dir=tmp_path, n_steps=15)

    def test_workflow_clutter_handling(self, tmp_path):
        """Workflow 6: Clutter handling completes."""
        from track_management_workflows import workflow_clutter_handling

        workflow_clutter_handling(output_dir=tmp_path, n_steps=10)
