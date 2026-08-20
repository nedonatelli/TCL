"""Migration tools for transitioning v1.x tracking pipelines to v2.0.0.

Provides utilities for analyzing legacy code, converting data formats,
and generating v2.0.0 template code using TrackDatabaseManager and
TrackHDF5Storage.

Examples
--------
Analyze a legacy tracking script:

>>> from pytcl.io.migration import MigrationHelper
>>> helper = MigrationHelper()
>>> analysis = helper.analyze_v1_code("legacy_tracker.py")  # doctest: +SKIP
>>> print(analysis.recommendations)  # doctest: +SKIP

Convert legacy pickle tracks to SQL:

>>> helper.convert_legacy_tracks_to_sql(  # doctest: +SKIP
...     legacy_data={"trk_0": {"states": states, "times": times}},
...     output_db="new_tracks.db",
... )

Generate a v2.0.0 template:

>>> template = helper.generate_v2_template(backend="sql")
>>> print(template)  # doctest: +SKIP
"""

import os
import re
from typing import Any, Dict, List

import numpy as np


class AnalysisResult:
    """Result of analyzing a v1.x tracking pipeline.

    Attributes
    ----------
    filter_types : list of str
        Detected filter types (e.g., ['kf', 'ekf', 'ukf']).
    storage_patterns : list of str
        Detected storage patterns (e.g., ['pickle', 'numpy', 'csv']).
    recommendations : list of str
        Suggested migration steps.
    estimated_complexity : str
        'low', 'medium', or 'high'.
    detected_imports : list of str
        pytcl imports found in the source.
    """

    def __init__(self) -> None:
        self.filter_types: List[str] = []
        self.storage_patterns: List[str] = []
        self.recommendations: List[str] = []
        self.estimated_complexity: str = "low"
        self.detected_imports: List[str] = []

    def __repr__(self) -> str:
        return (
            f"AnalysisResult(filters={self.filter_types}, "
            f"storage={self.storage_patterns}, "
            f"complexity={self.estimated_complexity})"
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            "=== Migration Analysis ===",
            f"Detected filters: {', '.join(self.filter_types) or 'none'}",
            f"Storage patterns: {', '.join(self.storage_patterns) or 'none'}",
            f"Estimated complexity: {self.estimated_complexity}",
            "",
            "Recommendations:",
        ]
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        return "\n".join(lines)


class MigrationHelper:
    """Utilities for migrating v1.x tracking pipelines to v2.0.0.

    Provides code analysis, data conversion, and template generation.

    Examples
    --------
    >>> helper = MigrationHelper()
    >>> result = helper.analyze_v1_code("my_tracker.py")  # doctest: +SKIP
    >>> print(result.summary())  # doctest: +SKIP
    """

    # Patterns for detecting filter usage
    _FILTER_PATTERNS: Dict[str, List[str]] = {
        "kf": [r"\bkf_predict", r"\bkf_update", r"\bkf_predict_update"],
        "ekf": [r"ekf_predict", r"ekf_update", r"ekf_predict_auto"],
        "ukf": [r"\bukf_predict", r"\bukf_update"],
        "ckf": [r"ckf_predict", r"ckf_update"],
        "srkf": [r"srkf_predict", r"srkf_update", r"sr_ukf_predict"],
        "imm": [r"IMMEstimator", r"imm_predict", r"imm_update"],
        "particle": [
            r"bootstrap_pf",
            r"particle_mean",
            r"initialize_particles",
        ],
        "mht": [r"MHTTracker", r"MHTConfig"],
        "jpda": [r"jpda_probabilities", r"jpda_update", r"jpda\b"],
        "multi_target": [r"MultiTargetTracker", r"gnn_association"],
        "information": [r"information_filter", r"srif_predict"],
        "hinf": [r"hinf_predict", r"hinf_update"],
    }

    _STORAGE_PATTERNS: Dict[str, List[str]] = {
        "pickle": [r"pickle\.dump", r"pickle\.load", r"\.pkl", r"\.pickle"],
        "numpy": [r"np\.save\b", r"np\.load\b", r"\.npy", r"\.npz"],
        "csv": [r"\.csv", r"np\.savetxt", r"np\.loadtxt", r"to_csv"],
        "json": [r"json\.dump", r"json\.load"],
        "hdf5": [r"h5py", r"\.h5\b", r"\.hdf5"],
        "sql": [r"sqlite3", r"TrackDatabaseManager", r"SQLStorage"],
    }

    def analyze_v1_code(self, source: str) -> AnalysisResult:
        """Analyze a v1.x tracking pipeline for migration.

        Parameters
        ----------
        source : str
            Either a file path or a string of Python source code.

        Returns
        -------
        AnalysisResult
            Analysis with detected patterns and recommendations.
        """
        # Read source if it's a file path
        if os.path.isfile(source):
            with open(source) as f:
                code = f.read()
        else:
            code = source

        result = AnalysisResult()

        # Detect filter types
        for filter_name, patterns in self._FILTER_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code):
                    if filter_name not in result.filter_types:
                        result.filter_types.append(filter_name)
                    break

        # Detect storage patterns
        for storage_name, patterns in self._STORAGE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, code):
                    if storage_name not in result.storage_patterns:
                        result.storage_patterns.append(storage_name)
                    break

        # Detect pytcl imports
        import_pattern = r"from\s+pytcl[\w.]*\s+import\s+[\w\s,()]*"
        result.detected_imports = re.findall(import_pattern, code)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        # Estimate complexity
        n_filters = len(result.filter_types)
        has_multi = any(
            f in result.filter_types for f in ["mht", "jpda", "multi_target", "imm"]
        )
        if n_filters <= 1 and not has_multi:
            result.estimated_complexity = "low"
        elif n_filters <= 3 or has_multi:
            result.estimated_complexity = "medium"
        else:
            result.estimated_complexity = "high"

        return result

    def _generate_recommendations(self, result: AnalysisResult) -> List[str]:
        """Generate migration recommendations based on analysis."""
        recs: List[str] = []

        # Storage recommendations
        if "pickle" in result.storage_patterns:
            recs.append(
                "Replace pickle storage with TrackDatabaseManager (SQL) "
                "for real-time tracking or TrackHDF5Storage for archival."
            )
        if "numpy" in result.storage_patterns:
            recs.append(
                "Replace np.save/np.load with TrackHDF5Storage for "
                "compressed, queryable track storage."
            )
        if not result.storage_patterns or result.storage_patterns == ["csv"]:
            recs.append(
                "Add TrackDatabaseManager for structured track lifecycle "
                "management with detection association."
            )

        # Filter-specific recommendations
        for ftype in result.filter_types:
            if ftype == "kf":
                recs.append(
                    "Use KalmanTrackAdapter from pytcl.io.compat to wrap "
                    "KF predict/update with automatic SQL persistence."
                )
            elif ftype == "ekf":
                recs.append(
                    "Use EKFTrackAdapter from pytcl.io.compat to wrap "
                    "EKF predict/update with automatic SQL persistence."
                )
            elif ftype == "ukf":
                recs.append(
                    "Use UKFTrackAdapter from pytcl.io.compat to wrap "
                    "UKF predict/update with automatic SQL persistence."
                )
            elif ftype == "imm":
                recs.append(
                    "Use IMMTrackAdapter from pytcl.io.compat to persist "
                    "the combined IMM state (mode probabilities stay on the "
                    "live filter object; they are not persisted)."
                )
            elif ftype == "particle":
                recs.append(
                    "Use ParticleFilterTrackAdapter from pytcl.io.compat "
                    "to store weighted mean/covariance from particle cloud."
                )
            elif ftype in ("mht", "multi_target"):
                recs.append(
                    "Use TrackerDatabaseAdapter from pytcl.io.compat to "
                    "wrap your tracker with automatic track persistence."
                )
            elif ftype == "srkf":
                recs.append(
                    "Use store_filter_result() from pytcl.io.compat to "
                    "store square-root filter results (auto-converts S to P)."
                )

        # General recommendations
        recs.append(
            "After migration, use TrackHDF5Storage.import_from_sql() to "
            "archive completed missions to compressed HDF5."
        )

        return recs

    def convert_legacy_tracks_to_sql(
        self,
        legacy_data: Dict[str, Dict[str, Any]],
        output_db: str,
    ) -> int:
        """Convert legacy track data to a SQL database.

        Parameters
        ----------
        legacy_data : dict
            Dictionary of track_id -> {"states": ndarray (N, state_dim),
            "covariances": ndarray (N, state_dim, state_dim) or None,
            "timestamps": ndarray (N,) or None}.
            Covariances default to identity if not provided.
            Timestamps default to 0..N-1 if not provided.
        output_db : str
            Path to the output SQLite database.

        Returns
        -------
        int
            Number of tracks converted.
        """
        from pytcl.io.track_database import TrackDatabaseManager

        db = TrackDatabaseManager(output_db)
        db.open(mode="w")

        count = 0
        for tid, data in legacy_data.items():
            states = np.asarray(data["states"], dtype=np.float64)
            n_steps, state_dim = states.shape

            if "covariances" in data and data["covariances"] is not None:
                covs = np.asarray(data["covariances"], dtype=np.float64)
            else:
                covs = np.array([np.eye(state_dim) for _ in range(n_steps)])

            if "timestamps" in data and data["timestamps"] is not None:
                times = np.asarray(data["timestamps"], dtype=np.float64)
            else:
                times = np.arange(n_steps, dtype=np.float64)

            db.initiate_track(tid, states[0], covs[0], float(times[0]))
            if n_steps > 1:
                db.store_track_history(tid, states[1:], covs[1:], times[1:])
            db.confirm_track(tid)
            count += 1

        db.close()
        return count

    def convert_legacy_tracks_to_hdf5(
        self,
        legacy_data: Dict[str, Dict[str, Any]],
        output_h5: str,
        scenario_id: str = "migrated",
    ) -> int:
        """Convert legacy track data to an HDF5 archive.

        Parameters
        ----------
        legacy_data : dict
            Same format as convert_legacy_tracks_to_sql.
        output_h5 : str
            Path to the output HDF5 file.
        scenario_id : str
            Scenario identifier. Default is 'migrated'.

        Returns
        -------
        int
            Number of tracks converted.
        """
        from pytcl.io.hdf5_track_storage import TrackHDF5Storage

        tracks: Dict[str, Dict[str, Any]] = {}
        for tid, data in legacy_data.items():
            states = np.asarray(data["states"], dtype=np.float64)
            n_steps, state_dim = states.shape

            if "covariances" in data and data["covariances"] is not None:
                covs = np.asarray(data["covariances"], dtype=np.float64)
            else:
                covs = np.array([np.eye(state_dim) for _ in range(n_steps)])

            if "timestamps" in data and data["timestamps"] is not None:
                times = np.asarray(data["timestamps"], dtype=np.float64)
            else:
                times = np.arange(n_steps, dtype=np.float64)

            tracks[tid] = {
                "states": states,
                "covariances": covs,
                "timestamps": times,
            }

        store = TrackHDF5Storage(output_h5)
        store.open(mode="w")
        store.store_tracking_scenario(scenario_id, tracks)
        store.close()

        return len(tracks)

    def generate_v2_template(
        self,
        backend: str = "sql",
        filter_type: str = "kf",
        n_targets: int = 3,
    ) -> str:
        """Generate v2.0.0 template code for a tracking pipeline.

        Parameters
        ----------
        backend : str
            Target backend: 'sql', 'hdf5', or 'both'. Default is 'sql'.
        filter_type : str
            Filter type. Default is 'kf'. Only three distinct templates
            exist -- ('sql', 'kf'), ('sql', 'ekf') which 'ukf' also uses,
            and one apiece for the 'hdf5' and 'both' backends. Anything else,
            including 'imm' and 'particle', falls back to the nearest
            template for that backend; the returned source names which one it
            is in its own header.
        n_targets : int
            Number of targets in the template. Default is 3.

        Returns
        -------
        str
            Python source code for the tracking pipeline template.

        Notes
        -----
        The parameter list previously read "'kf', 'ekf', 'ukf', 'imm',
        'particle'" as though each had its own template. Four templates
        exist. ``'imm'`` and ``'particle'`` return Kalman scaffolding, and
        the 'hdf5' and 'both' backends ignore ``filter_type`` entirely --
        deliberate, but not what the list implied.
        """
        templates = {
            ("sql", "kf"): self._template_sql_kf,
            ("sql", "ekf"): self._template_sql_ekf,
            ("sql", "ukf"): self._template_sql_ekf,  # same structure
            ("hdf5", "kf"): self._template_hdf5_kf,
            ("both", "kf"): self._template_both_kf,
        }

        key = (backend, filter_type)
        if key not in templates:
            # Deliberate graceful fallback (see test_fallback_template): an
            # unrecognised combination returns the nearest template rather
            # than failing. The template's own header names what it is, so
            # the substitution is visible in the output.
            for fallback_filter in ["kf", "ekf"]:
                if (backend, fallback_filter) in templates:
                    key = (backend, fallback_filter)
                    break
            else:
                key = ("sql", "kf")

        return templates[key](n_targets)

    @staticmethod
    def _template_sql_kf(n_targets: int) -> str:
        return f'''"""v2.0.0 Tracking Pipeline: SQL + Kalman Filter."""
import numpy as np
from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
from pytcl.io import TrackDatabaseManager
from pytcl.io.compat import KalmanTrackAdapter

# System model
dt = 1.0
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
Q = 0.1 * np.eye(4)
R = np.eye(2) * 4.0
P0 = np.diag([R[0, 0], 1.0, R[1, 1], 1.0])

# Create database
db = TrackDatabaseManager("tracking.db")
db.open(mode="w")

# Create adapters for each track
adapters = []
for i in range({n_targets}):
    adapter = KalmanTrackAdapter(db, f"trk_{{i:02d}}", F, H, Q, R)
    adapters.append(adapter)

# Main tracking loop
n_steps = 100
for k in range(n_steps):
    timestamp = k * dt
    for i, adapter in enumerate(adapters):
        if k == 0:
            x0 = np.zeros(4)  # Replace with initial state
            adapter.initialize(x0, P0, timestamp)
        else:
            z = np.zeros(2)  # Replace with actual measurement
            adapter.predict_update(z, timestamp)

# Query results
for i in range({n_targets}):
    history = db.get_track_history(f"trk_{{i:02d}}")
    print(f"Track {{i}}: {{len(history['timestamps'])}} state entries")

db.close()
'''

    @staticmethod
    def _template_sql_ekf(n_targets: int) -> str:
        return f'''"""v2.0.0 Tracking Pipeline: SQL + EKF/UKF."""
import numpy as np
from pytcl.io import TrackDatabaseManager
from pytcl.io.compat import EKFTrackAdapter

# Nonlinear dynamics model
def f(x):
    """State transition function."""
    dt = 1.0
    return np.array([x[0] + x[1] * dt, x[1], x[2] + x[3] * dt, x[3]])

def F_jac(x):
    """Jacobian of f."""
    dt = 1.0
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

def h(x):
    """Measurement function (range-bearing)."""
    r = np.sqrt(x[0] ** 2 + x[2] ** 2)
    theta = np.arctan2(x[2], x[0])
    return np.array([r, theta])

def H_jac(x):
    """Jacobian of h."""
    r = max(np.sqrt(x[0] ** 2 + x[2] ** 2), 1e-10)
    return np.array([
        [x[0] / r, 0, x[2] / r, 0],
        [-x[2] / r ** 2, 0, x[0] / r ** 2, 0],
    ])

Q = 0.1 * np.eye(4)
R = np.diag([1.0, np.radians(2) ** 2])
P0 = np.eye(4) * 10.0

db = TrackDatabaseManager("ekf_tracking.db")
db.open(mode="w")

adapters = []
for i in range({n_targets}):
    adapter = EKFTrackAdapter(db, f"trk_{{i:02d}}", f, F_jac, h, H_jac, Q, R)
    adapters.append(adapter)

# Main loop
for k in range(100):
    for i, adapter in enumerate(adapters):
        if k == 0:
            adapter.initialize(np.array([10, 1, 10, 0.5]), P0, 0.0)
        else:
            adapter.predict(float(k))
            z = np.zeros(2)  # Replace with actual measurement
            adapter.update(z, float(k))

db.close()
'''

    @staticmethod
    def _template_hdf5_kf(n_targets: int) -> str:
        return f'''"""v2.0.0 Tracking Pipeline: HDF5 Archival."""
import numpy as np
from pytcl.io import TrackHDF5Storage

# Store pre-computed tracks to HDF5
store = TrackHDF5Storage("scenario.h5", compression="gzip", compression_level=4)
store.open(mode="w")

tracks = {{}}
for i in range({n_targets}):
    n_steps = 100
    states = np.cumsum(np.random.randn(n_steps, 4) * 0.1, axis=0)
    covs = np.array([np.eye(4) for _ in range(n_steps)])
    times = np.arange(n_steps, dtype=float)
    tracks[f"trk_{{i:02d}}"] = {{
        "states": states,
        "covariances": covs,
        "timestamps": times,
    }}

store.store_tracking_scenario("mission_001", tracks)

# Query archived data
traj = store.get_track_trajectory("trk_00", start_time=10, end_time=50)
print(f"Trajectory slice: {{traj['states'].shape}}")

store.close()
'''

    @staticmethod
    def _template_both_kf(n_targets: int) -> str:
        return f'''"""v2.0.0 Tracking Pipeline: SQL Real-Time + HDF5 Archival."""
import numpy as np
from pytcl.dynamic_estimation.kalman import kf_predict, kf_update
from pytcl.io import TrackDatabaseManager, TrackHDF5Storage
from pytcl.io.compat import KalmanTrackAdapter

# --- Phase 1: Real-time tracking with SQL ---
dt = 1.0
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
Q = 0.1 * np.eye(4)
R = np.eye(2) * 4.0

db = TrackDatabaseManager("realtime.db")
db.open(mode="w")

for i in range({n_targets}):
    adapter = KalmanTrackAdapter(db, f"trk_{{i:02d}}", F, H, Q, R)
    adapter.initialize(np.zeros(4), np.eye(4) * 10, 0.0)
    for k in range(1, 100):
        z = np.zeros(2)  # Replace with measurement
        adapter.predict_update(z, float(k))
    db.confirm_track(f"trk_{{i:02d}}")

# --- Phase 2: Archive to HDF5 ---
store = TrackHDF5Storage("archive.h5")
store.open(mode="w")
store.import_from_sql(db, scenario_id="mission_001")
store.close()
db.close()

print("Real-time tracking + archival complete.")
'''

    @staticmethod
    def generate_migration_checklist() -> str:
        """Generate a migration validation checklist.

        Returns
        -------
        str
            Markdown-formatted checklist.
        """
        return """# v1.x → v2.0.0 Migration Checklist

## Pre-Migration
- [ ] Identify all filter types used (KF, EKF, UKF, IMM, PF, MHT)
- [ ] Catalog existing storage formats (pickle, numpy, csv)
- [ ] Run `MigrationHelper.analyze_v1_code()` on tracking scripts
- [ ] Review recommendations and plan adapter integration

## Data Migration
- [ ] Convert legacy track files using `convert_legacy_tracks_to_sql()`
- [ ] Verify round-trip data integrity (original vs converted)
- [ ] Archive historical data to HDF5 with `convert_legacy_tracks_to_hdf5()`

## Code Migration
- [ ] Replace manual state storage with adapter classes:
  - [ ] KalmanTrackAdapter for KF/linear filters
  - [ ] EKFTrackAdapter for Extended Kalman filters
  - [ ] UKFTrackAdapter for Unscented Kalman filters
  - [ ] IMMTrackAdapter for Interacting Multiple Model
  - [ ] ParticleFilterTrackAdapter for particle filters
  - [ ] TrackerDatabaseAdapter for MultiTargetTracker/MHT
- [ ] Replace pickle/numpy saves with TrackDatabaseManager
- [ ] Add HDF5 archival step for completed missions

## Validation
- [ ] All track states match between v1.x and v2.0.0 (< 1e-10 tolerance)
- [ ] Covariance matrices remain positive definite after round-trip
- [ ] Detection-track associations are preserved
- [ ] Timestamps are monotonically ordered
- [ ] Track lifecycle states (TENTATIVE/CONFIRMED/DEAD) are correct
- [ ] Query performance is acceptable for typical operations (benchmark
      against your own workload -- no fixed latency target is asserted
      here)

## Post-Migration
- [ ] Remove legacy pickle/numpy storage code
- [ ] Update documentation and examples
- [ ] Run full test suite with new storage backend
"""
