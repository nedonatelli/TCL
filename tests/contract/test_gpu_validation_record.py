"""The GPU layer's hardware-validation record must not go stale.

The CuPy layer can only be validated on real NVIDIA hardware (no CI
runner has one), so validation is a manual gpu.yml run whose result is
recorded in pytcl/gpu/VALIDATION.json. This gate fails when code
commits touch pytcl/gpu/ after the recorded validated_through_commit:
silently shipping GPU changes that no hardware has executed is the
vacuous-gate failure mode the v2.11.0 audit flagged. Like the CuPy
test gates, it enforces only when PYTCL_REQUIRE_CUPY=1 (set on the
GPU box); elsewhere it reports staleness as a skip so the build stays
green on machines that cannot validate.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "pytcl" / "gpu" / "VALIDATION.json"


def _commits_after(validated_commit: str) -> list:
    out = subprocess.run(
        [
            "git",
            "log",
            "--format=%h %s",
            f"{validated_commit}..HEAD",
            "--",
            "pytcl/gpu/",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if out.returncode != 0:
        pytest.skip("git history unavailable (sdist or shallow clone)")
    return [line for line in out.stdout.splitlines() if line.strip()]


def test_validation_record_is_current():
    record = json.loads(RECORD.read_text())
    for key in ("validated_date", "hardware", "validated_through_commit"):
        assert record.get(key), f"VALIDATION.json missing {key}"

    newer = _commits_after(record["validated_through_commit"])
    if not newer:
        return
    message = (
        "pytcl/gpu/ has commits after the recorded hardware validation "
        f"({record['validated_date']}, through "
        f"{record['validated_through_commit']}):\n  "
        + "\n  ".join(newer)
        + "\nRun the gpu workflow on real hardware and update "
        "pytcl/gpu/VALIDATION.json."
    )
    if os.environ.get("PYTCL_REQUIRE_CUPY") == "1":
        pytest.fail(message)
    pytest.skip("UNVALIDATED GPU CHANGES: " + message)
