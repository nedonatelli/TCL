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

The record file itself is excluded from the watched path: recording a
validation commits VALIDATION.json, and counting that commit made every
record stale the moment it was written (446aa51 did exactly that).
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
RECORD = REPO / "pytcl" / "gpu" / "VALIDATION.json"


def _commits_after(validated_commit: str, head: str = "HEAD") -> list[str]:
    out = subprocess.run(
        [
            "git",
            "log",
            "--format=%H %s",
            f"{validated_commit}..{head}",
            "--",
            "pytcl/gpu/",
            ":(exclude)pytcl/gpu/VALIDATION.json",
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


def test_record_file_itself_is_not_a_gpu_change():
    # Full SHAs: an abbreviated %h grows past 7 characters as the object
    # count grows, which would break a literal comparison.
    assert _commits_after(
        "e919189f4b0b9c545f6e9639fb6d6372ed9e85b4",
        head="781ebce5f1be842d39fa743d6aa05b2f443d0046",
    ) == [
        "04007f18beb4a60e405aeee51f2e21f4305148d8 "
        "fix: io and gpu correctness (v2.11.1 Tier 1, PR 6 of 8) (#203)",
        "df1aa1722f231b30ca35aacec223d96bf960616b "
        "fix: estimation-stack correctness (v2.11.1 Tier 1, PR 4 of 8) (#196)",
    ]
