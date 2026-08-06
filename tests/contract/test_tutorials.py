"""Execute every tutorial script in docs/tutorials/ and require it to succeed.

The tutorials shipped for months with no gate at all: nothing ran them, and
they had drifted badly enough that ``particle_filters.py`` crashed under
NumPy 2 and its "systematic resampling" was a weight-ignoring random
permutation. Two defects were invisible even to a successful exit code: the
scripts wrote their HTML output to a cwd-relative ``../_static`` -- outside
the repository when run from the root -- and printed characters outside
cp1252, which crashes on Windows redirected stdout.

Scripts run in a subprocess, marked ``examples`` and deselected from the
normal run; CI executes them in the dedicated examples job. The environment
pins ``PYTHONIOENCODING=cp1252`` so the Windows console failure mode fails
here first.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TUTORIALS_DIR = REPO_ROOT / "docs" / "tutorials"
TUTORIALS = sorted(TUTORIALS_DIR.glob("*.py"))

# The regression this guards: output written via a cwd-relative path landed
# in the repository's parent directory.
ESCAPED_OUTPUT = REPO_ROOT.parent / "_static"

TIMEOUT_S = 300

pytestmark = pytest.mark.examples


def _env():
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "cp1252"
    return env


def test_the_suite_actually_collects_tutorials():
    """An empty glob must fail loudly, not pass by testing nothing."""
    assert len(TUTORIALS) >= 10, sorted(p.name for p in TUTORIALS)


@pytest.mark.parametrize("script", TUTORIALS, ids=lambda p: p.name)
def test_tutorial_runs_clean(script):
    assert not ESCAPED_OUTPUT.exists(), (
        f"{ESCAPED_OUTPUT} exists before the run; remove it so escape "
        "detection means something"
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )

    assert result.returncode == 0, (
        f"{script.name} exited {result.returncode}\n"
        f"stdout tail:\n{result.stdout[-2000:]}\n"
        f"stderr tail:\n{result.stderr[-2000:]}"
    )
    assert not ESCAPED_OUTPUT.exists(), (
        f"{script.name} wrote output outside the repository "
        f"({ESCAPED_OUTPUT}); its OUTPUT_DIR must be anchored to __file__"
    )
