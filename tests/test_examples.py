"""Execute every script in examples/ and require it to succeed.

The example scripts are user-facing documentation: they are the first thing
someone runs after installing the package. Until this existed nothing checked
them, and they had drifted -- ``terrain_demo.py`` called ``compute_horizon``
with parameters the function does not accept and swallowed the resulting
TypeError, so it had never once computed a horizon.

Scripts run in a subprocess, so a crash, a stray ``sys.exit``, or a segfault
in one cannot take the test session with it. They are marked ``examples`` and
deselected from the normal test run (``-m "not examples"``); CI runs them in a
dedicated job.
"""

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
GUARD_DIR = pathlib.Path(__file__).resolve().parent / "example_guard"

# examples/data/generate_datasets.py is a fixture generator, not a demo, and is
# run once as a fixture below rather than as a test case of its own.
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.py"))

# Wall-clock ceiling per script. The slowest legitimate example takes about a
# minute; anything past this is a hang, not slowness.
TIMEOUT_S = 300

pytestmark = pytest.mark.examples


def _env():
    env = dict(os.environ)
    env["PYTCL_SHOW_PLOTS"] = "0"  # write HTML instead of opening a browser
    env["MPLBACKEND"] = "Agg"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(GUARD_DIR), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    return env


@pytest.fixture(scope="session", autouse=True)
def sample_datasets():
    """Generate the sample data a few examples read from disk."""
    generator = EXAMPLES_DIR / "data" / "generate_datasets.py"
    if not generator.exists():
        return
    subprocess.run(
        [sys.executable, str(generator)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        timeout=TIMEOUT_S,
        check=True,
    )


def test_examples_are_discovered():
    """A glob that matches nothing would make every other test here vacuous."""
    assert len(EXAMPLES) >= 25, f"only found {len(EXAMPLES)} examples"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script):
    """The script exits 0 with no traceback."""
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )

    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-25:])
        pytest.fail(
            f"{script.name} exited {result.returncode}\n"
            f"--- stderr (last 25 lines) ---\n{tail}"
        )

    # A script can print a traceback and still exit 0 if it catches broadly.
    assert "Traceback (most recent call last)" not in result.stdout, (
        f"{script.name} printed a traceback but exited 0 -- it is swallowing "
        "an exception that should either be handled or allowed to propagate"
    )
