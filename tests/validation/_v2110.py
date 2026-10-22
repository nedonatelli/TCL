"""Load a pytcl module exactly as shipped in v2.11.0, from git history.

Backward-compatibility tests compare against, and write fixtures with, the
released code rather than a hand-written imitation of it.
"""

import functools
import subprocess
import types
from pathlib import Path

import pytest

# The v2.11.0 release commit (tag v2.11.0).
V2110_COMMIT = "0d78788cf2eaab8102388b1f5a23094575fb450c"

_ROOT = Path(__file__).resolve().parents[2]


@functools.lru_cache(maxsize=None)
def _load(relpath: str) -> types.ModuleType:
    source = subprocess.run(
        ["git", "show", f"{V2110_COMMIT}:{relpath}"],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    name = "v2110_" + relpath.removesuffix(".py").replace("/", "_")
    module = types.ModuleType(name)
    module.__file__ = f"{V2110_COMMIT}:{relpath}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def shipped_module(relpath: str) -> types.ModuleType:
    """Return the v2.11.0 version of ``relpath`` (e.g. ``pytcl/io/x.py``).

    Skips outside a git checkout (an unpacked sdist); inside one, a missing
    commit is an error, so CI cannot pass these tests by skipping them.
    """
    if not (_ROOT / ".git").exists():
        pytest.skip("needs the git history of a source checkout")
    return _load(relpath)
