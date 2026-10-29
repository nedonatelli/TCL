"""Load a pytcl module exactly as shipped in v2.11.1, from git history.

Backward-compatibility tests compare against, and write fixtures with, the
released code rather than a hand-written imitation of it.
"""

import functools
import os
import re
import shutil
import subprocess
import types
from pathlib import Path

import pytest

# The v2.11.1 release commit (tag v2.11.1).
V2111_COMMIT = "781ebce5f1be842d39fa743d6aa05b2f443d0046"

_ROOT = Path(__file__).resolve().parents[2]


def _unavailable(reason: str) -> None:
    message = f"v2.11.1 reference code unavailable: {reason}"
    if os.environ.get("CI"):
        pytest.fail(message, pytrace=False)
    pytest.skip(
        f"{message} -- run `git fetch --unshallow` to enable (CI fails instead)"
    )


@functools.lru_cache(maxsize=None)
def _load(relpath: str) -> types.ModuleType:
    if shutil.which("git") is None:
        _unavailable("git executable not found")
    # Asks git rather than testing for _ROOT/.git: mutmut runs the suite from
    # its mutants/ copy, which has no .git of its own but sits inside the
    # checkout, and the .git test failed its clean run there.
    inside = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=_ROOT,
        capture_output=True,
    )
    if inside.returncode != 0:
        _unavailable("not a git checkout")
    have = subprocess.run(
        ["git", "cat-file", "-e", f"{V2111_COMMIT}^{{commit}}"],
        cwd=_ROOT,
        capture_output=True,
    )
    if have.returncode != 0:
        _unavailable(f"commit {V2111_COMMIT[:8]} not in this checkout (shallow clone?)")
    shown = subprocess.run(
        ["git", "show", f"{V2111_COMMIT}:{relpath}"],
        cwd=_ROOT,
        capture_output=True,
    )
    if shown.returncode != 0:
        _unavailable(f"git show failed for {relpath}")
    name = "v2111_" + relpath.removesuffix(".py").replace("/", "_")
    module = types.ModuleType(name)
    module.__file__ = f"{V2111_COMMIT}:{relpath}"
    module.__package__ = relpath.rsplit("/", 1)[0].replace("/", ".")
    # numba cannot cache a function whose source file is a git object name.
    source = re.sub(
        rb"^(@(?:numba\.)?n?jit\(.*?)\bcache=True",
        rb"\1cache=False",
        shown.stdout,
        flags=re.MULTILINE,
    )
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def shipped_module(relpath: str) -> types.ModuleType:
    """Return the v2.11.1 version of ``relpath`` (e.g. ``pytcl/io/x.py``).

    Its imports, relative ones included, resolve to the current package, so
    only the code in ``relpath`` itself is the shipped version.

    Skips when the history is unavailable (unpacked sdist, shallow clone, no
    git); under ``CI`` it fails instead, so CI cannot pass by skipping.
    """
    return _load(relpath)
