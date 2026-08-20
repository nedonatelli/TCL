"""Execute the python code fences in tracked markdown files.

The rst equivalent (``test_docs_code_blocks.py``) has existed since the
v2.0.0 audit, but markdown was never covered -- and that gap is where
``ARCHITECTURE.md``'s graceful-degradation example importing the nonexistent
``pytcl.gravity.egm2008`` lived for months. 26 fences across README,
CONTRIBUTING, the architecture docs and module templates were executable by
nothing.

Same rules as the rst gate: fences on a page run cumulatively in one
subprocess, pages are isolated from each other, and deliberately
non-runnable pages are excluded by name with a written reason.
"""

import pathlib
import re
import subprocess
import sys
import tempfile

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

# Files whose python fences are deliberately not runnable, with the reason.
EXCLUDED = {
    "docs/MODULE_TEMPLATE.md": "template placeholders ({module_name} etc.), not code",
    "docs/modules/kalman_linear.md": "API-reference excerpts that elide setup",
    "docs/architecture/ADR-001-geophysical-caching.md": "design sketch predating the implemented API",
    "docs/architecture/ADR-002-lazy-loading-architecture.md": "before/after design sketches, not current API",
    "docs/architecture/module-interdependencies.md": "illustrative import diagrams",
    "CONTRIBUTING.md": "process snippets (test-writing examples with placeholder names)",
    "docs/architecture/ARCHITECTURE.md": "pattern sketches that deliberately elide setup, same rationale as architecture.rst's exclusion from the rst gate",
    "examples/README.md": "example-structure templates (docstring skeletons), not code",
    "examples/data/README.md": "snippets load downloaded datasets from repo-relative paths",
    "tests/fixtures/ais/SOURCES.md": "provenance notes importing from the test suite, not the library",
}

_FENCE_RE = re.compile(r"^```python\s*$(.*?)^```\s*$", re.M | re.S)

TIMEOUT_S = 300

pytestmark = pytest.mark.examples


def _tracked_markdown():
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    return sorted(p for p in out if not p.startswith("docs/_build"))


PAGES = [p for p in _tracked_markdown() if p not in EXCLUDED]


def extract_fences(md_path: pathlib.Path):
    text = md_path.read_text(encoding="utf-8", errors="replace")
    fences = []
    for match in _FENCE_RE.finditer(text):
        start_line = text[: match.start()].count("\n") + 2
        fences.append((start_line, match.group(1)))
    return fences


def run_page(md_path: pathlib.Path):
    fences = extract_fences(md_path)
    if not fences:
        return None
    parts = ["import matplotlib; matplotlib.use('Agg')"]
    for start, code in fences:
        parts.append(f"print('FENCE {start}', flush=True)\n{code}")
    with tempfile.TemporaryDirectory() as workdir:
        script = pathlib.Path(workdir) / f"{md_path.stem}_fences.py"
        script.write_text("\n".join(parts), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )


def test_exclusions_are_not_stale():
    tracked = set(_tracked_markdown())
    missing = set(EXCLUDED) - tracked
    assert not missing, f"EXCLUDED names files that no longer exist: {missing}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p)
def test_markdown_fences_execute(page):
    result = run_page(REPO_ROOT / page)
    if result is None:
        pytest.skip("no python fences in this file")
    if result.returncode != 0:
        tail = result.stderr[-2000:]
        last = tail.strip().splitlines()[-1] if tail.strip() else ""
        if "ModuleNotFoundError" in last and "'pytcl" not in last:
            pytest.skip(f"optional dependency absent: {last}")
        reached = re.findall(r"^FENCE (\d+)", result.stdout, re.M)
        at = reached[-1] if reached else "?"
        pytest.fail(f"{page}: fence at line {at} failed\nstderr tail:\n{tail}")
