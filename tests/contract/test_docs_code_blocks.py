"""Execute the python code blocks of every prose documentation page.

The v2.0.0 accuracy audit executed every ``.. code-block:: python`` on every
guide page and found roughly 250 defects: imports that do not resolve,
invented signatures, 2-value unpacks of 6-field NamedTuples, and examples
whose printed output was fabricated. Executing the INS/GNSS tutorial found a
real library bug (``loose_coupled_update`` ignoring position fixes). This
gate keeps all of those pages executable so they cannot drift again.

Blocks on a page run cumulatively, in order, in one process -- later blocks
may use names defined by earlier ones, exactly as a reader following the
page would. Each page runs in a subprocess so one page cannot poison
another. A page whose failure is a missing optional dependency or a missing
external data file skips, per the repo's ``_data_skip`` convention.

Marked ``examples``: deselected from the normal run, executed by the
dedicated CI examples job.
"""

import pathlib
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DOCS = REPO_ROOT / "docs"

# Pages whose python blocks are deliberately not runnable, with the reason.
# Everything else under these globs must execute.
EXCLUDED = {
    "migration_v1_to_v2.rst": "its 'before' snippets demonstrate the v1 API "
    "and are required to raise under v2",
    "architecture.rst": "its blocks deliberately elide setup context; "
    "tests/contract/test_docs_architecture.py executes the page with the "
    "context supplied",
}

PAGE_GLOBS = (
    "*.rst",
    "user_guide/*.rst",
    "tutorials/*.rst",
    "examples/**/*.rst",
)

PAGES = sorted(
    page
    for pattern in PAGE_GLOBS
    for page in DOCS.glob(pattern)
    if page.name not in EXCLUDED
)

# Failure classes that mean "environment lacks an optional piece", not "the
# page is wrong": the repo-wide _data_skip convention, applied to subprocess
# stderr since the blocks run out of process.
SKIP_MARKERS = ("FileNotFoundError", "DependencyError", "ModuleNotFoundError")

TIMEOUT_S = 300

pytestmark = pytest.mark.examples

_BLOCK_RE = re.compile(r"^(\s*)\.\. code-block:: python\s*$")


def extract_blocks(rst_path):
    """Return [(first_line_number, code)] for each python block on a page."""
    lines = rst_path.read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        m = _BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        base_indent = len(m.group(1))
        i += 1
        while i < len(lines) and (
            not lines[i].strip() or re.match(r"^\s+:\w", lines[i])
        ):
            i += 1
        block = []
        indent = None
        start_line = i + 1
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                block.append("")
                i += 1
                continue
            cur = len(line) - len(line.lstrip())
            if cur <= base_indent:
                break
            if indent is None:
                indent = cur
            block.append(line[indent:] if len(line) >= indent else line.strip())
            i += 1
        while block and not block[-1]:
            block.pop()
        if block:
            blocks.append((start_line, "\n".join(block)))
    return blocks


def run_page(rst_path):
    """Execute a page's blocks cumulatively in a subprocess.

    Returns the completed process; the harness script prints the failing
    block's first line number before re-raising, so assertion messages can
    point at the page source.
    """
    blocks = extract_blocks(rst_path)
    if not blocks:
        return None
    parts = ["import matplotlib; matplotlib.use('Agg')"]
    for start, code in blocks:
        parts.append(f"# --- block at line {start}")
        parts.append(f"print('BLOCK {start}', flush=True)\n{code}")
    harness = "\n".join(parts)
    # A real source file, in a scratch working directory: numba's cache=True
    # needs a file-backed function to locate, and any output a block writes
    # relatively lands in the scratch dir instead of the repository (a page
    # block writing tracks.h5 to the repo root is how this was learned).
    with tempfile.TemporaryDirectory() as workdir:
        script = pathlib.Path(workdir) / f"{rst_path.stem}_blocks.py"
        script.write_text(harness, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
        )


def test_the_suite_actually_collects_pages():
    assert len(PAGES) > 40, sorted(p.name for p in PAGES)


def test_the_exclusion_list_is_not_stale():
    all_names = {p.name for pattern in PAGE_GLOBS for p in DOCS.glob(pattern)}
    missing = set(EXCLUDED) - all_names
    assert not missing, f"EXCLUDED entries for pages that no longer exist: {missing}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(DOCS)))
def test_page_blocks_execute(page):
    result = run_page(page)
    if result is None:
        pytest.skip("no python code blocks on this page")
    if result.returncode != 0:
        tail = result.stderr[-2500:]
        last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
        if any(marker in last_line for marker in SKIP_MARKERS):
            pytest.skip(f"optional dependency or data file absent: {last_line}")
        blocks_reached = re.findall(r"^BLOCK (\d+)", result.stdout, re.MULTILINE)
        at = blocks_reached[-1] if blocks_reached else "?"
        pytest.fail(
            f"{page.relative_to(DOCS)}: block at line {at} failed\nstderr tail:\n{tail}"
        )


class TestTheGateActuallyFires:
    """Negative controls: a gate that cannot fail verifies nothing."""

    def _page(self, tmp_path, body):
        p = tmp_path / "synthetic.rst"
        p.write_text(textwrap.dedent(body), encoding="utf-8")
        return p

    def test_a_raising_block_fails(self, tmp_path):
        page = self._page(
            tmp_path,
            """
            Title
            =====

            .. code-block:: python

               raise RuntimeError("the gate must see this")
            """,
        )
        result = run_page(page)
        assert result.returncode != 0
        assert "the gate must see this" in result.stderr

    def test_blocks_run_cumulatively_and_in_order(self, tmp_path):
        page = self._page(
            tmp_path,
            """
            Title
            =====

            .. code-block:: python

               shared = 21

            .. code-block:: python

               print("RESULT", shared * 2)
            """,
        )
        result = run_page(page)
        assert result.returncode == 0, result.stderr
        assert "RESULT 42" in result.stdout

    def test_a_verbatim_audit_defect_fails(self, tmp_path):
        """The 2-value unpack of kf_update that three pages shipped."""
        page = self._page(
            tmp_path,
            """
            Title
            =====

            .. code-block:: python

               import numpy as np
               from pytcl.dynamic_estimation.kalman import kf_update

               x, P = kf_update(
                   np.zeros(2), np.eye(2), np.zeros(1),
                   np.array([[1.0, 0.0]]), np.eye(1),
               )
            """,
        )
        result = run_page(page)
        assert result.returncode != 0
        assert "too many values to unpack" in result.stderr

    def test_extraction_sees_nested_blocks(self, tmp_path):
        page = self._page(
            tmp_path,
            """
            Title
            =====

            .. note::

               .. code-block:: python

                  print("RESULT nested")
            """,
        )
        result = run_page(page)
        assert result is not None, "extractor missed an indented block"
        assert "RESULT nested" in result.stdout

    def test_a_blockless_page_is_reported_as_such(self, tmp_path):
        page = self._page(tmp_path, "Title\n=====\n\nProse only.\n")
        assert run_page(page) is None
