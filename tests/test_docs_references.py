"""The documentation must point at the real example scripts.

``docs/examples/`` once held its own copy of all 34 example scripts alongside
the ``.rst`` pages. Nothing referenced them -- every ``literalinclude`` and
``:download:`` in the docs already resolved to ``examples/`` at the repository
root -- so the copies simply drifted. By the time they were removed,
``docs/examples/terrain_demo.py`` differed from ``examples/terrain_demo.py`` by
238 diff lines, and none of the bug fixes applied to the canonical scripts had
ever reached them.

Two things are checked here:

1. ``docs/`` contains no second copy of an example script.
2. Every file the docs include or offer for download actually exists, so a
   renamed or moved script cannot silently leave a dead link behind. Sphinx
   emits only a warning for these, and the docs CI gate fails on ERROR.
"""

import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
EXAMPLES = REPO_ROOT / "examples"

# literalinclude:: path | :download:`text <path>` | include:: path
REFERENCE = re.compile(r"(?:literalinclude::|include::|:download:`[^<]*<)\s*([^\s>`]+)")

# docs/tutorials/ is a separate deliverable, not a copy of examples/: the
# README and docs/TUTORIALS.md document it as ten standalone tutorial modules,
# and pyproject.toml carries per-file lint rules for it.
ALLOWED_SCRIPT_DIRS = {DOCS / "tutorials"}


def _rst_files():
    return sorted(DOCS.rglob("*.rst"))


def test_docs_have_rst_files():
    """A glob that matched nothing would make the rest of this vacuous."""
    assert len(_rst_files()) > 10


def test_no_duplicate_example_scripts_under_docs():
    """Example scripts live in examples/ only."""
    example_names = {p.name for p in EXAMPLES.glob("*.py")}

    duplicates = []
    for script in DOCS.rglob("*.py"):
        # Skip Sphinx's own directories: _build holds generated copies of every
        # :download: target, which are output, not source.
        if any(part.startswith("_") for part in script.relative_to(DOCS).parts):
            continue
        if any(script.is_relative_to(d) for d in ALLOWED_SCRIPT_DIRS):
            continue
        if script.name in example_names or script.parent.name == "examples":
            duplicates.append(script.relative_to(REPO_ROOT))

    assert not duplicates, (
        "these duplicate the canonical scripts in examples/ and will drift "
        f"out of sync: {sorted(map(str, duplicates))}. The docs already "
        "reference examples/ directly via literalinclude."
    )


@pytest.mark.parametrize("rst", _rst_files(), ids=lambda p: str(p.relative_to(DOCS)))
def test_included_files_exist(rst):
    """No literalinclude or :download: may point at a missing file."""
    missing = []
    for match in REFERENCE.finditer(rst.read_text(encoding="utf-8")):
        target = (rst.parent / match.group(1)).resolve()
        if not target.exists():
            missing.append(match.group(1))

    assert not missing, (
        f"{rst.relative_to(REPO_ROOT)} references files that do not exist: {missing}"
    )
