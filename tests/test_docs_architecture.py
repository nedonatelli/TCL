"""The architecture page must describe the library that exists.

Before this existed, ``docs/architecture.rst`` claimed 153 modules organised
into 8 subsystems, named a ``pytcl.geophysical`` package and six other
directories that were never created, and carried code examples with 17 imports
that could not resolve. None of it was checked, so none of it stayed true.

Two things are verified here:

1. Every module and public-name count in the Package Reference table matches
   what the package actually contains.
2. Every ``pytcl`` import appearing in a code block on the page resolves, and
   every name imported from a module exists on it.

The import check runs over the whole of ``docs/``. It started life with an
allowlist: 92 of the 244 pytcl imports in the docs did not resolve, across 16
pages. Those are all fixed, so ``STALE_PAGES`` is empty and every page is
checked. A page listed there that is *not* broken also fails the suite, so the
list cannot quietly drift back into use.

Entries are keyed by path relative to ``docs/``, not by filename: ten basenames
are ambiguous under ``docs/`` (``coordinate_systems.rst`` alone appears four
times), and matching on the basename both skipped pages that should have been
checked and made the result depend on the order ``rglob`` happens to return,
which differs between Linux and macOS.
"""

import importlib
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
ARCHITECTURE = DOCS / "architecture.rst"
PYTCL = REPO_ROOT / "pytcl"

IMPORT = re.compile(
    r"^[ \t]*(?:from[ \t]+([\w.]+)[ \t]+import[ \t]+([\w,\t ]+)|import[ \t]+([\w.]+))[ \t]*$",
    re.M,
)

# Pages whose code examples reference an API that does not exist. This is
# empty: every pytcl import in docs/ resolves. Adding an entry here is a
# regression, not a workaround -- fix the page instead.
STALE_PAGES: set[str] = set()


def _module_count(package):
    d = PYTCL / package
    return len([p for p in d.rglob("*.py") if p.name != "__init__.py"])


def _public_count(package):
    m = importlib.import_module(f"pytcl.{package}")
    names = getattr(m, "__all__", None)
    if names is None:
        names = [n for n in dir(m) if not n.startswith("_")]
    return len(names)


def _table_rows():
    """(package, modules, public) parsed from the Package Reference table."""
    text = ARCHITECTURE.read_text(encoding="utf-8")
    body = text.split("Package Reference", 1)[1].split("Composition Examples", 1)[0]
    rows = []
    for m in re.finditer(
        r"^   \* - ``(\w+)``\n     - (\d+)\n     - (\d+)\n", body, re.M
    ):
        rows.append((m.group(1), int(m.group(2)), int(m.group(3))))
    return rows


def _broken_imports(text):
    broken = []
    for m in IMPORT.finditer(text):
        mod, names, plain = m.groups()
        target = (mod or plain or "").strip()
        if not target.startswith("pytcl"):
            continue
        try:
            module = importlib.import_module(target)
        except Exception as exc:
            broken.append(f"{target} ({type(exc).__name__})")
            continue
        for raw in (names or "").split(","):
            # "from x import y as z" -- the imported name is y
            name = raw.strip().split(" as ")[0].strip()
            if name and not hasattr(module, name):
                broken.append(f"{target}.{name}")
    return broken


def test_table_was_parsed():
    """A parse that silently matched nothing would make the counts vacuous."""
    rows = _table_rows()
    assert len(rows) >= 18, f"parsed only {len(rows)} rows from the table"


@pytest.mark.parametrize("package,modules,public", _table_rows(), ids=lambda v: str(v))
def test_package_counts_match_reality(package, modules, public):
    assert _module_count(package) == modules, (
        f"architecture.rst says {package} has {modules} modules, "
        f"found {_module_count(package)}"
    )
    assert _public_count(package) == public, (
        f"architecture.rst says {package} exports {public} public names, "
        f"found {_public_count(package)}"
    )


def test_documented_packages_are_complete():
    """Every implemented package appears in the table."""
    implemented = {
        d.name
        for d in PYTCL.iterdir()
        if d.is_dir()
        and not d.name.startswith(("_", "."))
        and any(p.name != "__init__.py" for p in d.rglob("*.py"))
    }
    listed = {r[0] for r in _table_rows()}
    assert implemented == listed, (
        f"missing from the table: {sorted(implemented - listed)}; "
        f"listed but not implemented: {sorted(listed - implemented)}"
    )


def test_placeholder_packages_are_named():
    """Empty packages are called out, so they don't look like an omission."""
    empty = {
        d.name
        for d in PYTCL.iterdir()
        if d.is_dir()
        and not d.name.startswith(("_", "."))
        and not any(p.name != "__init__.py" for p in d.rglob("*.py"))
    }
    text = ARCHITECTURE.read_text(encoding="utf-8")
    missing = [p for p in empty if f"``{p}``" not in text]
    assert not missing, f"empty packages not mentioned on the page: {sorted(missing)}"


def test_architecture_page_imports_resolve():
    broken = _broken_imports(ARCHITECTURE.read_text(encoding="utf-8"))
    assert not broken, "architecture.rst has unresolvable imports:\n  " + "\n  ".join(
        broken
    )


@pytest.mark.parametrize(
    "page",
    sorted(
        p
        for p in DOCS.rglob("*.rst")
        if not any(x.startswith("_") for x in p.relative_to(DOCS).parts)
        and p.relative_to(DOCS).as_posix() not in STALE_PAGES
    ),
    ids=lambda p: str(p.relative_to(DOCS)),
)
def test_other_pages_imports_resolve(page):
    """Pages outside STALE_PAGES must keep their examples importable."""
    broken = _broken_imports(page.read_text(encoding="utf-8"))
    assert not broken, f"{page.relative_to(REPO_ROOT)}:\n  " + "\n  ".join(broken)


def _mermaid_blocks(text):
    """The indented body of every ``.. mermaid::`` directive."""
    blocks, cur, inside = [], [], False
    for line in text.splitlines():
        if line.strip() == ".. mermaid::":
            inside, cur = True, []
            continue
        if not inside:
            continue
        if not line.strip():
            cur.append("")
            continue
        if line.startswith("   "):
            cur.append(line[3:])
            continue
        inside = False
        blocks.append("\n".join(cur).strip())
    if inside and cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


MERMAID_TYPES = (
    "graph ",
    "flowchart ",
    "sequenceDiagram",
    "classDiagram",
    "stateDiagram",
)


def test_mermaid_diagrams_are_structurally_sound():
    """Structural checks on the diagrams; mermaid renders them client-side.

    This is not a mermaid parser -- the diagrams were validated against
    mermaid 11 when written -- but it catches the breakage that silently
    produces a blank figure in the browser: an empty directive body, an
    unrecognised diagram type, or unbalanced brackets and quotes in a label.
    """
    blocks = _mermaid_blocks(ARCHITECTURE.read_text(encoding="utf-8"))
    assert len(blocks) >= 4, f"expected at least 4 diagrams, found {len(blocks)}"

    for i, block in enumerate(blocks, 1):
        first = block.splitlines()[0].strip()
        assert first.startswith(MERMAID_TYPES), (
            f"diagram {i} starts with {first!r}, which is not a mermaid "
            f"diagram declaration"
        )
        for name, opener, closer in (("square", "[", "]"), ("round", "(", ")")):
            assert block.count(opener) == block.count(closer), (
                f"diagram {i} has unbalanced {name} brackets: "
                f"{block.count(opener)} {opener} vs {block.count(closer)} {closer}"
            )
        assert block.count('"') % 2 == 0, f"diagram {i} has an odd number of quotes"


def test_stale_page_list_is_accurate():
    """A page on the stale list that is now clean should be taken off it."""
    still_broken = set()
    for rel in STALE_PAGES:
        page = DOCS / rel
        if not page.exists():
            continue
        if _broken_imports(page.read_text(encoding="utf-8")):
            still_broken.add(rel)
    fixed = STALE_PAGES - still_broken
    assert not fixed, (
        f"these pages no longer have broken imports; remove them from "
        f"STALE_PAGES: {sorted(fixed)}"
    )


def test_stale_entries_exist():
    """Guard against a typo silently disabling a check."""
    missing = [rel for rel in STALE_PAGES if not (DOCS / rel).exists()]
    assert not missing, f"STALE_PAGES names files that do not exist: {missing}"
