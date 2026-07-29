"""
Structural checks on the tutorial notebooks.

These run in the ordinary (fast) test suite. Full execution happens in the
``notebooks`` CI job via ``pytest --nbval-lax``, which takes about a minute;
the checks here catch the two failure modes that job has actually hit, in
under a second, so they surface on any test run rather than only in CI.

Background: the notebook CI step used to end with
``|| echo "Notebook validation completed with warnings"``, which swallowed the
exit code. The job reported success while 13 cells across two notebooks were
broken -- one importing ``networkx``, which is not a dependency of this
project, and one carrying stale outputs on never-run cells.
"""

import ast
import importlib.util
import json
import pathlib

import pytest

NOTEBOOK_DIR = pathlib.Path(__file__).resolve().parent.parent / "docs" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))


def _code_cells(notebook_path):
    data = json.loads(notebook_path.read_text())
    return [c for c in data["cells"] if c["cell_type"] == "code"]


def _source(cell):
    return "".join(cell["source"])


def test_notebooks_exist():
    """Guard against the glob silently matching nothing."""
    assert NOTEBOOKS, f"no notebooks found under {NOTEBOOK_DIR}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_no_stale_outputs_on_unrun_cells(notebook):
    """A cell with outputs but no execution count is an inconsistent state.

    nbval rejects these with "Unrun reference cell has outputs". It happens
    when a notebook is hand-edited or partially stripped.
    """
    offenders = [
        i
        for i, cell in enumerate(_code_cells(notebook))
        if cell.get("outputs") and cell.get("execution_count") is None
    ]
    assert not offenders, (
        f"{notebook.name}: code cells {offenders} have stored outputs but no "
        "execution_count; clear their outputs or re-run the notebook"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_are_available(notebook):
    """Every module a notebook imports must actually be installable here.

    Catches undeclared dependencies (the `networkx` case) without executing
    the notebook.
    """
    missing = set()
    for cell in _code_cells(notebook):
        source = _source(cell)
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # Cells may contain IPython magics that are not valid Python.
            continue

        # Imports inside a try/except are deliberate optional dependencies
        # (the GPU notebook probes for cupy this way); only unguarded imports
        # are required to resolve.
        guarded = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        guarded.add(id(sub))

        for node in ast.walk(tree):
            if id(node) in guarded:
                continue
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import, not a distribution
                    continue
                names = [(node.module or "").split(".")[0]]
            else:
                continue
            for name in names:
                if name and importlib.util.find_spec(name) is None:
                    missing.add(name)

    assert not missing, (
        f"{notebook.name} imports modules that are not available: "
        f"{sorted(missing)}. Either declare them as dependencies or remove the import."
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_code_cells_parse(notebook):
    """Every code cell must be syntactically valid Python."""
    for i, cell in enumerate(_code_cells(notebook)):
        source = _source(cell)
        if any(line.strip().startswith(("%", "!")) for line in source.splitlines()):
            continue  # IPython magics / shell escapes
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{notebook.name} code cell {i} is not valid Python: {exc}")
