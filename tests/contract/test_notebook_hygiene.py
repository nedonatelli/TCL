"""
Structural checks on the tutorial notebooks.

These run in the ordinary (fast) test suite. Full execution happens in the
``notebooks`` CI job via ``pytest --nbval-lax``, which takes about a minute;
the checks here catch the failure modes that job has actually hit, in under a
second, so they surface on any test run rather than only in CI.

Background: the notebook CI step used to end with
``|| echo "Notebook validation completed with warnings"``, which swallowed the
exit code. Removing it exposed three problems the job had been reporting as
success -- an import of ``networkx``, which this project does not depend on
and the notebook never used; stale outputs on never-run cells, which nbval
rejects; and a job that installed only the ``[dev]`` extra while every
notebook plots with plotly from ``[visualization]``.
"""

import ast
import json
import pathlib
import re
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 predates tomllib
    tomllib = None

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
NOTEBOOK_DIR = REPO_ROOT / "docs" / "notebooks"
NOTEBOOKS = sorted(NOTEBOOK_DIR.glob("*.ipynb"))

# Distributions whose import name differs from their name on PyPI.
IMPORT_NAME_OVERRIDES = {
    "pywavelets": "pywt",
    "scikit-learn": "sklearn",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "nrl-tracker": "pytcl",
}


def _declared_import_names():
    """Import names for every distribution pyproject declares, plus stdlib."""
    pyproject = tomllib.loads(
        (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    project = pyproject["project"]

    requirements = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements.extend(extra)

    names = set(sys.stdlib_module_names) | {"pytcl"}
    for requirement in requirements:
        # Strip extras, version specifiers, markers: "numpy[foo]>=1.2 ; py>'3'"
        dist = re.split(r"[\[<>=!~;\s]", requirement, maxsplit=1)[0].strip().lower()
        if not dist:
            continue
        names.add(IMPORT_NAME_OVERRIDES.get(dist, dist.replace("-", "_")))
    return names


def _code_cells(notebook_path):
    # Explicit encoding: notebooks are UTF-8 by spec, but read_text() defaults
    # to the locale encoding, which is cp1252 on the Windows CI runners.
    data = json.loads(notebook_path.read_text(encoding="utf-8"))
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


@pytest.mark.skipif(tomllib is None, reason="tomllib requires Python 3.11+")
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_imports_are_declared_dependencies(notebook):
    """Every module a notebook imports must be a declared dependency.

    Checked against pyproject rather than against whatever happens to be
    installed, because the jobs that run this test do not all install the
    same extras -- the ``test`` job has no plotly, but the notebooks
    legitimately use it. Asserting on the ambient environment would fail
    there for the wrong reason.
    """
    declared = _declared_import_names()
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
                if name and name not in declared:
                    missing.add(name)

    assert not missing, (
        f"{notebook.name} imports {sorted(missing)}, which "
        "pyproject.toml does not declare as dependencies. Either add them to "
        "the appropriate extra or remove the import."
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
