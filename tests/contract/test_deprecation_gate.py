"""pytcl's own deprecations fail the test suite; third-party ones do not.

The v2.11 gate was ``error::DeprecationWarning:pytcl.*`` followed by
``default::DeprecationWarning``. It never fired, for two independent reasons:
the module field matches the frame the warning is attributed to, and every
pytcl site warns with ``stacklevel>=2``, so the caller (a test module) was
matched, never pytcl; and pytest applies the last matching filter, so the
``default`` rule listed after it overrode it anyway.

The gate now keys on a category instead of a module. These tests run the
repo's own ``filterwarnings`` list, read from ``pyproject.toml``, in a
sub-session, so a reordering or a rewrite of the config is caught here.
"""

import ast
import pathlib
import tomllib

pytest_plugins = ["pytester"]

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTCL_ROOT = ROOT / "pytcl"


def _repo_filterwarnings() -> list[str]:
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["tool"]["pytest"]["ini_options"]["filterwarnings"]


def _filterwarnings_toml() -> str:
    entries = "".join(f"    {rule!r},\n" for rule in _repo_filterwarnings())
    return f"[tool.pytest.ini_options]\nfilterwarnings = [\n{entries}]\n"


_FILTERWARNINGS_FROM_REPO_PYPROJECT = _filterwarnings_toml()


def _category_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _warn_calls_with_category(root: pathlib.Path, category: str) -> list[str]:
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not (isinstance(func, ast.Attribute) and func.attr == "warn"):
                continue
            if not (isinstance(func.value, ast.Name) and func.value.id == "warnings"):
                continue
            cats = [kw.value for kw in node.keywords if kw.arg == "category"]
            if len(node.args) >= 2:
                cats.append(node.args[1])
            if any(_category_name(c) == category for c in cats):
                offenders.append(f"{path.relative_to(root.parent)}:{node.lineno}")
    return offenders


def test_every_pytcl_deprecation_uses_the_gated_category():
    """AST scan: warnings.warn(..., DeprecationWarning) anywhere in pytcl/ fails."""
    offenders = _warn_calls_with_category(PYTCL_ROOT, "DeprecationWarning")
    assert offenders == []


def test_scan_finds_a_plain_deprecation(tmp_path):
    (tmp_path / "m.py").write_text(
        "import warnings\n"
        "warnings.warn('a', DeprecationWarning, stacklevel=2)\n"
        "warnings.warn('b', category=DeprecationWarning)\n"
        "warnings.warn('c', PytclDeprecationWarning)\n"
    )
    found = _warn_calls_with_category(tmp_path, "DeprecationWarning")
    assert [f.rsplit(":", 1)[1] for f in found] == ["2", "3"]


def test_gate_turns_a_pytcl_deprecation_into_an_error(pytester):
    pytester.makepyprojecttoml(_FILTERWARNINGS_FROM_REPO_PYPROJECT)
    pytester.makepyfile(
        "from pytcl.atmosphere.thermosphere import SimplifiedThermosphere\n"
        "def test_x():\n    SimplifiedThermosphere()\n"
    )
    result = pytester.runpytest("-p", "no:randomly")
    result.assert_outcomes(failed=1)


def test_gate_leaves_third_party_deprecations_visible_but_passing(pytester):
    pytester.makepyprojecttoml(_FILTERWARNINGS_FROM_REPO_PYPROJECT)
    pytester.makepyfile(
        "import warnings\ndef test_x():\n    warnings.warn('x', DeprecationWarning)\n"
    )
    pytester.runpytest("-p", "no:randomly").assert_outcomes(passed=1, warnings=1)
