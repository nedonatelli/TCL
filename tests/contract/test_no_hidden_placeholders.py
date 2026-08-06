"""Exported code may not admit incompleteness only where callers cannot see it.

``pytcl.atmosphere.NRLMSISE00`` was exported, its class docstring called it "a
comprehensive thermosphere model", and it returned a sea-level density 44%
below truth. The source said why, in a comment:

    # These are placeholder structures that would be populated from data files

Nothing connected that admission to the confident docstring above it, and no
validation test measured the gap, so a caller had no way to tell. It was
renamed to ``SimplifiedThermosphere`` and its error measured and pinned in
``tests/validation/test_thermosphere_limits.py`` (gh-79).

The rule here: **a module that concedes in its source that its values are not
real may not export names unless a validation test measures the gap.** Fix a
failure by removing a stale admission, by not exporting the name, or by
measuring what the shortfall costs and pinning it.

Note what is deliberately *not* flagged. A docstring saying "this uses a
simplified construction" is doing its job -- the caller reads it. A sentinel
``# Placeholder`` beside ``append(-1)`` is about a variable, not the
implementation. A first version of this test matched those and reported five
modules that hide nothing; the pattern below is narrow because the failure is
specific: disclosure the caller never sees.

The codebase currently matches nothing, so the controls at the bottom carry
the weight. Without them this file would pass whether or not it worked.
"""

import ast
import importlib
import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

ADMISSION = re.compile(
    r"placeholder (?:structure|value|data|table|coefficient)s?"
    r"|would be populated"
    r"|not the full (?:model|implementation|algorithm)"
    r"|in a (?:real|full|complete|production) implementation"
    r"|requires? (?:the )?(?:full|extensive|official) .{0,30}(?:table|data|coefficient)"
    r"|(?:returns?|gives?) (?:a )?(?:dummy|fake|arbitrary) ",
    re.I,
)

# Modules allowed to concede, each because a validation test measures the gap.
# Empty today: the one entry that would have belonged here,
# pytcl/atmosphere/thermosphere.py, was rewritten in gh-79 and no longer
# concedes anything. New entries must name the test that justifies them.
DISCLOSED: "dict[str, str]" = {}


def _validation_names():
    out = subprocess.run(
        [
            "grep",
            "-rhoE",
            r"[A-Za-z_][A-Za-z0-9_]*",
            str(ROOT / "tests" / "validation"),
        ],
        capture_output=True,
        text=True,
    )
    return set(out.stdout.split())


def _exported(package):
    try:
        module = importlib.import_module(package)
    except Exception:
        return set()
    return {n for n in dir(module) if not n.startswith("_")}


def offending_names(source, exported):
    """Exported names defined in a source file that concedes incompleteness.

    Split out so the controls below can drive it with synthetic input rather
    than needing a real offender in the tree.
    """
    if not ADMISSION.search(source):
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
        and node.name in exported
    ]


def _conceding_modules():
    found = []
    for path in sorted((ROOT / "pytcl").rglob("*.py")):
        if "__pycache__" in str(path) or path.name.startswith("_"):
            continue
        package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
        names = offending_names(
            path.read_text(encoding="utf-8", errors="ignore"), _exported(package)
        )
        if names:
            found.append((str(path.relative_to(ROOT)), names))
    return found


_VALIDATED = _validation_names()


def test_the_scan_reads_the_source():
    """Guard the guard: a scan that walked nothing would pass silently."""
    assert sum(1 for _ in (ROOT / "pytcl").rglob("*.py")) > 100


def test_no_exported_name_hides_an_undisclosed_placeholder():
    offenders = []
    for module, names in _conceding_modules():
        if module in DISCLOSED:
            continue
        untested = [n for n in names if n not in _VALIDATED]
        if untested:
            offenders.append(f"{module}: {', '.join(sorted(untested))}")

    assert not offenders, (
        "These modules concede in their source that their values are not real, "
        "and export names no validation test measures:\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nRemove the admission if it is stale, stop exporting the name, or "
        "measure the gap in tests/validation/ and add the module to DISCLOSED."
    )


@pytest.mark.parametrize("module", sorted(DISCLOSED))
def test_disclosed_entries_are_not_stale(module):
    """An allowlist entry for a module that no longer concedes would silently
    permit a future placeholder in the same file."""
    path = ROOT / module
    assert path.exists(), f"{module} is gone; remove it from DISCLOSED"
    assert ADMISSION.search(path.read_text(encoding="utf-8")), (
        f"{module} no longer concedes anything; remove it from DISCLOSED."
    )


class TestTheRuleActuallyFires:
    """The codebase matches nothing, so these carry the weight.

    Every assertion above is satisfied by a detector that never fires. These
    are not.
    """

    def test_the_historical_case_is_caught(self):
        """Verbatim from pytcl/atmosphere/nrlmsise00.py before gh-79."""
        source = (
            '"""A comprehensive thermosphere model."""\n'
            "# NRLMSISE-00 Coefficients (simplified structure)\n"
            "# Note: Full model requires extensive coefficient tables from NOAA\n"
            "# These are placeholder structures that would be populated from"
            " data files\n"
            "class NRLMSISE00:\n"
            "    pass\n"
        )
        assert offending_names(source, {"NRLMSISE00"}) == ["NRLMSISE00"]

    def test_a_sentinel_comment_is_not_flagged(self):
        """``# Placeholder`` beside ``append(-1)`` is about a variable."""
        source = (
            "class KDTree:\n"
            "    def build(self):\n"
            "        self._left.append(-1)  # Placeholder\n"
        )
        assert offending_names(source, {"KDTree"}) == []

    def test_an_honest_docstring_is_not_flagged(self):
        """Disclosure a caller reads is the fix, not the defect."""
        source = (
            "class CoverTree:\n"
            '    """Uses a simplified version of the original construction."""\n'
        )
        assert offending_names(source, {"CoverTree"}) == []

    def test_an_unexported_name_is_not_flagged(self):
        """Internal code may be as incomplete as it likes."""
        source = (
            "# these are placeholder values that would be populated later\n"
            "class Helper:\n    pass\n"
        )
        assert offending_names(source, set()) == []

    def test_a_private_name_is_not_flagged(self):
        source = "# returns a dummy value for now\ndef _private():\n    pass\n"
        assert offending_names(source, {"_private"}) == []
