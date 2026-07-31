"""Every exported function must be exercised by at least one test.

`associated_legendre_scaled` was exported from the gravity package and had zero
callers. It scaled along the wrong axis: the addition-theorem invariant it must
satisfy was off by a factor of 1e199 at degree 2000, and it blocked EGM2008 the
whole time it shipped. The suite was green throughout. It was found by hand
during the v2 audit, not by any gate.

This is that gate. It walks the package, classifies everything each module
publishes, and requires every published *function* to be reached by at least one
test.

**What counts as published.** ``__all__`` when a module declares one; otherwise
every non-underscore name the module defines itself. The second half was added
in gh-53, after the gate reported 98.6% coverage while twenty-two modules --
including `core.array_utils`, `transforms.fourier` and `astronomical.special_orbits`
-- declared no ``__all__`` and so contributed nothing to the denominator. A
percentage measured over the modules that opted in is not a percentage of the
public API. Widening the walk added 21 functions to the count after
identity-deduplication, of which all but one were already covered; the one that
was not is a min-cost-flow solver in a module slated for deletion.

Three decisions, from the spec in gh-47:

**Test reference, not internal caller.** For a library most public API
legitimately has no caller inside the library -- it exists to be called by
users. Gating on internal callers would flag most of the exported functions and
be discarded as noise. The `associated_legendre_scaled` failure is caught by the
test-reference rule and would not have been caught by a caller rule.

**Not line coverage.** A symbol can show covered lines because an unrelated
caller passed through the same file. Requiring that a test actually reach the
export is what makes the check mean something.

**Functions only.** Most exported classes are result types -- a caller receives
a ``CRBResult`` without ever writing the name -- so gating on them would produce
noise that gets suppressed rather than fixed. The classification is in place so
that a narrower class rule can be added later.

**Reached, not executed.** This is the gate's honest limit. Coverage here means
a test *references* the export, read from the AST -- not that the reference ran
and asserted something. The two diverge for data-gated tests: a test of a loader
whose multi-gigabyte input CI does not have marks that loader covered while
skipping in every CI run. `load_egm_coefficients` and `load_emm_coefficients`
left this allowlist that way in gh-51, and their invariant is enforced on a
developer machine with the data files, not in CI. Where that matters, the fix is
a companion test of the mechanism that runs everywhere -- as `TestMakeReadonly`
does for gh-51 -- rather than a change to this gate. Reading a falling allowlist
count as "that much more is verified" overstates it by however many entries are
data-gated.

Coverage is resolved by **object identity**, not by name. Ten exported function
names bind to two different implementations (``factorial`` exists in both
``combinatorics`` and ``special_functions``; ``get_cache_info`` in both
``reference_frames`` and ``great_circle``), so a name-keyed check lets one test
mark both covered and lets one allowlist entry exempt both. Imports in the test
suite are read from the AST and resolved to the object they actually bind, which
also means a name appearing only in a comment, a docstring or a string literal
does not count as coverage.
"""

import ast
import importlib
import inspect
import pathlib
import pkgutil
import typing

import pytest

import pytcl

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

# Exported functions no test reaches, keyed by "<defining module>:<name>" and
# carrying the reason each is not covered. The key is module-qualified because
# ten exported function names bind to two distinct implementations; a bare name
# would exempt both.
#
# **This is empty, and it should stay empty.** It began at 27 entries when the
# gate landed in gh-47 and reached zero in gh-24. Getting there took four
# changes: a suite for the cache-control surface, reference tests for the
# geomagnetic coefficient factories and six singletons, a linear-programming
# oracle for the min-cost-flow solver, and finally deleting two dead modules
# rather than writing tests to legitimize them.
#
# Three rules keep it honest: an entry that is now covered fails, an entry
# naming a symbol that no longer exists fails, and an entry that is not a
# function fails.
#
# Do not add an entry to silence a new export. Write the test.
UNCOVERED: dict[str, str] = {}

# A traversal that silently found nothing would make every assertion here pass.
# Raised from 1000 when gh-53 widened the walk to modules without ``__all__``;
# the figure is a floor against a broken traversal, so it wants to sit just
# under the real count rather than far below it.
MINIMUM_EXPECTED_EXPORTS = 1300
# Modules that fail to import contribute no exports. A handful is expected when
# optional extras are absent; a large number means the walk itself is broken.
MAXIMUM_UNIMPORTABLE_MODULES = 12


def _classify(obj: object) -> str:
    """function / class / module / constant.

    ``inspect.isfunction`` is deliberately not the test: it is False for
    ``lru_cache`` wrappers and numba dispatchers, which would drop six exported
    callables -- five of them in ``pytcl.gpu`` -- out of the gate entirely.
    """
    if inspect.isclass(obj):
        return "class"
    if inspect.ismodule(obj):
        return "module"
    # typing constructs report callable() -- Literal["a", "b"] does -- but they
    # are aliases, not entry points, and gating them is meaningless.
    if getattr(obj, "__module__", None) == "typing" or typing.get_origin(obj):
        return "constant"
    if callable(obj) and not isinstance(obj, type):
        return "function"
    return "constant"


class Export:
    """One exported object, and every path that reaches it."""

    def __init__(self, name: str, obj: object, module: str) -> None:
        self.name = name
        self.kind = _classify(obj)
        self.defining_module = getattr(obj, "__module__", module) or module
        self.reachable_from: set[str] = {module}

    @property
    def key(self) -> str:
        return f"{self.defining_module}:{self.name}"


def _published_names(module: object, module_name: str) -> list[str]:
    """What a module publishes.

    ``__all__`` when the module declares one. When it does not, every
    non-underscore name it *defines itself* -- because that is what
    ``from module import *`` gives a caller, and what Sphinx documents.

    Twenty-two modules declare no ``__all__``, and reading them as publishing
    nothing was how the gate came to miss 116 public functions (gh-53). A
    module that omits ``__all__`` has not opted out of being public; it has
    only declined to be explicit, and the conservative reading of that is to
    treat everything visible as public.

    ``__module__`` filters out names the module imported rather than defined,
    so a helper re-exported from elsewhere is attributed to the module that
    owns it and is not counted twice.
    """
    declared = getattr(module, "__all__", None)
    if declared is not None:
        return list(declared)
    return [
        name
        for name in dir(module)
        if not name.startswith("_")
        and getattr(getattr(module, name, None), "__module__", None) == module_name
    ]


def _collect_exports() -> tuple[dict[int, Export], int]:
    """Everything the package publishes, keyed by object identity.

    Identity rather than name: ten exported function names bind to two
    different implementations, and a name-keyed map silently merges them.

    A module that cannot be imported because an optional dependency is absent
    contributes nothing rather than aborting, so the verdict does not depend on
    which extras happen to be installed. The count of such modules is returned
    so that a broken walk cannot hide behind that tolerance.
    """
    exports: dict[int, Export] = {}
    unimportable = 0
    for info in pkgutil.walk_packages(pytcl.__path__, "pytcl."):
        try:
            module = importlib.import_module(info.name)
        except (ImportError, AttributeError, OSError) as exc:  # missing extra
            del exc
            unimportable += 1
            continue
        for name in _published_names(module, info.name):
            obj = getattr(module, name, None)
            if obj is None:
                continue
            existing = exports.get(id(obj))
            if existing is None:
                exports[id(obj)] = Export(name, obj, info.name)
            else:
                existing.reachable_from.add(info.name)
    return exports, unimportable


def _test_bindings() -> tuple[set[tuple[str, str]], set[str]]:
    """What the test suite imports, read from the AST.

    Returns the set of ``(module, name)`` pairs bound by ``from x import y``,
    and the set of bare identifiers used in code. Parsing rather than scanning
    text means a name in a comment, a docstring or a string literal is not
    mistaken for coverage.
    """
    qualified: set[tuple[str, str]] = set()
    bare: set[str] = set()
    this_file = pathlib.Path(__file__).name
    for path in TESTS_DIR.rglob("*.py"):
        if path.name == this_file:
            continue  # this file names exports only in order to exempt them
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                for alias in node.names:
                    qualified.add((node.module, alias.name))
                    bare.add(alias.name)
            elif isinstance(node, ast.Name):
                bare.add(node.id)
            elif isinstance(node, ast.Attribute):
                bare.add(node.attr)
    return qualified, bare


EXPORTS, UNIMPORTABLE = _collect_exports()
QUALIFIED, BARE = _test_bindings()

# A name that identifies exactly one export can be matched bare; an ambiguous
# one has to be reached through the module it was imported from.
_NAME_COUNTS: dict[str, int] = {}
for _export in EXPORTS.values():
    _NAME_COUNTS[_export.name] = _NAME_COUNTS.get(_export.name, 0) + 1


def _is_covered(export: Export) -> bool:
    # The defining module counts alongside the re-exporting packages. Several
    # modules here define public symbols without declaring __all__ themselves
    # (pytcl.core.validation is one), so the walk never records them as a
    # reachable path even though importing from them is perfectly ordinary.
    paths = export.reachable_from | {export.defining_module}
    if any((module, export.name) in QUALIFIED for module in paths):
        return True
    # unambiguous names may be matched without resolving the import
    return _NAME_COUNTS[export.name] == 1 and export.name in BARE


def test_traversal_found_the_package():
    """Guard against a walk that silently collects nothing."""
    assert len(EXPORTS) >= MINIMUM_EXPECTED_EXPORTS, (
        f"only discovered {len(EXPORTS)} exports; the package walk is broken, "
        "which would make every other assertion in this file vacuous"
    )


def test_unimportable_modules_are_the_exception():
    """Tolerating a missing extra must not tolerate a broken traversal."""
    assert UNIMPORTABLE <= MAXIMUM_UNIMPORTABLE_MODULES, (
        f"{UNIMPORTABLE} modules failed to import. A few is expected when "
        "optional extras are absent; this many means the walk is broken, and "
        "every export in those modules is escaping the gate."
    )


def test_every_exported_function_is_reached_by_a_test():
    """The gate. An exported function with no test is a defect, not a gap."""
    missing = sorted(
        export.key
        for export in EXPORTS.values()
        if export.kind == "function"
        and not _is_covered(export)
        and export.key not in UNCOVERED
    )
    assert not missing, (
        f"{len(missing)} exported function(s) are not reached by any test:\n"
        + "\n".join(f"    {key}" for key in missing)
        + "\n\nWrite a test that calls it, or -- only if it genuinely cannot "
        "be tested -- add it to UNCOVERED in this file, keyed exactly as "
        "shown above, with the reason. The allowlist is expected to shrink."
    )


def test_allowlist_has_no_stale_entries():
    """An entry that is now covered must be removed, so the list shrinks."""
    by_key = {export.key: export for export in EXPORTS.values()}
    now_covered = sorted(
        f"{key}  (was: {reason})"
        for key, reason in UNCOVERED.items()
        if key in by_key and _is_covered(by_key[key])
    )
    assert not now_covered, (
        "these are now reached by a test; remove them from UNCOVERED:\n"
        + "\n".join(f"    {entry}" for entry in now_covered)
    )


def test_allowlist_names_real_exports():
    """A typo, or a symbol since removed, must not sit here silently."""
    known = {export.key for export in EXPORTS.values()}
    unknown = sorted(
        f"{key}  (was: {reason})"
        for key, reason in UNCOVERED.items()
        if key not in known
    )
    assert not unknown, (
        "UNCOVERED names symbols that are not exported under that module; "
        "they were renamed, moved or removed:\n"
        + "\n".join(f"    {entry}" for entry in unknown)
    )


def test_allowlist_entries_are_functions():
    """Only functions are gated, so only functions belong in the allowlist."""
    by_key = {export.key: export for export in EXPORTS.values()}
    wrong_kind = sorted(
        f"{key} is a {by_key[key].kind}"
        for key in UNCOVERED
        if key in by_key and by_key[key].kind != "function"
    )
    assert not wrong_kind, (
        "these allowlist entries are not functions, so they were never "
        "gated:\n" + "\n".join(f"    {entry}" for entry in wrong_kind)
    )


@pytest.mark.parametrize("key", sorted(UNCOVERED))
def test_every_allowlist_entry_states_a_reason(key):
    reason = UNCOVERED[key]
    assert reason.strip(), f"UNCOVERED[{key!r}] needs a reason"


def test_reports_coverage_of_the_public_surface(capsys):
    """Report the numbers so movement between releases is measurable."""
    functions = [e for e in EXPORTS.values() if e.kind == "function"]
    covered = [e for e in functions if _is_covered(e)]
    kinds: dict[str, int] = {}
    for export in EXPORTS.values():
        kinds[export.kind] = kinds.get(export.kind, 0) + 1
    with capsys.disabled():
        print(
            f"\n  public surface: {len(EXPORTS)} exports "
            f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})"
            f"\n  exported functions reached by a test: "
            f"{len(covered)}/{len(functions)} "
            f"({100 * len(covered) / len(functions):.1f}%), "
            f"{len(UNCOVERED)} allowlisted"
        )
    assert covered, "no exported function is reached by any test"
