"""Every exported function must be exercised by at least one test.

`associated_legendre_scaled` was exported from the gravity package and had zero
callers. It scaled along the wrong axis: the addition-theorem invariant it must
satisfy was off by a factor of 1e199 at degree 2000, and it blocked EGM2008 the
whole time it shipped. The suite was green throughout. It was found by hand
during the v2 audit, not by any gate.

This is that gate. It walks ``__all__`` across the package, classifies each
export, and requires every exported *function* to appear in at least one test.

Three decisions, from the spec in gh-47:

**Test reference, not internal caller.** For a library most public API
legitimately has no caller inside the library -- it exists to be called by
users. Gating on internal callers would flag most of the 919 exported functions
and be discarded as noise. The `associated_legendre_scaled` failure is caught
by the test-reference rule and would not have been caught by a caller rule.

**Not line coverage.** A symbol can show covered lines because an unrelated
caller passed through the same file. Requiring that a test *names* the export is
what makes the check mean something.

**Functions only.** 61 exported classes have no test reference, but most are
result types -- a caller receives a ``CRBResult`` without ever writing the name
-- so gating on them would produce noise that gets suppressed rather than
fixed. The classification is in place so that a narrower class rule can be added
later.
"""

import importlib
import inspect
import pathlib
import pkgutil
import re

import pytest

import pytcl

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTS_DIR = REPO_ROOT / "tests"

# Exported functions that no test references, each with the reason it is not
# covered. This list must shrink. Two rules keep it honest: an entry that is
# now covered fails the suite (test_allowlist_has_no_stale_entries), and an
# entry naming a symbol that no longer exists fails too.
#
# Do not add an entry to silence a new export. Write the test.
UNCOVERED: dict[str, str] = {
    # --- pytcl.logging_config: module slated for removal in gh-24 -----------
    "TimingContext": "logging_config is dead code, removal tracked by gh-24",
    "configure_logging": "logging_config is dead code, removal tracked by gh-24",
    "get_logger": "logging_config is dead code, removal tracked by gh-24",
    "timed": "logging_config is dead code, removal tracked by gh-24",
    # --- cache control: an entire published surface, unexercised -----------
    # Ten functions across five caches. Their behaviour under concurrent and
    # repeated use is also what gh-26 raises, so the tests want writing with
    # that in view rather than one at a time.
    "clear_geodesy_cache": "cache-control surface untested, see gh-26",
    "clear_great_circle_cache": "cache-control surface untested, see gh-26",
    "clear_legendre_cache": "cache-control surface untested, see gh-26",
    "clear_magnetic_cache": "cache-control surface untested, see gh-26",
    "clear_transformation_cache": "cache-control surface untested, see gh-26",
    "configure_magnetic_cache": "cache-control surface untested, see gh-26",
    "get_cache_info": "cache-control surface untested, see gh-26",
    "get_geodesy_cache_info": "cache-control surface untested, see gh-26",
    "get_legendre_cache_info": "cache-control surface untested, see gh-26",
    "get_magnetic_cache_info": "cache-control surface untested, see gh-26",
    # --- coefficient factories and loaders ---------------------------------
    "create_igrf13_coefficients": "coefficient factory, needs a reference set",
    "create_wmm2020_coefficients": "coefficient factory, needs a reference set",
    "create_wmm2025_coefficients": "coefficient factory, needs a reference set",
    "load_egm_coefficients": "requires the EGM2008 data file (~1.3 GB)",
    "load_emm_coefficients": "requires the EMM data file",
    "magnetic_field_spherical": "needs a published reference value",
    # --- astronomical ------------------------------------------------------
    "format_tle": "round trip against parse_tle would cover this",
    "gps_to_utc": "needs a published leap-second reference",
    "precession_angles_iau76": "needs IAU published angles to compare against",
    # --- other -------------------------------------------------------------
    "true_airspeed_from_mach": "needs a reference value",
    "validate_query_input": "internal validation helper, exported unnecessarily",
}

# A traversal that silently found nothing would make every assertion here pass.
MINIMUM_EXPECTED_EXPORTS = 1000


def _classify(obj) -> str:
    if inspect.isfunction(obj) or inspect.isbuiltin(obj):
        return "function"
    if inspect.isclass(obj):
        return "class"
    if inspect.ismodule(obj):
        return "module"
    return "constant"


def _exports() -> dict[str, tuple[str, str]]:
    """Every name in an ``__all__``, mapped to (kind, defining module).

    A module that cannot be imported because an optional dependency is absent
    contributes nothing rather than aborting, so the verdict does not depend on
    which extras happen to be installed.
    """
    found: dict[str, tuple[str, str]] = {}
    for info in pkgutil.walk_packages(pytcl.__path__, "pytcl."):
        try:
            module = importlib.import_module(info.name)
        except Exception:
            continue
        for name in getattr(module, "__all__", None) or []:
            if name in found:
                continue
            found[name] = (_classify(getattr(module, name, None)), info.name)
    return found


def _names_referenced_by_tests() -> set[str]:
    """Every identifier appearing anywhere under tests/."""
    seen: set[str] = set()
    for path in TESTS_DIR.rglob("*.py"):
        if path.name == pathlib.Path(__file__).name:
            continue  # this file names exports only to exempt them
        seen.update(re.findall(r"\b\w+\b", path.read_text(encoding="utf-8")))
    return seen


EXPORTS = _exports()
REFERENCED = _names_referenced_by_tests()


def test_traversal_found_the_package():
    """Guard against a walk that silently collects nothing."""
    assert len(EXPORTS) >= MINIMUM_EXPECTED_EXPORTS, (
        f"only discovered {len(EXPORTS)} exports; the package walk is broken, "
        "which would make every other assertion in this file vacuous"
    )


def test_every_exported_function_is_referenced_by_a_test():
    """The gate. An exported function with no test is a defect, not a gap."""
    uncovered = sorted(
        name
        for name, (kind, _) in EXPORTS.items()
        if kind == "function" and name not in REFERENCED and name not in UNCOVERED
    )
    detail = "\n".join(f"    {n:38} {EXPORTS[n][1]}" for n in uncovered)
    assert not uncovered, (
        f"{len(uncovered)} exported function(s) are not referenced by any "
        f"test:\n{detail}\n\n"
        "Write a test that calls it, or -- only if it genuinely cannot be "
        "tested -- add it to UNCOVERED in this file with the reason. The "
        "allowlist is expected to shrink."
    )


def test_allowlist_has_no_stale_entries():
    """An entry that is now covered must be removed, so the list shrinks."""
    now_covered = sorted(n for n in UNCOVERED if n in REFERENCED)
    assert not now_covered, (
        f"these are now referenced by a test; remove them from UNCOVERED: {now_covered}"
    )


def test_allowlist_names_real_exports():
    """A typo, or a symbol that has been removed, must not sit here silently."""
    unknown = sorted(n for n in UNCOVERED if n not in EXPORTS)
    assert not unknown, (
        f"UNCOVERED names symbols that are not exported: {unknown}. They were "
        "renamed or removed; drop the entries."
    )


def test_allowlist_entries_are_functions():
    """Only functions are gated, so only functions belong in the allowlist."""
    wrong_kind = sorted(
        f"{n} ({EXPORTS[n][0]})"
        for n in UNCOVERED
        if EXPORTS.get(n, ("", ""))[0] != "function"
    )
    assert not wrong_kind, (
        f"these allowlist entries are not functions, so they were never "
        f"gated: {wrong_kind}"
    )


def test_reports_coverage_of_the_public_surface(capsys):
    """Report the numbers so movement between releases is measurable."""
    functions = [n for n, (k, _) in EXPORTS.items() if k == "function"]
    covered = [n for n in functions if n in REFERENCED]
    with capsys.disabled():
        kinds: dict[str, int] = {}
        for _, (kind, _) in EXPORTS.items():
            kinds[kind] = kinds.get(kind, 0) + 1
        print(
            f"\n  public surface: {len(EXPORTS)} exports "
            f"({', '.join(f'{v} {k}' for k, v in sorted(kinds.items()))})"
            f"\n  exported functions covered by a test: "
            f"{len(covered)}/{len(functions)} "
            f"({100 * len(covered) / len(functions):.1f}%), "
            f"{len(UNCOVERED)} allowlisted"
        )
    assert covered, "no exported function is referenced by any test"


@pytest.mark.parametrize("name", sorted(UNCOVERED))
def test_every_allowlist_entry_states_a_reason(name):
    reason = UNCOVERED[name]
    assert reason and len(reason) > 15, (
        f"UNCOVERED[{name!r}] needs a reason explaining why it cannot be "
        f"tested; got {reason!r}"
    )
