"""What is public is a decision, and this is where it is recorded.

78 names were tested but not reachable from their package. Most were
omissions rather than decisions: ``wrap_to_2pi`` was exported and
``wrap_to_360`` was not; ``min_cost_flow_simplex`` was private while its
result type was public and its input type ``FlowEdge`` was not, so it could
not be called even if imported. 56 were exported; the 22 below stay internal.

Nothing prevented that from happening, and nothing prevented it recurring:
the survey that found them was a script, not a test. This is that script,
with the outcome written down.

**A name is public if it can be imported from its package.** Not if it is in
``__all__``: ``pytcl.gpu`` resolves its exports through a lazy module-level
``__getattr__``, so names absent from ``__all__`` and from ``dir()`` still
import fine. An earlier version of this check used ``__all__`` and reported
reachable names as missing.

Adding a tested public-looking function without exporting it fails this test.
Fix it by exporting the name, by prefixing it with an underscore, or -- if it
is genuinely internal and genuinely needs a plain name -- by adding it to
``INTERNAL`` with a reason.
"""

import ast
import importlib
import pathlib
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]

# Deliberately not exported. Every entry needs a reason: an unexplained name
# here is indistinguishable from the omissions this test exists to catch.
INTERNAL = {
    # Backend dispatch. The module is _backend.py -- the underscore already
    # says internal, and these are the types it dispatches on.
    "pytcl.gpu.Backend": "dispatch base class, pytcl/gpu/_backend.py",
    "pytcl.gpu.CuPyBackend": "dispatch implementation",
    "pytcl.gpu.MLXBackend": "dispatch implementation",
    "pytcl.gpu.get_compute_backend": "dispatch entry point, used across pytcl.gpu",
    "pytcl.gpu.compute_backend_available": "dispatch probe",
    # Kalman and IMM internals. Real algorithms, but reached through the
    # filters rather than called directly.
    "pytcl.dynamic_estimation.kalman.compute_innovation_likelihood": "filter internal",
    "pytcl.dynamic_estimation.kalman.compute_mahalanobis_distance": "filter internal",
    "pytcl.dynamic_estimation.kalman.compute_matrix_sqrt": "filter internal",
    "pytcl.dynamic_estimation.kalman.compute_merwe_weights": "sigma-point internal",
    "pytcl.dynamic_estimation.kalman.ensure_symmetric": "filter internal",
    "pytcl.dynamic_estimation.combine_estimates": "IMM internal",
    "pytcl.dynamic_estimation.compute_mixing_probabilities": "IMM internal",
    "pytcl.dynamic_estimation.mix_states": "IMM internal",
    # Aliased on export because the bare name says nothing about which cache.
    "pytcl.navigation.get_cache_info": "exported as get_great_circle_cache_info",
    "pytcl.astronomical.get_cache_info": "exported as get_transformation_cache_info",
    # Implementation details.
    "pytcl.core.validated_array_input": "decorator used inside pytcl.core",
    "pytcl.core.format_maturity_badge": "docs tooling",
    "pytcl.io.AnalysisResult": "internal to the migration helper",
    "pytcl.astronomical.unkozai_mean_motion": "SGP4 internal",
    "pytcl.gravity.spherical_harmonic_sum_high_degree": "computational kernel",
    "pytcl.gravity.clenshaw_geoid": "computational kernel, not a parser",
    "pytcl.magnetism.create_test_coefficients": "test scaffolding, per its name",
}


def _tested_names():
    """Every identifier appearing anywhere under tests/.

    Crude on purpose: the question is whether a name is exercised at all, and
    a false positive here only means a name is held to the export rule.
    """
    out = subprocess.run(
        ["grep", "-rhoE", r"[A-Za-z_][A-Za-z0-9_]*", str(ROOT / "tests")],
        capture_output=True,
        text=True,
    )
    return set(out.stdout.split())


def _is_reachable(package, name):
    """Can a caller write ``from <package> import <name>``?"""
    return hasattr(importlib.import_module(package), name)


def _public_definitions():
    """(package, name) for every public-looking definition outside a private module."""
    found = []
    for path in sorted((ROOT / "pytcl").rglob("*.py")):
        if "__pycache__" in str(path) or path.name.startswith("_"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        package = ".".join(path.relative_to(ROOT).with_suffix("").parts[:-1])
        for node in tree.body:
            if isinstance(
                node, (ast.FunctionDef, ast.ClassDef)
            ) and not node.name.startswith("_"):
                found.append((package, node.name))
    return found


_TESTED = _tested_names()
_DEFINITIONS = _public_definitions()


def test_the_scan_finds_definitions():
    """Guard the guard: an empty scan makes the rule below vacuous."""
    assert len(_DEFINITIONS) > 500, f"only {len(_DEFINITIONS)} definitions found"


def test_reachability_check_works():
    """And guard the mechanism: if hasattr always answered True the rule would
    never fire. These two are known to sit on opposite sides."""
    assert _is_reachable("pytcl.navigation", "compute_dop")
    assert not _is_reachable("pytcl.core", "validated_array_input")


def test_every_tested_public_name_is_reachable_or_declared_internal():
    unreachable = []
    for package, name in _DEFINITIONS:
        if name not in _TESTED:
            continue
        try:
            if _is_reachable(package, name):
                continue
        except Exception:
            continue
        if f"{package}.{name}" in INTERNAL:
            continue
        unreachable.append(f"{package}.{name}")

    assert not unreachable, (
        "These are tested but cannot be imported from their package:\n  "
        + "\n  ".join(sorted(unreachable))
        + "\n\nExport the name, prefix it with an underscore, or add it to "
        "INTERNAL in this file with a reason."
    )


@pytest.mark.parametrize("dotted", sorted(INTERNAL))
def test_internal_entries_are_still_internal(dotted):
    """An allowlist nobody prunes becomes a lie.

    If one of these is exported later, the entry is stale and should be
    deleted rather than left implying a decision that no longer holds.
    """
    package, _, name = dotted.rpartition(".")
    assert not _is_reachable(package, name), (
        f"{dotted} is now exported; remove it from INTERNAL."
    )
