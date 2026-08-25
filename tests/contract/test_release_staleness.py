"""Gates for the release-facing prose that keeps going stale.

Each test covers a staleness class that has actually bitten:

1. Version stamps scattered across prose files drifting from the real
   version (ROADMAP still said v2.6.0 during v2.7.0 prep).
2. Hardcoded counts (README hero line and badges, parity-inventory
   closing paragraph) drifting from measured reality.
3. "Unported"/"missing" prose claims outliving the port that falsified
   them (docs/api/atmosphere.rst still called the ray tracers unported
   two merged PRs after they shipped).

Like the dead-parameter gate's ALLOWED dict, the CLAIMED_ABSENT registry
below is self-policing: entries must both (a) name symbols that do not
exist and (b) match text actually present in the claiming file, so stale
registry entries fail from the other direction.
"""

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent


def _read(rel):
    return (REPO / rel).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# 1. Version stamps
# --------------------------------------------------------------------------


def _pyproject_version():
    m = re.search(r'^version = "(\d+\.\d+\.\d+)"', _read("pyproject.toml"), re.M)
    assert m, "version not found in pyproject.toml"
    return m.group(1)


def test_version_stamps_agree():
    version = _pyproject_version()
    import pytcl

    assert pytcl.__version__ == version

    m = re.search(r"^## \[(\d+\.\d+\.\d+)\]", _read("CHANGELOG.md"), re.M)
    assert m and m.group(1) == version, (
        f"CHANGELOG's newest release entry is {m and m.group(1)}, "
        f"pyproject says {version}"
    )

    m = re.search(r"\*\*Current Version:\*\* v(\d+\.\d+\.\d+)", _read("ROADMAP.md"))
    assert m and m.group(1) == version, (
        f"ROADMAP.md header says v{m and m.group(1)}, pyproject says {version}"
    )

    m = re.search(r"Current State \(v(\d+\.\d+\.\d+)\)", _read("docs/roadmap.rst"))
    assert m and m.group(1) == version, (
        f"docs/roadmap.rst Current State says v{m and m.group(1)}, "
        f"pyproject says {version}"
    )

    contributing = _read("CONTRIBUTING.md")
    for pattern, label in [
        (r"\*\*Version:\*\* v(\d+\.\d+\.\d+)", "CONTRIBUTING version line"),
        (r"Current metrics \(v(\d+\.\d+\.\d+)\)", "CONTRIBUTING metrics header"),
    ]:
        m = re.search(pattern, contributing)
        assert m and m.group(1) == version, (
            f"{label} says v{m and m.group(1)}, pyproject says {version}"
        )


def test_claude_md_version_matches():
    m = re.search(r"\*\*Version:\*\* (\d+\.\d+\.\d+)", _read("CLAUDE.md"))
    assert m and m.group(1) == _pyproject_version()


# --------------------------------------------------------------------------
# 2. Hardcoded counts
# --------------------------------------------------------------------------


def _count_functions():
    count = 0
    for path in (REPO / "pytcl").rglob("*.py"):
        count += sum(
            1
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("def ")
        )
    return count


def _count_modules():
    return sum(
        1 for p in (REPO / "pytcl").rglob("*.py") if "__pycache__" not in p.parts
    )


@pytest.fixture(scope="module")
def collected_test_count():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    m = re.search(r"(\d+) tests collected", result.stdout)
    assert m, f"could not parse collection output:\n{result.stdout[-2000:]}"
    return int(m.group(1))


def test_readme_hero_counts_current(collected_test_count):
    """The README hero line may round down, but not overstate or lag far."""
    readme = _read("README.md")
    m = re.search(
        r"\*\*([\d,]+)\+ functions\*\* \| \*\*([\d,]+) modules\*\* \| "
        r"\*\*([\d,]+)\+ tests\*\* \| \*\*(\d+)% coverage\*\*",
        readme,
    )
    assert m, "README hero counts line not found or format changed"
    fn_claim, mod_claim, test_claim = (
        int(m.group(1).replace(",", "")),
        int(m.group(2).replace(",", "")),
        int(m.group(3).replace(",", "")),
    )

    fn_actual = _count_functions()
    assert fn_claim <= fn_actual <= fn_claim + 200, (
        f"README claims {fn_claim}+ functions, measured {fn_actual}; "
        "update the hero line (both directions gate: no overstatement, "
        "no lagging more than 200 behind)"
    )
    mod_actual = _count_modules()
    assert mod_claim == mod_actual, (
        f"README claims {mod_claim} modules, measured {mod_actual}"
    )
    assert test_claim <= collected_test_count <= test_claim + 1500, (
        f"README claims {test_claim}+ tests, collected {collected_test_count}"
    )

    badge = re.search(r"badge/tests-(\d+)%2B%20passing", readme)
    assert badge and int(badge.group(1)) == test_claim, (
        "README tests badge disagrees with the hero line"
    )


def test_coverage_claims_are_synchronized():
    """Every hardcoded coverage % equals the CONTRIBUTING measurement.

    CONTRIBUTING is the single source: it cites the CI run that measured
    its number. The README badge/hero and docs/index.rst must match it
    (rounded down to a whole percent), so a re-measurement updates
    everything or the gate fails.
    """
    m = re.search(r"\*\*Coverage:\*\* (\d+)\.(\d+)%", _read("CONTRIBUTING.md"))
    assert m, "CONTRIBUTING coverage measurement not found"
    whole = int(m.group(1))

    readme = _read("README.md")
    hero = re.search(r"\*\*(\d+)% coverage\*\*", readme)
    assert hero and int(hero.group(1)) == whole, (
        f"README hero says {hero and hero.group(1)}%, CONTRIBUTING measures "
        f"{m.group(0)}"
    )
    badge = re.search(r"badge/coverage-(\d+)%25", readme)
    assert badge and int(badge.group(1)) == whole, (
        f"README coverage badge says {badge and badge.group(1)}%, "
        f"CONTRIBUTING measures {m.group(0)}"
    )
    idx = re.search(r"(\d+)% coverage", _read("docs/index.rst"))
    assert idx and int(idx.group(1)) == whole, (
        f"docs/index.rst says {idx and idx.group(1)}%, CONTRIBUTING "
        f"measures {m.group(0)}"
    )


def test_parity_inventory_closing_counts(collected_test_count):
    text = _read("docs/matlab_parity_inventory.rst")
    m = re.search(
        r"test suite of ([\d,]+)\+\s*\ncases that includes (\d+) validation files",
        text,
    )
    assert m, "parity inventory closing counts not found or format changed"
    test_claim = int(m.group(1).replace(",", ""))
    val_claim = int(m.group(2))
    assert test_claim <= collected_test_count <= test_claim + 1500
    val_actual = len(list((REPO / "tests" / "validation").glob("*.py")))
    assert val_claim == val_actual, (
        f"inventory claims {val_claim} validation files, found {val_actual}"
    )


# --------------------------------------------------------------------------
# 3. "Unported" prose claims
# --------------------------------------------------------------------------

# Every prose claim that something is NOT ported, registered as (file,
# text that carries the claim, dotted names that would exist if the claim
# became false). When a port ships under one of these names, this gate
# fails and forces the prose update that docs/api/atmosphere.rst missed
# for two releases.
CLAIMED_ABSENT = {
    "docs/api/atmosphere.rst": {
        "text": "What remains unported",
        "symbols": [
            "pytcl.atmosphere.nrlmsise00",
            "pytcl.atmosphere.thermosphere.jacchia_atmos_param",
        ],
    },
    "ROADMAP.md": {
        "text": "Refraction suite remainder",
        "symbols": [
            "pytcl.atmosphere.nrlmsise00",
            "pytcl.atmosphere.thermosphere.jacchia_atmos_param",
            "pytcl.atmosphere.models.speed_of_sound_gas_table",
        ],
    },
}


def _resolves(dotted):
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            mod = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        obj = mod
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return False
        return True
    return False


def test_unported_claims_have_not_shipped():
    for rel, claim in CLAIMED_ABSENT.items():
        text = _read(rel)
        assert claim["text"] in text, (
            f"{rel} no longer contains the registered claim text "
            f"{claim['text']!r} -- update CLAIMED_ABSENT to match the prose"
        )
        for dotted in claim["symbols"]:
            assert not _resolves(dotted), (
                f"{rel} claims {dotted} is unported, but it now resolves -- "
                "update the prose (and then this registry)"
            )


def test_claimed_absent_registry_is_not_stale():
    """Registered files must exist; empty symbol lists are meaningless."""
    for rel, claim in CLAIMED_ABSENT.items():
        assert (REPO / rel).exists(), f"{rel} disappeared; update CLAIMED_ABSENT"
        assert claim["symbols"], f"{rel} entry has no symbols to check"
