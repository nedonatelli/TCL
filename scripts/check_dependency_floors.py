"""Read the runtime dependency floors from pyproject.toml, or verify them.

ci.yml's ``floors`` job resolves every direct dependency to its lowest
allowed version and runs the suite there. Two things can make that job test
something other than the floors: installing the test runner can pull a
runtime dependency upward, and a resolver change can pick a different
version. This script closes both.

Usage::

    python scripts/check_dependency_floors.py --write-constraints PATH
    python scripts/check_dependency_floors.py --check

``--write-constraints`` writes one ``name==floor`` line per runtime
dependency, for ``uv pip install --constraint``. It imports only the
standard library, so it runs in an environment that has nothing but the
runtime installed.

``--check`` exits 1 and lists every runtime dependency whose installed
version is not equal to its floor (compared as ``packaging`` versions, so
``0.7`` equals ``0.7.0``), or that is not installed at all.
"""

import argparse
import re
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _raw_dependencies(pyproject: Path) -> list[tuple[str, str]]:
    """``(name, specifier string)`` for each ``[project] dependencies`` entry."""
    with pyproject.open("rb") as f:
        specs = tomllib.load(f)["project"]["dependencies"]
    return [(_NAME.match(spec).group(0), spec) for spec in specs]


def declared_floors(pyproject: Path) -> dict[str, str]:
    """Map each runtime dependency to the version after its ``>=``."""
    floors = {}
    for name, spec in _raw_dependencies(pyproject):
        rest = spec[len(name) :].strip()
        if not rest.startswith(">=") or "," in rest or ";" in rest:
            raise ValueError(f"{spec!r} is not a bare '>=' floor")
        floors[name] = rest[2:].strip()
    return floors


def off_floor(
    floors: dict[str, str], installed: Callable[[str], str | None]
) -> list[str]:
    """One line per dependency whose installed version differs from its floor."""
    from packaging.version import Version

    problems = []
    for name, floor in floors.items():
        version = installed(name)
        if version is None:
            problems.append(f"{name}: not installed (floor {floor})")
        elif Version(version) != Version(floor):
            problems.append(f"{name}: installed {version}, floor {floor}")
    return problems


def _installed_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-constraints", metavar="PATH", type=Path)
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    floors = declared_floors(PYPROJECT)
    if args.write_constraints is not None:
        args.write_constraints.write_text(
            "".join(f"{name}=={floor}\n" for name, floor in floors.items()),
            encoding="utf-8",
        )
        return 0

    problems = off_floor(floors, _installed_version)
    if problems:
        print("Installed runtime dependencies are not at their declared floors:")
        for line in problems:
            print(f"  {line}")
        return 1
    print("All runtime dependencies are at their declared floors:")
    for name, floor in floors.items():
        print(f"  {name} {_installed_version(name)} (floor {floor})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
