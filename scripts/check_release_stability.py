"""Check a diff against the maturity registry's stability promises.

The registry (``pytcl.core.maturity``) is a per-module promise about
breakage: STABLE modules change API only in major bumps, MATURE ones may
adjust in minors. Nothing enforced that until v2.6's audit ran the check by
hand and immediately caught a STABLE module (``core.validation``) with a
breaking change queued for a minor release.

This mechanizes that check: given a base ref, list every changed pytcl
module alongside its registered level, most binding first. It cannot decide
whether a change is *breaking* -- that stays a human judgement -- but it
guarantees the judgement is made looking at the right list.

Usage::

    uv run python scripts/check_release_stability.py [BASE_REF]

BASE_REF defaults to ``main``. Exit status is 0 unless ``--strict`` is given
and a STABLE module changed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pytcl.core.maturity import MODULE_MATURITY, MaturityLevel  # noqa: E402

CONTRACTS = {
    MaturityLevel.STABLE: "API frozen -- breaking changes only in MAJOR bumps",
    MaturityLevel.MATURE: "minor API adjustments allowed in minor versions",
    MaturityLevel.EXPERIMENTAL: "may change in any release",
    MaturityLevel.DEPRECATED: "scheduled for removal",
}


def changed_modules(base_ref: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    ).stdout.split()
    return sorted(
        path.removeprefix("pytcl/").removesuffix(".py").replace("/", ".")
        for path in out
        if path.startswith("pytcl/") and path.endswith(".py")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_ref", nargs="?", default="main")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any STABLE module changed",
    )
    args = parser.parse_args()

    rows = []
    for module in changed_modules(args.base_ref):
        level = MODULE_MATURITY.get(module)
        if level is not None:
            rows.append((level, module))

    if not rows:
        print(f"No registered pytcl modules changed vs {args.base_ref}.")
        return 0

    rows.sort(key=lambda r: (-int(r[0]), r[1]))
    print(f"Registered modules changed vs {args.base_ref}:\n")
    for level, module in rows:
        print(f"  {level.name:12s} {module}")
    print()
    for level in dict.fromkeys(level for level, _ in rows):
        print(f"  {level.name}: {CONTRACTS[level]}")

    stable_changed = [m for level, m in rows if level is MaturityLevel.STABLE]
    if stable_changed:
        print(
            "\nSTABLE modules changed -- verify each change is non-breaking, "
            "or plan a major bump, or reclassify with a written rationale:"
        )
        for module in stable_changed:
            print(f"  pytcl.{module}")
        if args.strict:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
