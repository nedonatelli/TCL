"""Require a CHANGELOG entry for a pull request that changes ``pytcl/``.

Every behavior change gets an entry under ``## [Unreleased]``
(CONTRIBUTING). Reviews enforced that alone, and v2.12.0's whole-branch
review still found four changes with no entry after every task review had
passed. This is a prompt, not proof: it checks that a PR touching the
package adds a line to the Unreleased section, not that the line describes
the change. A PR that changes no behavior (a refactor, a docstring, a
release PR, which empties the section) carries the ``no-changelog`` label,
and the workflow skips the check.

Usage::

    python scripts/check_changelog_entry.py BASE HEAD

BASE and HEAD are git refs; the diff is ``BASE...HEAD`` (from the merge
base). Exits 1 when ``pytcl/`` changed and the Unreleased section gained no
non-blank line.
"""

import argparse
import re
import subprocess
import sys

CHANGELOG = "CHANGELOG.md"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def needs_entry(changed_paths: list[str]) -> bool:
    return any(path.startswith("pytcl/") for path in changed_paths)


def unreleased_lines(changelog_text: str) -> range:
    """1-based line numbers of the Unreleased section's body."""
    lines = changelog_text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith("## [Unreleased]")),
        None,
    )
    if start is None:
        raise ValueError("CHANGELOG.md has no '## [Unreleased]' heading")
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")),
        len(lines),
    )
    return range(start + 2, end + 1)


def added_lines(unified_diff: str) -> set[int]:
    """New-file line numbers of the non-blank lines a ``-U0`` diff adds."""
    added = set()
    line_no = None
    for line in unified_diff.splitlines():
        m = _HUNK.match(line)
        if m:
            line_no = int(m.group(1))
        elif line_no is not None and line.startswith("+"):
            if line[1:].strip():
                added.add(line_no)
            line_no += 1
    return added


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base")
    parser.add_argument("head")
    args = parser.parse_args(argv)
    span = f"{args.base}...{args.head}"
    if not needs_entry(_git("diff", "--name-only", span).splitlines()):
        print("No pytcl/ change; no CHANGELOG entry needed.")
        return 0
    section = unreleased_lines(_git("show", f"{args.head}:{CHANGELOG}"))
    if added_lines(_git("diff", "-U0", span, "--", CHANGELOG)) & set(section):
        print("pytcl/ changed and ## [Unreleased] gained an entry.")
        return 0
    print(
        "pytcl/ changed but CHANGELOG.md's ## [Unreleased] section gained no "
        "entry. Add one (ending with a 'What to do:' sentence), or, if the "
        "change alters no behavior, label the PR 'no-changelog'."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
