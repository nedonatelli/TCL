"""Printed output must survive the Windows console encoding.

On Windows, Python encodes stdout with the locale codepage -- cp1252 on a
default install -- whenever stdout is a pipe or a file rather than a console.
A character outside that codepage raises UnicodeEncodeError and kills the
script, so a demo that looks fine in a terminal dies as soon as anyone
redirects it to a log.

Eight of the thirty example scripts used to fail this way, most on their
opening banner, from box-drawing characters. CI runs the examples on Ubuntu
only and cannot catch it, hence this check.

Non-ASCII is fine in comments, docstring prose, and plot titles and axis
labels: those are read from UTF-8 source or written into an HTML figure and
never pass through the console encoder. Only what reaches ``print`` matters.
"""

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EXAMPLES = sorted((REPO_ROOT / "examples").glob("*.py"))

# The characters a default Windows install can actually encode.
CP1252 = set()
for _byte in range(256):
    try:
        CP1252.add(bytes([_byte]).decode("cp1252"))
    except UnicodeDecodeError:
        pass


def _unencodable(text):
    return sorted({c for c in text if ord(c) > 127 and c not in CP1252})


def _printed_strings(tree):
    """Every string literal reachable from a print() argument."""
    for node in ast.walk(tree):
        is_print = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        )
        if not is_print:
            continue
        for arg in node.args:
            for sub in ast.walk(arg):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    yield sub.lineno, sub.value


def test_examples_are_discovered():
    assert len(EXAMPLES) >= 25, f"only found {len(EXAMPLES)} examples"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_printed_output_is_cp1252_safe(script):
    tree = ast.parse(script.read_text(encoding="utf-8"))

    offenders = []
    for lineno, value in _printed_strings(tree):
        bad = _unencodable(value)
        if bad:
            rendered = ", ".join(f"U+{ord(c):04X} {c!r}" for c in bad)
            offenders.append(f"  line {lineno}: {rendered}")

    assert not offenders, (
        f"{script.name} prints characters Windows cannot encode when stdout is "
        "redirected:\n"
        + "\n".join(offenders)
        + "\n\nUse ASCII in printed strings: '=' and '-' for rules, '->' for "
        "arrows, 'pi'/'sigma'/'sqrt'/'inf' spelled out. Reproduce with:\n"
        f"    PYTHONIOENCODING=cp1252 python {script.relative_to(REPO_ROOT)} > /dev/null"
    )
