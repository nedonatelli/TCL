"""A NamedTuple's documented Attributes must be its actual fields.

The post-v2.5.0 audit found result-type docstrings drifting from their
fields in both directions -- ``INSGNSSState.error_state`` documented a size
nothing produces, ``Spectrogram.power`` documented a type the field does not
always hold -- and those were caught by hand. This pins the mechanical half:
every attribute documented in a NamedTuple's ``Attributes`` section must be
a real field (no phantoms), and every field must be documented (no silent
additions). The sweep that introduced this gate found zero violations; the
gate exists so that stays true.
"""

import ast
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
PACKAGE = REPO_ROOT / "pytcl"

_ATTRIBUTES_SECTION = re.compile(
    r"Attributes\n\s*-+\n(.*?)(?=\n\s*\n\s*\w+\n\s*-+\n|\Z)", re.S
)
_ATTRIBUTE_NAME = re.compile(r"^\s{0,4}(\w+) ?:", re.M)


def _mismatches(package: pathlib.Path):
    findings = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any("NamedTuple" in ast.unparse(b) for b in node.bases):
                continue
            fields = [
                s.target.id
                for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
            ]
            doc = ast.get_docstring(node) or ""
            section = _ATTRIBUTES_SECTION.search(doc)
            if not section:
                continue
            documented = _ATTRIBUTE_NAME.findall(section.group(1))
            phantom = sorted(set(documented) - set(fields))
            undocumented = sorted(set(fields) - set(documented))
            if phantom or undocumented:
                findings.append(
                    f"{path.relative_to(package.parent)}:{node.lineno} "
                    f"{node.name}: phantom={phantom} undocumented={undocumented}"
                )
    return findings


def test_namedtuple_attributes_sections_match_their_fields():
    findings = _mismatches(PACKAGE)
    assert not findings, (
        "NamedTuple docstrings out of step with their fields "
        "(phantom = documented but no such field; undocumented = field with "
        "no Attributes entry):\n  " + "\n  ".join(findings)
    )


class TestTheGateActuallyFires:
    def test_phantom_and_undocumented_are_both_found(self, tmp_path):
        (tmp_path / "synthetic.py").write_text(
            "from typing import NamedTuple\n\n\n"
            "class Result(NamedTuple):\n"
            '    """A result.\n\n'
            "    Attributes\n"
            "    ----------\n"
            "    x : float\n"
            "        Real field.\n"
            "    ghost : float\n"
            "        Documented, does not exist.\n"
            '    """\n\n'
            "    x: float\n"
            "    hidden: float\n",
            encoding="utf-8",
        )
        findings = _mismatches(tmp_path)
        assert len(findings) == 1
        assert "phantom=['ghost']" in findings[0]
        assert "undocumented=['hidden']" in findings[0]
