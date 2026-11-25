"""Check CONTRIBUTING's recorded coverage against CI's own measurement.

The contract tests keep every hardcoded coverage figure equal to the one in
CONTRIBUTING.md, but nothing compared that figure to a real measurement:
v2.12.0's release PR merged with v2.11.1's 94.60% while CI measured
95.25%. ci.yml's coverage cell runs this after writing ``coverage.xml``.

Usage::

    python scripts/check_coverage_claim.py coverage.xml

Exits 1 when the measured total (statements plus branches, as pytest-cov
reports it under ``--cov-branch``) differs from the recorded figure by more
than ``TOLERANCE`` points, either way; re-measure and update CONTRIBUTING's
metrics block (and the figures synchronized with it) to fix.
"""

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

TOLERANCE = 1.0
CONTRIBUTING = Path(__file__).resolve().parents[1] / "CONTRIBUTING.md"


def measured_percent(xml_text: str) -> float:
    """Total coverage from a Cobertura report: covered over valid, statements
    and branches together."""
    root = ET.fromstring(xml_text)
    covered = int(root.get("lines-covered")) + int(root.get("branches-covered"))
    valid = int(root.get("lines-valid")) + int(root.get("branches-valid"))
    return 100.0 * covered / valid


def recorded_percent(contributing_text: str) -> float:
    m = re.search(r"\*\*Coverage:\*\* (\d+\.\d+)%", contributing_text)
    if m is None:
        raise ValueError("CONTRIBUTING.md has no '**Coverage:** NN.NN%' line")
    return float(m.group(1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("coverage_xml", type=Path)
    args = parser.parse_args(argv)
    measured = measured_percent(args.coverage_xml.read_text(encoding="utf-8"))
    recorded = recorded_percent(CONTRIBUTING.read_text(encoding="utf-8"))
    print(f"measured {measured:.2f}%, CONTRIBUTING records {recorded:.2f}%")
    if abs(measured - recorded) > TOLERANCE:
        print(
            f"Coverage claim is stale by {measured - recorded:+.2f} points "
            f"(tolerance {TOLERANCE}): update CONTRIBUTING.md's metrics block "
            "with this run's figure and ID, then the figures the contract "
            "tests keep in sync with it."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
