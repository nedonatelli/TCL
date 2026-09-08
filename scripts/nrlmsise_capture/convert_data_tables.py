"""Regenerate pytcl/atmosphere/_nrlmsise00_data.py from the vendored C.

Parses the array initializers in csrc/nrlmsise00/nrlmsise-00_data.c
into numpy arrays, preserving C's zero-fill of partially-initialized
arrays (ptm is declared [50] with 10 initializers).
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "csrc" / "nrlmsise00" / "nrlmsise-00_data.c"
DEST = REPO / "pytcl" / "atmosphere" / "_nrlmsise00_data.py"

HEADER = '''"""NRLMSISE-00 model coefficients.

Generated from the NRL-modified reference implementation's
nrlmsise-00_data.c (vendored at csrc/nrlmsise00) by
scripts/nrlmsise_capture/convert_data_tables.py. Arrays declared
larger than their initializer are zero-filled, as in C. Do not edit
by hand.
"""

import numpy as np

'''


def main() -> int:
    src = SRC.read_text()
    decls = re.findall(r"double\s+(\w+)\s*((?:\[\d+\])+)\s*=\s*\{(.*?)\};", src, re.S)
    lines = [HEADER.rstrip("\n"), ""]
    for name, dims_s, body in decls:
        dims = tuple(int(d) for d in re.findall(r"\[(\d+)\]", dims_s))
        nums = re.findall(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
            re.sub(r"/\*.*?\*/", "", body, flags=re.S),
        )
        vals = [float(x) for x in nums]
        expected = 1
        for d in dims:
            expected *= d
        assert len(vals) <= expected, f"{name}: {len(vals)} > {expected}"
        vals += [0.0] * (expected - len(vals))
        lines.append(f"{name.upper()} = np.array([")
        for i in range(0, len(vals), 6):
            lines.append("    " + ", ".join(repr(v) for v in vals[i : i + 6]) + ",")
        shape = ", ".join(str(d) for d in dims)
        lines.append(f"]).reshape({shape}{',' if len(dims) == 1 else ''})")
        lines.append("")
        print(f"{name}: dims={dims} initializers={len(nums)}")
    DEST.write_text("\n".join(lines))
    print(f"wrote {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
