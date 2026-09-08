"""Regenerate the NRLMSISE-00 oracle fixture grid.

Usage: generate_grid.py <path-to-compiled-oracle-driver>

Writes tests/fixtures/nrlmsise00/grid_in.txt and grid_out.txt. The
input format per record (whitespace separated) is

    mode doy sec alt_km g_lat g_long lst f107a f107 ap a0..a6

with mode 0 = gtd7 (scalar ap), 1 = storm mode (7-element ap array),
2 = gtd7d; see scripts/nrlmsise_capture/oracle_driver.c.
"""

import itertools
import random
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "nrlmsise00"


def build_records():
    records = []
    alts = [0, 5, 10, 30, 50, 70, 72.5, 85, 100, 110, 120, 200, 400, 700, 1000]
    geo = [(60, -70), (-45, 120), (0, 0), (75, -180), (-80, 30)]
    solar = [(150, 150, 4), (70, 70, 4), (200, 180, 40)]
    doys = [172, 1, 265]
    for alt, (lat, lon), (f107a, f107, ap), doy, sec in itertools.product(
        alts, geo, solar, doys, [29000.0]
    ):
        records.append(
            (0, doy, sec, alt, lat, lon, 16.0, f107a, f107, ap, *([0.0] * 7))
        )
    for lst in [0.0, 8.05, 23.9]:
        for alt in [100, 400]:
            records.append(
                (0, 172, 29000.0, alt, 60, -70, lst, 150, 150, 4, *([0.0] * 7))
            )
    for sec in [0.0, 43200.0, 86400.0]:
        records.append((0, 172, sec, 400, 60, -70, 16.0, 150, 150, 4, *([0.0] * 7)))
    aph = (48.8, 60.0, 39.0, 27.0, 18.0, 22.5, 15.4)
    for alt in [85, 120, 200, 400, 700]:
        records.append((1, 310, 50000.0, alt, 55, 10, 16.0, 140, 180, 48.8, *aph))
    for alt in [200, 400, 700, 1000]:
        records.append((2, 172, 29000.0, alt, 60, -70, 16.0, 150, 150, 4, *([0.0] * 7)))
    rng = random.Random(2026)
    for _ in range(300):
        mode = rng.choice([0, 0, 0, 1, 2])
        records.append(
            (
                mode,
                rng.randint(1, 366),
                rng.uniform(0, 86400),
                rng.uniform(0, 1000),
                rng.uniform(-90, 90),
                rng.uniform(-180, 180),
                16.0,
                rng.uniform(65, 250),
                rng.uniform(65, 300),
                rng.uniform(0, 300),
                *[rng.uniform(0, 300) for _ in range(7)],
            )
        )
    return records


def main() -> int:
    driver = sys.argv[1]
    records = build_records()
    inp = "\n".join(" ".join(f"{v:.10g}" for v in r) for r in records) + "\n"
    out = subprocess.run(
        [driver], input=inp, capture_output=True, text=True, check=True
    )
    n_out = len(out.stdout.strip().splitlines())
    if n_out != len(records):
        raise SystemExit(f"oracle produced {n_out} outputs for {len(records)} records")
    (FIXTURE_DIR / "grid_in.txt").write_text(inp)
    (FIXTURE_DIR / "grid_out.txt").write_text(out.stdout)
    print(f"wrote {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
