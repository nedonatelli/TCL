"""The runtime floors in pyproject.toml are the ones CI installs and tests.

A ``>=`` floor nobody installs is a guess: v2.11 declared ``numpy>=1.24.0``
while no CI cell had run anything older than the lock's numpy 2.4. The
``floors`` job in ci.yml installs exactly these versions, and
``check_dependency_floors.py --check`` fails that job if the resolver moved
any of them.
"""

import importlib.util
import pathlib
import subprocess
import sys
from importlib import metadata

from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
SCRIPT = ROOT / "scripts" / "check_dependency_floors.py"

_spec = importlib.util.spec_from_file_location("check_dependency_floors", SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
declared_floors = _mod.declared_floors
_raw_dependencies = _mod._raw_dependencies
off_floor = _mod.off_floor


def test_declared_floors_are_decision_4():
    assert declared_floors(PYPROJECT) == {
        "numpy": "2.3.2",
        "scipy": "1.16.1",
        "numba": "0.63.0",
        "h5py": "3.15.0",
        "msgspec": "0.20.0",
        "loguru": "0.7",
        "rich": "13",
    }


def test_every_runtime_dependency_has_a_floor():
    assert all(
        spec.startswith(name) and ">=" in spec
        for name, spec in _raw_dependencies(PYPROJECT)
    )


def test_check_fails_exactly_when_an_installed_version_is_off_its_floor():
    # The dev venv sits above the floors (numpy 2.4.x), so here --check must
    # fail and name numpy. The same test runs inside ci.yml's floors job,
    # where every version equals its floor and --check must pass.
    off = [
        name
        for name, floor in declared_floors(PYPROJECT).items()
        if Version(metadata.version(name)) != Version(floor)
    ]
    proc = subprocess.run(
        [sys.executable, SCRIPT, "--check"], capture_output=True, text=True
    )
    if off:
        assert proc.returncode == 1
        assert all(name in proc.stdout for name in off)
    else:
        assert proc.returncode == 0, proc.stdout


def test_off_floor_compares_versions_not_strings():
    floors = {"loguru": "0.7", "numpy": "2.3.2", "rich": "13"}
    at = {"loguru": "0.7.0", "numpy": "2.3.2", "rich": "13.0.0"}
    assert off_floor(floors, at.get) == []
    above = {**at, "numpy": "2.4.6"}
    assert off_floor(floors, above.get) == ["numpy: installed 2.4.6, floor 2.3.2"]
    missing = {"loguru": "0.7.0", "numpy": "2.3.2"}
    assert off_floor(floors, missing.get) == ["rich: not installed (floor 13)"]


def test_constraints_file_pins_every_floor(tmp_path):
    out = tmp_path / "floors.txt"
    subprocess.run([sys.executable, SCRIPT, "--write-constraints", out], check=True)
    assert out.read_text(encoding="utf-8").splitlines() == [
        f"{name}=={floor}" for name, floor in declared_floors(PYPROJECT).items()
    ]
