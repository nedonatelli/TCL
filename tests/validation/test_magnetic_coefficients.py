"""The three geomagnetic coefficient factories, against their official tables.

``create_wmm2020_coefficients``, ``create_wmm2025_coefficients`` and
``create_igrf13_coefficients`` each embed a model's spherical-harmonic
coefficients as a literal table in the source and parse it at import time. Until
now no test reached any of them (gh-49), so nothing checked that the transcribed
tables match what NOAA and IAGA published, and nothing checked that the parser
put each value in the right place.

That combination is how this package was wrong before: structural tests passed
while WMM magnetism was roughly 180 degrees out. A transposed ``g``/``h`` column
or an off-by-one in ``n``/``m`` produces a field that is smooth, plausible, and
wrong, and no property test derived from the coefficients themselves can see it.
The only way to catch it is to compare against the published numbers.

So these are REFERENCE-class tests. The reference side is vendored verbatim under
``tests/fixtures/magnetism/`` -- the IAGA ``igrf13coeffs.txt`` and the two NOAA
``.COF`` files -- with provenance and checksums in ``SOURCES.md`` there. Vendoring
rather than downloading means these run in CI with no network and no optional
dependency, which is what makes the check real rather than skipped. Comparison is
exact: these are transcriptions, so any difference at all is a defect.
"""

import pathlib

import numpy as np
import pytest

from pytcl.magnetism.igrf import create_igrf13_coefficients
from pytcl.magnetism.wmm import (
    create_wmm2020_coefficients,
    create_wmm2025_coefficients,
)

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "magnetism"


def _read_wmm_cof(path: pathlib.Path) -> dict[tuple[int, int], tuple[float, ...]]:
    """Parse an official WMM ``.COF`` file: ``n m g h g_dot h_dot`` per line.

    The header line and the trailing ``9999...`` sentinel do not have six
    fields, so length is enough to select the coefficient rows.
    """
    coefficients = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) != 6:
            continue
        n, m = int(fields[0]), int(fields[1])
        coefficients[(n, m)] = tuple(float(x) for x in fields[2:])
    return coefficients


def _read_igrf13_2020(
    path: pathlib.Path,
) -> dict[tuple[str, int, int], tuple[float, float]]:
    """Parse the 2020.0 epoch and 2020-25 secular-variation columns from IAGA.

    ``igrf13coeffs.txt`` is a wide table: one row per (g/h, n, m), one column per
    epoch from 1900.0 to 2020.0, then a final secular-variation column. The two
    columns are located by their header labels rather than by position, so a
    future IGRF generation appending an epoch does not silently shift the
    comparison onto the wrong column.
    """
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    header = lines[1].split()
    epoch_column = header.index("2020.0")
    secular_column = header.index("2020-25")

    coefficients = {}
    for line in lines[2:]:
        fields = line.split()
        if len(fields) < 3:
            continue
        kind, n, m = fields[0], int(fields[1]), int(fields[2])
        coefficients[(kind, n, m)] = (
            float(fields[epoch_column]),
            float(fields[secular_column]),
        )
    return coefficients


class TestWorldMagneticModel:
    """WMM-2020 and WMM-2025 against the NOAA coefficient files."""

    MODELS = [
        ("WMM2020", create_wmm2020_coefficients, "WMM_2020.COF", 2020.0),
        ("WMM2025", create_wmm2025_coefficients, "WMM_2025.COF", 2025.0),
    ]
    IDS = [name for name, _, _, _ in MODELS]

    @pytest.mark.parametrize("name,factory,filename,epoch", MODELS, ids=IDS)
    def test_every_coefficient_matches_the_official_file(
        self, name, factory, filename, epoch
    ):
        """All 90 (n, m) pairs, all four values each. Exact.

        This is the assertion the other tests in this file exist to support. A
        transposed column or a shifted index shows up here and nowhere else.
        """
        official = _read_wmm_cof(FIXTURES / filename)
        assert len(official) == 90, (
            f"{filename}: expected 90 coefficient rows for a degree-12 model, "
            f"read {len(official)} -- the fixture or the parser is wrong, so "
            f"the comparison below would be vacuous"
        )

        coefficients = factory()
        mismatches = []
        for (n, m), values in sorted(official.items()):
            ours = (
                coefficients.g[n, m],
                coefficients.h[n, m],
                coefficients.g_dot[n, m],
                coefficients.h_dot[n, m],
            )
            for label, reference, actual in zip(
                ("g", "h", "g_dot", "h_dot"), values, ours
            ):
                if reference != actual:
                    mismatches.append(
                        f"{label}[{n},{m}]: ours={actual} official={reference}"
                    )

        assert not mismatches, (
            f"{name}: {len(mismatches)} coefficient(s) differ from {filename}:\n  "
            + "\n  ".join(mismatches[:20])
        )

    @pytest.mark.parametrize("name,factory,filename,epoch", MODELS, ids=IDS)
    def test_epoch_and_degree_are_declared_correctly(
        self, name, factory, filename, epoch
    ):
        """The epoch drives secular-variation extrapolation, so it must be right.

        A wrong epoch produces a field that is correct at one instant and drifts
        linearly away from truth, which is exactly the failure that looks like a
        modeling error rather than a typo.
        """
        coefficients = factory()
        assert coefficients.epoch == epoch
        assert coefficients.n_max == 12, (
            f"{name}: WMM is a degree-12 model, got n_max={coefficients.n_max}"
        )

    @pytest.mark.parametrize("name,factory,filename,epoch", MODELS, ids=IDS)
    def test_the_file_header_names_the_model_it_is_used_for(
        self, name, factory, filename, epoch
    ):
        """Guard against the two fixtures being swapped.

        Every coefficient assertion above would still pass if both models were
        compared against the same file, so check the file says what it should.
        """
        header = (FIXTURES / filename).read_text(encoding="utf-8").splitlines()[0]
        assert f"{epoch:.1f}" in header
        assert name.replace("WMM", "WMM-") in header, (
            f"{filename} header does not name {name}: {header!r}"
        )


class TestInternationalGeomagneticReferenceField:
    """IGRF-13 at epoch 2020.0, against the IAGA distribution."""

    def test_every_coefficient_matches_the_official_file(self):
        """All 195 (g/h, n, m) rows, main field and secular variation. Exact."""
        official = _read_igrf13_2020(FIXTURES / "igrf13coeffs.txt")
        assert len(official) == 195, (
            f"expected 195 rows for a degree-13 model, read {len(official)} -- "
            f"the fixture or the parser is wrong"
        )

        coefficients = create_igrf13_coefficients()
        mismatches = []
        for (kind, n, m), (main, secular) in sorted(official.items()):
            table = coefficients.g if kind == "g" else coefficients.h
            rate = coefficients.g_dot if kind == "g" else coefficients.h_dot
            if table[n, m] != main:
                mismatches.append(
                    f"{kind}[{n},{m}]: ours={table[n, m]} official={main}"
                )
            if rate[n, m] != secular:
                mismatches.append(
                    f"{kind}_dot[{n},{m}]: ours={rate[n, m]} official={secular}"
                )

        assert not mismatches, (
            f"IGRF-13: {len(mismatches)} value(s) differ from igrf13coeffs.txt:\n  "
            + "\n  ".join(mismatches[:20])
        )

    def test_epoch_and_degree_are_declared_correctly(self):
        coefficients = create_igrf13_coefficients()
        assert coefficients.epoch == 2020.0
        assert coefficients.n_max == 13, (
            f"IGRF-13 is a degree-13 model, got n_max={coefficients.n_max}"
        )

    def test_the_module_singleton_is_the_same_model(self):
        """``igrf`` defaults to a module-level ``IGRF13`` built at import.

        A caller who passes no coefficients gets that object, so it has to hold
        what the factory produces -- otherwise the tested path and the default
        path are different models.
        """
        from pytcl.magnetism.igrf import IGRF13

        fresh = create_igrf13_coefficients()
        np.testing.assert_array_equal(IGRF13.g, fresh.g)
        np.testing.assert_array_equal(IGRF13.h, fresh.h)
        np.testing.assert_array_equal(IGRF13.g_dot, fresh.g_dot)
        np.testing.assert_array_equal(IGRF13.h_dot, fresh.h_dot)
        assert (IGRF13.epoch, IGRF13.n_max) == (fresh.epoch, fresh.n_max)


class TestSharedStructure:
    """Invariants every one of the three models must satisfy.

    These are cheap and they catch a whole parser failure mode that per-value
    comparison against a single file cannot: a table read into the wrong shape
    still matches wherever the two happen to agree.
    """

    FACTORIES = [
        ("WMM2020", create_wmm2020_coefficients),
        ("WMM2025", create_wmm2025_coefficients),
        ("IGRF13", create_igrf13_coefficients),
    ]
    IDS = [name for name, _ in FACTORIES]

    @pytest.mark.parametrize("name,factory", FACTORIES, ids=IDS)
    def test_the_order_zero_sectorial_terms_are_zero(self, name, factory):
        """``h[n,0]`` has no meaning: the m=0 term has no sine component.

        The official files carry 0.0 in that column. A parser that shifted a
        column would put a real coefficient here.
        """
        coefficients = factory()
        assert np.all(coefficients.h[:, 0] == 0.0), (
            f"{name}: h[n,0] is not identically zero, so the sine terms are "
            f"misaligned: {coefficients.h[:, 0]}"
        )
        assert np.all(coefficients.h_dot[:, 0] == 0.0), (
            f"{name}: h_dot[n,0] is not identically zero"
        )

    @pytest.mark.parametrize("name,factory", FACTORIES, ids=IDS)
    def test_no_coefficient_has_order_above_its_degree(self, name, factory):
        """``m > n`` is undefined for a spherical harmonic and must stay empty.

        Everything above the diagonal is padding. A value there means an index
        pair was written transposed.
        """
        coefficients = factory()
        upper = np.triu_indices(coefficients.n_max + 1, k=1)
        for label in ("g", "h", "g_dot", "h_dot"):
            table = getattr(coefficients, label)
            assert np.all(table[upper] == 0.0), (
                f"{name}: {label} has nonzero entries where m > n, so at least "
                f"one (n, m) pair was stored transposed"
            )

    @pytest.mark.parametrize("name,factory", FACTORIES, ids=IDS)
    def test_the_dipole_dominates_and_points_the_right_way(self, name, factory):
        """``g[1,0]`` near -29,500 nT is the axial dipole, and it is negative.

        Its sign is why a compass needle points north. This is the assertion
        that would have failed when the package was 180 degrees wrong, and it
        holds for every geomagnetic model of the modern era.
        """
        coefficients = factory()
        axial_dipole = coefficients.g[1, 0]
        assert -30000.0 < axial_dipole < -29000.0, (
            f"{name}: axial dipole g[1,0] = {axial_dipole} nT is outside the "
            f"range every modern geomagnetic model occupies; a sign error here "
            f"reverses the field"
        )
        assert abs(axial_dipole) > abs(coefficients.g[2, 0]) * 5, (
            f"{name}: the dipole term does not dominate the quadrupole, so the "
            f"degrees are misordered"
        )

    @pytest.mark.parametrize("name,factory", FACTORIES, ids=IDS)
    def test_all_four_tables_are_square_and_sized_to_the_degree(self, name, factory):
        coefficients = factory()
        expected = (coefficients.n_max + 1, coefficients.n_max + 1)
        for label in ("g", "h", "g_dot", "h_dot"):
            assert getattr(coefficients, label).shape == expected, (
                f"{name}: {label} has shape "
                f"{getattr(coefficients, label).shape}, expected {expected}"
            )

    @pytest.mark.parametrize("name,factory", FACTORIES, ids=IDS)
    def test_the_degree_zero_monopole_is_zero(self, name, factory):
        """There are no magnetic monopoles, so degree 0 carries no field."""
        coefficients = factory()
        for label in ("g", "h", "g_dot", "h_dot"):
            assert getattr(coefficients, label)[0, 0] == 0.0, (
                f"{name}: {label}[0,0] is nonzero, which would be a magnetic monopole"
            )


def test_the_three_models_are_distinct():
    """A copy-paste between factories would make two of them identical.

    WMM-2020, WMM-2025 and IGRF-13 are different models at different epochs.
    Every assertion above is per-model and would still pass if two factories
    returned the same table.
    """
    wmm2020 = create_wmm2020_coefficients()
    wmm2025 = create_wmm2025_coefficients()
    igrf13 = create_igrf13_coefficients()

    assert not np.array_equal(wmm2020.g, wmm2025.g), (
        "WMM-2020 and WMM-2025 returned identical main-field coefficients"
    )
    # IGRF-13 is degree 13 and WMM degree 12, so compare the common block.
    common = slice(0, 13)
    assert not np.array_equal(igrf13.g[common, common], wmm2020.g[common, common]), (
        "IGRF-13 and WMM-2020 returned identical main-field coefficients"
    )
