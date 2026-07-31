"""A cached loader must not let one caller corrupt another's data.

Two callers that load the same terrain region received the *same* ``DEMGrid``,
and either could write to its array. Against the real GEBCO file:

    same DEMGrid object handed to both callers: True
    caller B's grid at [0, 0]: 1287.0 -> -99999.0

Caller A did nothing unusual -- it wrote to an array it was handed. Nothing in
the signature or the type said that array was shared. There was no exception:
caller B carried on computing viewsheds and line-of-sight on terrain reading
-99999 m where the ground is at 1287 m, and because the corruption depends on
which caller ran first, the same pipeline could differ between runs. See gh-51.

The invariant asserted here is about **what a caller can observe**, not about
how the guarantee is achieved: *a caller cannot change what another caller
sees*. A loader satisfies it by returning an immutable type, by returning a
fresh object each call, or by returning a read-only view. Phrasing it that way
means the test survives a change of mechanism, and it also describes the caches
that were already safe.

Scope, from an audit of all 29 cached functions in the library: the exposure is
cached *loaders returning a mutable container*. Most cached routines return a
float, int, bool or tuple. The ones whose annotations mention arrays --
associated Legendre, the precession and nutation matrices, the Cholesky update
-- cache nested tuples and rebuild per call, so they were never at risk. Test
object identity rather than reading the type hints.
"""

import numpy as np
import pytest

from pytcl.core.exceptions import DependencyError

# Loaders that hold a mutable container behind a cache. Each entry is
# (label, a zero-argument callable, the attribute holding the mutable payload).
# The data files are too large for CI, so a loader whose inputs are absent is
# reported as skipped rather than silently passing.
_REGION = dict(
    lat_min=np.radians(35.0),
    lat_max=np.radians(35.2),
    lon_min=np.radians(-120.0),
    lon_max=np.radians(-119.8),
)

_data_skip = (FileNotFoundError, DependencyError)


def _gebco():
    from pytcl.terrain.loaders import load_gebco

    return load_gebco(**_REGION)


def _earth2014():
    from pytcl.terrain.loaders import load_earth2014

    return load_earth2014(layer="SUR", **_REGION)


def _egm():
    from pytcl.gravity.egm import load_egm_coefficients

    return load_egm_coefficients(model="EGM96", n_max=8)


def _emm():
    from pytcl.magnetism.emm import load_emm_coefficients

    return load_emm_coefficients(model="WMMHR2025", n_max=8)


CACHED_LOADERS = [
    ("gebco", _gebco, "data"),
    ("earth2014", _earth2014, "data"),
    ("egm", _egm, "C"),
    ("emm", _emm, "g"),
]
IDS = [name for name, _, _ in CACHED_LOADERS]


def _load_twice(loader):
    """Two independent loads, or a skip naming what was unavailable."""
    try:
        return loader(), loader()
    except _data_skip as exc:
        pytest.skip(f"data file unavailable: {type(exc).__name__}: {exc}")


@pytest.mark.parametrize("name,loader,payload", CACHED_LOADERS, ids=IDS)
def test_one_caller_cannot_corrupt_another(name, loader, payload):
    """The invariant. Writing to what you were handed must not affect anyone else.

    Either the write raises -- because the shared array is read-only -- or the
    two callers hold independent arrays. Both satisfy the contract; silently
    accepting the write and changing the other caller's data does not.
    """
    first, second = _load_twice(loader)
    a = getattr(first, payload)
    b = getattr(second, payload)

    original = float(b[0, 0])
    sentinel = original - 12345.0
    corrupted = (
        f"{name}: writing to one caller's data changed another caller's from "
        f"{original} to {{}}. A cached loader must not hand out a shared "
        f"mutable array."
    )

    if not a.flags.writeable:
        # Satisfied by refusing the write. Assert it is refused loudly, and
        # that the other caller is unaffected -- checking only that a
        # ValueError came out would pass for an array that raised after
        # writing.
        with pytest.raises(ValueError, match="read-only"):
            a[0, 0] = sentinel
        assert float(b[0, 0]) == original, corrupted.format(b[0, 0])
        return

    # Satisfied instead by the two callers holding independent arrays.
    try:
        a[0, 0] = sentinel
        assert float(b[0, 0]) == original, corrupted.format(b[0, 0])
    finally:
        a[0, 0] = original


@pytest.mark.parametrize("name,loader,payload", CACHED_LOADERS, ids=IDS)
def test_the_cache_is_actually_being_used(name, loader, payload):
    """Guard against the invariant holding for the wrong reason.

    If the loader stopped caching altogether the corruption test would pass
    trivially -- two callers cannot alias what was never shared -- while the
    cache's purpose was lost. So assert object identity: comparing the arrays
    equal would also hold for a non-caching loader that re-read the file, which
    is precisely the case this is here to exclude.
    """
    first, second = _load_twice(loader)
    assert first is second, (
        f"{name}: two loads of the same region returned different objects, so "
        f"the loader is no longer caching and the aliasing invariant above "
        f"holds vacuously"
    )
    assert getattr(first, payload) is getattr(second, payload)


@pytest.mark.parametrize("name,loader,payload", CACHED_LOADERS, ids=IDS)
def test_a_caller_can_still_obtain_a_writable_grid(name, loader, payload):
    """Read-only sharing must not block legitimate modification.

    A caller that needs to change the data copies it. That must work, and the
    copy must be independent of the cached original.
    """
    first, _ = _load_twice(loader)
    original = getattr(first, payload)

    working = original.copy()
    assert working.flags.writeable, (
        f"{name}: a copy of the grid data is not writable, so a caller has no "
        f"way to modify terrain at all"
    )

    before = float(original[0, 0])
    working[0, 0] = before - 999.0
    assert float(original[0, 0]) == before, (
        f"{name}: writing to a copy changed the cached original"
    )


class TestMakeReadonly:
    """The mechanism itself, which needs no data files.

    The loader tests above skip wherever their multi-gigabyte inputs are
    absent, which in CI is all of them. These run everywhere, so the guarantee
    is enforced rather than merely enforceable on a machine that happens to
    have 7 GB of terrain.
    """

    def test_marks_an_array_unwritable(self):
        from pytcl.core.array_utils import make_readonly

        array = np.array([1.0, 2.0, 3.0])
        assert array.flags.writeable
        make_readonly(array)
        assert not array.flags.writeable

    def test_a_write_raises_rather_than_being_ignored(self):
        """The whole point: refuse loudly instead of silently accepting."""
        from pytcl.core.array_utils import make_readonly

        array = np.array([[1.0, 2.0], [3.0, 4.0]])
        make_readonly(array)
        with pytest.raises(ValueError, match="read-only"):
            array[0, 0] = 99.0
        assert array[0, 0] == 1.0

    def test_marks_several_arrays_at_once(self):
        """Coefficient sets carry four related arrays; marking is per call."""
        from pytcl.core.array_utils import make_readonly

        a, b, c = (np.ones(2) for _ in range(3))
        make_readonly(a, b, c)
        assert not any(x.flags.writeable for x in (a, b, c))

    def test_a_copy_of_a_readonly_array_is_writable(self):
        """A caller that needs to modify the data must have a way to."""
        from pytcl.core.array_utils import make_readonly

        shared = np.array([1.0, 2.0, 3.0])
        make_readonly(shared)

        working = shared.copy()
        assert working.flags.writeable
        working[0] = 99.0
        assert shared[0] == 1.0, "writing to the copy changed the original"

    def test_reading_and_arithmetic_still_work(self):
        """Read-only restricts writes, not use."""
        from pytcl.core.array_utils import make_readonly

        array = np.array([1.0, 2.0, 3.0])
        make_readonly(array)

        assert float(array.sum()) == 6.0
        assert float((array * 2).sum()) == 12.0  # produces a new array
        np.testing.assert_array_equal(array[1:], [2.0, 3.0])

    def test_marking_twice_is_harmless(self):
        from pytcl.core.array_utils import make_readonly

        array = np.ones(3)
        make_readonly(array)
        make_readonly(array)
        assert not array.flags.writeable


def test_at_least_one_loader_was_exercised_or_all_were_reported_skipped():
    """A fully-skipped run must not read as a passing one.

    Every loader here is data-gated. If the files are absent the invariant is
    unverified, and that should be visible in the run rather than inferred from
    a green tick.
    """
    exercised = []
    for name, loader, _ in CACHED_LOADERS:
        try:
            loader()
            exercised.append(name)
        except _data_skip:
            continue

    if not exercised:
        pytest.skip(
            "no cached loader could run: none of the data files are present, so "
            "the aliasing invariant is unverified here. TestMakeReadonly covers "
            "the mechanism."
        )
    assert exercised
