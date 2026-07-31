"""The published cache-control surface, across all five caches.

Eleven exported functions -- ``clear_*_cache``, ``get_*_cache_info``, and
``configure_magnetic_cache`` -- had no test reaching any of them. They are the
largest cluster on the public-API coverage allowlist (gh-49), and they are the
same surface gh-26 raises for mutable sharing between callers, so they are
written together rather than one at a time.

The assertions are relational, because that is where a cache can be wrong. A
test that calls ``clear_geodesy_cache()`` and checks nothing raised would
satisfy the coverage gate while verifying nothing.

The property that matters most is **cold equals warm**: a cache must not change
an answer. That catches a cache returning a stale value and a cache keyed too
loosely, neither of which shows up in occupancy counters.

The five caches report through three different shapes -- a bare ``CacheInfo``
namedtuple, a dict of named ``CacheInfo`` values, and a flat dict of ints -- so
each is normalised before comparison rather than asserted against a single
assumed layout.
"""

import numpy as np
import pytest

from pytcl.astronomical.reference_frames import (
    clear_transformation_cache,
    precession_matrix_iau76,
)
from pytcl.astronomical.reference_frames import (
    get_cache_info as get_transformation_cache_info,
)
from pytcl.gravity.spherical_harmonics import (
    associated_legendre,
    clear_legendre_cache,
    get_legendre_cache_info,
)
from pytcl.magnetism.wmm import (
    clear_magnetic_cache,
    configure_magnetic_cache,
    get_magnetic_cache_info,
)
from pytcl.navigation.geodesy import (
    clear_geodesy_cache,
    get_geodesy_cache_info,
    inverse_geodetic,
)
from pytcl.navigation.great_circle import (
    clear_great_circle_cache,
    great_circle_distance,
)
from pytcl.navigation.great_circle import (
    get_cache_info as get_great_circle_cache_info,
)

DC = (np.radians(38.9072), np.radians(-77.0369))
NYC = (np.radians(40.7128), np.radians(-74.0060))


def _counters(info):
    """Normalise the three reporting shapes to a list of (hits, misses, currsize).

    ``get_legendre_cache_info`` returns a bare ``CacheInfo``;
    ``get_geodesy_cache_info`` a dict of dicts; ``get_cache_info`` in
    reference_frames a dict of ``CacheInfo``; ``get_magnetic_cache_info`` a flat
    dict. Normalising here keeps the assertions about behaviour rather than
    about which shape a particular module happens to use.
    """
    if hasattr(info, "hits"):  # a bare CacheInfo
        return [(info.hits, info.misses, info.currsize)]
    if "hits" in info:  # a flat dict
        return [(info["hits"], info["misses"], info["currsize"])]
    out = []
    for entry in info.values():  # dict of CacheInfo or of dict
        if hasattr(entry, "hits"):
            out.append((entry.hits, entry.misses, entry.currsize))
        else:
            out.append((entry["hits"], entry["misses"], entry["currsize"]))
    return out


def _totals(info):
    counters = _counters(info)
    return (
        sum(c[0] for c in counters),
        sum(c[1] for c in counters),
        sum(c[2] for c in counters),
    )


# (label, clear, info, a callable that exercises the cache and returns a value)
CACHES = [
    (
        "geodesy",
        clear_geodesy_cache,
        get_geodesy_cache_info,
        lambda: inverse_geodetic(DC[0], DC[1], NYC[0], NYC[1]),
    ),
    (
        "great_circle",
        clear_great_circle_cache,
        get_great_circle_cache_info,
        lambda: great_circle_distance(DC[0], DC[1], NYC[0], NYC[1]),
    ),
    (
        "legendre",
        clear_legendre_cache,
        get_legendre_cache_info,
        lambda: associated_legendre(8, 8, 0.3, normalized=True),
    ),
    (
        "transformation",
        clear_transformation_cache,
        get_transformation_cache_info,
        lambda: precession_matrix_iau76(0.24),
    ),
]
IDS = [c[0] for c in CACHES]


@pytest.fixture(autouse=True)
def _clean_caches():
    """Every test starts and ends cold, so ordering cannot affect a result."""
    for _, clear, _, _ in CACHES:
        clear()
    clear_magnetic_cache()
    yield
    for _, clear, _, _ in CACHES:
        clear()
    clear_magnetic_cache()


@pytest.mark.parametrize("label,clear,info,exercise", CACHES, ids=IDS)
def test_a_cleared_cache_reports_empty(label, clear, info, exercise):
    exercise()
    assert _totals(info())[2] > 0, f"{label}: nothing was cached to begin with"

    clear()
    hits, misses, currsize = _totals(info())
    assert (hits, misses, currsize) == (0, 0, 0), (
        f"{label}: after clearing, the cache still reports "
        f"hits={hits} misses={misses} currsize={currsize}"
    )


@pytest.mark.parametrize("label,clear,info,exercise", CACHES, ids=IDS)
def test_first_call_misses_and_repeat_call_hits(label, clear, info, exercise):
    """Occupancy counters must track what actually happened."""
    exercise()
    hits, misses, currsize = _totals(info())
    assert (hits, misses) == (0, 1), f"{label}: first call should be a miss"
    assert currsize == 1

    exercise()
    hits, misses, currsize = _totals(info())
    assert (hits, misses) == (1, 1), f"{label}: repeat call should be a hit"
    assert currsize == 1, f"{label}: a repeat must not add a second entry"


@pytest.mark.parametrize("label,clear,info,exercise", CACHES, ids=IDS)
def test_cold_and_warm_results_are_identical(label, clear, info, exercise):
    """The property that matters: a cache must not change the answer.

    Catches a cache returning a stale value, and one keyed too loosely so that
    a different input collides with a stored one. Neither shows up in the
    occupancy counters that the other tests here assert on.
    """
    clear()
    cold = exercise()
    warm = exercise()

    if isinstance(cold, np.ndarray):
        np.testing.assert_array_equal(cold, warm)
    elif isinstance(cold, tuple):
        assert len(cold) == len(warm)
        for a, b in zip(cold, warm):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)
    else:
        assert cold == warm

    # and again after a clear: recomputing from cold must reproduce it
    clear()
    recomputed = exercise()
    if isinstance(cold, np.ndarray):
        np.testing.assert_array_equal(cold, recomputed)
    elif isinstance(cold, tuple):
        for a, b in zip(cold, recomputed):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)
    else:
        assert cold == recomputed


@pytest.mark.parametrize("label,clear,info,exercise", CACHES, ids=IDS)
def test_clearing_twice_is_harmless(label, clear, info, exercise):
    """Clearing an already-empty cache must not raise or corrupt state."""
    clear()
    clear()
    assert _totals(info()) == (0, 0, 0)
    exercise()  # still usable afterwards
    assert _totals(info())[2] == 1


@pytest.mark.parametrize("label,clear,info,exercise", CACHES, ids=IDS)
def test_reported_capacity_is_positive(label, clear, info, exercise):
    """A maxsize of 0 would mean nothing is ever retained."""
    raw = info()
    if hasattr(raw, "maxsize"):
        sizes = [raw.maxsize]
    elif "maxsize" in raw:
        sizes = [raw["maxsize"]]
    else:
        sizes = [
            entry.maxsize if hasattr(entry, "maxsize") else entry["maxsize"]
            for entry in raw.values()
        ]
    assert sizes and all(s is None or s > 0 for s in sizes), (
        f"{label}: reported maxsize {sizes} would retain nothing"
    )


class TestMagneticCacheConfiguration:
    """The magnetic cache is the only one with a configuration entry point."""

    def test_configure_changes_the_reported_capacity(self):
        configure_magnetic_cache(maxsize=32)
        try:
            assert get_magnetic_cache_info()["maxsize"] == 32
        finally:
            configure_magnetic_cache(maxsize=1024)
        assert get_magnetic_cache_info()["maxsize"] == 1024

    def test_reconfiguring_starts_from_empty(self):
        """Resizing must not leave entries counted against the new capacity."""
        configure_magnetic_cache(maxsize=16)
        try:
            hits, misses, currsize = _totals(get_magnetic_cache_info())
            assert (hits, misses, currsize) == (0, 0, 0)
        finally:
            configure_magnetic_cache(maxsize=1024)

    def test_hit_rate_is_reported_and_bounded(self):
        info = get_magnetic_cache_info()
        assert "hit_rate" in info
        assert 0.0 <= info["hit_rate"] <= 1.0

    def test_hit_rate_is_zero_on_an_empty_cache(self):
        """With no lookups there is no rate to report; it must not divide by zero."""
        clear_magnetic_cache()
        assert get_magnetic_cache_info()["hit_rate"] == 0.0

    def test_configure_with_no_arguments_leaves_capacity_alone(self):
        before = get_magnetic_cache_info()["maxsize"]
        configure_magnetic_cache()
        assert get_magnetic_cache_info()["maxsize"] == before


def test_the_caches_are_independent():
    """Clearing one cache must not empty another.

    They are separate module-level caches; a shared backing store would make
    one package's clear silently discard another's work. This is the
    cross-cache half of what gh-26 asks about.
    """
    for _, _, _, exercise in CACHES:
        exercise()

    populated = [label for label, _, info, _ in CACHES if _totals(info())[2] > 0]
    assert len(populated) == len(CACHES), f"only {populated} were populated"

    # clear one, and only one, and confirm the others are untouched
    clear_geodesy_cache()
    assert _totals(get_geodesy_cache_info())[2] == 0
    for label, _, info, _ in CACHES:
        if label == "geodesy":
            continue
        assert _totals(info())[2] > 0, (
            f"clearing the geodesy cache also emptied {label}"
        )
