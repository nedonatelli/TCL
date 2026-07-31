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
each is normalized before comparison rather than asserted against a single
assumed layout.
"""

from collections.abc import Callable
from typing import NamedTuple

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
    _DEFAULT_CACHE_SIZE,
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


class Counters(NamedTuple):
    """One cache's reported state, independent of how the module reports it."""

    hits: int
    misses: int
    currsize: int
    maxsize: int | None


def _read(info) -> list[Counters]:
    """Normalize the three reporting shapes the five caches use.

    ``get_legendre_cache_info`` returns a bare ``CacheInfo``;
    ``get_magnetic_cache_info`` a flat dict of ints; the other three a dict
    keyed by routine, holding either ``CacheInfo`` or a nested dict. Reading
    them in one place keeps every assertion about behavior rather than about
    which shape a particular module happens to use -- and stops the branching
    being re-implemented per test.
    """

    def one(entry) -> Counters:
        if hasattr(entry, "hits"):  # CacheInfo namedtuple
            return Counters(entry.hits, entry.misses, entry.currsize, entry.maxsize)
        return Counters(
            entry["hits"], entry["misses"], entry["currsize"], entry.get("maxsize")
        )

    if hasattr(info, "hits") or "hits" in info:
        return [one(info)]
    return [one(entry) for entry in info.values()]


def _totals(info) -> Counters:
    parts = _read(info)
    return Counters(
        sum(p.hits for p in parts),
        sum(p.misses for p in parts),
        sum(p.currsize for p in parts),
        None,
    )


def _assert_same(first, second, what: str) -> None:
    """Equality that copes with an array, a tuple of floats, or a scalar."""
    if isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second, err_msg=what)
    elif isinstance(first, tuple):
        assert len(first) == len(second), what
        for a, b in zip(first, second):
            np.testing.assert_allclose(a, b, rtol=0, atol=0, err_msg=what)
    else:
        assert first == second, what


class Cache(NamedTuple):
    """A cache and the three published entry points that control it."""

    label: str
    clear: Callable[[], None]
    info: Callable[[], object]
    exercise: Callable[[], object]


CACHES = [
    # magnetic is handled separately: it is the only one with a configure
    # entry point, and its assertions live in its own class below.
    Cache(
        "geodesy",
        clear_geodesy_cache,
        get_geodesy_cache_info,
        lambda: inverse_geodetic(DC[0], DC[1], NYC[0], NYC[1]),
    ),
    Cache(
        "great_circle",
        clear_great_circle_cache,
        get_great_circle_cache_info,
        lambda: great_circle_distance(DC[0], DC[1], NYC[0], NYC[1]),
    ),
    Cache(
        "legendre",
        clear_legendre_cache,
        get_legendre_cache_info,
        lambda: associated_legendre(8, 8, 0.3, normalized=True),
    ),
    Cache(
        "transformation",
        clear_transformation_cache,
        get_transformation_cache_info,
        # J2000.0 plus a decade; the parameter is a Julian date, and 0.24 is not
        # one -- any future input validation would reject it.
        lambda: precession_matrix_iau76(2455197.5),
    ),
]
IDS = [c.label for c in CACHES]


@pytest.fixture(autouse=True)
def _clean_caches():
    """Every test starts and ends cold, so ordering cannot affect a result."""
    for cache in CACHES:
        cache.clear()
    clear_magnetic_cache()
    yield
    for cache in CACHES:
        cache.clear()
    clear_magnetic_cache()


@pytest.mark.parametrize("cache", CACHES, ids=IDS)
def test_a_cleared_cache_reports_empty(cache):
    cache.exercise()
    assert _totals(cache.info()).currsize > 0, (
        f"{cache.label}: nothing was cached to begin with"
    )

    cache.clear()
    state = _totals(cache.info())
    assert (state.hits, state.misses, state.currsize) == (0, 0, 0), (
        f"{cache.label}: after clearing, the cache still reports "
        f"hits={state.hits} misses={state.misses} currsize={state.currsize}"
    )


@pytest.mark.parametrize("cache", CACHES, ids=IDS)
def test_first_call_misses_and_repeat_call_hits(cache):
    """Occupancy counters must track what actually happened."""
    cache.exercise()
    state = _totals(cache.info())
    assert (state.hits, state.misses) == (0, 1), (
        f"{cache.label}: first call should be a miss"
    )
    assert state.currsize == 1

    cache.exercise()
    state = _totals(cache.info())
    assert (state.hits, state.misses) == (1, 1), (
        f"{cache.label}: repeat call should be a hit"
    )
    assert state.currsize == 1, f"{cache.label}: a repeat must not add a second entry"


@pytest.mark.parametrize("cache", CACHES, ids=IDS)
def test_cold_and_warm_results_are_identical(cache):
    """The property that matters: a cache must not change the answer.

    Catches a cache returning a stale value, and one keyed too loosely so that
    a different input collides with a stored one. Neither shows up in the
    occupancy counters the other tests here read.
    """
    cache.clear()
    cold = cache.exercise()
    warm = cache.exercise()
    _assert_same(
        cold, warm, f"{cache.label}: the warm result differs from the cold one"
    )

    cache.clear()
    recomputed = cache.exercise()
    _assert_same(
        cold, recomputed, f"{cache.label}: recomputing after a clear changed the result"
    )


@pytest.mark.parametrize("cache", CACHES, ids=IDS)
def test_clearing_twice_is_harmless(cache):
    """Clearing an already-empty cache must not raise or corrupt state."""
    cache.clear()
    cache.clear()
    state = _totals(cache.info())
    assert (state.hits, state.misses, state.currsize) == (0, 0, 0)
    cache.exercise()  # still usable afterwards
    assert _totals(cache.info()).currsize == 1


@pytest.mark.parametrize("cache", CACHES, ids=IDS)
def test_reported_capacity_is_positive(cache):
    """A maxsize of 0 would mean nothing is ever retained."""
    sizes = [part.maxsize for part in _read(cache.info())]
    assert sizes and all(s is None or s > 0 for s in sizes), (
        f"{cache.label}: reported maxsize {sizes} would retain nothing"
    )


class TestMagneticCacheConfiguration:
    """The magnetic cache is the only one with a configuration entry point."""

    def test_configure_changes_the_reported_capacity(self):
        configure_magnetic_cache(maxsize=32)
        try:
            assert get_magnetic_cache_info()["maxsize"] == 32
        finally:
            configure_magnetic_cache(maxsize=_DEFAULT_CACHE_SIZE)
        assert get_magnetic_cache_info()["maxsize"] == _DEFAULT_CACHE_SIZE

    def test_reconfiguring_starts_from_empty(self):
        """Resizing must not leave entries counted against the new capacity."""
        configure_magnetic_cache(maxsize=16)
        try:
            state = _totals(get_magnetic_cache_info())
            assert (state.hits, state.misses, state.currsize) == (0, 0, 0)
        finally:
            configure_magnetic_cache(maxsize=_DEFAULT_CACHE_SIZE)

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
    for entry in CACHES:
        entry.exercise()

    populated = [c.label for c in CACHES if _totals(c.info()).currsize > 0]
    assert len(populated) == len(CACHES), f"only {populated} were populated"

    # clear one, and only one, and confirm the others are untouched
    clear_geodesy_cache()
    assert _totals(get_geodesy_cache_info()).currsize == 0
    for entry in CACHES:
        if entry.label == "geodesy":
            continue
        assert _totals(entry.info()).currsize > 0, (
            f"clearing the geodesy cache also emptied {entry.label}"
        )
