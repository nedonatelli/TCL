"""
Region-cubature point sets: bounded-domain integration, true-measure weights.

Unlike :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`
(which targets the Gaussian weight N(0, I) and normalizes ``sum(w) == 1`` for
direct use as ``E[f(X)] ~= sum_i w_i f(x_i)``), every generator in this module
targets a bounded geometric region with the *Lebesgue* weight ``1`` and
reports the region's TRUE measure: weights sum to the region's volume, not to
1. This module currently covers the ``Cube_Space`` region (``[-1, 1]^n``,
``sum(w) == 2**n``) and the ``Simplex`` region (the standard n-simplex
``{x >= 0, sum(x) <= 1}``, ``sum(w) == 1/n!``). A caller wanting the
probability-normalized version divides by the returned volume
(``weights / weights.sum()``), which is always safe; going the other way
(recovering the true measure from a pre-normalized rule) is not possible
without already knowing the volume. See
``docs/superpowers/specs/2026-08-16-region-cubature-design.md`` Section 1 for
the full rationale and the conventions pinned for the two region families
(``Sphere``, ``Spherical_Surface``) not yet ported here.

Ported from the Tracker Component Library's ``Cubature_Points/Cube_Space``
and ``Cubature_Points/Simplex`` collections (top-level, general-dimension
files only; the fixed-``n=2``/``n=3`` ``Cube/``, ``Square/``, ``Tetrahedra/``,
and ``Triangles/`` subdirectories are out of scope for this module -- see the
design spec's Section 8). Within each ported file, only the general-dimension
algorithms (and, for ``seventhOrderNDimCubPoints.m``, the one algorithm
MATLAB itself defaults to per dimension, since that file has no general-``n``
algorithm at all) are transcribed; the remaining fixed-``n`` literature
variants each file also offers are deferred for the same reason the
``Cube``/``Square``/``Tetrahedra``/``Triangles`` subdirectories are (no
current pytcl consumer, narrow value versus the general-``n`` rule already
covered). ``thirdOrderSimplexCubPoints.m``'s two fixed-dimension algorithms
(10, ``n=2``; 11, ``n=5``) fall into this same deferred category even though
the file is otherwise general-``n`` -- ten other algorithms in that same file
already cover general ``n`` at degree 3, so the fixed-dimension pair adds no
degree/dimension coverage a general-``n`` algorithm doesn't already provide.

**Two corrected MATLAB defects (both verified by direct inspection of the
pinned source, not merely accuracy checks).** These are provable
dimension-mismatch bugs, not degree-exactness disputes: the affected
lines, run in real MATLAB, produce a ``points``/``weights`` pair whose
sizes do not match each other:

1. ``firstOrderNDimCubPoints.m`` algorithm 1 (the file's *default*
   algorithm, ``2**n`` points): the source sets
   ``w = 1/2^numDim*ones(numDim,1)*V`` -- an ``numDim``-length vector --
   against ``xi = PMCombos(ones(numDim,1))``, which is ``numDim x 2^numDim``.
   Except at ``numDim == 2^numDim`` (never, for integer ``numDim >= 1``),
   this is a straight shape mismatch: MATLAB's ``[xi.', w]`` horizontal
   concatenation (as ``scripts/matlab_capture/capture_region_rules.m`` uses)
   errors outright rather than silently returning a wrong answer. The
   evidently-intended formula -- uniform weight ``V / 2**n`` on each of the
   ``2**n`` vertices, the natural equal-weight vertex rule and the only
   choice consistent with ``sum(w) == V`` -- is what :func:`cube_cubature_points`
   implements for ``degree=1, algorithm=1``.
2. ``thirdOrderNDimCubPoints.m`` algorithm 0 (the file's default algorithm,
   ``2*n`` points, "with the correction listed in Table I of [2]" per its
   own docstring -- a *different*, already-applied correction, orthogonal to
   this bug) at odd ``n``: every other assignment in the branch indexes
   columns directly with ``i = 1:(2*numDim)``, but the odd-``n`` row uses
   ``xi(numDim, i+1)`` -- shifted by one, so column ``2*numDim + 1`` is
   referenced (growing ``xi`` past its declared ``2*numDim`` columns) while
   column 1 of that row is left at its zero-initialized default. ``w`` stays
   fixed at ``2*numDim`` entries (computed earlier, unaffected by the
   growth), so the same shape mismatch results. This bites the exact
   dimensions (``n = 3, 5``) the design spec's capture-case sweep
   (``n = 2..5``) would otherwise exercise -- see the note in
   ``scripts/matlab_capture/capture_region_rules.m``. The correction used
   here drops the ``+1`` (``xi[n-1, :] = (-1)**i / sqrt(3)`` for
   ``i = 1..2*n``, matching every other assignment in the same branch);
   verified against the degree-3 monomial oracle in
   ``tests/unit/test_region_cubature.py``.

Both corrections were checked against every monomial of total degree <= the
rule's claimed degree using this module's own closed-form cube-integral
oracle (see ``tests/unit/test_region_cubature.py``'s
``cube_monomial_integral``), not merely assumed from the "obvious fix."

**A related, non-crashing point-count discrepancy (``ninthOrderNDimCubPoints.m``,
not counted as a defect above).** MATLAB's own comment there reads "The
formula for the number of points as given in the text is incorrect" --
referring to Stroud's *published* point-count formula
``4*(n**4-5*n**3+14*n**2-7*n+3)/3``, which MATLAB nonetheless uses to
preallocate ``xi``/``w``. The construction that follows fills strictly
fewer entries (e.g. 177, not 180, at ``n=4``): the true count, from the
actual ``fullSymPerms`` block sizes, is
``1 + 4*n + 16*C(n,2) + 8*C(n,3) + 32*C(n,4)``. Because MATLAB only ever
*indexes into* the preallocated arrays (never reads past what it wrote),
the unfilled trailing slots stay at their zero-initialized default --
harmless, zero-weight duplicate origin points, not a shape mismatch like
defects 1-2 above. :func:`cube_cubature_points` builds only the true,
non-redundant point set (the formula above), one degenerate-point-free
rule with the same integral, not MATLAB's padded 180.

**``ninthOrderNDimCubPoints.m``'s two algorithms are genuinely different
rules, not a relabeling.** ``degree=9``'s shared quartic
(``I2*I6 - I4^2`` etc.) has two roots; algorithm 0 ("variant 1" per
MATLAB's own comment) binds the smaller root to ``u`` and the larger to
``v``, algorithm 1 ("variant 2") binds them the other way. Because the
downstream formulas (``B`` through ``J``) and the point patterns they
weight (e.g. the ``(v, v, v, 0, ...)`` block always carries weight ``H``)
are written in terms of the ``u``/``v`` *symbols*, not "the larger/smaller
root," and are NOT symmetric under exchanging which root each symbol
names (``D``'s formula is not ``E``'s formula with ``u`` and ``v``
swapped), the two algorithms place that triple-repeated block at
different radii -- algorithm 0's at the larger root (~0.90618 for
``n=4``), algorithm 1's at the smaller (~0.53847) -- and are not equal as
point sets. Both are independently verified degree-9 exact (measured
worst-monomial error ~1e-13 for both, at ``n`` = 4, 5, 6); see
:func:`cube_cubature_points`'s ``algorithm`` parameter and
``tests/unit/test_region_cubature.py``'s ``TestCubeDegree9`` for both.

**A third corrected MATLAB defect, found in the ``Simplex`` port
(``thirdOrderSimplexCubPoints.m`` algorithm 3, formula T_n 3-4).** This one
is a NaN-poisoning indeterminate form, not a shape mismatch -- verified
exactly with symbolic algebra (not floating-point approximation), so it is
not a case of "the numbers looked wrong": at ``n == 2`` the file's own
parameter-defining cubic
(``2*(n-2)*(n+1)*(n+3)*r**3 - (5*n**2+5*n-18)*r**2 + 4*n*r - 1 == 0``) has
two real roots, ``r = 1/6`` and ``r = 1/2``, and BOTH make the weight
formula's numerator ``(n - 2)`` and its denominator
``(1 - 2*n*r**2 - 2*(1 - n*r)**2)`` simultaneously and exactly zero --
confirmed by direct symbolic substitution, not a near-zero measured
residual. ``0.0 / 0.0`` is ``NaN`` in IEEE 754 arithmetic; real MATLAB run at
``numDim == 2`` with this algorithm would return a weight vector poisoned
with ``NaN`` at every entry that touches ``B``, not merely an inaccurate
answer. MATLAB's own domain guard (``numDim < 7``) does not exclude
``numDim == 2``, so nothing in the source stops this from happening. Rather
than invent a limiting formula Stroud's text does not supply (the
`port-fidelity-over-invention` convention's bar), :func:`simplex_cubature_points`
hardens the guard to ``3 <= n < 7`` for this algorithm -- consistent with a
dedicated ``n=2``, third-order simplex formula already existing elsewhere in
the same file (algorithm 10, not ported here; see the "deferred" note above),
which is the natural reading of why Stroud's T_n 3-4 was likely never meant
to cover ``n == 2`` in the first place.

**A related, non-defect root-selection choice (same algorithm).** MATLAB's
own comment says "we just choose the first real root," which is an artifact
of whatever order its ``roots()`` builtin happens to return real roots in --
not independently reproducible without a live MATLAB session, and this
algorithm has no MATLAB fixture capture to check it against (spec Section
6's capture case list only exercises algorithm 0). At ``n == 5`` the same
cubic has a genuine algebraic double root at ``r = 1/6`` alongside a simple
root at ``r = 1/8`` (confirmed symbolically); both independently satisfy the
degree-3 monomial oracle (measured worst-monomial error ~1e-18 for either
choice). :func:`simplex_cubature_points` deterministically selects the
SMALLEST real root at every supported ``n`` (a well-defined, reproducible
tie-break, verified exact via the oracle at every ``n`` it ported this
algorithm for -- not merely assumed to match whichever root MATLAB's own
run happened to pick).

**``fourthOrderSimplexCubPoints.m``'s own zero-weight stripping (not a
defect -- already handled by MATLAB itself).** Unlike ``ninthOrderNDimCubPoints.m``
(module docstring above), this file explicitly strips zero-weight points
before returning (``sel=~(w==0); w=w(sel); xi=xi(:,sel);``). Its ``B4``
weight term carries an exact factor of ``(4 - n)``, so at ``n == 4`` (and
only there) one whole point block (30 points, computed with valid
coordinates but identically zero weight) is dropped, reducing the point
count from the general formula ``C(n+4,4)`` (70 at ``n=4``) down to 40.
:func:`simplex_cubature_points` reproduces this exact-equality filter for
degree 4, matching MATLAB's own behavior rather than MATLAB's padding
artifact from the ninth-order cube case (there was no port defect to fix
here; both sides already agree by construction).

References
----------
A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
Prentice-Hall, 1971.

R. Cools, "An encyclopedia of cubature formulas," Journal of Complexity,
vol. 19, no. 3, pp. 445-453, Jun. 2003.

D. F. Crouse, "The Tracker Component Library," IEEE AESS Magazine, 2017.
"""

import itertools
import math
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


def _pm_combos(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """All sign flips of the nonzero entries of x (MATLAB's PMCombos)."""
    x = np.asarray(x, dtype=np.float64)
    nz = np.flatnonzero(x)
    out = []
    for signs in itertools.product((1.0, -1.0), repeat=len(nz)):
        p = x.copy()
        p[nz] = x[nz] * np.array(signs)
        out.append(p)
    return np.array(out)


def _multiset_perms(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """All distinct permutations of x, no sign flips (MATLAB's
    genAllMultisetPermutations)."""
    x = np.asarray(x, dtype=np.float64)
    perms = sorted(set(itertools.permutations(x.tolist())))
    return np.array(perms)


def _full_sym_perms(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Every distinct permutation of the multiset of entries of x, each
    signed with every sign combination of its nonzero entries (MATLAB's
    fullSymPerms)."""
    x = np.asarray(x, dtype=np.float64)
    if np.all(x == 0):
        return x.reshape(1, -1)
    perms = set(itertools.permutations(x.tolist()))
    return np.vstack([_pm_combos(np.array(p)) for p in perms])


def _tensor_grid_rule(
    nodes: NDArray[np.floating], weights: NDArray[np.floating], n: int
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """n-fold tensor product of a 1-D rule (nodes, weights) over [-1, 1]^n."""
    nodes = np.asarray(nodes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    idx = np.array(list(itertools.product(range(len(nodes)), repeat=n)))
    points = nodes[idx]
    w = np.prod(weights[idx], axis=1)
    return points, w


def _simplex_bary_block(tail: NDArray[np.floating], n: int) -> NDArray[np.floating]:
    """Every ``Simplex`` algorithm below builds its point blocks the same
    way: an (n+1)-length "extended" barycentric-style vector (``tail``) is
    passed to MATLAB's ``genAllMultisetPermutations``, and only the first
    ``n`` coordinates of each unique permutation are kept -- the implicit
    ``(n+1)``th coordinate is ``1 - sum(other coords)``, dropped because the
    simplex's own points are represented in ``R^n``, not the (n+1)-length
    barycentric form. ``_multiset_perms`` (used for ``Cube_Space`` above)
    already computes exactly MATLAB's "every unique permutation of a
    multiset" semantics, so it is reused here rather than re-implemented."""
    return _multiset_perms(np.asarray(tail, dtype=np.float64))[:, :n]


def _cube_degree1(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 1
    V = 2.0**n
    if algorithm == 0:  # Cn 1-1, 1 point.
        return np.zeros((1, n)), np.array([V])
    if algorithm == 1:  # Cn 1-2, 2^n points. See module docstring, defect 1.
        points = _pm_combos(np.ones(n))
        weights = np.full(points.shape[0], V / 2.0**n)
        return points, weights
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 1 (general-n); expected 0 or 1"
    )


def _cube_degree2(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = 2.0**n
    if algorithm == 0:  # Cn 2-1, n+1 points.
        num_pts = n + 1
        i = np.arange(num_pts)
        xi = np.zeros((n, num_pts))
        for k in range(1, n // 2 + 1):
            xi[2 * k - 2, :] = np.sqrt(2.0 / 3.0) * np.cos(2 * i * k * np.pi / (n + 1))
            xi[2 * k - 1, :] = np.sqrt(2.0 / 3.0) * np.sin(2 * i * k * np.pi / (n + 1))
        if n % 2 != 0:
            xi[n - 1, :] = (-1.0) ** i / np.sqrt(3.0)
        w = np.full(num_pts, V / (n + 1))
        return xi.T, w
    if algorithm == 1:  # Cn 2-2, 2n+1 points.
        r = np.sqrt(3.0) / 6.0
        B1, B2, B3 = V, -r * V, r * V
        p0 = (2.0 * r) * np.ones((1, n))
        base1 = np.concatenate([[1.0], r * np.ones(n - 1)])
        base2 = np.concatenate([[-1.0], r * np.ones(n - 1)])
        points = np.vstack([p0, _multiset_perms(base1), _multiset_perms(base2)])
        w = np.concatenate([[B1], np.full(n, B2), np.full(n, B3)])
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 2 (general-n); expected 0 or 1"
    )


def _cube_degree3(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = 2.0**n
    if algorithm == 0:  # Cn 3-1 (Table-I corrected), 2n points.
        num_pts = 2 * n
        i = np.arange(1, num_pts + 1)
        xi = np.zeros((n, num_pts))
        for k in range(1, n // 2 + 1):
            xi[2 * k - 2, :] = np.sqrt(2.0 / 3.0) * np.cos((2 * k - 1) * i * np.pi / n)
            xi[2 * k - 1, :] = np.sqrt(2.0 / 3.0) * np.sin((2 * k - 1) * i * np.pi / n)
        if n % 2 != 0:
            # Corrected: see module docstring, defect 2. MATLAB indexes
            # this row with `i+1`, overrunning the 2n columns declared
            # for this branch; `i` directly (as every other assignment
            # in this branch uses) is the correction, verified below.
            xi[n - 1, :] = (-1.0) ** i / np.sqrt(3.0)
        w = np.full(num_pts, V / (2 * n))
        return xi.T, w
    if algorithm == 1:  # Cn 3-3, 2n+1 points.
        B0 = V * (3.0 - n) / 3.0
        B1 = V / 6.0
        e1 = np.zeros(n)
        e1[0] = 1.0
        points = np.vstack([np.zeros((1, n)), _full_sym_perms(e1)])
        w = np.concatenate([[B0], np.full(2 * n, B1)])
        return points, w
    if algorithm == 2:  # Cn 3-4, 2^n points.
        r = np.sqrt(3.0) / 3.0
        points = _pm_combos(r * np.ones(n))
        w = np.full(points.shape[0], V / 2.0**n)
        return points, w
    if algorithm == 3:  # Cn 3-5, 2^n+1 points.
        B0 = (2.0 / 3.0) * V
        B1 = (1.0 / (3.0 * 2.0**n)) * V
        points = np.vstack([np.zeros((1, n)), _pm_combos(np.ones(n))])
        w = np.concatenate([[B0], np.full(2**n, B1)])
        return points, w
    if algorithm == 4:  # Cn 3-6, 3^n points.
        nodes = np.array([-1.0, 0.0, 1.0])
        weights = np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0])
        return _tensor_grid_rule(nodes, weights, n)
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 3 (general-n); "
        "expected one of 0-4"
    )


def _cube_degree5(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = 2.0**n
    if algorithm == 0:  # Cn 5-2, 2n^2+1 points.
        r = np.sqrt(3.0 / 5.0)
        B0 = V * (25.0 * n**2 - 115.0 * n + 162.0) / 162.0
        B1 = V * (70.0 - 25.0 * n) / 162.0
        B2 = V * 25.0 / 324.0
        v1 = np.zeros(n)
        v1[0] = r
        v2 = np.zeros(n)
        v2[0] = r
        v2[1] = r
        points = np.vstack([np.zeros((1, n)), _full_sym_perms(v1), _full_sym_perms(v2)])
        w = np.concatenate([[B0], np.full(2 * n, B1), np.full(2 * n * (n - 1), B2)])
        return points, w
    if algorithm == 1:  # Cn 5-3, 3n^2+3n+1 points.
        r = np.sqrt(7.0 / 15.0)
        s = np.sqrt((7.0 + np.sqrt(24.0)) / 15.0)
        t = np.sqrt((7.0 - np.sqrt(24.0)) / 15.0)
        B0 = V * (5.0 * n**2 - 15.0 * n + 14.0) / 14.0
        B1 = V * 25.0 / 168.0
        B2 = -V * 25.0 * (n - 2.0) / 168.0
        B3 = V * 5.0 / 48.0
        B4 = -V * 5.0 * (n - 2.0) / 48.0

        rr = np.concatenate([[r, r], np.zeros(n - 2)])
        neg_rr = np.concatenate([[-r, -r], np.zeros(n - 2)])
        r0 = np.zeros(n)
        r0[0] = r
        st = np.concatenate([[s, -t], np.zeros(n - 2)])
        neg_st = np.concatenate([[-s, t], np.zeros(n - 2)])
        s0 = np.zeros(n)
        s0[0] = s
        t0 = np.zeros(n)
        t0[0] = t

        points = np.vstack(
            [
                np.zeros((1, n)),
                _multiset_perms(rr),
                _multiset_perms(neg_rr),
                _full_sym_perms(r0),
                _multiset_perms(st),
                _multiset_perms(neg_st),
                _full_sym_perms(s0),
                _full_sym_perms(t0),
            ]
        )
        w = np.concatenate(
            [
                [B0],
                np.full(_multiset_perms(rr).shape[0], B1),
                np.full(_multiset_perms(neg_rr).shape[0], B1),
                np.full(2 * n, B2),
                np.full(_multiset_perms(st).shape[0], B3),
                np.full(_multiset_perms(neg_st).shape[0], B3),
                np.full(2 * n, B4),
                np.full(2 * n, B4),
            ]
        )
        return points, w
    if algorithm == 2:  # Cn 5-4, 2^n+2n points.
        r = np.sqrt((5.0 * n + 4.0) / 30.0)
        s = np.sqrt((5.0 * n + 4.0) / (15.0 * n - 12.0))
        B1 = 40.0 * V / (5.0 * n + 4.0) ** 2
        B2 = 2.0 ** (-n) * ((5.0 * n - 4.0) / (5.0 * n + 4.0)) ** 2 * V
        r0 = np.zeros(n)
        r0[0] = r
        points = np.vstack([_full_sym_perms(r0), _pm_combos(s * np.ones(n))])
        w = np.concatenate([np.full(2 * n, B1), np.full(2**n, B2)])
        return points, w
    if algorithm == 3:  # Cn 5-5, 2^n+2n+1 points.
        r = np.sqrt(2.0 / 5.0)
        B0 = V * (8.0 - 5.0 * n) / 9.0
        B1 = V * 5.0 / 18.0
        B2 = V * 1.0 / (9.0 * 2.0**n)
        r0 = np.zeros(n)
        r0[0] = r
        points = np.vstack(
            [np.zeros((1, n)), _full_sym_perms(r0), _pm_combos(np.ones(n))]
        )
        w = np.concatenate([[B0], np.full(2 * n, B1), np.full(2**n, B2)])
        return points, w
    if algorithm == 4:  # Cn 5-6, 2^(n+1)-1 points.
        s = 1.0 / np.sqrt(3.0)
        points_list = [np.zeros((1, n))]
        weights_list = [np.array([4.0 / (5.0 * n + 4.0) * V])]
        vec = s * np.ones(n)
        for k in range(1, n + 1):
            vec[k - 1] = np.sqrt((5.0 * k + 4.0) / 15.0)
            sub = vec[k - 1 :]
            combos = _pm_combos(sub)
            num_cur = combos.shape[0]
            block = np.zeros((num_cur, n))
            block[:, k - 1 :] = combos
            points_list.append(block)
            wk = 5.0 * 2.0 ** (k - n + 1) * V / ((5.0 * k - 1.0) * (5.0 * k + 4.0))
            weights_list.append(np.full(num_cur, wk))
        return np.vstack(points_list), np.concatenate(weights_list)
    if algorithm == 5:  # Cn 5-7, n*2^n+1 points.
        B0 = V * 4.0 / (5.0 * n + 4.0)
        B1 = V * 5.0 * 2.0 ** (-n) / (5.0 * n + 4.0)
        r = np.sqrt(
            (5.0 * n + 4.0 + 2.0 * (n - 1.0) * np.sqrt(5.0 * n + 4.0)) / (15.0 * n)
        )
        s = np.sqrt((5.0 * n + 4.0 - 2.0 * np.sqrt(5.0 * n + 4.0)) / (15.0 * n))
        base = np.concatenate([[r], s * np.ones(n - 1)])
        points = np.vstack([np.zeros((1, n)), _full_sym_perms(base)])
        w = np.concatenate([[B0], np.full(n * 2**n, B1)])
        return points, w
    if algorithm == 6:  # Cn 5-8, 2^n*(n+1) points. Requires n >= 3: the
        # radicand of s below is negative at n=2 (measured -0.074), a
        # domain restriction inherent to the formula itself, not checked
        # in the MATLAB source (which would silently return NaN/complex).
        if n < 3:
            raise ValueError(f"algorithm 6 (Cn 5-8) requires n >= 3, got {n}")
        r = np.sqrt(
            (5.0 * n - 2.0 * np.sqrt(5.0) + 2.0 * (n - 1.0) * np.sqrt(5.0 * n + 5.0))
            / (15.0 * n)
        )
        s = np.sqrt(
            (5.0 * n - 2.0 * np.sqrt(5.0) - 2.0 * np.sqrt(5.0 * n + 5.0)) / (15.0 * n)
        )
        t = np.sqrt((5.0 + 2.0 * np.sqrt(5.0)) / 15.0)
        base = np.concatenate([[r], s * np.ones(n - 1)])
        points = np.vstack([_full_sym_perms(base), _pm_combos(t * np.ones(n))])
        w_uniform = V / (2.0**n * (n + 1))
        w = np.full(points.shape[0], w_uniform)
        return points, w
    if algorithm == 7:  # Cn 5-9, 3^n points.
        nodes = np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)])
        weights = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        return _tensor_grid_rule(nodes, weights, n)
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 5 (general-n); "
        "expected one of 0-7"
    )


def _cube_degree7(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0 if n == 2 else 5
    V = 2.0**n
    if algorithm == 0:  # C2 7-1, 12 points, n=2 only.
        if n != 2:
            raise ValueError(f"algorithm 0 requires n == 2, got {n}")
        r = np.sqrt(6.0 / 7.0)
        s = np.sqrt((114.0 - 3.0 * np.sqrt(583.0)) / 287.0)
        t = np.sqrt((114.0 + 3.0 * np.sqrt(583.0)) / 287.0)
        r0 = np.array([r, 0.0])
        points = np.vstack(
            [_full_sym_perms(r0), _pm_combos([s, s]), _pm_combos([t, t])]
        )
        B1 = (49.0 / 810.0) * V
        B2 = ((178981.0 + 2769.0 * np.sqrt(583.0)) / 1888920.0) * V
        B3 = ((178981.0 - 2769.0 * np.sqrt(583.0)) / 1888920.0) * V
        w = np.concatenate([np.full(4, B1), np.full(4, B2), np.full(4, B3)])
        return points, w
    if algorithm == 5:  # C3 7-2, 34 points, n=3 only.
        if n != 3:
            raise ValueError(f"algorithm 5 requires n == 3, got {n}")
        r = np.sqrt(6.0 / 7.0)
        s = np.sqrt((960.0 - 3.0 * np.sqrt(28798.0)) / 2726.0)
        t = np.sqrt((960.0 + 3.0 * np.sqrt(28798.0)) / 2726.0)
        r0 = np.array([r, 0.0, 0.0])
        rr0 = np.array([r, r, 0.0])
        points = np.vstack(
            [
                _full_sym_perms(r0),
                _full_sym_perms(rr0),
                _pm_combos([s, s, s]),
                _pm_combos([t, t, t]),
            ]
        )
        B1 = (1078.0 / 29160.0) * V
        B2 = (343.0 / 29160.0) * V
        B3 = ((774.0 * t**2 - 230.0) / (9720.0 * (t**2 - s**2))) * V
        B4 = ((230.0 - 774.0 * s**2) / (9720.0 * (t**2 - s**2))) * V
        w = np.concatenate(
            [np.full(6, B1), np.full(12, B2), np.full(8, B3), np.full(8, B4)]
        )
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 7; only 0 (n=2) and "
        "5 (n=3) are ported -- no general-n degree-7 formula exists in "
        "MATLAB for this region (see module docstring)"
    )


def _cube_degree9_roots(n: int) -> Tuple[float, float, float]:
    """The two radii (+disc, -disc branches of the shared quartic) that
    ``_cube_degree9_build`` binds to ``u``/``v`` differently per algorithm,
    plus the cube volume ``V``."""
    V = 2.0**n
    I2, I4, I6, I8 = V / 3.0, V / 5.0, V / 7.0, V / 9.0
    disc = np.sqrt(
        I2**2 * I8**2
        + 4.0 * I4**3 * I8
        + 4.0 * I2 * I6**3
        - 6.0 * I2 * I4 * I6 * I8
        - 3.0 * I4**2 * I6**2
    )
    root_plus = np.sqrt((I2 * I8 - I4 * I6 + disc) / (2.0 * (I2 * I6 - I4**2)))
    root_minus = np.sqrt((I2 * I8 - I4 * I6 - disc) / (2.0 * (I2 * I6 - I4**2)))
    return V, root_plus, root_minus


def _cube_degree9_build(
    n: int, u: float, v: float, V: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Shared point/weight construction for both Cn 9-1 variants, given
    which root has already been bound to ``u`` vs ``v`` by the caller (see
    ``_cube_degree9``)."""
    I2, I4 = V / 3.0, V / 5.0
    I22, I42, I44, I62 = V / 9.0, V / 15.0, V / 25.0, V / 21.0
    I222, I422, I2222 = V / 27.0, V / 45.0, V / 81.0

    F = (I62 - I44) / (4.0 * u**2 * v**2 * (u**2 - v**2) ** 2)
    H = (I422 - (n - 3.0) * I2222) / (8.0 * v**8)
    Ic = (I422 - v**2 * I222) / (16.0 * (n - 3.0) * u**6 * (u**2 - v**2))
    J = (I2222 - 16.0 * u**8 * Ic) / (16.0 * v**8)
    E = (
        (u**2 * I22 - I42) / (4.0 * v**4 * (u**2 - v**2))
        - F * u**2 / v**2
        - 2.0 * (n - 2.0) * (H + (n - 3.0) * J)
    )
    D = (
        (I42 - v**2 * I22) / (4.0 * u**4 * (u**2 - v**2))
        - F * v**2 / u**2
        - 2.0 * (n - 2.0) * (n - 3.0) * Ic
    )
    C = (u**2 * I2 - I4) / (2.0 * v**2 * (u**2 - v**2)) - 2.0 * (n - 1.0) * (
        E + F + (n - 2.0) * (H + (2.0 / 3.0) * (n - 3.0) * J)
    )
    B = (I4 - v**2 * I2) / (2.0 * u**2 * (u**2 - v**2)) - 2.0 * (n - 1.0) * (
        D + F + (2.0 / 3.0) * (n - 2.0) * (n - 3.0) * Ic
    )
    A = V - 2.0 * n * (
        B
        + C
        + (n - 1.0)
        * (D + E + 2.0 * F + (1.0 / 3.0) * (n - 2.0) * (2.0 * H + (n - 3.0) * (Ic + J)))
    )

    e_u = np.zeros(n)
    e_u[0] = u
    e_v = np.zeros(n)
    e_v[0] = v
    e_uu = np.zeros(n)
    e_uu[0] = u
    e_uu[1] = u
    e_vv = np.zeros(n)
    e_vv[0] = v
    e_vv[1] = v
    e_uv = np.zeros(n)
    e_uv[0] = u
    e_uv[1] = v
    e_vvv = np.zeros(n)
    e_vvv[:3] = v
    e_uuuu = np.zeros(n)
    e_uuuu[:4] = u
    e_vvvv = np.zeros(n)
    e_vvvv[:4] = v

    blocks = [
        (np.zeros((1, n)), A),
        (e_u, B),
        (e_v, C),
        (e_uu, D),
        (e_vv, E),
        (e_uv, F),
        (e_vvv, H),
        (e_uuuu, Ic),
        (e_vvvv, J),
    ]
    points_list = []
    weights_list = []
    for vec, wt in blocks:
        pts = vec if vec.ndim == 2 else _full_sym_perms(vec)
        points_list.append(pts)
        weights_list.append(np.full(pts.shape[0], wt))
    return np.vstack(points_list), np.concatenate(weights_list)


def _cube_degree9(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if algorithm not in (0, 1):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 9; expected 0 or 1"
        )
    if n < 4:
        raise ValueError(f"degree-9 cube rule requires n >= 4, got {n}")
    V, root_plus, root_minus = _cube_degree9_roots(n)
    # Algorithm 0 ("variant 1" per MATLAB's own comment) binds v to the
    # +disc root, u to the -disc root; algorithm 1 ("variant 2") binds
    # them the other way. Every downstream formula and point-pattern
    # (e.g. the vvv-block always gets weight H) is written in terms of
    # u/v symbols, not in terms of "the larger/smaller root" -- and is
    # NOT symmetric under swapping which root each symbol names (D's
    # formula, e.g., is not E's formula with u and v exchanged). So this
    # is not cosmetic relabeling: the two algorithms are genuinely
    # different point sets, both degree-9 exact (see module docstring).
    if algorithm == 0:
        u, v = root_minus, root_plus
    else:
        u, v = root_plus, root_minus
    return _cube_degree9_build(n, u, v, V)


def cube_cubature_points(
    n: int, degree: int, algorithm: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Cubature points for the ``n``-dimensional cube ``[-1, 1]^n``.

    Counterpart of the MATLAB TCL's ``Cube_Space`` top-level, general-
    dimension files (see this module's docstring for the exact per-degree
    algorithm coverage and the two corrected upstream defects). Every
    ``degree`` here is exact through that total polynomial degree -- verified
    against the closed-form cube-monomial oracle in
    ``tests/unit/test_region_cubature.py`` for the ``(n, degree, algorithm)``
    grid its test classes sweep; no wider claim is made (per the
    claims-inherit-measurement-range convention).

    **Weight convention (region measure, not probability).** ``weights``
    sum to the cube's volume ``2**n``, NOT to 1 -- this module targets the
    plain Lebesgue measure on ``[-1, 1]^n``, unlike
    :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`'s
    Gaussian-weight rules. A caller wanting a probability-normalized rule
    divides by the volume themselves: ``weights / weights.sum()``.

    Parameters
    ----------
    n : int
        Dimension, n >= 1 (algorithm/degree combinations below may require
        more, e.g. degree 9 requires n >= 4).
    degree : int
        Polynomial degree the rule is exact through. One of 1, 2, 3, 5, 7, 9
        -- the degrees MATLAB's ``Cube_Space`` top-level files provide.
        Degrees 4, 6, 8 have no file in this directory (MATLAB provides only
        odd degrees plus degree 2, matching Gaussian-quadrature convention
        of even-order-exact rules living at the next odd degree up).
    algorithm : int, optional
        Which MATLAB algorithm variant to use; see each degree's section in
        the module docstring for the ported subset (default+general-n
        algorithms only -- fixed-``n=2``/``n=3`` literature variants are
        deferred, no current pytcl consumer). Default None reproduces
        MATLAB's own default selection for that degree/n:

        - degree 1: algorithm 1 (2^n points; algorithm 0 is 1 point).
        - degree 2: algorithm 0 (n+1 points; algorithm 1 is 2n+1 points).
        - degree 3: algorithm 0 (2n points, Table-I-corrected Cn 3-1).
        - degree 5: algorithm 0 (2n^2+1 points, Cn 5-2).
        - degree 7: algorithm 0 if n == 2, else algorithm 5 if n == 3 (no
          other n is supported -- no general-n degree-7 cube formula exists
          in MATLAB).
        - degree 9: algorithm 0 (n >= 4 required). Algorithm 1 is a
          genuinely DIFFERENT rule (MATLAB's own comment calls it "variant
          2" of the same Cn 9-1 formula, not a relabeling of algorithm 0)
          -- both are degree-9 exact; see this module's docstring for how
          they differ.

    Returns
    -------
    points : ndarray
        Shape (num_points, n).
    weights : ndarray
        Shape (num_points,), summing to ``2**n`` (the cube's volume), not 1.
        Commonly contains negative entries at larger ``n`` -- inherent to
        these Stroud formulas, not suppressed. Measured examples: degree 2
        algorithm 1 (every ``n`` tested, ``n >= 2``); degree 3 algorithm 1
        (``n >= 4``); degree 5's DEFAULT algorithm 0 (``n >= 3``) and
        algorithms 1, 3; degree 9's DEFAULT algorithm (``n >= 4``, i.e.
        every ``n`` this function accepts for that degree). Covariances or
        other quantities assembled from these points must not use a
        sqrt-of-weights factorization.

    Examples
    --------
    >>> pts, w = cube_cubature_points(3, 3)
    >>> pts.shape
    (6, 3)
    >>> round(float(w.sum()), 12)
    8.0
    >>> round(float(np.sum(w * pts[:, 0] ** 2)), 9)  # integral of x^2 over [-1,1]^3
    2.666666667

    References
    ----------
    A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
    Prentice-Hall, 1971, Formulas Cn 1-1/1-2, Cn 2-1/2-2, Cn 3-1/3-3/3-4/
    3-5/3-6, Cn 5-2 through 5-9, C2 7-1, C3 7-2, Cn 9-1, pp. 229-266.

    R. Cools, "An encyclopedia of cubature formulas," Journal of
    Complexity, vol. 19, no. 3, pp. 445-453, Jun. 2003.
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if degree == 1:
        return _cube_degree1(n, algorithm)
    if degree == 2:
        if n < 2:
            raise ValueError(f"degree 2 requires n >= 2, got {n}")
        return _cube_degree2(n, algorithm)
    if degree == 3:
        if n < 2:
            raise ValueError(f"degree 3 requires n >= 2, got {n}")
        return _cube_degree3(n, algorithm)
    if degree == 5:
        if n < 2:
            raise ValueError(f"degree 5 requires n >= 2, got {n}")
        return _cube_degree5(n, algorithm)
    if degree == 7:
        return _cube_degree7(n, algorithm)
    if degree == 9:
        return _cube_degree9(n, algorithm)
    raise ValueError(f"unsupported degree {degree}; expected one of 1, 2, 3, 5, 7, 9")


def _simplex_degree2(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm not in (None, 0):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 2 (general-n); "
            "expected 0 (MATLAB's secondOrderSimplexCubPoints.m has no "
            "algorithm parameter at all)"
        )
    V = 1.0 / math.factorial(n)
    r = 0.5
    B = V * (2.0 - n) / ((n + 1) * (n + 2))
    C = V * 4.0 / ((n + 1) * (n + 2))
    vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
    edges = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [r, r]]), n)
    points = np.vstack([vertices, edges])
    w = np.concatenate([np.full(vertices.shape[0], B), np.full(edges.shape[0], C)])
    return points, w


def _simplex_degree3_alg3_root(n: int) -> float:
    """The parameter ``r`` of T_n 3-4, one real root of MATLAB's own cubic.

    See the module docstring's third corrected defect: at n == 2 EVERY real
    root of this cubic makes the weight formula an exact 0/0 (verified
    symbolically), so callers must exclude n == 2 before calling this
    (:func:`simplex_cubature_points` does). For n >= 3 the cubic has either
    one real root or (n == 5 only) a real double root plus a real simple
    root; this function deterministically returns the SMALLEST real root --
    a reproducible tie-break, not a guess, since both n=5 candidates are
    independently degree-3 exact (module docstring).
    """
    coeffs = [
        2.0 * (n - 2) * (n + 1) * (n + 3),
        -(5.0 * n**2 + 5.0 * n - 18.0),
        4.0 * n,
        -1.0,
    ]
    roots = np.roots(coeffs)
    real_roots = sorted(roots[np.abs(roots.imag) < 1e-8].real)
    uniq = []
    for rv in real_roots:
        if not uniq or abs(rv - uniq[-1]) > 1e-6:
            uniq.append(float(rv))
    return min(uniq)


def _simplex_degree3(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = 1.0 / math.factorial(n)
    if algorithm == 0:  # T_n 3-1, n+2 points.
        r = 1.0 / (n + 1)
        s = 1.0 / (n + 3)
        t = 3.0 / (n + 3)
        B = -V * (n + 1) ** 2 / (4.0 * (n + 2))
        C = V * (n + 3) ** 2 / (4.0 * (n + 1) * (n + 2))
        center = np.full((1, n), r)
        block = _simplex_bary_block(np.concatenate([np.full(n, s), [t]]), n)
        points = np.vstack([center, block])
        w = np.concatenate([[B], np.full(block.shape[0], C)])
        return points, w
    if algorithm == 1:  # T_n 3-2, 2n+2 points.
        r = (2.0 * n + 5.0 - np.sqrt(4.0 * n + 13.0)) / (2.0 * (n + 1) * (n + 3))
        s = 1.0 - n * r
        B = V * (1.0 - np.sqrt(4.0 * n + 13.0)) / (2.0 * (n + 1) * (n + 2) * (n + 3))
        C = (
            V
            * (2.0 * n**2 + 10.0 * n + 11.0 + np.sqrt(4.0 * n + 13.0))
            / (2.0 * (n + 1) * (n + 2) * (n + 3))
        )
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block = _simplex_bary_block(np.concatenate([np.full(n, r), [s]]), n)
        points = np.vstack([vertices, block])
        w = np.concatenate([np.full(vertices.shape[0], B), np.full(block.shape[0], C)])
        return points, w
    if algorithm == 2:  # T_n 3-3, 2n+3 points.
        r = 1.0 / (n + 1)
        s = 1.0 / n
        A = V * (3.0 - n) * (n + 1) ** 2 / ((n + 2) * (n + 3))
        B = V * 3.0 / ((n + 1) * (n + 2) * (n + 3))
        C = V * n**3 / ((n + 1) * (n + 2) * (n + 3))
        center = np.full((1, n), r)
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block = _simplex_bary_block(np.concatenate([np.full(n, s), [0.0]]), n)
        points = np.vstack([center, vertices, block])
        w = np.concatenate(
            [[A], np.full(vertices.shape[0], B), np.full(block.shape[0], C)]
        )
        return points, w
    if algorithm == 3:  # T_n 3-4, (n+1)(n+2)/2 points. n=2 excluded: see
        # module docstring's third corrected defect (exact 0/0 there).
        if n < 3 or n >= 7:
            raise ValueError(
                f"algorithm 3 (T_n 3-4) requires 3 <= n < 7, got {n} -- "
                "MATLAB's own guard is only n<7, but n=2 makes the weight "
                "formula an exact 0/0 (see module docstring)"
            )
        r = _simplex_degree3_alg3_root(n)
        t = 0.5
        s = 1.0 - n * r
        denom = 1.0 - 2.0 * n * r**2 - 2.0 * (1.0 - n * r) ** 2
        B = V * (n - 2) / ((n + 1) * (n + 2) * denom)
        C = (2.0 / n) * (V / (n + 1) - B)
        vertices = _simplex_bary_block(np.concatenate([np.full(n, r), [s]]), n)
        block = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [t, t]]), n)
        points = np.vstack([vertices, block])
        w = np.concatenate([np.full(vertices.shape[0], B), np.full(block.shape[0], C)])
        return points, w
    if algorithm == 4:  # T_n 3-5, (n+1)(n+4)/2 points, n>=3.
        if n < 3:
            raise ValueError(f"algorithm 4 (T_n 3-5) requires n >= 3, got {n}")
        r = 0.5
        s = 1.0 / n
        B = V * (6.0 - n) / ((n + 1) * (n + 2) * (n + 3))
        C = V * 8.0 * (n - 3) / ((n - 2) * (n + 1) * (n + 2) * (n + 3))
        D = V * n**3 / ((n - 2) * (n + 1) * (n + 2) * (n + 3))
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block1 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [r, r]]), n)
        block2 = _simplex_bary_block(np.concatenate([np.full(n, s), [0.0]]), n)
        points = np.vstack([vertices, block1, block2])
        w = np.concatenate(
            [
                np.full(vertices.shape[0], B),
                np.full(block1.shape[0], C),
                np.full(block2.shape[0], D),
            ]
        )
        return points, w
    if algorithm == 5:  # T_n 3-7, (n^3+5n+12)/6 points. n>=3: MATLAB does
        # not check this, but A and C both divide by (n-2) (hardened here,
        # same pattern as cube's algorithm-6 n>=3 guard).
        if n < 3:
            raise ValueError(f"algorithm 5 (T_n 3-7) requires n >= 3, got {n}")
        r = 1.0 / (n + 1)
        s = 1.0 / 3.0
        A = V * (n + 1) ** 2 * (n - 3) / ((n - 2) * (n + 2) * (n + 3))
        B = V * (9.0 - n) / (2.0 * (n + 1) * (n + 2) * (n + 3))
        C = V * 27.0 / ((n - 2) * (n + 1) * (n + 2) * (n + 3))
        center = np.full((1, n), r)
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [s, s, s]]), n)
        points = np.vstack([center, vertices, block])
        w = np.concatenate(
            [[A], np.full(vertices.shape[0], B), np.full(block.shape[0], C)]
        )
        return points, w
    if algorithm == 6:  # T_n 3-8, (n^3+11n+12)/6 points. n>=2: MATLAB does
        # not check this, but A/B/C all divide by (n-1) (hardened here);
        # satisfied automatically by this module's n>=2 floor.
        r = 1.0 / n
        s = 1.0 / 3.0
        A = (
            V
            * (-(n**2) + 11.0 * n - 12.0)
            / (2.0 * (n - 1) * (n + 1) * (n + 2) * (n + 3))
        )
        B = V * n**3 / ((n - 1) * (n + 1) * (n + 2) * (n + 3))
        C = V * 27.0 / ((n - 1) * (n + 1) * (n + 2) * (n + 3))
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block1 = _simplex_bary_block(np.concatenate([np.full(n, r), [0.0]]), n)
        block2 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [s, s, s]]), n)
        points = np.vstack([vertices, block1, block2])
        w = np.concatenate(
            [
                np.full(vertices.shape[0], A),
                np.full(block1.shape[0], B),
                np.full(block2.shape[0], C),
            ]
        )
        return points, w
    if algorithm == 7:  # T_n 3-9, (n+1)(n+2)(n+3)/6 points.
        r = 1.0 / 3.0
        s = 2.0 / 3.0
        B = V * (n**2 - 4.0 * n + 6.0) / ((n + 1) * (n + 2) * (n + 3))
        C = V * (27.0 - 9.0 * n) / (2.0 * (n + 1) * (n + 2) * (n + 3))
        D = V * 27.0 / ((n + 1) * (n + 2) * (n + 3))
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        block1 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [r, s]]), n)
        block2 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [r, r, r]]), n)
        points = np.vstack([vertices, block1, block2])
        w = np.concatenate(
            [
                np.full(vertices.shape[0], B),
                np.full(block1.shape[0], C),
                np.full(block2.shape[0], D),
            ]
        )
        return points, w
    if algorithm in (8, 9):  # T_n 3-10 / T_n 3-11, n>=3 and n!=5 (MATLAB's
        # own guard: both divide by (n-5)).
        if n < 3 or n == 5:
            raise ValueError(
                f"algorithm {algorithm} (T_n 3-10/3-11) requires n >= 3 "
                f"and n != 5, got {n}"
            )
        s = 1.0 / 3.0
        t = 1.0 / (n - 2)
        block1 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [s, s, s]]), n)
        block2 = _simplex_bary_block(
            np.concatenate([np.full(n - 2, t), [0.0, 0.0, 0.0]]), n
        )
        if algorithm == 8:  # T_n 3-10, (n^3-n+3)/3 points.
            r = 1.0 / (n + 1)
            A = (
                V
                * (3.0 - n)
                * (n - 12.0)
                * (n + 1) ** 2
                / (3.0 * (n - 2) * (n + 2) * (n + 3))
            )
            B = (
                V
                * 54.0
                * (3.0 * n - 11.0)
                / ((n - 5) * (n - 2) * (n - 1) * (n + 1) * (n + 2) * (n + 3))
            )
            C = (
                V
                * 2.0
                * (n - 2) ** 2
                * (n - 9.0)
                / ((n - 5) * (n - 1) * (n + 1) * (n + 2) * (n + 3))
            )
            center = np.full((1, n), r)
            points = np.vstack([center, block1, block2])
            w = np.concatenate(
                [[A], np.full(block1.shape[0], B), np.full(block2.shape[0], C)]
            )
            return points, w
        # T_n 3-11, (n^3+2n+3)/3 points.
        A = V * (12.0 - n) / (2.0 * (n + 1) * (n + 2) * (n + 3))
        B = V * 27.0 * (n - 7.0) / ((n - 5) * (n - 1) * (n + 1) * (n + 2) * (n + 3))
        C = V * 6.0 * (n - 2) ** 2 / ((n - 5) * (n - 1) * (n + 1) * (n + 2) * (n + 3))
        vertices = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        points = np.vstack([vertices, block1, block2])
        w = np.concatenate(
            [
                np.full(vertices.shape[0], A),
                np.full(block1.shape[0], B),
                np.full(block2.shape[0], C),
            ]
        )
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 3 (general-n); "
        "expected one of 0-9 (10 and 11 are the fixed-n=2/n=5 literature "
        "variants T_2 3-1 and T_5 3-1, deferred -- see module docstring)"
    )


def _simplex_degree4(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm not in (None, 0):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 4 (general-n); "
            "expected 0 (MATLAB's fourthOrderSimplexCubPoints.m has no "
            "algorithm parameter at all)"
        )
    if n < 3:
        raise ValueError(f"degree 4 simplex rule requires n >= 3, got {n}")
    V = 1.0 / math.factorial(n)
    r, s, t = 0.25, 0.75, 0.5
    B1 = (
        V
        * (-3.0 * n**3 + 17.0 * n**2 - 58.0 * n + 72.0)
        / (3.0 * (n + 1) * (n + 2) * (n + 3) * (n + 4))
    )
    B2 = (
        V
        * 16.0
        * (n**2 - 5.0 * n + 12.0)
        / (3.0 * (n + 1) * (n + 2) * (n + 3) * (n + 4))
    )
    B3 = V * 4.0 * (n**2 - 9.0 * n + 12.0) / ((n + 1) * (n + 2) * (n + 3) * (n + 4))
    B4 = V * 64.0 * (4.0 - n) / (2.0 * (n + 1) * (n + 2) * (n + 3) * (n + 4))
    B5 = V * 256.0 / ((n + 1) * (n + 2) * (n + 3) * (n + 4))

    b1 = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
    b2 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [r, s]]), n)
    b3 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [t, t]]), n)
    b4 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [r, r, t]]), n)
    # n >= 3 is already guaranteed by the guard above, so this block (which
    # needs n-3 >= 0) always applies here.
    b5 = _simplex_bary_block(np.concatenate([np.zeros(n - 3), [r, r, r, r]]), n)

    points = np.vstack([b1, b2, b3, b4, b5])
    w = np.concatenate(
        [
            np.full(b1.shape[0], B1),
            np.full(b2.shape[0], B2),
            np.full(b3.shape[0], B3),
            np.full(b4.shape[0], B4),
            np.full(b5.shape[0], B5),
        ]
    )
    # Get rid of zero-weight points -- exactly reproduces MATLAB's own
    # `sel=~(w==0)` filter (module docstring: not a defect, B4 is exactly 0
    # at n==4 and only there). Exact-equality comparison is safe: B4 is
    # `V * 64 * (4 - n) / denom`, and `(4 - n)` is exact integer arithmetic
    # in double precision, so B4 rounds to exactly 0.0 at n==4.
    sel = w != 0.0
    return points[sel], w[sel]


def _simplex_degree5(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        if n >= 4:
            algorithm = 0
        elif n == 2:
            algorithm = 1
        else:
            algorithm = 2
    V = 1.0 / math.factorial(n)
    if algorithm == 0:  # T_n 5-2, C(n+5,5) points, n>=4.
        if n < 4:
            raise ValueError(f"algorithm 0 (T_n 5-2) requires n >= 4, got {n}")
        r, s, u, v = 0.2, 0.8, 0.4, 0.6
        n_fact_rat = math.factorial(n) / math.factorial(n + 5)
        B1 = (
            V
            * (12.0 * n**4 - 82.0 * n**3 + 477.0 * n**2 - 1277.0 * n + 1440.0)
            * n_fact_rat
            / 12.0
        )
        B2 = (
            V
            * 25.0
            * (-3.0 * n**3 + 19.0 * n**2 - 96.0 * n + 170.0)
            * n_fact_rat
            / 12.0
        )
        B3 = V * 25.0 * (-(n**3) + 13.0 * n**2 - 47.0 * n + 65.0) * n_fact_rat / 6.0
        B4 = V * 125.0 * (n**2 - 6.0 * n + 20.0) * n_fact_rat / 3.0
        B5 = V * 125.0 * (n**2 - 11.0 * n + 20.0) * n_fact_rat / 4.0
        B6 = V * 625.0 * (5.0 - n) * n_fact_rat / 2.0
        B7 = V * 3125.0 * n_fact_rat

        b1 = _simplex_bary_block(np.concatenate([np.zeros(n), [1.0]]), n)
        b2 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [r, s]]), n)
        b3 = _simplex_bary_block(np.concatenate([np.zeros(n - 1), [u, v]]), n)
        b4 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [r, r, v]]), n)
        b5 = _simplex_bary_block(np.concatenate([np.zeros(n - 2), [r, u, u]]), n)
        b6 = _simplex_bary_block(np.concatenate([np.zeros(n - 3), [r, r, r, u]]), n)
        b7 = _simplex_bary_block(np.concatenate([np.zeros(n - 4), [r, r, r, r, r]]), n)
        points = np.vstack([b1, b2, b3, b4, b5, b6, b7])
        w = np.concatenate(
            [
                np.full(b1.shape[0], B1),
                np.full(b2.shape[0], B2),
                np.full(b3.shape[0], B3),
                np.full(b4.shape[0], B4),
                np.full(b5.shape[0], B5),
                np.full(b6.shape[0], B6),
                np.full(b7.shape[0], B7),
            ]
        )
        return points, w
    if algorithm == 1:  # T_2 5-1, 7 points, n=2 only.
        if n != 2:
            raise ValueError(f"algorithm 1 (T_2 5-1) requires n == 2, got {n}")
        t = 1.0 / 3.0
        r = (6.0 - np.sqrt(15.0)) / 21.0
        u = (6.0 + np.sqrt(15.0)) / 21.0
        s = (9.0 + 2.0 * np.sqrt(15.0)) / 21.0
        v = (9.0 - 2.0 * np.sqrt(15.0)) / 21.0
        A = V * 9.0 / 40.0
        B = V * (155.0 - np.sqrt(15.0)) / 1200.0
        C = V * (155.0 + np.sqrt(15.0)) / 1200.0
        points = np.array([[t, t], [r, r], [r, s], [s, r], [u, u], [u, v], [v, u]])
        w = np.array([A, B, B, B, C, C, C])
        return points, w
    if algorithm == 2:  # T_3 5-1, 15 points, n=3 only.
        if n != 3:
            raise ValueError(f"algorithm 2 (T_3 5-1) requires n == 3, got {n}")
        r = 0.25
        s1 = (7.0 - np.sqrt(15.0)) / 34.0
        s2 = (7.0 + np.sqrt(15.0)) / 34.0
        u = (10.0 - 2.0 * np.sqrt(15.0)) / 40.0
        t1 = (13.0 + 3.0 * np.sqrt(15.0)) / 34.0
        t2 = (13.0 - 3.0 * np.sqrt(15.0)) / 34.0
        v = (10.0 + 2.0 * np.sqrt(15.0)) / 40.0
        A = V * 16.0 / 135.0
        B1 = V * (2665.0 + 14.0 * np.sqrt(15.0)) / 37800.0
        B2 = V * (2665.0 - 14.0 * np.sqrt(15.0)) / 37800.0
        C = V * 20.0 / 378.0
        points = np.array(
            [
                [r, r, r],
                [s1, s1, s1],
                [s1, s1, t1],
                [s1, t1, s1],
                [t1, s1, s1],
                [s2, s2, s2],
                [s2, s2, t2],
                [s2, t2, s2],
                [t2, s2, s2],
                [u, u, v],
                [u, v, u],
                [v, u, u],
                [v, v, u],
                [v, u, v],
                [u, v, v],
            ]
        )
        w = np.array([A] + [B1] * 4 + [B2] * 4 + [C] * 6)
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 5; expected 0 (n>=4), "
        "1 (n==2), or 2 (n==3)"
    )


def simplex_cubature_points(
    n: int, degree: int, algorithm: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Cubature points for the standard n-simplex ``{x >= 0, sum(x) <= 1}``.

    Counterpart of the MATLAB TCL's ``Simplex`` top-level, general-dimension
    files (see this module's docstring for the exact per-degree algorithm
    coverage and the corrected upstream defect). Every ``degree`` here is
    exact through that total polynomial degree -- verified against the
    closed-form Dirichlet-integral simplex-monomial oracle in
    ``tests/unit/test_region_cubature.py`` for the ``(n, degree, algorithm)``
    grid its test classes sweep; no wider claim is made (per the
    claims-inherit-measurement-range convention).

    **Weight convention (region measure, not probability).** ``weights``
    sum to the simplex's volume ``1 / n!``, NOT to 1 -- this module targets
    the plain Lebesgue measure on the simplex, unlike
    :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`'s
    Gaussian-weight rules. A caller wanting a probability-normalized rule
    divides by the volume themselves: ``weights / weights.sum()``.

    Parameters
    ----------
    n : int
        Dimension, n >= 2 (the design spec's tested/captured dimension
        range starts at n=2; algorithm/degree combinations below may
        require more, e.g. degree 4 requires n >= 3). n=1 (the degenerate
        1-D "simplex", the interval [0, 1]) is out of scope for this port
        even where a formula would evaluate without error there.
    degree : int
        Polynomial degree the rule is exact through. One of 2, 3, 4, 5 --
        the degrees MATLAB's ``Simplex`` top-level files provide.
    algorithm : int, optional
        Which MATLAB algorithm variant to use; see each degree's section in
        the module docstring for the ported subset. Default None reproduces
        MATLAB's own default selection for that degree/n:

        - degree 2: algorithm 0 (the only variant; MATLAB's
          ``secondOrderSimplexCubPoints.m`` takes no algorithm argument).
        - degree 3: algorithm 0 ((n+2) points, T_n 3-1). Algorithms 1-9 are
          the other general-``n`` variants (each with its own dimension
          restriction, some hardened beyond MATLAB's own checks -- see
          module docstring); algorithms 10 (T_2 3-1, n=2 only) and 11
          (T_5 3-1, n=5 only) are the fixed-dimension literature variants,
          not ported (module docstring).
        - degree 4: algorithm 0 (the only variant; MATLAB's
          ``fourthOrderSimplexCubPoints.m`` takes no algorithm argument),
          n >= 3 required.
        - degree 5: algorithm 0 (T_n 5-2, n >= 4) if n >= 4; algorithm 1
          (T_2 5-1, 7 points) if n == 2; algorithm 2 (T_3 5-1, 15 points) if
          n == 3 -- matching MATLAB's own per-``n`` default selection.

    Returns
    -------
    points : ndarray
        Shape (num_points, n).
    weights : ndarray
        Shape (num_points,), summing to ``1 / n!`` (the simplex's volume),
        not 1. Contains negative entries for several algorithms at some
        dimensions -- e.g. degree 3 algorithm 0's ``B`` weight is negative
        for every ``n`` this function accepts -- inherent to these Stroud
        formulas, not suppressed.

    Examples
    --------
    >>> pts, w = simplex_cubature_points(2, 2)
    >>> pts.shape
    (6, 2)
    >>> round(float(w.sum()), 12)
    0.5
    >>> round(float(np.sum(w * pts[:, 0])), 6)  # integral of x over the 2-simplex
    0.166667

    References
    ----------
    A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
    Prentice-Hall, 1971, Formulas T_n 2-2, T_n 3-1/3-2/3-3/3-4/3-5/3-7/
    3-8/3-9/3-10/3-11, T_n 4-1, T_n 5-2, T_2 5-1, T_3 5-1, pp. 307-315.

    R. Cools, "An encyclopedia of cubature formulas," Journal of
    Complexity, vol. 19, no. 3, pp. 445-453, Jun. 2003.
    """
    if n < 2:
        raise ValueError(f"dimension must be >= 2, got {n}")
    if degree == 2:
        return _simplex_degree2(n, algorithm)
    if degree == 3:
        return _simplex_degree3(n, algorithm)
    if degree == 4:
        return _simplex_degree4(n, algorithm)
    if degree == 5:
        return _simplex_degree5(n, algorithm)
    raise ValueError(f"unsupported degree {degree}; expected one of 2, 3, 4, 5")
