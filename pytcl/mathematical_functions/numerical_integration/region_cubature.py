"""
Region-cubature point sets: bounded-domain integration, true-measure weights.

Unlike :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`
(which targets the Gaussian weight N(0, I) and normalizes ``sum(w) == 1`` for
direct use as ``E[f(X)] ~= sum_i w_i f(x_i)``), every generator in this module
targets a bounded geometric region with the *Lebesgue* weight ``1`` and
reports the region's TRUE measure: weights sum to the region's volume, not to
1. Currently this module covers only the ``Cube_Space`` region,
``[-1, 1]^n``, ``sum(w) == 2**n``. A caller wanting the probability-normalized
version divides by the returned volume (``weights / weights.sum()``), which
is always safe; going the other way (recovering the true measure from a
pre-normalized rule) is not possible without already knowing the volume. See
``docs/superpowers/specs/2026-08-16-region-cubature-design.md`` Section 1 for
the full rationale and the conventions pinned for the three region families
(``Simplex``, ``Sphere``, ``Spherical_Surface``) not yet ported here.

Ported from the Tracker Component Library's
``Cubature_Points/Cube_Space`` collection (top-level, general-dimension
files only; the fixed-``n=2``/``n=3`` ``Cube/`` and ``Square/``
subdirectories are out of scope for this module -- see the design spec's
Section 8). Within each ported file, only the general-dimension algorithms
(and, for ``seventhOrderNDimCubPoints.m``, the one algorithm MATLAB itself
defaults to per dimension, since that file has no general-``n`` algorithm at
all) are transcribed; the remaining fixed-``n`` literature variants each file
also offers are deferred for the same reason the ``Cube``/``Square``
subdirectories are (no current pytcl consumer, narrow value versus the
general-``n`` rule already covered).

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

References
----------
A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
Prentice-Hall, 1971.

R. Cools, "An encyclopedia of cubature formulas," Journal of Complexity,
vol. 19, no. 3, pp. 445-453, Jun. 2003.

D. F. Crouse, "The Tracker Component Library," IEEE AESS Magazine, 2017.
"""

import itertools
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


def _cube_degree9(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if algorithm not in (0, 1):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 9; expected 0 or 1 "
            "(both variants of Cn 9-1 produce the identical rule -- see "
            "docstring)"
        )
    if n < 4:
        raise ValueError(f"degree-9 cube rule requires n >= 4, got {n}")
    V = 2.0**n
    I2, I4, I6, I8 = V / 3.0, V / 5.0, V / 7.0, V / 9.0
    I22, I42, I44, I62 = V / 9.0, V / 15.0, V / 25.0, V / 21.0
    I222, I422, I2222 = V / 27.0, V / 45.0, V / 81.0

    disc = np.sqrt(
        I2**2 * I8**2
        + 4.0 * I4**3 * I8
        + 4.0 * I2 * I6**3
        - 6.0 * I2 * I4 * I6 * I8
        - 3.0 * I4**2 * I6**2
    )
    v = np.sqrt((I2 * I8 - I4 * I6 + disc) / (2.0 * (I2 * I6 - I4**2)))
    u = np.sqrt((I2 * I8 - I4 * I6 - disc) / (2.0 * (I2 * I6 - I4**2)))

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
        - degree 9: algorithm 0 (n >= 4 required; algorithm 1 produces the
          identical rule under a relabeled derivation, see the degree-9
          section below).

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
