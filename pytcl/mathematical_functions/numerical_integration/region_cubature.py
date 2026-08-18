"""
Region-cubature point sets: bounded-domain integration, true-measure weights.

Unlike :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`
(which targets the Gaussian weight N(0, I) and normalizes ``sum(w) == 1`` for
direct use as ``E[f(X)] ~= sum_i w_i f(x_i)``), every generator in this module
targets a bounded geometric region with the *Lebesgue* weight ``1`` (or, for
the ``Sphere`` region, ``|x|**alpha``) and reports the region's TRUE measure:
weights sum to the region's volume, not to 1. This module currently covers
the ``Cube_Space`` region (``[-1, 1]^n``, ``sum(w) == 2**n``), the
``Simplex`` region (the standard n-simplex ``{x >= 0, sum(x) <= 1}``,
``sum(w) == 1/n!``), the ``Sphere`` region (the unit ball
``{|x| <= 1}``, weight ``|x|**alpha``, ``sum(w) == 2/(n+alpha) * pi**(n/2) /
gamma(n/2)`` -- the unit ball volume ``pi**(n/2) / gamma(n/2+1)`` when
``alpha == 0``), and the ``Spherical_Surface`` region (the unit sphere
``S^(n-1) = {|x| == 1}``, weight 1, ``sum(w) == 2*pi**(n/2) / gamma(n/2)``
-- the sphere's surface area). A caller wanting the probability-normalized
version divides by the returned volume (``weights / weights.sum()``), which
is always safe; going the other way (recovering the true measure from a
pre-normalized rule) is not possible without already knowing the volume.
See the region-cubature port-feasibility design spec (local-only, untracked
-- see CONTRIBUTING.md's ``docs/superpowers/`` policy), Section 1, for the
full rationale.

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

Also ported (this module's third region): the Tracker Component Library's
``Cubature_Points/Sphere`` collection (solid unit ball), Tier-1 top-level
files only -- ``secondOrderSpherCubPoints.m``, ``thirdOrderSpherCubPoints.m``
(all 5 algorithms), ``fifthOrderSpherCubPoints.m`` (all 10 algorithms),
``seventhOrderSpherCubPoints.m`` (all 6 algorithms), and
``arbOrderSpherCubPoints.m`` (general dimension, general ORDER, folded into
:func:`ball_cubature_points` as the dispatch target for every odd
``degree >= 9`` -- see "The ``arbOrderSpherCubPoints.m`` port" below). Two
files in the same directory are explicitly EXCLUDED, with reasons:

- ``ninthOrderSpherCubPoints.m`` and ``eleventhOrderSpherCubPoints.m`` are
  hardcoded to ``n == 2`` (Tier 3 per the design spec's Section 8) -- no
  general-``n`` value, same rationale as the deferred ``Cube``/``Square``/
  ``Tetrahedra``/``Triangles`` subdirectories above. (``arbOrderSpherCubPoints``
  reaches degree 9/11 for general ``n`` anyway, at a different, larger
  point count, so nothing here is a coverage gap.)
- ``spherSurfPoints2SpherPoints.m`` converts an existing
  ``Spherical_Surface`` rule into a ball rule of matching order; superseded
  in practice now that :func:`ball_cubature_points` exists via direct
  construction (design spec Section 8) -- even now that
  :func:`spherical_surface_cubature_points` exists too (below), this
  converter file itself remains unported since nothing in this module
  needs it.

**The ``arbOrderSpherCubPoints.m`` port (general dimension, general order,
general real ``alpha``).** ``degree = 2*order - 1`` for any caller-chosen
``order``; :func:`ball_cubature_points` dispatches every odd ``degree >= 9``
here (degrees 2/3/5/7 keep using the dedicated named-formula files above,
which are more efficient point counts at those specific degrees). MATLAB's
own construction (Stroud's Theorem 2.6-2, a recursive "generalized polar
coordinates" decomposition) needs two 1-D quadrature families from the
separate ``quadraturePoints1D.m`` dispatcher (outside this ported subset):
algorithm 3 (Gegenbauer, weight ``(1-x**2)**(c1-1/2)`` on ``(-1,1)``) for
the ``n-1`` angular coordinates, and algorithm 8 (radial, weight ``|x|**c1``
on ``(-1,1)``, ``c1 = numDim-1+alpha``) for the radial coordinate.

- **Gegenbauer piece**: direct correspondence to
  ``scipy.special.roots_jacobi(m, c1-0.5, c1-0.5)`` -- both target weight
  ``(1-x**2)**(c1-0.5)`` and their total masses match algebraically
  (``sqrt(pi)*Gamma(c1+0.5)/Gamma(c1+1)`` either way). MATLAB special-cases
  ``c1 == 0`` to a separate closed-form arcsine rule because its OWN
  hand-rolled three-term recurrence is exactly singular there (division by
  a term that is identically zero at ``c1=0``, confirmed by inspection of
  ``quadraturePoints1D.m``'s case-3 coefficients); ``scipy.special.roots_jacobi``
  uses a different, numerically robust algorithm and needs no such special
  case -- verified directly: ``roots_jacobi(n, -0.5, -0.5)`` matches the
  arcsine closed form to machine precision at every ``n`` tested (1, 2, 3,
  5, 7).
- **Radial piece**: NOT a ``roots_jacobi`` substitution for MATLAB's own
  algorithm-8 recursion (that recursion is restricted to nonnegative
  INTEGER ``c1`` by construction, per its own docstring -- it cannot serve
  a general real ``alpha`` at all, which :func:`ball_cubature_points`
  needs). Instead, DERIVED from scratch via the classical even-weight
  "symmetrization" technique (Gautschi): substituting ``t = x**2`` maps
  weight ``|x|**c1`` on ``(-1,1)`` to a plain power weight
  ``t**((c1-1)/2)`` on ``(0,1)`` for the even-degree test-polynomial part;
  an ``m``-point ``x``-domain rule (exact through degree ``2m-1``) reduces
  to a ``floor(m/2)``-point Gauss-Jacobi problem on ``(0,1)`` (mapped from
  ``scipy.special.roots_jacobi`` on ``(-1,1)`` via the standard affine
  ``t=(x+1)/2`` change of variable, with the matching ``(1/2)**(p+1)``
  Jacobian rescale on the weights) at even ``m``, plus one additional fixed
  node at the origin (weight determined by the zeroth-moment/total-mass
  residual) at odd ``m``. This derivation was independently verified TWO
  ways before use, not merely asserted: (1) it reproduces MATLAB's own
  algorithm-8 three-term-recurrence construction (separately, exactly
  transcribed for this one-time check) to machine precision at every
  INTEGER ``c1`` tested (0 through 4, ``m`` = 3 through 6); (2) it matches
  brute-force ``scipy.integrate.quad`` at every monomial degree ``<= 2m-1``
  for NON-integer ``c1`` (0.5, 1.5, 2.5) that MATLAB's own algorithm cannot
  evaluate at all -- exactly the general-``alpha`` case this port needs and
  MATLAB's own code cannot provide.
- **Full construction**: independently re-verified end-to-end (both pieces
  plus the recursive point/weight assembly together) against this module's
  own ball-monomial oracle at ``order`` in ``{5, 6, 7}`` (degrees 9, 11,
  13), ``n`` in ``{2, 3, 4}``, ``alpha`` in ``{0.0, 1.5}`` -- every case
  exact through its claimed degree (worst-monomial error ~1e-15) and sharp
  (a real, non-roundoff residual at degree+1, ~1e-3 to ~1e-5) -- see
  ``tests/unit/test_region_cubature.py``'s ``TestBallArbOrder``.
- **A genuine MATLAB size-mismatch inefficiency, not a defect** (found by
  inspection, distinct from the two corrected shape-mismatch defects
  below): the source preallocates ``xi``/``w`` with
  ``numPoints = numRVals * numGegenbauerPoints`` (``= order**2``), but the
  nested loop that follows fills ``order**numDim`` columns total whenever
  ``numDim > 2`` -- more than was preallocated. MATLAB auto-grows arrays on
  out-of-bound assignment (no error, unlike the two Cube_Space defects
  below, which crash outright), and every column the loop visits gets
  fully populated in row-major order with no gaps, so the final result is
  correct, just built via an inefficient reallocate-and-copy path in real
  MATLAB. This module builds the correctly-sized ``(order**numDim, numDim)``
  array directly.

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

**A confirmed MATLAB documentation defect, not a code defect (found in the
``Sphere`` port): the ``alpha`` weighting-function sign.** Every
alpha-dependent ``Sphere`` file's docstring (``thirdOrderSpherCubPoints.m``,
``fifthOrderSpherCubPoints.m``, ``arbOrderSpherCubPoints.m``) describes the
weighting function as ``sum(x.^2)^(-alpha/2)``, i.e. ``|x|**(-alpha)``. The
CODE says otherwise: every one of those files' ``V``/coefficient formulas
divides by ``(numDim + alpha)`` (e.g. ``V = 2/(numDim+alpha) *
pi^(numDim/2) / gamma(numDim/2)``), and the same docstrings independently
state the valid domain as ``alpha > -numDim``. Both of those are consistent
ONLY with a weight of ``|x|**(+alpha)``: the standard closed-form radial
ball integral ``integral_{unit ball} |x|**s dx = (2*pi**(n/2)/gamma(n/2)) /
(s+n)``, valid for ``s+n > 0``, has denominator ``(s+n)`` and domain
``s > -n`` -- matching the code's ``(numDim+alpha)``/``alpha > -numDim``
only when ``s = alpha`` (i.e. weight ``|x|**alpha``), not ``s = -alpha``.
This was checked three independent ways -- against
``thirdOrderSpherCubPoints.m`` algorithm 0, ``fifthOrderSpherCubPoints.m``
algorithms 0/2/5/7, and ``arbOrderSpherCubPoints.m``'s own radial quadrature
domain (``c1 = numDim-1+alpha >= 0``, again only consistent with
``+alpha``) -- and against the elementary calculus derivation above, not
merely assumed. This module's ``alpha`` parameter and
:func:`ball_cubature_points`'s docstring therefore state the weight as
``|x|**alpha`` (matching what the CODE actually computes), diverging from
both the MATLAB docstrings' literal text and the design spec's Section 5.1
oracle formula (which inherited the same ``-alpha`` sign from the MATLAB
docstrings without cross-checking it against the formulas) --
``tests/unit/test_region_cubature.py``'s ``ball_monomial_integral`` oracle
uses the corrected ``(n + alpha + sum(a))`` denominator accordingly.

**A fourth corrected MATLAB defect (a wrong-formula bug, not a
crash/shape-mismatch), found in ``seventhOrderSpherCubPoints.m`` algorithm 2
(``S2 7-2``, 16 points, ``n == 2`` only).** The source's manual
trigonometric construction sets BOTH coordinate rows from ``cos``:
``xi(1,1:8)=r1*cos(...); xi(2,1:8)=r1*cos(...)`` (and likewise for the
``r2`` block) -- every one of the 16 points therefore lies exactly on the
line ``x == y``, collapsing what should be a 2-D point set onto a 1-D
subspace. Verified NUMERICALLY (not merely by inspection) against this
module's own ball-monomial oracle: the literal transcription fails at
degree <= 7 monomials it should be exact for -- measured residuals ~0.39
at ``(2,2)`` (oracle value ~0.131) and exactly wrong-sign/nonzero at
``(1,1)`` (oracle value 0, literal-transcription value ~0.785) -- while
using ``sin`` for the second row (the natural cos/sin pairing used by
EVERY other angular construction in this codebase, e.g.
``secondOrderSpherCubPoints.m``'s own ``cos``/``sin`` pair) matches the
oracle to float64 tolerance through degree 7. :func:`ball_cubature_points`
implements the corrected (``cos``, ``sin``) pairing for degree 7 algorithm 2.

Also ported (this module's fourth region): the Tracker Component Library's
``Cubature_Points/Spherical_Surface`` collection (unit sphere surface),
Tier-1 top-level files only -- ``firstOrderSpherSurfCubPoints.m`` (despite
its own header text reading "third-order," an audit-known docstring typo
contradicted by the 2-point antipodal construction and its own citation --
treated as a docstring error, not transcribed; the BODY, which computes a
valid degree-1 rule, is what is ported), ``thirdOrderSpherSurfCubPoints.m``
(all 4 algorithms), ``fifthOrderSpherSurfCubPoints.m`` (algorithms 0-8, see
the fifth confirmed finding below), and ``seventhOrderSpherSurfCubPoints.m``
(all 9 algorithms -- algorithm 0 reuses the private
:func:`_seventh_order_sphere_surface_alg0` helper A3 already ported as a
``ball_cubature_points`` degree-7 dependency, promoted here to also serve
as :func:`spherical_surface_cubature_points`'s own degree-7 algorithm-0
implementation, exactly as this module's earlier note anticipated ("a
future task porting Spherical_Surface may promote or generalize this
helper")). Two files are EXCLUDED, both ``n=3``-only (Tier 3 per the design
spec's Section 8, same rationale as ``ninthOrderSpherCubPoints.m``/
``eleventhOrderSpherCubPoints.m``'s exclusion from :func:`ball_cubature_points`):
``ninthOrderSpherSurfCubPoints.m`` and ``eleventhOrderSpherSurfCubPoints.m``
-- both degrees are reachable for ANY ``n`` via the general odd-degree
``arbOrderSpherSurfCubPoints.m`` reuse path below, so nothing is lost.

**Reuse, not reimplementation, for two further files** (design spec Section
4, inventory rows 179-180): ``fourteenthOrderSpherSurfCubPoints.m`` (``n=3``
only) wraps
:func:`~pytcl.mathematical_functions.numerical_integration.cubature_points._fourteenth_order_unit_sphere_points_3d`,
and ``arbOrderSpherSurfCubPoints.m``/``arbOrder2DSpherSurfCubPoints.m``
(general ``n``, general odd degree >= 9) wrap
:func:`~pytcl.mathematical_functions.numerical_integration.cubature_points._sphere_surface_points`
-- both existing private helpers normalize to ``sum(w) == 1``, so
:func:`spherical_surface_cubature_points` rescales their output by the
closed-form surface area ``2*pi**(n/2)/gamma(n/2)`` rather than re-deriving
either construction. The reused general-order construction is a genuinely
DIFFERENT algorithm from MATLAB's own Gegenbauer-based
``arbOrderSpherSurfCubPoints.m`` (dimension-recursive Gauss-Jacobi vs
Stroud's Theorem 2.7-3) -- both produce a general-``n``, general-order rule
for the uniform surface measure, and the design spec's own capture case for
this file exists specifically to confirm LOW-ORDER-MOMENT agreement between
the two constructions despite their different point sets, not to check
raw-coordinate equality (which would be invalid for two different
constructions of the same rule family, the same reasoning the LCD
validation convention states explicitly for its own flat-optimum-manifold
case).

**A fifth confirmed MATLAB defect, a documentation-only mismatch (found in
``fifthOrderSpherSurfCubPoints.m``), verified by direct inspection of the
switch statement, not merely by reading the header comment.** The file's
own docstring claims 10 algorithms exist, indices 0-9: index 8 described as
"Algorithm U3 5-4 of [1], requiring 20 points," index 9 described as
"Algorithm U3 5-5 of [1], requiring 30 points." The actual ``switch``
statement, however, has cases only for indices 0 through 8 -- and the code
at ``case 8`` computes the 30-point U3 5-5 formula the docstring attributes
to index 9, not the 20-point U3 5-4 formula the docstring attributes to
index 8. So real MATLAB, called with ``algorithm=8``, returns the 30-point
formula; the 20-point U3 5-4 formula is simply never implemented anywhere
in the file; and ``algorithm=9`` hits the ``otherwise`` branch and raises
"Unknown algorithm specified." Per the port-fidelity-over-invention
convention, this module transcribes what the CODE actually computes (index
8 -> the 30-point formula) rather than what the docstring claims, and caps
the supported range at 0-8 to match what is actually reachable in real
MATLAB -- see :func:`_sphsurf_degree5`'s algorithm-8 branch and its
``ValueError`` message for algorithms outside 0-8.

**A related, non-defect finding: three degree-7 algorithms compute the
IDENTICAL rule.** ``seventhOrderSpherSurfCubPoints.m`` algorithms 0
("Formula I" of Stroud (1968)), 3 ("the formula from" Stroud (1967)), and 4
("Un 7-1" of Stroud's 1971 book) each place the exact same three weight
formulas onto the exact same three point blocks (an axis block at
``k=1``, an axis block at ``k=2``, and an all-nonzero block at ``k=n``) --
verified by matching each algorithm's coefficient formula symbol-for-symbol
against the other two (not merely by both passing the same degree-7
exactness oracle, which two DIFFERENT degree-7-exact rules could also both
pass) and confirmed numerically in
``tests/unit/test_region_cubature.py``'s
``test_algorithm_0_3_4_compute_identical_rule``. Three different textbook
citations for the same underlying Stroud construction, not three distinct
rules -- each is still transcribed independently (not delegated to a
shared implementation) to keep this module's per-algorithm dispatch a
faithful mirror of MATLAB's own per-algorithm ``case`` structure, matching
how ``ball_cubature_points`` keeps ``thirdOrderSpherCubPoints.m``'s five
algorithms independently transcribed even where some literature formulas
coincide.

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
from scipy.special import roots_jacobi

from pytcl.mathematical_functions.numerical_integration.cubature_points import (
    _fourteenth_order_unit_sphere_points_3d,
    _sphere_surface_points,
)


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


def _ball_volume(n: int, alpha: float) -> float:
    """integral_{unit n-ball} |x|**alpha dx = 2/(n+alpha) * pi**(n/2) / gamma(n/2).

    Degenerates to the plain unit-ball volume pi**(n/2)/gamma(n/2+1) at
    alpha=0. See the module docstring's confirmed-documentation-defect note
    for why this uses +alpha, not the -alpha MATLAB's own docstrings claim.
    """
    if n + alpha <= 0.0:
        raise ValueError(f"require n + alpha > 0, got n={n}, alpha={alpha}")
    return 2.0 / (n + alpha) * (math.pi ** (n / 2.0) / math.gamma(n / 2.0))


def _ball_degree2(
    n: int, algorithm: Optional[int], alpha: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm not in (None, 0):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 2 (general-n); "
            "expected 0 (MATLAB's secondOrderSpherCubPoints.m has no "
            "algorithm parameter at all)"
        )
    if alpha != 0.0:
        raise ValueError(
            "degree 2 (secondOrderSpherCubPoints.m) does not expose alpha "
            "(weight fixed at 1); got alpha != 0"
        )
    V = _ball_volume(n, 0.0)
    num_pts = n + 1
    i = np.arange(num_pts)
    xi = np.zeros((n, num_pts))
    for k in range(1, n // 2 + 1):
        xi[2 * k - 2, :] = np.sqrt(2.0 / (n + 2.0)) * np.cos(
            2 * i * k * np.pi / (n + 1)
        )
        xi[2 * k - 1, :] = np.sqrt(2.0 / (n + 2.0)) * np.sin(
            2 * i * k * np.pi / (n + 1)
        )
    if n % 2 != 0:
        xi[n - 1, :] = (-1.0) ** i / np.sqrt(n + 2.0)
    w = np.full(num_pts, V / num_pts)
    return xi.T, w


def _ball_degree3(
    n: int, algorithm: Optional[int], alpha: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if algorithm == 0:  # Sn 3-1, 2n points, alpha-capable.
        V = _ball_volume(n, alpha)
        r = math.sqrt((n + alpha) / (n + alpha + 2.0))
        base = np.zeros(n)
        base[0] = r
        points = _full_sym_perms(base)
        w = np.full(2 * n, V / (2.0 * n))
        return points, w
    if algorithm == 1:  # Sn 3-2, 2^n points, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 1 (Sn 3-2) requires alpha == 0")
        r = math.sqrt(1.0 / (n + 2.0))
        points = _pm_combos(r * np.ones(n))
        V = _ball_volume(n, 0.0)
        w = np.full(points.shape[0], V * 2.0 ** (-n))
        return points, w
    if algorithm == 2:  # S2 3-1, 4 points, n=2, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 2 (S2 3-1) requires alpha == 0")
        if n != 2:
            raise ValueError(f"algorithm 2 (S2 3-1) requires n == 2, got {n}")
        V = _ball_volume(2, 0.0)
        points = _full_sym_perms(np.array([1.0 / math.sqrt(2.0), 0.0]))
        w = np.full(4, V / 4.0)
        return points, w
    if algorithm == 3:  # S2 3-2, 4 points, n=2, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 3 (S2 3-2) requires alpha == 0")
        if n != 2:
            raise ValueError(f"algorithm 3 (S2 3-2) requires n == 2, got {n}")
        V = _ball_volume(2, 0.0)
        points = _full_sym_perms(np.array([0.5, 0.5]))
        w = np.full(4, V / 4.0)
        return points, w
    if algorithm == 4:  # S3 3-1, 6 points, n=3, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 4 (S3 3-1) requires alpha == 0")
        if n != 3:
            raise ValueError(f"algorithm 4 (S3 3-1) requires n == 3, got {n}")
        V = _ball_volume(3, 0.0)
        r = math.sqrt(3.0 / 5.0)
        points = _full_sym_perms(np.array([r, 0.0, 0.0]))
        w = np.full(6, V / 6.0)
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 3 (general-n); "
        "expected one of 0-4"
    )


def _ball_degree5(
    n: int, algorithm: Optional[int], alpha: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if algorithm == 0:  # Sn 5-2, 2n^2+1 points, alpha-capable.
        V = _ball_volume(n, alpha)
        r = math.sqrt(3.0 * (n + alpha + 2.0) / ((n + 2.0) * (n + alpha + 4.0)))
        B2 = (
            V
            * (n + 2.0)
            * (n + alpha)
            * (n + alpha + 4.0)
            / (36.0 * n * (n + alpha + 2.0) ** 2)
        )
        B1 = (
            V
            * (4.0 - n)
            * (n + 2.0)
            * (n + alpha)
            * (n + alpha + 4.0)
            / (18.0 * n * (n + alpha + 2.0) ** 2)
        )
        B0 = V - 2.0 * n * B1 - 2.0 * n * (n - 1.0) * B2
        v1 = np.zeros(n)
        v1[0] = r
        v2 = np.zeros(n)
        v2[0] = r
        v2[1] = r
        points = np.vstack([np.zeros((1, n)), _full_sym_perms(v1), _full_sym_perms(v2)])
        w = np.concatenate([[B0], np.full(2 * n, B1), np.full(2 * n * (n - 1), B2)])
        return points, w
    if algorithm == 1:  # Sn 5-3, 2^n+2n points, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 1 (Sn 5-3) requires alpha == 0")
        r = math.sqrt((n + 4.0 - math.sqrt(2.0 * (n + 4.0))) / (n + 4.0))
        s = math.sqrt(
            (n * (n + 4.0) + 2.0 * math.sqrt(2.0 * (n + 4.0)))
            / ((n**2 + 2.0 * n - 4.0) * (n + 4.0))
        )
        V = _ball_volume(n, 0.0)
        B1 = V / ((n + 2.0) * (n + 4.0) * r**4)
        B2 = V / (2.0**n * (n + 2.0) * (n + 4.0) * s**4)
        v1 = np.zeros(n)
        v1[0] = r
        points = np.vstack([_full_sym_perms(v1), _pm_combos(s * np.ones(n))])
        w = np.concatenate([np.full(2 * n, B1), np.full(2**n, B2)])
        return points, w
    if algorithm == 2:  # Sn 5-4, 2^(n+1)-1 points, alpha-capable.
        s = math.sqrt((n + alpha + 2.0) / ((n + 2.0) * (n + alpha + 4.0)))
        V = _ball_volume(n, alpha)
        num_points = 2 ** (n + 1) - 1
        xi = np.zeros((n, num_points))
        w = np.zeros(num_points)
        vec = np.full(n, s)
        b_pow_sum = 0.0
        cur_start = 0
        for k in range(1, n + 1):
            r = math.sqrt(
                (k + 2.0) * (n + alpha + 2.0) / ((n + 2.0) * (n + alpha + 4.0))
            )
            B = (
                V
                * 2.0 ** (k - n)
                * (n + 2.0)
                * (n + alpha)
                * (n + alpha + 4.0)
                / (n * (k + 1.0) * (k + 2.0) * (n + alpha + 2.0) ** 2)
            )
            vec[k - 1] = r
            combos = _pm_combos(vec[k - 1 :])
            num_cur = combos.shape[0]
            xi[k - 1 :, cur_start : cur_start + num_cur] = combos.T
            w[cur_start : cur_start + num_cur] = B
            cur_start += num_cur
            b_pow_sum += 2.0 ** (n - k + 1) * B
        if alpha == 0.0:
            w[-1] = 4.0 * V / (n + 2.0) ** 2
        else:
            w[-1] = V - b_pow_sum
        return xi.T, w
    if algorithm == 3:  # Sn 5-5, n*2^n+1 points, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 3 (Sn 5-5) requires alpha == 0")
        V = _ball_volume(n, 0.0)
        r = math.sqrt(
            (n + 2.0 + (n - 1.0) * math.sqrt(2.0 * (n + 2.0))) / (n * (n + 4.0))
        )
        s = math.sqrt((n + 2.0 - math.sqrt(2.0 * (n + 2.0))) / (n * (n + 4.0)))
        B0 = 4.0 * V / (n + 2.0) ** 2
        B1 = V * (n + 4.0) / (2.0**n * (n + 2.0) ** 2)
        base = np.concatenate([[r], s * np.ones(n - 1)])
        points = np.vstack([np.zeros((1, n)), _full_sym_perms(base)])
        w = np.concatenate([[B0], np.full(n * 2**n, B1)])
        return points, w
    if algorithm == 4:  # Sn 5-6, 2^n*(n+1) points, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 4 (Sn 5-6) requires alpha == 0")
        V = _ball_volume(n, 0.0)
        r = math.sqrt(
            (
                n * (n + 4.0)
                + 2.0 * math.sqrt(n + 4.0)
                + (n - 1.0) * math.sqrt(2.0 * (n + 1.0) * (n + 2.0) * (n + 4.0))
            )
            / (n * (n + 2.0) * (n + 4.0))
        )
        s = math.sqrt(
            (
                n * (n + 4.0)
                + 2.0 * math.sqrt(n + 4.0)
                - math.sqrt(2.0 * (n + 1.0) * (n + 2.0) * (n + 4.0))
            )
            / (n * (n + 2.0) * (n + 4.0))
        )
        t = math.sqrt((n + 4.0 - 2.0 * math.sqrt(n + 4.0)) / ((n + 2.0) * (n + 4.0)))
        B = V / (2.0**n * (n + 1.0))
        base = np.concatenate([[r], s * np.ones(n - 1)])
        points = np.vstack([_full_sym_perms(base), _pm_combos(t * np.ones(n))])
        w = np.full(points.shape[0], B)
        return points, w
    if algorithm == 5:  # S2 5-1, 7 points, n=2, alpha-capable.
        if n != 2:
            raise ValueError(f"algorithm 5 (S2 5-1) requires n == 2, got {n}")
        V = _ball_volume(2, alpha)
        r = math.sqrt((alpha + 4.0) / (alpha + 6.0))
        s = math.sqrt((alpha + 4.0) / (4.0 * (alpha + 6.0)))
        t = math.sqrt(3.0 * (alpha + 4.0) / (4.0 * (alpha + 6.0)))
        A = 4.0 * V / (alpha + 4.0) ** 2
        B = V * (alpha + 2.0) * (alpha + 6.0) / (6.0 * (alpha + 4.0) ** 2)
        points = np.vstack([np.zeros((1, 2)), _pm_combos([r, 0.0]), _pm_combos([s, t])])
        w = np.concatenate([[A], np.full(6, B)])
        return points, w
    if algorithm == 6:  # S2 5-2, 9 points, n=2, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 6 (S2 5-2) requires alpha == 0")
        if n != 2:
            raise ValueError(f"algorithm 6 (S2 5-2) requires n == 2, got {n}")
        V = _ball_volume(2, 0.0)
        r = 1.0 / math.sqrt(2.0)
        A = V / 6.0
        B = V / 24.0
        points = np.vstack(
            [np.zeros((1, 2)), _full_sym_perms(np.array([r, 0.0])), _pm_combos([r, r])]
        )
        w = np.concatenate([np.full(5, A), np.full(4, B)])
        return points, w
    if algorithm == 7:  # S3 5-1, 13 points, n=3, alpha-capable.
        if n != 3:
            raise ValueError(f"algorithm 7 (S3 5-1) requires n == 3, got {n}")
        V = _ball_volume(3, alpha)
        r = math.sqrt((alpha + 5.0) * (5.0 + math.sqrt(5.0)) / (10.0 * (alpha + 7.0)))
        s = math.sqrt((alpha + 5.0) * (5.0 - math.sqrt(5.0)) / (10.0 * (alpha + 7.0)))
        B0 = V * 4.0 / (alpha + 5.0) ** 2
        B1 = V * (alpha + 3.0) * (alpha + 7.0) / (12.0 * (alpha + 5.0) ** 2)
        points = np.vstack(
            [
                np.zeros((1, 3)),
                _pm_combos([r, s, 0.0]),
                _pm_combos([0.0, r, s]),
                _pm_combos([s, 0.0, r]),
            ]
        )
        w = np.concatenate([[B0], np.full(12, B1)])
        return points, w
    if algorithm == 8:  # S3 5-2, 21 points, n=3, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 8 (S3 5-2) requires alpha == 0")
        if n != 3:
            raise ValueError(f"algorithm 8 (S3 5-2) requires n == 3, got {n}")
        V = _ball_volume(3, 0.0)
        r = math.sqrt((15.0 + 5.0 * math.sqrt(5.0)) / 42.0)
        s = math.sqrt((15.0 - 5.0 * math.sqrt(5.0)) / 42.0)
        t = math.sqrt(5.0 / 21.0)
        B0 = 4.0 * V / 25.0
        B1 = 21.0 * V / 500.0
        points = np.vstack(
            [
                np.zeros((1, 3)),
                _pm_combos([r, s, 0.0]),
                _pm_combos([0.0, r, s]),
                _pm_combos([s, 0.0, r]),
                _pm_combos([t, t, t]),
            ]
        )
        w = np.concatenate([[B0], np.full(20, B1)])
        return points, w
    if algorithm == 9:  # S4 5-1, 25 points, n=4, alpha=0.
        if alpha != 0.0:
            raise ValueError("algorithm 9 (S4 5-1) requires alpha == 0")
        if n != 4:
            raise ValueError(f"algorithm 9 (S4 5-1) requires n == 4, got {n}")
        V = _ball_volume(4, 0.0)
        r = math.sqrt(3.0 / 8.0)
        B1 = V / 27.0
        B0 = V / 9.0
        points = np.vstack(
            [np.zeros((1, 4)), _full_sym_perms(np.array([r, r, 0.0, 0.0]))]
        )
        w = np.concatenate([[B0], np.full(24, B1)])
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 5 (general-n); "
        "expected one of 0-9"
    )


def _seventh_order_sphere_surface_alg0(
    n: int,
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Spherical_Surface/seventhOrderSpherSurfCubPoints.m algorithm 0
    (Formula I of [1] below, n >= 3), ported ONLY as the dependency
    :func:`ball_cubature_points`'s degree-7 algorithm 0 needs -- real MATLAB's
    seventhOrderSpherCubPoints.m algorithm 0 calls this internally. NOT a
    general Spherical_Surface port (that region family is not yet in this
    module at all -- see the module docstring's Sphere exclusions); a future
    task porting Spherical_Surface may promote or generalize this helper.

    Weights sum to the unit sphere surface area 2*pi**(n/2)/gamma(n/2), the
    same true-measure convention as the rest of this module (verified: the
    two ball-degree-7-algorithm-0 rescale factors A(1)+A(2) below reduce
    algebraically to exactly 1/n, so A(1)*sum(B)+A(2)*sum(B) collapses to
    the ball volume _ball_volume(n, 0.0) -- an internal-consistency check on
    the port, not merely assumed).

    References
    ----------
    [1] A. H. Stroud, "Some seventh degree integration formulas for the
        surface of an n-sphere," Numerische Mathematik, vol. 11, no. 3,
        pp. 273-276, Mar. 1968.
    """
    I1 = 2.0 * math.pi ** (n / 2.0) / math.gamma(n / 2.0)
    A1 = (8.0 - n) / (n * (n + 2.0) * (n + 4.0)) * I1
    A2 = 4.0 / (n * (n + 2.0) * (n + 4.0)) * I1
    A3 = 2.0 ** (-n) * n**3 / (n * (n + 2.0) * (n + 4.0)) * I1
    v1 = np.zeros(n)
    v1[0] = 1.0
    v2 = np.zeros(n)
    v2[0] = 1.0 / math.sqrt(2.0)
    v2[1] = 1.0 / math.sqrt(2.0)
    v3 = np.full(n, 1.0 / math.sqrt(n))
    xi1 = _full_sym_perms(v1)
    xi2 = _full_sym_perms(v2)
    xi3 = _full_sym_perms(v3)
    points = np.vstack([xi1, xi2, xi3])
    w = np.concatenate(
        [
            np.full(xi1.shape[0], A1),
            np.full(xi2.shape[0], A2),
            np.full(xi3.shape[0], A3),
        ]
    )
    return points, w


def _ball_degree7(
    n: int, algorithm: Optional[int], alpha: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if alpha != 0.0:
        raise ValueError(
            "degree 7 (seventhOrderSpherCubPoints.m) does not expose alpha "
            "(weight fixed at 1); got alpha != 0"
        )
    if algorithm == 0:  # Sn 7-2, 2^(n+1)+4n^2 points, n>=3.
        if n < 3:
            raise ValueError(f"algorithm 0 (Sn 7-2) requires n >= 3, got {n}")
        u, B = _seventh_order_sphere_surface_alg0(n)
        disc = math.sqrt(2.0 * (n + 2.0) * (n + 4.0))
        r1 = math.sqrt(((n + 2.0) * (n + 4.0) - 2.0 * disc) / ((n + 4.0) * (n + 6.0)))
        r2 = math.sqrt(((n + 2.0) * (n + 4.0) + 2.0 * disc) / ((n + 4.0) * (n + 6.0)))
        A1 = (2.0 * (n + 2.0) ** 2 - (n - 2.0) * disc) / (4.0 * n * (n + 2.0) ** 2)
        A2 = (2.0 * (n + 2.0) ** 2 + (n - 2.0) * disc) / (4.0 * n * (n + 2.0) ** 2)
        points = np.vstack([r1 * u, r2 * u])
        w = np.concatenate([A1 * B, A2 * B])
        return points, w
    if algorithm == 1:  # S2 7-1, 12 points, n=2, corrected B2 coefficient
        # (4 -> 41) already applied by MATLAB itself, per its own docstring.
        if n != 2:
            raise ValueError(f"algorithm 1 (S2 7-1) requires n == 2, got {n}")
        V = _ball_volume(2, 0.0)
        r = math.sqrt(3.0 / 4.0)
        s = math.sqrt((27.0 - 3.0 * math.sqrt(29.0)) / 104.0)
        t = math.sqrt((27.0 + 3.0 * math.sqrt(29.0)) / 104.0)
        B1 = V * 2.0 / 27.0
        B2 = V * (551.0 + 41.0 * math.sqrt(29.0)) / 6264.0
        B3 = V * (551.0 - 41.0 * math.sqrt(29.0)) / 6264.0
        points = np.vstack(
            [
                _full_sym_perms(np.array([r, 0.0])),
                _pm_combos([s, s]),
                _pm_combos([t, t]),
            ]
        )
        w = np.concatenate([np.full(4, B1), np.full(4, B2), np.full(4, B3)])
        return points, w
    if algorithm == 2:  # S2 7-2, 16 points, n=2. Corrected: see module
        # docstring's fourth defect -- second coordinate row uses sin, not
        # the cos MATLAB's source literally has (verified numerically
        # against the ball-monomial oracle).
        if n != 2:
            raise ValueError(f"algorithm 2 (S2 7-2) requires n == 2, got {n}")
        V = _ball_volume(2, 0.0)
        r1 = math.sqrt((3.0 - math.sqrt(3.0)) / 6.0)
        r2 = math.sqrt((3.0 + math.sqrt(3.0)) / 6.0)
        w = np.full(16, V / 16.0)
        k = np.arange(1, 9)
        xi = np.zeros((2, 16))
        xi[0, 0:8] = r1 * np.cos((2 * k - 1) * np.pi / 8.0)
        xi[1, 0:8] = r1 * np.sin((2 * k - 1) * np.pi / 8.0)
        xi[0, 8:16] = r2 * np.cos((2 * k - 1) * np.pi / 8.0)
        xi[1, 8:16] = r2 * np.sin((2 * k - 1) * np.pi / 8.0)
        return xi.T, w
    if algorithm == 3:  # S3 7-2, 32 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 3 (S3 7-2) requires n == 3, got {n}")
        V = _ball_volume(3, 0.0)
        r = math.sqrt((1715.0 - 7.0 * math.sqrt(17770.0)) / 2817.0)
        s = math.sqrt((1715.0 + 7.0 * math.sqrt(17770.0)) / 2817.0)
        t = math.sqrt(7.0 / 18.0)
        u = math.sqrt(7.0 / 27.0)
        B1 = (
            V
            * (2965.0 * math.sqrt(17770.0) + 227816.0)
            / (72030.0 * math.sqrt(17770.0))
        )
        B2 = (
            V
            * (2965.0 * math.sqrt(17770.0) - 227816.0)
            / (72030.0 * math.sqrt(17770.0))
        )
        B3 = 324.0 * V / 12005.0
        B4 = 2187.0 * V / 96040.0
        points = np.vstack(
            [
                _full_sym_perms(np.array([r, 0.0, 0.0])),
                _full_sym_perms(np.array([s, 0.0, 0.0])),
                _full_sym_perms(np.array([t, t, 0.0])),
                _pm_combos([u, u, u]),
            ]
        )
        w = np.concatenate(
            [np.full(6, B1), np.full(6, B2), np.full(12, B3), np.full(8, B4)]
        )
        return points, w
    if algorithm == 4:  # S3 7-3, 33 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 4 (S3 7-3) requires n == 3, got {n}")
        V = _ball_volume(3, 0.0)
        r = math.sqrt((5.0 + math.sqrt(5.0)) / 18.0)
        s = math.sqrt((5.0 - math.sqrt(5.0)) / 18.0)
        u = math.sqrt((3.0 - math.sqrt(5.0)) / 6.0)
        v = math.sqrt((3.0 + math.sqrt(5.0)) / 6.0)
        t = 1.0 / math.sqrt(3.0)
        B0 = 16.0 * V / 175.0
        B1 = 81.0 * V / 1400.0
        B2 = 3.0 * V / 280.0
        points = np.vstack(
            [
                np.zeros((1, 3)),
                _pm_combos([r, s, 0.0]),
                _pm_combos([0.0, r, s]),
                _pm_combos([s, 0.0, r]),
                _pm_combos([u, v, 0.0]),
                _pm_combos([0.0, u, v]),
                _pm_combos([v, 0.0, u]),
                _pm_combos([t, t, t]),
            ]
        )
        w = np.concatenate([[B0], np.full(12, B1), np.full(20, B2)])
        return points, w
    if algorithm == 5:  # S4 7-2, 72 points, n=4.
        if n != 4:
            raise ValueError(f"algorithm 5 (S4 7-2) requires n == 4, got {n}")
        V = _ball_volume(4, 0.0)
        r = math.sqrt((39.0 - 3.0 * math.sqrt(41.0)) / 64.0)
        s = math.sqrt((39.0 + 3.0 * math.sqrt(41.0)) / 64.0)
        t = 1.0 / math.sqrt(2.0)
        u = 0.5
        B1 = V * (33.0 * math.sqrt(41.0) + 109.0) / (1440.0 * math.sqrt(41.0))
        B2 = V * (33.0 * math.sqrt(41.0) - 109.0) / (1440.0 * math.sqrt(41.0))
        B3 = V / 240.0
        B4 = V / 60.0
        points = np.vstack(
            [
                _full_sym_perms(np.array([r, 0.0, 0.0, 0.0])),
                _full_sym_perms(np.array([s, 0.0, 0.0, 0.0])),
                _full_sym_perms(np.array([t, t, 0.0, 0.0])),
                _full_sym_perms(np.array([u, u, u, 0.0])),
            ]
        )
        w = np.concatenate(
            [np.full(8, B1), np.full(8, B2), np.full(24, B3), np.full(32, B4)]
        )
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 7; expected one of 0-5"
    )


def _quad1d_gegenbauer(
    m: int, c1: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``quadraturePoints1D.m`` algorithm 3 equivalent: ``m``-point Gauss
    quadrature for weight ``(1-x**2)**(c1-0.5)`` on ``(-1, 1)``, ``c1 >
    -0.5``, exact through degree ``2*m-1``. Direct correspondence to
    ``scipy.special.roots_jacobi(m, c1-0.5, c1-0.5)`` -- see module
    docstring's "arbOrderSpherCubPoints.m port" note for the verification
    (including why, unlike MATLAB's own hand-rolled recursion, no ``c1==0``
    special case is needed here).
    """
    lam = c1 - 0.5
    return roots_jacobi(m, lam, lam)


def _quad1d_abs_x_pow(
    m: int, c1: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """General real-``c1`` (``c1 > -1``) Gauss quadrature for weight
    ``|x|**c1`` on ``(-1, 1)``, ``m`` points, exact through degree
    ``2*m-1``. NOT a transcription of ``quadraturePoints1D.m`` algorithm 8
    (that recursion only supports nonnegative INTEGER ``c1``); derived
    instead via the classical even-weight symmetrization technique and
    independently verified two ways -- see module docstring's
    "arbOrderSpherCubPoints.m port" note.
    """
    k = m // 2
    if m % 2 == 0:
        p1 = (c1 - 1.0) / 2.0
        x, w = roots_jacobi(k, 0.0, p1)
        t = (x + 1.0) / 2.0
        big_w = w * (0.5 ** (p1 + 1.0))
        a = big_w / 2.0
        nodes = np.concatenate([np.sqrt(t), -np.sqrt(t)])
        weights = np.concatenate([a, a])
    else:
        p1 = (c1 - 1.0) / 2.0
        p2 = (c1 + 1.0) / 2.0
        x, w = roots_jacobi(k, 0.0, p2)
        t = (x + 1.0) / 2.0
        big_w = w * (0.5 ** (p2 + 1.0))
        a = big_w / (2.0 * t)
        m0 = 1.0 / (p1 + 1.0)
        a0 = m0 - 2.0 * np.sum(a)
        nodes = np.concatenate([[0.0], np.sqrt(t), -np.sqrt(t)])
        weights = np.concatenate([[a0], a, a])
    return nodes, weights


def _ball_arb_order(
    n: int, order: int, alpha: float
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """``Sphere/arbOrderSpherCubPoints.m``: general-dimension, general-order
    (degree ``2*order-1``) ball rule via Stroud Theorem 2.6-2's recursive
    Gegenbauer/radial construction -- see module docstring for the
    ``quadraturePoints1D`` dependency derivations. Direct transcription of
    the MATLAB recursive point-assembly loop (1-based indices translated to
    0-based; ``index2NDim``'s odometer enumeration replaced with
    ``itertools.product``, order-independent so this is a faithful,
    not merely equivalent, substitution).
    """
    c1_radial = n - 1.0 + alpha
    if c1_radial <= -1.0:
        raise ValueError(
            f"alpha={alpha} requires n - 1 + alpha > -1 (i.e. alpha > -n), "
            f"got n - 1 + alpha = {c1_radial}"
        )
    r, ar = _quad1d_abs_x_pow(order, c1_radial)

    y = np.zeros((order, n - 1))
    ak = np.zeros((order, n - 1))
    for k in range(1, n):
        y_cur, ak_cur = _quad1d_gegenbauer(order, (k - 1) / 2.0)
        y[:, k - 1] = y_cur
        ak[:, k - 1] = ak_cur

    num_points = order**n
    xi = np.zeros((n, num_points))
    w = np.zeros(num_points)

    cur = 0
    for cur_r in range(order):
        for idx in itertools.product(range(order), repeat=n - 1):
            xi[n - 1, cur] = r[cur_r] * y[idx[n - 2], n - 2]
            w_prod = ak[idx[n - 2], n - 2]
            prod_recur = 1.0
            for k in range(n - 1, 1, -1):
                prod_recur *= math.sqrt(1.0 - y[idx[k - 1], k - 1] ** 2)
                xi[k - 1, cur] = r[cur_r] * prod_recur * y[idx[k - 2], k - 2]
                w_prod *= ak[idx[k - 2], k - 2]
            prod_recur *= math.sqrt(1.0 - y[idx[0], 0] ** 2)
            xi[0, cur] = r[cur_r] * prod_recur
            w[cur] = w_prod * ar[cur_r]
            cur += 1
    return xi.T, w


def ball_cubature_points(
    n: int, degree: int, algorithm: Optional[int] = None, alpha: float = 0.0
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Cubature points for the unit n-ball ``{x : |x| <= 1}``, weight ``|x|**alpha``.

    Counterpart of the MATLAB TCL's ``Sphere`` top-level, general-dimension
    files (see this module's docstring for the exact per-degree algorithm
    coverage, the two excluded ``Sphere`` files with reasons, and the
    corrected/confirmed issues specific to this region -- a
    documentation-only ``alpha``-sign defect and a wrong-formula defect in
    degree 7 algorithm 2). Every odd ``degree >= 9`` dispatches to a
    from-scratch general-order, general-``alpha`` construction (folding in
    ``arbOrderSpherCubPoints.m``, see module docstring); every other
    ``degree`` is exact through that total polynomial degree -- verified
    against the closed-form ball-monomial oracle in
    ``tests/unit/test_region_cubature.py`` for the ``(n, degree, algorithm,
    alpha)`` grid its test classes sweep; no wider claim is made (per the
    claims-inherit-measurement-range convention).

    **Weight convention (region measure, not probability).** ``weights`` sum
    to the ``|x|**alpha``-weighted unit-ball measure ``2/(n+alpha) *
    pi**(n/2) / gamma(n/2)`` (the plain unit-ball volume ``pi**(n/2) /
    gamma(n/2+1)`` when ``alpha == 0``), NOT to 1 -- this module targets the
    plain (alpha-weighted) Lebesgue measure on the unit ball, unlike
    :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`'s
    Gaussian-weight rules. A caller wanting a probability-normalized rule
    divides by the volume themselves: ``weights / weights.sum()``. NOTE the
    sign of ``alpha`` here is CORRECTED relative to MATLAB's own docstrings
    (which describe the weight as ``|x|**(-alpha)``) -- see the module
    docstring's "confirmed MATLAB documentation defect" note; this module's
    ``alpha`` matches what the MATLAB CODE actually computes, not what its
    comments claim.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    degree : int
        Polynomial degree the rule is exact through. One of 2, 3, 5, 7, or
        any ODD degree >= 9 (``9, 11, 13, ...``): 2/3/5/7 are the top-level,
        named-formula degrees MATLAB's ``Sphere`` directory provides;
        degree 9 and above dispatch to the general-order
        ``arbOrderSpherCubPoints.m`` port (module docstring) at
        ``order = (degree + 1) // 2`` -- an EVEN degree >= 9 (e.g. 10) has
        no MATLAB formula in this directory at all and raises ``ValueError``
        (``ninthOrderSpherCubPoints.m``/``eleventhOrderSpherCubPoints.m``
        are ``n == 2``-only named files, superseded here by the general-``n``
        ``arbOrderSpherCubPoints`` path at the same degrees -- see module
        docstring).
    algorithm : int, optional
        Which MATLAB algorithm variant to use; see each degree's section in
        the module docstring for the ported subset. Default None reproduces
        MATLAB's own default selection for that degree:

        - degree 2: algorithm 0 (the only variant; MATLAB's
          ``secondOrderSpherCubPoints.m`` takes no algorithm argument, and
          does not expose ``alpha`` at all -- weight fixed at 1).
        - degree 3: algorithm 0 (Sn 3-1, 2n points, alpha-capable).
          Algorithms 1 (2^n points, alpha=0), 2/3 (S2 3-1/3-2, n=2,
          alpha=0), 4 (S3 3-1, n=3, alpha=0) are the other ported variants.
        - degree 5: algorithm 0 (Sn 5-2, 2n^2+1 points, alpha-capable).
          Algorithms 1 (Sn 5-3, alpha=0), 2 (Sn 5-4, alpha-capable), 3
          (Sn 5-5, alpha=0), 4 (Sn 5-6, alpha=0) are general-n; 5 (S2 5-1,
          n=2, alpha-capable), 6 (S2 5-2, n=2, alpha=0), 7 (S3 5-1, n=3,
          alpha-capable), 8 (S3 5-2, n=3, alpha=0), 9 (S4 5-1, n=4, alpha=0)
          are the fixed-dimension variants.
        - degree 7: algorithm 0 (Sn 7-2, general n>=3) always, matching
          MATLAB's own unconditional default (``seventhOrderSpherCubPoints.m``
          does not auto-select per ``n`` the way ``fifthOrderSimplexCubPoints.m``
          does elsewhere in this module -- calling this with ``n == 2`` and
          no explicit ``algorithm`` raises ``ValueError`` from algorithm 0's
          own ``n >= 3`` guard, matching real MATLAB's behavior exactly; pass
          ``algorithm=2`` explicitly for the n=2 case). No algorithm here
          exposes ``alpha`` at all. Algorithm 0 depends on a private port of
          ``seventhOrderSpherSurfCubPoints.m`` algorithm 0 (see
          :func:`_seventh_order_sphere_surface_alg0`); algorithms 1 (S2 7-1,
          n=2), 2 (S2 7-2, n=2, corrected -- see module docstring), 3
          (S3 7-2, n=3), 4 (S3 7-3, n=3), 5 (S4 7-2, n=4) are the
          fixed-dimension variants.
        - degree >= 9 (odd): algorithm must be None or 0 (MATLAB's
          ``arbOrderSpherCubPoints.m`` takes no algorithm argument);
          alpha-capable for any real ``alpha > -n``.
    alpha : float, optional
        Exponent of the radial weighting function ``|x|**alpha``,
        ``alpha > -n``. Default 0.0 (plain Lebesgue ball measure). Not every
        algorithm supports ``alpha != 0`` -- see the per-degree notes above;
        an unsupported nonzero ``alpha`` raises ``ValueError``.

    Returns
    -------
    points : ndarray
        Shape (num_points, n).
    weights : ndarray
        Shape (num_points,), summing to the ``|x|**alpha``-weighted ball
        measure (see above), not 1.

    Examples
    --------
    >>> pts, w = ball_cubature_points(3, 3)
    >>> pts.shape
    (6, 3)
    >>> round(float(w.sum()), 9)  # unit 3-ball volume, 4*pi/3
    4.188790205
    >>> round(float(np.sum(w * pts[:, 0] ** 2)), 9)  # integral of x^2, 4*pi/15
    0.837758041

    References
    ----------
    A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
    Prentice-Hall, 1971, Formulas Sn 2-1, Sn 3-1/3-2, S2 3-1/3-2, S3 3-1,
    Sn 5-2 through 5-6, S2 5-1/5-2, S3 5-1/5-2, S4 5-1, Sn 7-2, S2 7-1/7-2,
    S3 7-2/7-3, S4 7-2, pp. 267-292.

    A. H. Stroud, "Some seventh degree integration formulas for the surface
    of an n-sphere," Numerische Mathematik, vol. 11, no. 3, pp. 273-276,
    Mar. 1968.

    R. Cools, "An encyclopedia of cubature formulas," Journal of
    Complexity, vol. 19, no. 3, pp. 445-453, Jun. 2003.
    """
    if n < 2:
        raise ValueError(f"dimension must be >= 2, got {n}")
    if degree == 2:
        return _ball_degree2(n, algorithm, alpha)
    if degree == 3:
        return _ball_degree3(n, algorithm, alpha)
    if degree == 5:
        return _ball_degree5(n, algorithm, alpha)
    if degree == 7:
        return _ball_degree7(n, algorithm, alpha)
    if degree >= 9 and degree % 2 == 1:
        if algorithm not in (None, 0):
            raise ValueError(
                f"algorithm {algorithm} not ported for degree {degree}; "
                "expected 0 (MATLAB's arbOrderSpherCubPoints.m has no "
                "algorithm parameter at all)"
            )
        order = (degree + 1) // 2
        return _ball_arb_order(n, order, alpha)
    raise ValueError(
        f"unsupported degree {degree}; expected 2, 3, 5, 7, or any odd degree >= 9"
    )


def _sphsurf_area(n: int) -> float:
    """integral_{S^(n-1)} dS(x) = 2*pi**(n/2) / gamma(n/2), the unit
    sphere's surface area (module docstring's Spherical_Surface ``V``).
    """
    return 2.0 * math.pi ** (n / 2.0) / math.gamma(n / 2.0)


def _regular_n_simplex_coords(n: int) -> NDArray[np.floating]:
    """``regularNSimplexCoords.m`` (Geometry, outside this ported subset),
    method 0 (cosine-based) -- the only method any Spherical_Surface caller
    below needs. Returns shape ``(n, n+1)``: the n+1 vertices of a regular
    n-simplex, each at unit distance from the origin, centroid 0 (columns
    are vertices, MATLAB's own orientation -- callers below transpose as
    needed for this module's ``(num_points, n)`` row convention).
    """
    points = np.zeros((n, n + 1))
    r = np.arange(1, n // 2 + 1)
    for k in range(n + 1):
        points[2 * r - 2, k] = np.cos(2 * r * k * np.pi / (n + 1))
        points[2 * r - 1, k] = np.sin(2 * r * k * np.pi / (n + 1))
        if n % 2 != 0:
            points[n - 1, k] = (-1.0) ** k / np.sqrt(2.0)
    return points / np.sqrt(n / 2.0)


def _regular_icosahedron_coords() -> NDArray[np.floating]:
    """``regularIcosahedronCoords.m`` (Geometry, outside this ported
    subset): the 12 vertices of a regular icosahedron, each at unit
    distance from the origin -- used by
    ``fifthOrderSpherSurfCubPoints.m`` algorithm 5 (U3 5-1, n=3 only).
    Returns shape (12, 3) (rows are vertices, this module's convention).
    """
    r = math.sqrt((5.0 + math.sqrt(5.0)) / 10.0)
    s = math.sqrt((5.0 - math.sqrt(5.0)) / 10.0)
    return np.array(
        [
            [r, s, 0.0],
            [r, -s, 0.0],
            [-r, s, 0.0],
            [-r, -s, 0.0],
            [0.0, r, s],
            [0.0, r, -s],
            [0.0, -r, s],
            [0.0, -r, -s],
            [s, 0.0, r],
            [s, 0.0, -r],
            [-s, 0.0, r],
            [-s, 0.0, -r],
        ]
    )


def _sphsurf_axis_block(n: int, k: int, val: float) -> NDArray[np.floating]:
    """``fullSymPerms`` of a vector with ``k`` entries equal to ``val`` and
    the rest zero -- the shared point-block pattern used by every
    Spherical_Surface "Un" formula below (Stroud's Un-family notation
    places ``k`` nonzero coordinates, typically ``val = 1/sqrt(k)``, at
    every position/sign combination)."""
    vec = np.concatenate([np.full(k, val), np.zeros(n - k)])
    return _full_sym_perms(vec)


def _sphsurf_degree1(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm not in (None, 0):
        raise ValueError(
            f"algorithm {algorithm} not ported for degree 1; expected 0 "
            "(MATLAB's firstOrderSpherSurfCubPoints.m has no algorithm "
            "parameter at all)"
        )
    V = _sphsurf_area(n)
    points = np.zeros((2, n))
    points[0, 0] = 1.0
    points[1, 0] = -1.0
    w = np.full(2, V / 2.0)
    return points, w


def _sphsurf_degree3(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = _sphsurf_area(n)
    if algorithm == 0:  # Un 3-1, 2n points.
        points = np.vstack([np.eye(n), -np.eye(n)])
        w = np.full(2 * n, V / (2.0 * n))
        return points, w
    if algorithm == 1:  # Un 3-2, 2^n points.
        r = 1.0 / math.sqrt(n)
        points = _pm_combos(r * np.ones(n))
        w = np.full(points.shape[0], V * 2.0 ** (-n))
        return points, w
    if algorithm == 2:  # Mysovskikh, 2*(n+1) points.
        v = _regular_n_simplex_coords(n)
        points = np.vstack([v.T, -v.T])
        w = np.full(2 * (n + 1), V / (2.0 * (n + 1)))
        return points, w
    if algorithm == 3:  # U3 3-1, 12 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 3 (U3 3-1) requires n == 3, got {n}")
        r = 1.0 / math.sqrt(2.0)
        points = _full_sym_perms(np.array([r, r, 0.0]))
        w = np.full(12, V / 12.0)
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 3; expected one of 0-3"
    )


def _sphsurf_degree5(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    V = _sphsurf_area(n)
    if algorithm == 0:  # Un 5-1, 2n^2 points. Negative weights for n>4.
        s = 1.0 / math.sqrt(2.0)
        B1 = (4.0 - n) / (2.0 * n * (n + 2.0)) * V
        B2 = 1.0 / (n * (n + 2.0)) * V
        xi1 = _sphsurf_axis_block(n, 1, 1.0)
        # MATLAB's own nested pair-loop (curIdx1=1:(numDim-1)) is simply
        # empty at n=1 (MATLAB's `1:0` range), leaving this second block
        # empty rather than erroring; matched here explicitly since
        # _sphsurf_axis_block's vectorized construction would otherwise
        # need a negative-length np.zeros(n - 2) at n=1.
        if n >= 2:
            xi2 = _full_sym_perms(np.concatenate([[s, s], np.zeros(n - 2)]))
        else:
            xi2 = np.zeros((0, n))
        points = np.vstack([xi1, xi2])
        w = np.concatenate([np.full(xi1.shape[0], B1), np.full(xi2.shape[0], B2)])
        return points, w
    if algorithm == 1:  # Un 5-2, 2^n+2n points.
        s = 1.0 / math.sqrt(n)
        B1 = 1.0 / (n * (n + 2.0)) * V
        B2 = n / (2.0**n * (n + 2.0)) * V
        xi1 = _sphsurf_axis_block(n, 1, 1.0)
        xi2 = _pm_combos(s * np.ones(n))
        points = np.vstack([xi1, xi2])
        w = np.concatenate([np.full(xi1.shape[0], B1), np.full(xi2.shape[0], B2)])
        return points, w
    if algorithm == 2:  # Un 5-3, 2^(n+1)-2 points.
        s = 1.0 / math.sqrt(n + 2.0)
        vec = np.full(n, s)
        points_list = []
        weights_list = []
        for k in range(1, n + 1):
            r = math.sqrt((k + 2.0) / (n + 2.0))
            B = V * 2.0 ** (k - n) * (n + 2.0) / (n * (k + 1.0) * (k + 2.0))
            vec[k - 1] = r
            combos = _pm_combos(vec[k - 1 :])
            num_cur = combos.shape[0]
            block = np.zeros((num_cur, n))
            block[:, k - 1 :] = combos
            points_list.append(block)
            weights_list.append(np.full(num_cur, B))
        return np.vstack(points_list), np.concatenate(weights_list)
    if algorithm == 3:  # Un 5-4, n*2^n points.
        u = math.sqrt(
            (n + 2.0 + (n - 1.0) * math.sqrt(2.0 * (n + 2.0))) / (n * (n + 2.0))
        )
        v = math.sqrt((n + 2.0 - math.sqrt(2.0 * (n + 2.0))) / (n * (n + 2.0)))
        num_points = n * 2**n
        w = np.full(num_points, V / num_points)
        blocks = []
        for spot in range(n):
            vec_cur = np.full(n, v)
            vec_cur[spot] = u
            blocks.append(_pm_combos(vec_cur))
        return np.vstack(blocks), w
    if algorithm == 4:  # Mysovskikh, (n+1)*(n+2) points. Negative weights
        # for n>7.
        v = _regular_n_simplex_coords(n)
        w1 = V * (7.0 - n) * n / (2.0 * (n + 1.0) ** 2 * (n + 2.0))
        w2 = V * 2.0 * (n - 1.0) ** 2 / (n * (n + 1.0) ** 2 * (n + 2.0))
        block1 = np.vstack([-v.T, v.T])
        edges = []
        for i in range(n):
            for j in range(i + 1, n + 1):
                y = (v[:, i] + v[:, j]) / 2.0
                y = y / np.linalg.norm(y)
                edges.append(-y)
                edges.append(y)
        edges = np.array(edges)
        points = np.vstack([block1, edges])
        w = np.concatenate([np.full(block1.shape[0], w1), np.full(edges.shape[0], w2)])
        return points, w
    if algorithm == 5:  # U3 5-1, 12 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 5 (U3 5-1) requires n == 3, got {n}")
        points = _regular_icosahedron_coords()
        w = np.full(12, V / 12.0)
        return points, w
    if algorithm == 6:  # U3 5-2, 14 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 6 (U3 5-2) requires n == 3, got {n}")
        s = 1.0 / math.sqrt(3.0)
        B1 = (8.0 / 120.0) * V
        B2 = (9.0 / 120.0) * V
        points = np.vstack(
            [_full_sym_perms(np.array([1.0, 0.0, 0.0])), _pm_combos([s, s, s])]
        )
        w = np.concatenate([np.full(6, B1), np.full(8, B2)])
        return points, w
    if algorithm == 7:  # U3 5-3, 18 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 7 (U3 5-3) requires n == 3, got {n}")
        s = 1.0 / math.sqrt(2.0)
        B1 = (1.0 / 30.0) * V
        B2 = (2.0 / 30.0) * V
        points = np.vstack(
            [
                _full_sym_perms(np.array([1.0, 0.0, 0.0])),
                _full_sym_perms(np.array([s, s, 0.0])),
            ]
        )
        w = np.concatenate([np.full(6, B1), np.full(12, B2)])
        return points, w
    if algorithm == 8:  # U3 5-5, 30 points, n=3. See module docstring's
        # fifth confirmed finding: MATLAB's own docstring mislabels this
        # (claims index 8 is a different, never-implemented 20-point
        # formula and index 9 is this 30-point one) -- this is what the
        # CODE at switch-case index 8 actually computes; index 9 does not
        # exist in the switch at all (raises "Unknown algorithm specified"
        # in real MATLAB).
        if n != 3:
            raise ValueError(f"algorithm 8 (U3 5-5) requires n == 3, got {n}")
        r = 0.5
        s = (math.sqrt(5.0) + 1.0) / 4.0
        t = (math.sqrt(5.0) - 1.0) / 4.0
        B = V / 30.0
        points = np.vstack(
            [
                _full_sym_perms(np.array([1.0, 0.0, 0.0])),
                _pm_combos([r, s, t]),
                _pm_combos([t, r, s]),
                _pm_combos([s, t, r]),
            ]
        )
        w = np.full(30, B)
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 5; expected one of "
        "0-8 (MATLAB's own docstring claims algorithms up to 9 exist, but "
        "only 0-8 are actually reachable via the switch statement -- see "
        "module docstring)"
    )


def _sphsurf_degree7(
    n: int, algorithm: Optional[int]
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    if algorithm is None:
        algorithm = 0
    if n < 2:
        # seventhOrderSpherSurfCubPoints.m's own header states "numDim>=2"
        # for the FILE as a whole (unlike third/fifth order, which state no
        # such bound); individual algorithms further restrict n above.
        raise ValueError(
            f"degree 7 (seventhOrderSpherSurfCubPoints.m) requires n >= 2, got {n}"
        )
    if algorithm == 0:  # Formula I from [1], 2^n+2n^2 points, n>=3.
        if n < 3:
            raise ValueError(f"algorithm 0 (Formula I) requires n >= 3, got {n}")
        return _seventh_order_sphere_surface_alg0(n)
    if algorithm == 1:  # Formula II from [1], n>=4.
        if n < 4:
            raise ValueError(f"algorithm 1 (Formula II) requires n >= 4, got {n}")
        V = _sphsurf_area(n)
        A1 = (14.0 - n) / (2.0 * n * (n + 2.0) * (n + 4.0)) * V
        A2 = 27.0 / (4.0 * n * (n + 2.0) * (n + 4.0) * (n - 3.0)) * V
        A3 = n**3 * (n - 5.0) / (2.0**n * n * (n + 2.0) * (n + 4.0) * (n - 3.0)) * V
        xi1 = _sphsurf_axis_block(n, 1, 1.0)
        xi2 = _sphsurf_axis_block(n, 3, 1.0 / math.sqrt(3.0))
        xi3 = _sphsurf_axis_block(n, n, 1.0 / math.sqrt(n))
        points = np.vstack([xi1, xi2, xi3])
        w = np.concatenate(
            [
                np.full(xi1.shape[0], A1),
                np.full(xi2.shape[0], A2),
                np.full(xi3.shape[0], A3),
            ]
        )
        return points, w
    if algorithm == 2:  # Formula III from [1], n>=4.
        if n < 4:
            raise ValueError(f"algorithm 2 (Formula III) requires n >= 4, got {n}")
        V = _sphsurf_area(n)
        A1 = 4.0 * (14.0 - n) / (n * (n + 2.0) * (n + 4.0) * (n - 2.0)) * V
        A2 = (
            27.0
            * (n - 8.0)
            / (2.0 * n * (n + 2.0) * (n + 4.0) * (n - 2.0) * (n - 3.0))
            * V
        )
        A3 = (
            n**3
            * (n**2 - 9.0 * n + 38.0)
            / (2.0**n * n * (n + 2.0) * (n + 4.0) * (n - 2.0) * (n - 3.0))
            * V
        )
        xi1 = _sphsurf_axis_block(n, 2, 1.0 / math.sqrt(2.0))
        xi2 = _sphsurf_axis_block(n, 3, 1.0 / math.sqrt(3.0))
        xi3 = _sphsurf_axis_block(n, n, 1.0 / math.sqrt(n))
        points = np.vstack([xi1, xi2, xi3])
        w = np.concatenate(
            [
                np.full(xi1.shape[0], A1),
                np.full(xi2.shape[0], A2),
                np.full(xi3.shape[0], A3),
            ]
        )
        return points, w
    if algorithm == 3:  # Formula from [2], 2^n+2n^2 points, n>=3. Verified
        # (TestSpherSurfDegree7's test_algorithm_0_3_4_compute_identical_rule)
        # to compute the IDENTICAL rule as algorithm 0 -- the same three
        # weight formulas assigned to the same three point blocks, just
        # built in a different order (module docstring).
        if n < 3:
            raise ValueError(f"algorithm 3 requires n >= 3, got {n}")
        V = _sphsurf_area(n)
        A1 = (8.0 - n) / (n * (n + 2.0) * (n + 4.0)) * V
        A2 = 2.0 ** (-n) * n**3 / (n * (n + 2.0) * (n + 4.0)) * V
        A3 = 4.0 / (n * (n + 2.0) * (n + 4.0)) * V
        xi1 = _sphsurf_axis_block(n, 1, 1.0)
        xi2 = _sphsurf_axis_block(n, n, 1.0 / math.sqrt(n))
        xi3 = _sphsurf_axis_block(n, 2, 1.0 / math.sqrt(2.0))
        points = np.vstack([xi1, xi2, xi3])
        w = np.concatenate(
            [
                np.full(xi1.shape[0], A1),
                np.full(xi2.shape[0], A2),
                np.full(xi3.shape[0], A3),
            ]
        )
        return points, w
    if algorithm == 4:  # Un 7-1 from [3], 2^n+2n^2 points, n>=2. Also
        # verified identical to algorithms 0 and 3 (module docstring).
        V = _sphsurf_area(n)
        B = (8.0 - n) / (n * (n + 2.0) * (n + 4.0)) * V
        C = (2.0 ** (-n) * n**3) / (n * (n + 2.0) * (n + 4.0)) * V
        D = 4.0 / (n * (n + 2.0) * (n + 4.0)) * V
        xi1 = _sphsurf_axis_block(n, 1, 1.0)
        xi2 = _pm_combos((1.0 / math.sqrt(n)) * np.ones(n))
        xi3 = _sphsurf_axis_block(n, 2, 1.0 / math.sqrt(2.0))
        points = np.vstack([xi1, xi2, xi3])
        w = np.concatenate(
            [
                np.full(xi1.shape[0], B),
                np.full(xi2.shape[0], C),
                np.full(xi3.shape[0], D),
            ]
        )
        return points, w
    if algorithm == 5:  # Un 7-2 from [3], 2^n*(n+1) points. (n>=2 already
        # enforced by the blanket check above.)
        V = _sphsurf_area(n)
        r = math.sqrt(1.0 / n)
        s = math.sqrt(5.0 / (n + 4.0))
        t = math.sqrt(1.0 / (n + 4.0))
        A = -(n**2) / (2.0 ** (n + 3.0) * (n + 2.0)) * V
        B = (n + 4.0) ** 2 / (2.0 ** (n + 3.0) * n * (n + 2.0)) * V
        xi1 = _pm_combos(r * np.ones(n))
        xi2 = _full_sym_perms(np.concatenate([[s], t * np.ones(n - 1)]))
        points = np.vstack([xi1, xi2])
        w = np.concatenate([np.full(xi1.shape[0], A), np.full(xi2.shape[0], B)])
        return points, w
    if algorithm == 6:  # U3 7-1, 24 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 6 (U3 7-1) requires n == 3, got {n}")
        coeffs = [1.0, 0.0, -1.0, 0.0, 1.0 / 5.0, 0.0, -1.0 / 105.0]
        roots = np.roots(coeffs)
        real_pos = sorted(
            (rt.real for rt in roots if abs(rt.imag) < 1e-9 and rt.real > 0),
            reverse=True,
        )
        r, s, t = real_pos[0], real_pos[1], real_pos[2]
        u_vals = np.array([r, -r, s, -s, t, -t])
        v_vals = np.array([s, t, t, r, r, s])
        z_vals = np.array([t, s, r, t, s, r])
        points = np.zeros((24, 3))
        for i in range(6):
            points[i] = [u_vals[i], v_vals[i], z_vals[i]]
            points[i + 6] = [u_vals[i], -v_vals[i], -z_vals[i]]
            points[i + 12] = [u_vals[i], z_vals[i], -v_vals[i]]
            points[i + 18] = [u_vals[i], -z_vals[i], v_vals[i]]
        V = _sphsurf_area(3)
        w = np.full(24, V / 24.0)
        return points, w
    if algorithm == 7:  # U3 7-2, 26 points, n=3.
        if n != 3:
            raise ValueError(f"algorithm 7 (U3 7-2) requires n == 3, got {n}")
        s = 1.0 / math.sqrt(2.0)
        t = 1.0 / math.sqrt(3.0)
        V = _sphsurf_area(3)
        xi1 = _full_sym_perms(np.array([1.0, 0.0, 0.0]))
        xi2 = _full_sym_perms(np.array([s, s, 0.0]))
        xi3 = _pm_combos([t, t, t])
        points = np.vstack([xi1, xi2, xi3])
        w = np.concatenate(
            [
                np.full(6, (40.0 / 840.0) * V),
                np.full(12, (32.0 / 840.0) * V),
                np.full(8, (27.0 / 840.0) * V),
            ]
        )
        return points, w
    if algorithm == 8:  # U4 7-1, 48 points, n=4.
        if n != 4:
            raise ValueError(f"algorithm 8 (U4 7-1) requires n == 4, got {n}")
        s = 0.5
        t = 1.0 / math.sqrt(2.0)
        V = _sphsurf_area(4)
        xi1 = _full_sym_perms(np.array([1.0, 0.0, 0.0, 0.0]))
        xi2 = _pm_combos([s, s, s, s])
        xi3 = _full_sym_perms(np.array([t, t, 0.0, 0.0]))
        points = np.vstack([xi1, xi2, xi3])
        w = np.full(48, V / 48.0)
        return points, w
    raise ValueError(
        f"algorithm {algorithm} not ported for degree 7; expected one of 0-8"
    )


def spherical_surface_cubature_points(
    n: int, degree: int, algorithm: Optional[int] = None
) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Cubature points for the unit sphere surface ``S^(n-1) = {x : |x| = 1}``.

    Counterpart of the MATLAB TCL's ``Spherical_Surface`` top-level files
    (see this module's docstring for the full per-degree algorithm
    coverage). Degrees 1, 3, 5, and 7 are direct ports of the matching
    named-formula MATLAB file; degree 14 (n=3 only) and every odd degree
    >= 9 REUSE existing private helpers from
    :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`
    rather than re-deriving those constructions (design spec Section 4,
    rows 179-180 of its inventory): ``fourteenthOrderSpherSurfCubPoints.m``
    wraps :func:`~pytcl.mathematical_functions.numerical_integration.cubature_points._fourteenth_order_unit_sphere_points_3d`,
    and every general-``n``, general-odd-degree >= 9 case (superseding
    ``arbOrderSpherSurfCubPoints.m`` and, at ``n=2``,
    ``arbOrder2DSpherSurfCubPoints.m``) wraps
    :func:`~pytcl.mathematical_functions.numerical_integration.cubature_points._sphere_surface_points`
    -- both private helpers normalize to ``sum(w) == 1``, so this function
    rescales by the closed-form surface area (below) rather than
    transcribing a second, functionally-equivalent-but-differently-pointed
    construction. This is the one place ``region_cubature.py`` depends on
    ``cubature_points.py`` (one-directional; never the reverse -- module
    docstring).

    Every degree here is exact through that total polynomial degree --
    verified against the closed-form surface-monomial oracle
    (``sphere_surface_monomial_integral``) in
    ``tests/unit/test_region_cubature.py`` for the ``(n, degree,
    algorithm)`` grid its test classes sweep; no wider claim is made (per
    the claims-inherit-measurement-range convention). This also includes
    the two reused-construction cases: the wrapped
    ``_sphere_surface_points`` general path is checked at low order
    against the same oracle to confirm it agrees with the closed-form
    moments despite using a different point set than MATLAB's own
    ``arbOrderSpherSurfCubPoints`` construction would (design spec's
    stated purpose for that capture case).

    **Weight convention (region measure, not probability).** ``weights``
    sum to the unit sphere's surface area ``2*pi**(n/2) / gamma(n/2)``, NOT
    to 1 -- this module targets the plain (uniform) surface measure on
    ``S^(n-1)``, unlike
    :mod:`~pytcl.mathematical_functions.numerical_integration.cubature_points`'s
    Gaussian-weight rules (whose surface-adjacent helpers, e.g.
    :func:`~pytcl.mathematical_functions.numerical_integration.cubature_points.sphere_surface_to_gauss_points`,
    normalize to 1 because their consumers are Kalman-family filters
    computing ``E[f(X)]``). A caller wanting a probability-normalized rule
    divides by the surface area themselves: ``weights / weights.sum()``.

    Parameters
    ----------
    n : int
        Dimension, n >= 1 (individual algorithms below may require more,
        e.g. degree 14 requires n == 3).
    degree : int
        Polynomial degree the rule is exact through. One of 1, 3, 5, 7, 14,
        or any ODD degree >= 9 (9, 11, 13, ...): 1/3/5/7 are the top-level,
        named-formula degrees MATLAB's ``Spherical_Surface`` directory
        provides for general ``n``; 14 is the fixed-``n=3`` Stroud U3 14-1
        construction (reused, see above); odd degrees >= 9 dispatch to the
        general-``n``, general-order reused construction -- this also
        SUPERSEDES MATLAB's ``ninthOrderSpherSurfCubPoints.m`` and
        ``eleventhOrderSpherSurfCubPoints.m`` (both ``n=3``-only, Tier 3 per
        the design spec's Section 8 -- not individually ported here, same
        rationale as :func:`ball_cubature_points`'s analogous exclusion of
        ``ninthOrderSpherCubPoints.m``/``eleventhOrderSpherCubPoints.m``).
        An EVEN degree other than 14 (e.g. 10) has no MATLAB formula in
        this directory at all and raises ``ValueError``.
    algorithm : int, optional
        Which MATLAB algorithm variant to use; see each degree's section in
        the module docstring for the ported subset. Default None reproduces
        MATLAB's own default selection for that degree:

        - degree 1: algorithm 0 (the only variant; MATLAB's
          ``firstOrderSpherSurfCubPoints.m`` takes no algorithm argument).
        - degree 3: algorithm 0 (Un 3-1, 2n points). Algorithms 1 (Un 3-2,
          2^n points), 2 (Mysovskikh, 2*(n+1) points), 3 (U3 3-1, n=3, 12
          points) are the other ported variants.
        - degree 5: algorithm 0 (Un 5-1, 2n^2 points, negative weights for
          n>4). Algorithms 1-4 are general-n (Un 5-2 through Un 5-4, plus
          Mysovskikh); 5-8 are the n=3 fixed-dimension variants (U3 5-1
          through a 30-point formula MATLAB's own docstring mislabels --
          see module docstring's fifth confirmed finding). Algorithm index
          9 does not exist (MATLAB raises "Unknown algorithm specified").
        - degree 7: algorithm 0 (Formula I, general n>=3) always, matching
          MATLAB's own unconditional default (mirroring
          :func:`ball_cubature_points`'s degree-7 dispatch note -- calling
          this with n < 3 and no explicit algorithm raises ``ValueError``
          from algorithm 0's own guard). Algorithms 1-2 are general-n
          (n>=4); 3-4 are general-n (n>=3, n>=2 respectively) and are
          VERIFIED to compute the identical rule as algorithm 0 (see
          module docstring); 5 is general-n (n>=2); 6-8 are the fixed-n=3/
          n=3/n=4 variants.
        - degree 14: algorithm 0 (the only variant; MATLAB's
          ``fourteenthOrderSpherSurfCubPoints.m`` takes no algorithm
          argument, and only n=3 is supported).
        - degree >= 9 (odd): algorithm must be None or 0 (MATLAB's
          ``arbOrderSpherSurfCubPoints.m`` takes no algorithm parameter).

    Returns
    -------
    points : ndarray
        Shape (num_points, n).
    weights : ndarray
        Shape (num_points,), summing to the sphere surface area (see
        above), not 1.

    Examples
    --------
    >>> pts, w = spherical_surface_cubature_points(3, 3)
    >>> pts.shape
    (6, 3)
    >>> round(float(w.sum()), 9)  # surface area of S^2, 4*pi
    12.566370614
    >>> round(float(np.sum(w * pts[:, 0] ** 2)), 9)  # integral of x^2, 4*pi/3
    4.188790205

    References
    ----------
    A. H. Stroud, "Approximate Calculation of Multiple Integrals,"
    Prentice-Hall, 1971, Formulas Un 3-1/3-2, U3 3-1, Un 5-1 through 5-4,
    U3 5-1/5-2/5-3/5-5, Un 7-1/7-2, U3 7-1/7-2, U4 7-1, U3 14-1, pp. 292-302.

    A. H. Stroud, "Some seventh degree integration formulas for the
    surface of an n-sphere," Numerische Mathematik, vol. 11, no. 3,
    pp. 273-276, Mar. 1968.

    I. P. Mysovskikh, "The approximation of multiple integrals by using
    interpolatory cubature formulae," in Quantitative Approximation,
    R. A. DeVore and K. Scherer, eds., Academic Press, 1980, pp. 217-243.

    R. Cools, "An encyclopedia of cubature formulas," Journal of
    Complexity, vol. 19, no. 3, pp. 445-453, Jun. 2003.
    """
    if n < 1:
        raise ValueError(f"dimension must be >= 1, got {n}")
    if degree == 1:
        return _sphsurf_degree1(n, algorithm)
    if degree == 3:
        return _sphsurf_degree3(n, algorithm)
    if degree == 5:
        return _sphsurf_degree5(n, algorithm)
    if degree == 7:
        return _sphsurf_degree7(n, algorithm)
    if degree == 14:
        if algorithm not in (None, 0):
            raise ValueError(
                f"algorithm {algorithm} not ported for degree 14; expected "
                "0 (MATLAB's fourteenthOrderSpherSurfCubPoints.m has no "
                "algorithm parameter at all)"
            )
        if n != 3:
            raise ValueError(f"degree 14 requires n == 3, got {n}")
        pts, w = _fourteenth_order_unit_sphere_points_3d()
        return pts, w * _sphsurf_area(3)
    if degree >= 9 and degree % 2 == 1:
        if algorithm not in (None, 0):
            raise ValueError(
                f"algorithm {algorithm} not ported for degree {degree}; "
                "expected 0 (MATLAB's arbOrderSpherSurfCubPoints.m has no "
                "algorithm parameter at all)"
            )
        pts, w = _sphere_surface_points(n, degree)
        return pts, w * _sphsurf_area(n)
    raise ValueError(
        f"unsupported degree {degree}; expected 1, 3, 5, 7, 14, or any odd degree >= 9"
    )
