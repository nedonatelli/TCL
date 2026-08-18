Mathematical Functions
======================

The library provides a comprehensive set of mathematical functions
commonly used in tracking and estimation. Core routines are exported
from ``pytcl.mathematical_functions``; more specialized routines live in
the topical submodules shown below.

Special Functions
-----------------

Gamma Functions
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions import gamma, gammaln
   from pytcl.mathematical_functions.special_functions import (
       gammainc, gammaincc, digamma,
   )

   # Gamma function
   y = gamma(5.5)

   # Log-gamma (more numerically stable)
   y = gammaln(100)

   # Regularized incomplete gamma functions
   a, x = 2.0, 1.5
   y = gammainc(a, x)  # Lower incomplete
   y = gammaincc(a, x)  # Upper incomplete

Error Functions
^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions import erf, erfc, erfinv
   from pytcl.mathematical_functions.special_functions import erfcinv

   y = erf(x)
   y = erfinv(0.5)

Bessel Functions
^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions.special_functions import (
       besselj, bessely, besseli, besselk,
       spherical_jn, spherical_yn,
   )

   # Bessel functions of the first kind
   y = besselj(0, x)  # J_0(x)

   # Modified Bessel functions
   y = besseli(1, x)  # I_1(x)

Statistics
----------

Distributions
^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions.statistics import (
       Gaussian, ChiSquared, Exponential, Uniform,
   )

   # Gaussian distribution
   g = Gaussian(mean=0, var=1)
   pdf_val = g.pdf(0)
   cdf_val = g.cdf(1.96)
   samples = g.sample(1000)

Estimators
^^^^^^^^^^

.. code-block:: python

   import numpy as np

   from pytcl.mathematical_functions.statistics import (
       weighted_mean, weighted_cov,
       sample_mean, sample_var,
       median, mad,
   )

   x = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
   weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])

   # Weighted statistics
   mean = weighted_mean(x, weights)

   # weighted_cov expects samples with shape (n_samples, n_dims)
   data = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
   cov = weighted_cov(data, np.array([0.5, 0.3, 0.2]))

   # Robust estimators
   med = median(x)
   mad_val = mad(x)  # Median absolute deviation

Interpolation
-------------

1D Interpolation
^^^^^^^^^^^^^^^^

``linear_interp`` evaluates directly; the spline constructors return a
callable interpolator object.

.. code-block:: python

   from pytcl.mathematical_functions.interpolation import (
       linear_interp, cubic_spline, pchip, akima,
   )

   x = np.linspace(0.0, 10.0, 11)
   y = np.sin(x)
   x_new = np.linspace(0.0, 10.0, 51)

   # Linear interpolation
   y_new = linear_interp(x_new, x, y)

   # Cubic spline (smooth, may overshoot)
   spline = cubic_spline(x, y)
   y_new = spline(x_new)

   # PCHIP (shape-preserving)
   interp = pchip(x, y)
   y_new = interp(x_new)

Multidimensional
^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions.interpolation import (
       interp2d, interp3d, rbf_interpolate,
   )

   # 2D interpolation on a regular grid: returns an interpolator
   xg = np.linspace(0.0, 1.0, 5)
   yg = np.linspace(0.0, 1.0, 5)
   zg = np.outer(xg, yg)
   interp = interp2d(xg, yg, zg)
   z_new = interp([[0.5, 0.5], [0.2, 0.8]])

   # RBF for scattered data: returns an interpolator
   points = np.random.default_rng(0).random((20, 2))
   values = points[:, 0] ** 2 + points[:, 1]
   rbf = rbf_interpolate(points, values)
   z_new = rbf(np.array([[0.5, 0.5]]))

Numerical Integration
---------------------

Gaussian Quadrature
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from pytcl.mathematical_functions import gauss_legendre, gauss_hermite
   from pytcl.mathematical_functions.numerical_integration import (
       gauss_laguerre,
   )

   # Gauss-Legendre for [-1, 1]
   nodes, weights = gauss_legendre(n=5)

   # Gauss-Hermite for (-inf, inf) with exp(-x^2) weight
   nodes, weights = gauss_hermite(n=10)

Adaptive Integration
^^^^^^^^^^^^^^^^^^^^

Each routine returns a ``(value, error_estimate)`` tuple. For
``dblquad``, the integrand is ``f(y, x)`` and the inner (y) limits are
functions of x.

.. code-block:: python

   from pytcl.mathematical_functions import quad
   from pytcl.mathematical_functions.numerical_integration import (
       dblquad, tplquad,
   )

   # 1D integration
   result, error = quad(lambda x: np.sin(x), 0, np.pi)

   # 2D integration of x*y over the unit square
   result, error = dblquad(
       lambda y, x: x * y, 0, 1, lambda x: 0.0, lambda x: 1.0
   )

Cubature Points
^^^^^^^^^^^^^^^

``seventh_order_cubature_points(n, algorithm=None)`` returns degree-7
cubature rules for the standard normal N(0, I) -- Stroud's full algorithm
surface, ported from MATLAB TCL's ``seventhOrderCubPoints``. With
``algorithm=None`` it reproduces MATLAB's default dispatch: n == 1 picks
algorithm 9, n == 2 picks algorithm 2, and every other n picks algorithm 0.

.. list-table:: Seventh-order cubature algorithms (Stroud 1971)
   :header-rows: 1
   :widths: 10 30 15 20

   * - algorithm
     - rule
     - valid n
     - points
   * - 0
     - E_n^{r^2} 7-3, p. 319
     - n = 3..6 (see note)
     - 2*(2^n + 2n^2)
   * - 1
     - E_n^{r^2} 7-1, p. 318
     - 3, 4, 6, 7
     - 2^n + 2n^2 + 1
   * - 2
     - E_2^{r^2} 7-1, p. 324
     - 2
     - 12
   * - 3
     - E_2^{r^2} 7-2, p. 324
     - 2
     - 17 (see note)
   * - 4 / 5
     - E_3^{r^2} 7-1, p. 327
     - 3
     - 27
   * - 6 / 7
     - E_3^{r^2} 7-2, p. 328
     - 3
     - 33
   * - 8
     - E_4^{r^2} 7-1, p. 329
     - 4
     - 49 (see note)
   * - 9
     - quadraturePoints1D(4)
     - 1
     - 4

Each algorithm exactly integrates every polynomial of total degree <= 7
against N(0, I) for its own valid n. That has been verified only for the
(algorithm, n) pairs in the table above:

- Algorithm 0's code accepts any n >= 3, but the exactness claim is
  verified, and therefore bounded, to n = 3..6; n > 6 runs without error
  but is an unverified extrapolation, not a documented guarantee.
- Algorithms 3 and 8 deliberately diverge from MATLAB. MATLAB documents
  scale corrections for both (a 4/3 factor for algorithm 3's r and s; a
  sqrt(4/5) factor for algorithm 8's r, s, and t), but neither correction
  actually achieves degree-7 exactness, confirmed with exact symbolic
  arithmetic. Algorithm 8's real defect is a transcription typo -- the
  book's ``t = 3 + sqrt(3)`` is missing an outer square root; with
  ``t = sqrt(3 + sqrt(3))`` and Stroud's other coefficients unscaled, the
  rule is exact and no sqrt(4/5) correction is needed. Algorithm 3's
  16-point, no-origin layout is provably incapable of degree-7 exactness
  for any choice of its two radii and three weights; adding a 17th point
  at the origin supplies the missing degree of freedom and, with Stroud's
  original (unscaled) r and s, yields an exact rule. Neither corrected
  rule reproduces MATLAB's numeric output for that algorithm.
- Both corrected rules carry a negative weight (algorithm 3's 17-point
  rule has D = -2/3 at the origin), and algorithm 0's axis-shell weight
  turns negative for n > 8. Negative weights are inherent to these rules,
  not errors -- do not assemble covariances from these points with a
  sqrt-of-weights factorization.

.. code-block:: python

   from pytcl.mathematical_functions.numerical_integration import (
       seventh_order_cubature_points,
   )

   # Default dispatch for n=3 selects algorithm 0
   pts, w = seventh_order_cubature_points(3)
   assert pts.shape == (52, 3)
   assert round(float(w.sum()), 12) == 1.0

   # Corrected 17-point algorithm 3 (n=2): note the negative origin weight
   pts3, w3 = seventh_order_cubature_points(2, algorithm=3)
   assert pts3.shape == (17, 2)
   assert bool((w3 < 0).any())

Smolyak Sparse Grids
^^^^^^^^^^^^^^^^^^^^

``smolyak_points(n, level, algorithm=0)`` builds Smolyak sparse-grid
cubature over nested Genz-Keister 1-D sequences (see the Genz-Keister
section of :doc:`/advanced_kf_variants`). This is an original design with no MATLAB
TCL counterpart -- MATLAB provides the nested 1-D sequences but never the
Smolyak combination over them. It follows the standard construction from
Smolyak (1963), using the Genz-Keister sequences exactly as Genz and
Keister (1996) designed them to be used.

Each Smolyak level combines tensor products of a 1-D rule taken at
different resolutions. The 1-D rule at level q is the Genz-Keister rule at
the "milestone" m where the algorithm's q-th nu stage completes:

.. list-table:: Level to Genz-Keister milestone mapping
   :header-rows: 1
   :widths: 12 8 8 14 14

   * - algorithm
     - level
     - GK m
     - 1-D points
     - 1-D degree
   * - 0
     - 0
     - 0
     - 1
     - 1
   * - 0
     - 1
     - 1
     - 3
     - 5
   * - 0
     - 2
     - 4
     - 9
     - 15
   * - 0
     - 3
     - 9
     - 19
     - 29
   * - 1
     - 0
     - 0
     - 1
     - 1
   * - 1
     - 1
     - 1
     - 3
     - 5
   * - 1
     - 2
     - 5
     - 11
     - 19

Each algorithm's ladder deliberately stops below its Genz-Keister table's
top milestone (m=17 for algorithm 0, m=15 for algorithm 1): at that top
value the published double-precision constants lose exactness, so no
Smolyak level is built on it. This caps ``level`` at 3 for algorithm 0 and
2 for algorithm 1.

Because the 1-D point sets nest exactly, points shared across the
combination's tensor grids merge into single points with summed weights,
which is what keeps the point count far below the full tensor product's:

.. code-block:: python

   from pytcl.mathematical_functions.numerical_integration import (
       smolyak_points,
   )

   # n=2, level=1: cross of the 3-point GK rule, origin merged: 5, not 3**2
   pts, w = smolyak_points(2, 1)
   assert pts.shape == (5, 2)
   assert round(float(w.sum()), 12) == 1.0

   # Sparse grid vs. the equivalent-degree tensor Gauss-Hermite grid, at
   # level 2 (degree >= 5 for n=4..8): the sparse count grows polynomially
   # in n, the tensor count exponentially (3 ** n).
   pts4, w4 = smolyak_points(4, 2)
   pts8, w8 = smolyak_points(8, 2)
   assert pts4.shape[0] == 57  # vs. 3 ** 4 == 81
   assert pts8.shape[0] == 177  # vs. 3 ** 8 == 6561

Weights commonly go negative -- already at n=4, level=1 the origin's
weight is exactly -3 + 4*(2/3) = -1/3. This is a disclosed property of the
Smolyak combination (its coefficients alternate in sign) compounded by the
Genz-Keister rules' own negative weights, not suppressed or clamped. As
with the cubature rules above, do not assemble covariances from these
points with a sqrt-of-weights factorization.

.. code-block:: python

   pts, w = smolyak_points(8, 2)
   assert bool((w < 0).any())

**Measured exactness.** The generic theoretical floor for nested 1-D
sequences is total-degree exactness ``2 * level + 1``, but because the
Genz-Keister milestone degrees (1, 5, 15, 29) grow much faster than that,
low dimensions do measurably better. Rather than assert the generic floor,
the actual total-degree exactness was measured per (algorithm, n, level)
cell against closed-form N(0, I) moments, and every cell is sharp (the
next even degree fails):

.. list-table:: Measured exactness, algorithm 0 (n <= 8 only)
   :header-rows: 1
   :widths: 15 15 15 15 15

   * - n
     - level 0
     - level 1
     - level 2
     - level 3
   * - 1
     - 1
     - 5
     - 15
     - 29
   * - 2
     - 1
     - 3
     - 7
     - 11
   * - 3
     - 1
     - 3
     - 5
     - 9
   * - 4..8
     - 1
     - 3
     - 5
     - 7

.. list-table:: Measured exactness, algorithm 1 (n <= 6 only)
   :header-rows: 1
   :widths: 15 15 15 15

   * - n
     - level 0
     - level 1
     - level 2
   * - 1
     - 1
     - 5
     - 19
   * - 2
     - 1
     - 3
     - 7
   * - 3..6
     - 1
     - 3
     - 5

These tables are claims only for the (algorithm, n, level) cells they
list. For larger n the generic ``2 * level + 1`` floor is the standard
theoretical result but was not measured here; re-verify before relying on
it (see ``tests/unit/test_cubature_points.py::TestSmolyakPoints`` for the
measurement reproduced as tests).

Region Cubature (True-Measure Rules)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``pytcl.mathematical_functions.numerical_integration.region_cubature``
provides a second family of cubature rules, for four bounded geometric
regions instead of the Gaussian weight N(0, I) the rules above target:
the cube ``[-1, 1]^n``, the standard n-simplex ``{x >= 0, sum(x) <= 1}``,
the unit n-ball ``{|x| <= 1}`` (optionally weighted by ``|x|**alpha``),
and the unit sphere surface ``S^(n-1) = {|x| == 1}``.

**Measure convention -- read this before using these functions.** Every
rule above (``seventh_order_cubature_points``, ``smolyak_points``, and the
rest of this page) normalizes its weights to sum to 1, so
``E[f(X)] ~= sum_i w_i f(x_i)`` directly for X ~ N(0, I). The four
functions below do the OPPOSITE: their weights sum to the region's TRUE
measure, not to 1:

.. list-table:: Region cubature weight-sum convention
   :header-rows: 1
   :widths: 25 35 20

   * - function
     - region
     - sum(weights)
   * - ``cube_cubature_points(n, degree)``
     - ``[-1, 1]^n``
     - ``2**n``
   * - ``simplex_cubature_points(n, degree)``
     - ``{x >= 0, sum(x) <= 1}``
     - ``1 / n!``
   * - ``ball_cubature_points(n, degree, alpha=0.0)``
     - ``{|x| <= 1}``, weight ``|x|**alpha``
     - ``2/(n+alpha) * pi**(n/2) / gamma(n/2)``
   * - ``spherical_surface_cubature_points(n, degree)``
     - ``S^(n-1) = {|x| == 1}``
     - ``2 * pi**(n/2) / gamma(n/2)``

This is a deliberate divergence from the Gaussian-weight rules, not an
inconsistency: these four rules answer "what is the integral over this
region," a question with a real geometric answer that a forced
sum-to-1 renormalization would silently discard. A caller who wants the
probability-normalized version divides by the returned sum themselves
(``weights / weights.sum()``), always safe; recovering the true measure
from an already-normalized rule is not possible without separately
knowing the volume.

.. code-block:: python

   from pytcl.mathematical_functions.numerical_integration import (
       cube_cubature_points, simplex_cubature_points,
       ball_cubature_points, spherical_surface_cubature_points,
   )

   # Cube [-1, 1]^3: weights sum to the cube volume 2^3 = 8.
   pts, w = cube_cubature_points(3, degree=3)
   assert pts.shape == (6, 3)
   assert round(float(w.sum()), 12) == 8.0

   # Standard 3-simplex: weights sum to 1/3! = 1/6.
   pts, w = simplex_cubature_points(3, degree=3)
   assert pts.shape == (5, 3)
   assert round(float(w.sum()), 12) == round(1.0 / 6.0, 12)

   # Unit 3-ball: weights sum to the ball volume 4*pi/3.
   pts, w = ball_cubature_points(3, degree=3)
   assert pts.shape == (6, 3)
   assert round(float(w.sum()), 9) == round(4.0 * np.pi / 3.0, 9)

   # Unit sphere surface S^2: weights sum to the surface area 4*pi.
   pts, w = spherical_surface_cubature_points(3, degree=3)
   assert pts.shape == (6, 3)
   assert round(float(w.sum()), 9) == round(4.0 * np.pi, 9)
   # integral of x^2 over S^2 is 4*pi/3.
   assert round(float(np.sum(w * pts[:, 0] ** 2)), 9) == round(
       4.0 * np.pi / 3.0, 9
   )

Every function's ``degree`` selects among the MATLAB TCL degree-named
source files it was ported from (``cube_cubature_points`` supports
1, 2, 3, 5, 7, 9; ``simplex_cubature_points`` supports 2, 3, 4, 5;
``ball_cubature_points`` supports 2, 3, 5, 7, or any odd degree >= 9;
``spherical_surface_cubature_points`` supports 1, 3, 5, 7, 14, or any odd
degree >= 9), and ``algorithm`` selects among the degree-specific variants
each MATLAB file offers, defaulting to whichever variant MATLAB itself
defaults to. Each ``(n, degree, algorithm)`` combination the test suite
covers is verified exact against a closed-form monomial oracle for that
region -- see ``region_cubature.py``'s module docstring for the full
per-degree algorithm coverage and every MATLAB source defect found and
corrected during the port (a shape-mismatch crash in
``firstOrderNDimCubPoints.m``'s default algorithm, an off-by-one column
index in ``thirdOrderNDimCubPoints.m``, a NaN-poisoning indeterminate form
in one ``Simplex`` algorithm at n=2, a documentation-only sign error in
the ``Sphere`` region's ``alpha`` weighting exponent, a wrong-trigonometric-
function transcription bug in one ``Sphere`` degree-7 algorithm, and a
docstring/code index mismatch in one ``Spherical_Surface`` degree-5
algorithm). Like the Gaussian-weight rules above, negative weights occur
at several (region, degree, algorithm) combinations and are inherent to
the underlying formulas -- do not assemble covariances from these points
with a sqrt-of-weights factorization.

``spherical_surface_cubature_points`` reuses two existing private
constructions from ``cubature_points.py`` (the degree-14, n=3 Stroud
U3 14-1 rule, and the general-n, general-odd-degree Gauss-Jacobi surface
construction already used internally by ``spherical_radial_points``)
rather than re-deriving them, rescaled from their native sum-to-1
convention to this module's sum-to-surface-area convention.

Gaussian LCD Samples
^^^^^^^^^^^^^^^^^^^^

``gaussian_lcd_samples(n, num_points, force_cov_match=True, rng=None,
max_iter=1000)`` builds a Dirac-mixture (particle-set) approximation to
N(0, I_n) by numerically minimizing a modified Cramer-von Mises (CvM)
distance between the point set's localized CDF and the true Gaussian's,
ported from MATLAB TCL's ``GaussianLCDSamples``. Unlike the cubature rules
above, which are closed-form and return whatever fixed point count the
rule's polynomial-exactness degree dictates, LCD samples are built for an
arbitrary, caller-chosen ``num_points`` (any count ``>= 2*n``, even or
odd) via ``scipy.optimize.minimize(method="L-BFGS-B")`` -- there is no
"exactness degree" here, only a converged approximation quality.

**When to prefer this over a cubature rule.** LCD optimization is
comparatively expensive (the MATLAB source's own header comment calls the
algorithm "generally too slow ... for real-time systems") and carries no
polynomial-exactness guarantee, so it is not a drop-in replacement for the
fixed-degree sigma-point sets above. Prefer it when the point count itself
is the actual requirement -- e.g. a downstream particle filter or
Monte-Carlo consumer needs exactly ``num_points`` weighted samples and no
fixed-degree cubature rule happens to produce that count -- and the
optimization cost is acceptable (an offline / precomputed step, not inside
a real-time filter's per-step loop). Prefer a cubature rule instead
whenever a specific polynomial-exactness degree is what actually matters,
or points are needed on every filter step.

**The manifold caveat -- read this before comparing output across runs or
against MATLAB.** For ``n >= 2`` the CvM cost is provably invariant under
any global orthogonal (rotation/reflection) transform applied to every
point simultaneously, so a minimizer sits on a continuous flat manifold of
equally-optimal solutions, not an isolated point. Two correctly converged
runs -- including this port versus a MATLAB ``GaussianLCDSamples`` run
started from the identical seed matrix -- generically land on *different*
points of that manifold: same CvM cost, different raw coordinates. This is
expected, not a bug, and is why this port's own MATLAB-fixture tests never
compare raw coordinates for ``n >= 2`` (see
``tests/unit/test_lcd_samples.py::TestGaussianLCDSamplesMatlabFixtures``,
which instead compares the rotation-and-permutation-invariant Gram-matrix
eigenvalue spectrum, the CvM objective value, and the sample mean and
covariance). Only ``n == 1`` has a discrete symmetry group (``{+1, -1}``),
where raw coordinates are a meaningful comparison.

.. code-block:: python

   from pytcl.mathematical_functions.numerical_integration import (
       gaussian_lcd_samples,
   )

   # Small, fast case for this example: low max_iter keeps it quick and is
   # not a claim about converged accuracy at that setting (see the
   # function's own docstring for measured convergence behavior at the
   # default max_iter=1000).
   pts, w = gaussian_lcd_samples(
       2, 10, rng=np.random.default_rng(0), max_iter=50
   )
   assert pts.shape == (10, 2)
   assert w.shape == (10,)
   assert round(float(w.sum()), 12) == 1.0

   # force_cov_match=True (the default) whitens the result so the sample
   # mean and covariance match N(0, I) to float64 precision -- exactly by
   # construction, not a statistical approximation.
   assert bool(np.allclose(pts.mean(axis=0), 0.0, atol=1e-10))
   assert bool(np.allclose((pts * w[:, None]).T @ pts, np.eye(2), atol=1e-8))

Geometry
--------

.. code-block:: python

   from pytcl.mathematical_functions import (
       point_in_polygon,
       polygon_area,
       line_intersection,
       convex_hull,
   )
   from pytcl.mathematical_functions.geometry import point_to_line_distance

   square = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

   # Point-in-polygon test
   inside = point_in_polygon([0.5, 0.5], square)

   # Polygon area (shoelace formula)
   area = polygon_area(square)

   # Segment-segment intersection (endpoints; None if no intersection)
   intersection = line_intersection([0, 0], [1, 1], [0, 1], [1, 0])

   # Distance from a point to a line through two points
   d = point_to_line_distance([0.5, 1.0], [0.0, 0.0], [1.0, 0.0])

   # Convex hull returns (hull_points, hull_indices)
   pts = np.random.default_rng(1).random((10, 2))
   hull_points, hull_indices = convex_hull(pts)

Combinatorics
-------------

.. code-block:: python

   from pytcl.mathematical_functions import (
       factorial, n_choose_k, permutations, combinations,
   )
   from pytcl.mathematical_functions.combinatorics import (
       n_permute_k, stirling_second, bell_number,
   )

   # Binomial coefficient
   c = n_choose_k(10, 3)  # 120

   # Stirling numbers of the second kind
   s = stirling_second(5, 2)  # 15

   # Generate permutations
   for perm in permutations([1, 2, 3]):
       print(perm)

Matrix Operations
-----------------

.. code-block:: python

   from pytcl.mathematical_functions import (
       chol_semi_def, tria, matrix_sqrt, block_diag,
   )
   from pytcl.mathematical_functions.basic_matrix import (
       vandermonde, toeplitz, hankel,
   )

   P = np.array([[4.0, 1.0], [1.0, 3.0]])

   # Cholesky of semi-definite matrix
   L = chol_semi_def(P)  # L @ L.T = P

   # Matrix square root
   S = matrix_sqrt(P)  # S @ S = P

   # Block diagonal matrix (matrices passed as separate arguments)
   A = np.eye(2)
   B = np.ones((2, 2))
   M = block_diag(A, B)
