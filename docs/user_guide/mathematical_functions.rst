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
