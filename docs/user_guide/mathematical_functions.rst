Mathematical Functions
======================

The library provides a comprehensive set of mathematical functions
commonly used in tracking and estimation.

Special Functions
-----------------

Gamma Functions
^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       gamma, gammaln, gammainc, gammaincc, digamma,
   )

   # Gamma function
   y = gamma(5.5)

   # Log-gamma (more numerically stable)
   y = gammaln(100)

   # Incomplete gamma functions
   y = gammainc(a, x)  # Lower incomplete
   y = gammaincc(a, x)  # Upper incomplete

Error Functions
^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       erf, erfc, erfinv, erfcinv,
   )

   y = erf(x)
   y = erfinv(0.5)

Bessel Functions
^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       jn, yn, iv, kv,
       spherical_jn, spherical_yn,
   )

   # Bessel functions of the first kind
   y = jn(0, x)  # J_0(x)

   # Modified Bessel functions
   y = iv(1, x)  # I_1(x)

Statistics
----------

Distributions
^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
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

   from tracker_component_library.mathematical_functions import (
       weighted_mean, weighted_cov,
       sample_mean, sample_var,
       median, mad,
   )

   # Weighted statistics
   mean = weighted_mean(x, weights)
   cov = weighted_cov(x, weights)

   # Robust estimators
   med = median(x)
   mad_val = mad(x)  # Median absolute deviation

Interpolation
-------------

1D Interpolation
^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       linear_interp, cubic_spline, pchip, akima,
   )

   # Linear interpolation
   y_new = linear_interp(x_new, x, y)

   # Cubic spline (smooth, may overshoot)
   y_new = cubic_spline(x_new, x, y)

   # PCHIP (shape-preserving)
   y_new = pchip(x_new, x, y)

Multidimensional
^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       interp2d, interp3d, rbf_interpolate,
   )

   # 2D interpolation on regular grid
   z_new = interp2d(x_new, y_new, x, y, z)

   # RBF for scattered data
   z_new = rbf_interpolate(points_new, points, values)

Numerical Integration
---------------------

Gaussian Quadrature
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       gauss_legendre, gauss_hermite, gauss_laguerre,
   )

   # Gauss-Legendre for [-1, 1]
   nodes, weights = gauss_legendre(n=5)

   # Gauss-Hermite for (-inf, inf) with exp(-x^2) weight
   nodes, weights = gauss_hermite(n=10)

Adaptive Integration
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       quad, dblquad, tplquad,
   )

   # 1D integration
   result = quad(lambda x: np.sin(x), 0, np.pi)

   # 2D integration
   result = dblquad(lambda x, y: x*y, 0, 1, 0, 1)

Geometry
--------

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       point_in_polygon,
       polygon_area,
       line_intersection,
       distance_point_to_line,
       convex_hull,
   )

   # Point-in-polygon test
   inside = point_in_polygon(point, polygon_vertices)

   # Polygon area (shoelace formula)
   area = polygon_area(vertices)

   # Line-line intersection
   intersection = line_intersection(p1, d1, p2, d2)

Combinatorics
-------------

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       factorial, n_choose_k, n_permute_k,
       permutations, combinations,
       stirling1, stirling2, bell_number,
   )

   # Binomial coefficient
   c = n_choose_k(10, 3)  # 120

   # Generate permutations
   for perm in permutations([1, 2, 3]):
       print(perm)

Matrix Operations
-----------------

.. code-block:: python

   from tracker_component_library.mathematical_functions import (
       chol_semi_def, tria, matrix_sqrt,
       vandermonde, toeplitz, hankel,
       block_diag,
   )

   # Cholesky of semi-definite matrix
   L = chol_semi_def(P)

   # Matrix square root
   S = matrix_sqrt(P)  # P = S @ S.T

   # Block diagonal matrix
   M = block_diag([A, B, C])
