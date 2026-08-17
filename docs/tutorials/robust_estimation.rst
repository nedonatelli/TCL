Robust Estimation Tutorial
============================

This tutorial demonstrates robust regression methods for handling outliers
and non-Gaussian noise, comparing three estimators on the same
outlier-contaminated linear dataset.

Topics covered:

- Ordinary Least Squares (OLS) as the outlier-sensitive baseline
- RANSAC (Random Sample Consensus)
- Iteratively Reweighted Least Squares (IRLS) with a Huber weight function
- Comparing residual distributions across estimators

Contaminated Data
-------------------

The true model is ``y = 2x + 1``; 15% of the 100 points are corrupted with
large uniform outliers on top of the baseline Gaussian noise:

.. code-block:: python

   import numpy as np

   np.random.seed(42)
   x_data = np.linspace(0, 10, 100)
   y_data = 2 * x_data + 1 + np.random.randn(100) * 0.5

   outlier_idx = np.random.choice(100, 15, replace=False)
   y_data[outlier_idx] += np.random.uniform(-5, 5, 15)

Ordinary Least Squares
------------------------

.. code-block:: python

   A = np.vstack([x_data, np.ones_like(x_data)]).T
   params_ols = np.linalg.lstsq(A, y_data, rcond=None)[0]

OLS minimizes squared residuals over *all* points, so the 15% outlier
fraction pulls the fitted slope and intercept measurably off the true model.

RANSAC
------

RANSAC repeatedly fits a line to a minimal 2-point sample, keeps the sample
whose fit has the most inliers under a residual threshold, and reports the
inlier-only fit:

.. code-block:: python

   best_inliers = 0
   for _ in range(100):
       sample_idx = np.random.choice(100, 2, replace=False)
       params = np.linalg.lstsq(
           np.vstack([x_data[sample_idx], np.ones(2)]).T, y_data[sample_idx],
           rcond=None,
       )[0]
       inlier_mask = np.abs(y_data - (params[0] * x_data + params[1])) < 1.0
       if inlier_mask.sum() > best_inliers:
           best_inliers, best_params, best_inlier_mask = inlier_mask.sum(), params, inlier_mask

Because it never lets an outlier influence the fit once it is excluded as an
outlier, RANSAC recovers a slope/intercept close to the true model even with
15% contamination.

Iteratively Reweighted Least Squares
---------------------------------------

IRLS starts from the OLS fit and iteratively down-weights points with large
residuals using a Huber weight (``w = 1`` inside the threshold, ``w = c /
|residual|`` outside it), refitting a weighted least-squares problem each
iteration:

.. code-block:: python

   params = params_ols.copy()
   for _ in range(10):
       residuals = y_data - (params[0] * x_data + params[1])
       weights = np.where(np.abs(residuals) <= 1.0, 1.0, 1.0 / np.abs(residuals))
       W = np.diag(weights)
       A = np.vstack([x_data, np.ones_like(x_data)]).T
       params = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y_data, rcond=None)[0]

Unlike RANSAC, IRLS never fully discards a point -- even a downweighted
outlier still contributes a small residual pull -- so its fit sits between
OLS and RANSAC in practice.

Next Steps
----------

- See :doc:`/api/mathematical_functions` for the library's robust-statistics
  and optimization utilities
- See :doc:`data_association` for outlier rejection in a tracking context
  (gating serves a similar role to RANSAC's inlier threshold)
