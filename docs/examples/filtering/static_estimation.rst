Static Estimation
=================

This example demonstrates static estimation algorithms for parameter estimation, sensor calibration, and model fitting.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_estimation.html"></iframe>
   </div>

Overview
--------

Static estimation methods are fundamental for:

- **Sensor calibration**: Bias and scale factor estimation
- **Model fitting**: Parameter estimation from data
- **Regression analysis**: Relationship between variables
- **Data fusion**: Combining multiple measurements

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_ols.html"></iframe>
   </div>

Least Squares Methods
---------------------

**Ordinary Least Squares (OLS)**
   - Minimizes sum of squared residuals
   - Assumes equal measurement uncertainty
   - Optimal for Gaussian noise

**Weighted Least Squares (WLS)**
   - Accounts for varying measurement precision
   - Weights = 1/variance for optimal results
   - Essential for heteroscedastic data

**Total Least Squares (TLS)**
   - Errors in both dependent and independent variables
   - Also known as errors-in-variables regression
   - Avoids bias from noisy predictors

**Generalized Least Squares (GLS)**
   - Accounts for correlated measurement errors
   - Uses full covariance matrix

**Recursive Least Squares (RLS)**
   - Online/sequential estimation
   - Updates estimate with each new measurement
   - Useful for real-time applications

**Ridge Regression**
   - L2 regularization for ill-conditioned problems
   - Shrinks coefficients toward zero
   - Handles multicollinearity

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_robust_estimation.html"></iframe>
   </div>

Robust Estimation
-----------------

**Huber M-estimator**
   - Combines L2 (small residuals) and L1 (large residuals)
   - Less sensitive to outliers than OLS
   - Iteratively reweighted least squares (IRLS)

**Tukey Bisquare M-estimator**
   - Completely rejects large outliers (zero weight)
   - More aggressive outlier handling
   - Identifies outliers through weights

**RANSAC**
   - Random Sample Consensus
   - Robust to high outlier percentages
   - Separates inliers from outliers

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_ransac.html"></iframe>
   </div>

Maximum Likelihood
------------------

**MLE for Gaussian**
   - Estimates mean and variance
   - Optimal for Gaussian data

**Fisher Information**
   - Measures information in data about parameters
   - Determines estimation precision limits

**Cramer-Rao Bound**
   - Lower bound on estimator variance
   - Efficiency = CRB / actual variance

Model Selection
---------------

**AIC (Akaike Information Criterion)**
   - Balances fit quality and model complexity
   - Penalizes additional parameters
   - Lower is better

**BIC (Bayesian Information Criterion)**
   - Stronger penalty for complexity
   - Consistent model selection
   - Prefers simpler models than AIC

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/static_model_selection.html"></iframe>
   </div>

Code Highlights
---------------

The example demonstrates:

- OLS with ``ordinary_least_squares()``
- WLS with ``weighted_least_squares()``
- TLS with ``total_least_squares()``
- RLS with ``recursive_least_squares()``
- Ridge with ``ridge_regression()``
- Huber with ``huber_regression()``
- Tukey with ``tukey_regression()``
- RANSAC with ``ransac()``
- MLE with ``mle_gaussian()``
- Model selection with ``aic()``, ``bic()``

Source Code
-----------

.. literalinclude:: ../../../examples/static_estimation.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/static_estimation.py

See Also
--------

- :doc:`kalman_filter_comparison` - Dynamic estimation
- :doc:`performance_evaluation` - Tracking metrics
- :doc:`gaussian_mixtures` - Clustering and mixture models
