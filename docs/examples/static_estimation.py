"""
Static Estimation Example
=========================

This example demonstrates static estimation algorithms in PyTCL:

Least Squares Methods:
- Ordinary Least Squares (OLS)
- Weighted Least Squares (WLS)
- Total Least Squares (TLS)
- Generalized Least Squares (GLS)
- Recursive Least Squares (RLS)
- Ridge Regression

Robust Estimation:
- Huber M-estimator
- Tukey bisquare M-estimator
- RANSAC for outlier-robust fitting

Maximum Likelihood Estimation:
- MLE for Gaussian parameters
- Fisher Information and Cramer-Rao Bounds
- Model selection (AIC, BIC)

These methods are fundamental for parameter estimation, sensor calibration,
and model fitting in the presence of noise and outliers.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Global flag to control plotting
SHOW_PLOTS = True


from pytcl.static_estimation import (  # Least squares; Robust estimation; MLE and Fisher Information; Model selection
    aic,
    aicc,
    bic,
    cramer_rao_bound,
    efficiency,
    fisher_information_gaussian,
    fisher_information_numerical,
    generalized_least_squares,
    huber_regression,
    irls,
    mad,
    mle_gaussian,
    ordinary_least_squares,
    ransac,
    recursive_least_squares,
    ridge_regression,
    total_least_squares,
    tukey_regression,
    weighted_least_squares,
)


def demo_ordinary_least_squares():
    """Demonstrate ordinary least squares."""
    print("=" * 70)
    print("Ordinary Least Squares Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate linear regression data
    n_samples = 50
    x = np.linspace(0, 10, n_samples)
    true_slope = 2.5
    true_intercept = 1.0
    noise_std = 0.5

    y = true_intercept + true_slope * x + np.random.randn(n_samples) * noise_std

    # Design matrix [1, x]
    A = np.column_stack([np.ones(n_samples), x])

    # OLS solution
    result = ordinary_least_squares(A, y)

    print(f"\nTrue parameters: intercept={true_intercept}, slope={true_slope}")
    print(f"OLS estimate: intercept={result.x[0]:.4f}, slope={result.x[1]:.4f}")
    print(f"\nResidual sum of squares: {np.sum(result.residuals**2):.4f}")
    print(f"Matrix rank: {result.rank}")

    # Coefficient of determination
    ss_res = np.sum(result.residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - ss_res / ss_tot
    print(f"R² = {r_squared:.4f}")

    # Plot OLS fit
    if SHOW_PLOTS:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            f"Ordinary Least Squares (R² = {r_squared:.4f})",
            "Residual Plot"
        ])

        # Fit plot
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Data'
        ), row=1, col=1)

        x_line = np.linspace(x.min(), x.max(), 100)
        y_true_line = true_intercept + true_slope * x_line
        y_fit_line = result.x[0] + result.x[1] * x_line

        fig.add_trace(go.Scatter(
            x=x_line, y=y_true_line, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='True line'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_line, y=y_fit_line, mode='lines',
            line=dict(color='red', width=2),
            name='OLS fit'
        ), row=1, col=1)

        # Residuals plot
        residuals = y - (result.x[0] + result.x[1] * x)
        fig.add_trace(go.Scatter(
            x=x, y=residuals, mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Residuals', showlegend=False
        ), row=1, col=2)

        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="Residual", row=1, col=2)

        fig.update_layout(height=500, width=1000, showlegend=True)
        fig.write_html("static_ols.html")
        print("\n  [Plot saved to static_ols.html]")


def demo_weighted_least_squares():
    """Demonstrate weighted least squares."""
    print("\n" + "=" * 70)
    print("Weighted Least Squares Demo")
    print("=" * 70)

    np.random.seed(42)

    # Data with heteroscedastic noise (varying variance)
    n_samples = 50
    x = np.linspace(0, 10, n_samples)
    true_slope = 2.0
    true_intercept = 3.0

    # Noise increases with x
    noise_std = 0.2 + 0.1 * x
    y = true_intercept + true_slope * x + np.random.randn(n_samples) * noise_std

    A = np.column_stack([np.ones(n_samples), x])

    # OLS (ignores varying noise)
    result_ols = ordinary_least_squares(A, y)

    # WLS with weights = 1/variance
    weights = 1 / noise_std**2
    result_wls = weighted_least_squares(A, y, weights=weights)

    print(f"\nTrue parameters: intercept={true_intercept}, slope={true_slope}")
    print(
        f"\nOLS estimate: intercept={result_ols.x[0]:.4f}, slope={result_ols.x[1]:.4f}"
    )
    print(f"WLS estimate: intercept={result_wls.x[0]:.4f}, slope={result_wls.x[1]:.4f}")

    print("\nNote: WLS gives more weight to precise measurements (low variance)")
    print("and typically produces better estimates when noise is heteroscedastic.")


def demo_total_least_squares():
    """Demonstrate total least squares (errors-in-variables)."""
    print("\n" + "=" * 70)
    print("Total Least Squares Demo")
    print("=" * 70)

    np.random.seed(42)

    # Both x and y have measurement errors
    n_samples = 30
    x_true = np.linspace(0, 10, n_samples)
    true_slope = 1.5
    true_intercept = 2.0

    # Add noise to both x and y
    x_noise_std = 0.3
    y_noise_std = 0.5

    x = x_true + np.random.randn(n_samples) * x_noise_std
    y = true_intercept + true_slope * x_true + np.random.randn(n_samples) * y_noise_std

    A = np.column_stack([np.ones(n_samples), x])

    # OLS (assumes x is error-free)
    result_ols = ordinary_least_squares(A, y)

    # TLS (accounts for errors in x)
    result_tls = total_least_squares(A, y)

    print(f"\nTrue parameters: intercept={true_intercept}, slope={true_slope}")
    print(
        f"\nOLS estimate: intercept={result_ols.x[0]:.4f}, slope={result_ols.x[1]:.4f}"
    )
    print(f"TLS estimate: intercept={result_tls.x[0]:.4f}, slope={result_tls.x[1]:.4f}")

    print("\nNote: TLS is preferred when independent variables have measurement error.")
    print("OLS typically underestimates the true slope in this case.")


def demo_recursive_least_squares():
    """Demonstrate recursive least squares for online estimation."""
    print("\n" + "=" * 70)
    print("Recursive Least Squares Demo")
    print("=" * 70)

    np.random.seed(42)

    # Online parameter estimation
    n_samples = 100
    x = np.linspace(0, 10, n_samples)
    true_slope = 2.0
    true_intercept = 1.0
    noise_std = 0.3

    y = true_intercept + true_slope * x + np.random.randn(n_samples) * noise_std

    # Initialize RLS (2 parameters: intercept and slope)
    n_params = 2
    x_est = np.zeros(n_params)  # Initial parameter estimate
    P = np.eye(n_params) * 100.0  # Initial covariance (large uncertainty)

    print("\nOnline parameter estimation with RLS:")
    print("-" * 50)

    checkpoints = [10, 25, 50, 100]
    checkpoint_idx = 0

    for i in range(n_samples):
        # Measurement vector [1, x_i] for y = intercept + slope * x
        a = np.array([1.0, x[i]])
        y_i = y[i]

        # RLS update
        x_est, P = recursive_least_squares(x_est, P, a, y_i)

        # Print at checkpoints
        if checkpoint_idx < len(checkpoints) and i + 1 == checkpoints[checkpoint_idx]:
            print(
                f"  After {i+1:>3} samples: intercept={x_est[0]:.4f}, "
                f"slope={x_est[1]:.4f}"
            )
            checkpoint_idx += 1

    print(f"\nTrue values: intercept={true_intercept}, slope={true_slope}")
    print("\nNote: RLS converges to true values as more data arrives.")


def demo_ridge_regression():
    """Demonstrate ridge regression for ill-conditioned problems."""
    print("\n" + "=" * 70)
    print("Ridge Regression Demo")
    print("=" * 70)

    np.random.seed(42)

    # Create collinear predictors (ill-conditioned)
    n_samples = 30
    x1 = np.random.randn(n_samples)
    x2 = x1 + np.random.randn(n_samples) * 0.1  # x2 ≈ x1

    true_coef = np.array([1.0, 2.0, 3.0])  # [intercept, b1, b2]
    y = true_coef[0] + true_coef[1] * x1 + true_coef[2] * x2
    y += np.random.randn(n_samples) * 0.5

    A = np.column_stack([np.ones(n_samples), x1, x2])

    # OLS solution (may be unstable)
    result_ols = ordinary_least_squares(A, y)

    # Ridge regression with regularization
    lambdas = [0.0, 0.01, 0.1, 1.0]

    print(f"\nTrue coefficients: {true_coef}")
    print("\nEstimates with different regularization:")
    print("-" * 60)
    print(f"{'Lambda':>10} {'Intercept':>12} {'b1':>12} {'b2':>12}")
    print("-" * 60)

    for lam in lambdas:
        if lam == 0:
            x_hat = result_ols.x
        else:
            x_hat = ridge_regression(A, y, alpha=lam)  # Returns array directly
        print(f"{lam:>10.2f} {x_hat[0]:>12.4f} {x_hat[1]:>12.4f} " f"{x_hat[2]:>12.4f}")

    print("\nNote: Ridge regression shrinks coefficients toward zero,")
    print("which helps stabilize estimates for collinear predictors.")


def demo_robust_estimation():
    """Demonstrate robust estimation methods."""
    print("\n" + "=" * 70)
    print("Robust Estimation Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate data with outliers
    n_samples = 50
    n_outliers = 5

    x = np.linspace(0, 10, n_samples)
    true_slope = 2.0
    true_intercept = 1.0

    y = true_intercept + true_slope * x + np.random.randn(n_samples) * 0.5

    # Add outliers
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    y[outlier_idx] += np.random.randn(n_outliers) * 10

    A = np.column_stack([np.ones(n_samples), x])

    print(f"\nData: {n_samples} samples with {n_outliers} outliers")
    print(f"True parameters: intercept={true_intercept}, slope={true_slope}")

    # OLS (sensitive to outliers)
    result_ols = ordinary_least_squares(A, y)
    print(f"\nOLS: intercept={result_ols.x[0]:.4f}, slope={result_ols.x[1]:.4f}")

    # Huber M-estimator
    result_huber = huber_regression(A, y)
    print(f"Huber: intercept={result_huber.x[0]:.4f}, slope={result_huber.x[1]:.4f}")
    print(f"  Iterations: {result_huber.n_iter}, Converged: {result_huber.converged}")

    # Tukey bisquare M-estimator
    result_tukey = tukey_regression(A, y)
    print(f"Tukey: intercept={result_tukey.x[0]:.4f}, slope={result_tukey.x[1]:.4f}")
    print(f"  Iterations: {result_tukey.n_iter}, Converged: {result_tukey.converged}")

    # Analyze weights from Tukey estimator
    weights = result_tukey.weights
    detected_outliers = np.where(weights < 0.1)[0]
    print(f"\nDetected outliers (low weight): {detected_outliers}")
    print(f"True outlier indices: {sorted(outlier_idx)}")

    # Plot robust estimation comparison
    if SHOW_PLOTS:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "Robust Estimation: OLS vs M-estimators",
            "Tukey M-estimator Weights (red = true outliers)"
        ])

        # Fit comparison
        # Regular data points
        regular_mask = np.ones(n_samples, dtype=bool)
        regular_mask[outlier_idx] = False

        fig.add_trace(go.Scatter(
            x=x[regular_mask], y=y[regular_mask], mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Data'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x[outlier_idx], y=y[outlier_idx], mode='markers',
            marker=dict(color='red', size=10, opacity=0.8),
            name='Outliers'
        ), row=1, col=1)

        x_line = np.linspace(x.min(), x.max(), 100)

        fig.add_trace(go.Scatter(
            x=x_line, y=true_intercept + true_slope * x_line, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='True'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_line, y=result_ols.x[0] + result_ols.x[1] * x_line, mode='lines',
            line=dict(color='black', width=2),
            name='OLS'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_line, y=result_huber.x[0] + result_huber.x[1] * x_line, mode='lines',
            line=dict(color='magenta', width=2, dash='dash'),
            name='Huber'
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x_line, y=result_tukey.x[0] + result_tukey.x[1] * x_line, mode='lines',
            line=dict(color='cyan', width=2, dash='dot'),
            name='Tukey'
        ), row=1, col=1)

        # Weights from Tukey estimator
        colors = ['red' if i in outlier_idx else 'blue' for i in range(n_samples)]
        fig.add_trace(go.Scatter(
            x=x, y=weights, mode='markers',
            marker=dict(color=colors, size=8, opacity=0.7),
            name='Weights', showlegend=False
        ), row=1, col=2)

        fig.add_hline(y=0.1, line_dash="dash", line_color="red", row=1, col=2)

        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="x", row=1, col=2)
        fig.update_yaxes(title_text="Tukey weight", row=1, col=2)

        fig.update_layout(height=500, width=1200, showlegend=True)
        fig.write_html("static_robust_estimation.html")
        print("\n  [Plot saved to static_robust_estimation.html]")


def demo_ransac():
    """Demonstrate RANSAC for robust fitting."""
    print("\n" + "=" * 70)
    print("RANSAC Demo")
    print("=" * 70)

    np.random.seed(42)

    # Line fitting with many outliers
    n_inliers = 40
    n_outliers = 20

    # Inliers: points on line y = 2x + 1
    x_inliers = np.random.uniform(0, 10, n_inliers)
    y_inliers = 1 + 2 * x_inliers + np.random.randn(n_inliers) * 0.3

    # Outliers: random points
    x_outliers = np.random.uniform(0, 10, n_outliers)
    y_outliers = np.random.uniform(-5, 25, n_outliers)

    x = np.concatenate([x_inliers, x_outliers])
    y = np.concatenate([y_inliers, y_outliers])

    # Shuffle
    perm = np.random.permutation(len(x))
    x, y = x[perm], y[perm]

    A = np.column_stack([np.ones(len(x)), x])

    print(f"\nData: {n_inliers} inliers + {n_outliers} outliers = {len(x)} total")
    print("True line: y = 2x + 1")

    # OLS
    result_ols = ordinary_least_squares(A, y)
    print(f"\nOLS: y = {result_ols.x[1]:.4f}x + {result_ols.x[0]:.4f}")

    # RANSAC
    threshold = 1.0  # Inlier threshold (residual threshold)

    result_ransac = ransac(
        A, y, min_samples=2, residual_threshold=threshold, max_trials=100
    )

    print(f"RANSAC: y = {result_ransac.x[1]:.4f}x + {result_ransac.x[0]:.4f}")
    print(f"  Inliers found: {result_ransac.n_inliers}/{len(x)}")

    print("\nNote: RANSAC successfully recovers the true line despite")
    print("33% of the data being outliers.")

    # Plot RANSAC
    if SHOW_PLOTS:
        fig = go.Figure()

        # Determine inliers based on RANSAC residuals
        y_pred = result_ransac.x[0] + result_ransac.x[1] * x
        residuals = np.abs(y - y_pred)
        is_inlier = residuals < threshold

        fig.add_trace(go.Scatter(
            x=x[is_inlier], y=y[is_inlier], mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Inliers'
        ))

        fig.add_trace(go.Scatter(
            x=x[~is_inlier], y=y[~is_inlier], mode='markers',
            marker=dict(color='red', size=8, opacity=0.6),
            name='Outliers'
        ))

        x_line = np.linspace(x.min(), x.max(), 100)

        fig.add_trace(go.Scatter(
            x=x_line, y=1 + 2 * x_line, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='True line'
        ))

        fig.add_trace(go.Scatter(
            x=x_line, y=result_ols.x[0] + result_ols.x[1] * x_line, mode='lines',
            line=dict(color='black', width=2),
            name='OLS'
        ))

        fig.add_trace(go.Scatter(
            x=x_line, y=result_ransac.x[0] + result_ransac.x[1] * x_line, mode='lines',
            line=dict(color='red', width=2),
            name='RANSAC'
        ))

        fig.update_layout(
            title=f"RANSAC Line Fitting ({result_ransac.n_inliers} inliers / {len(x)} total)",
            xaxis_title="x",
            yaxis_title="y",
            height=500, width=800,
            showlegend=True
        )
        fig.write_html("static_ransac.html")
        print("\n  [Plot saved to static_ransac.html]")


def demo_mle_and_fisher():
    """Demonstrate MLE and Fisher information."""
    print("\n" + "=" * 70)
    print("Maximum Likelihood Estimation Demo")
    print("=" * 70)

    np.random.seed(42)

    # Generate Gaussian data
    n_samples = 100
    true_mean = 5.0
    true_std = 2.0

    data = true_mean + np.random.randn(n_samples) * true_std

    print(f"\nTrue parameters: mean={true_mean}, std={true_std}")
    print(f"Sample size: {n_samples}")

    # MLE for Gaussian parameters
    # theta[0] = mean, theta[1] = variance
    result = mle_gaussian(data)
    mle_mean = result.theta[0]
    mle_var = result.theta[1]
    mle_std = np.sqrt(mle_var)
    print(f"\nMLE estimates: mean={mle_mean:.4f}, std={mle_std:.4f}")
    print(f"Log-likelihood: {result.log_likelihood:.4f}")

    # Fisher information
    # For Gaussian: I(μ) = n/σ², I(σ²) = n/(2σ⁴)
    fisher_mean = n_samples / true_std**2
    fisher_var = n_samples / (2 * true_std**4)

    print(f"\nFisher information:")
    print(f"  I(mean) = {fisher_mean:.4f}")
    print(f"  I(variance) = {fisher_var:.4f}")

    # Cramer-Rao bounds
    crb_mean = 1 / fisher_mean
    crb_var = 1 / fisher_var

    print(f"\nCramer-Rao lower bounds (variance of unbiased estimator):")
    print(f"  Var(mean_hat) ≥ {crb_mean:.6f}")
    print(f"  Var(var_hat) ≥ {crb_var:.6f}")

    # Check actual MLE variance (through simulation)
    n_trials = 1000
    mean_estimates = []
    for _ in range(n_trials):
        sample = true_mean + np.random.randn(n_samples) * true_std
        result = mle_gaussian(sample)
        mean_estimates.append(result.theta[0])

    actual_variance = np.var(mean_estimates)
    print(f"\nSimulated variance of mean estimator: {actual_variance:.6f}")
    print(f"Efficiency: {crb_mean / actual_variance:.4f}")
    print("(Efficiency = 1.0 means estimator achieves the CRB)")


def demo_model_selection():
    """Demonstrate model selection using information criteria."""
    print("\n" + "=" * 70)
    print("Model Selection Demo")
    print("=" * 70)

    np.random.seed(42)

    # True model: quadratic y = 1 + 2x + 0.5x²
    n_samples = 50
    x = np.linspace(0, 5, n_samples)
    y_true = 1 + 2 * x + 0.5 * x**2
    y = y_true + np.random.randn(n_samples) * 0.5

    print("\nTrue model: y = 1 + 2x + 0.5x²")
    print("Comparing polynomial models of degree 1 through 5:")
    print("-" * 60)
    print(f"{'Degree':>6} {'RSS':>10} {'k':>4} {'AIC':>10} {'BIC':>10}")
    print("-" * 60)

    results = []
    for degree in range(1, 6):
        # Build design matrix for polynomial
        A = np.column_stack([x**i for i in range(degree + 1)])
        result = ordinary_least_squares(A, y)

        rss = np.sum(result.residuals**2)
        k = degree + 1  # Number of parameters
        n = n_samples

        # Compute log-likelihood (assuming Gaussian errors)
        sigma2 = rss / n
        log_lik = -n / 2 * np.log(2 * np.pi * sigma2) - rss / (2 * sigma2)

        aic_val = aic(log_lik, k)
        bic_val = bic(log_lik, k, n)

        results.append((degree, rss, k, aic_val, bic_val))
        print(f"{degree:>6} {rss:>10.4f} {k:>4} {aic_val:>10.2f} {bic_val:>10.2f}")

    # Find best models
    best_aic = min(results, key=lambda x: x[3])
    best_bic = min(results, key=lambda x: x[4])

    print(f"\nBest by AIC: degree {best_aic[0]}")
    print(f"Best by BIC: degree {best_bic[0]}")
    print("\nNote: True model is degree 2. AIC/BIC help avoid overfitting.")

    # Plot model selection
    if SHOW_PLOTS:
        fig = make_subplots(rows=1, cols=2, subplot_titles=[
            "Polynomial Model Fits",
            "Model Selection: AIC vs BIC"
        ])

        # Model fits
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color='blue', size=8, opacity=0.6),
            name='Data'
        ), row=1, col=1)

        x_line = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=1 + 2 * x_line + 0.5 * x_line**2, mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='True (deg 2)'
        ), row=1, col=1)

        # Fit degrees 1, 2, 3
        colors = ['red', 'green', 'purple']
        for i, deg in enumerate([1, 2, 3]):
            A_fit = np.column_stack([x_line**j for j in range(deg + 1)])
            A_data = np.column_stack([x**j for j in range(deg + 1)])
            coef = ordinary_least_squares(A_data, y).x
            y_fit = A_fit @ coef
            fig.add_trace(go.Scatter(
                x=x_line, y=y_fit, mode='lines',
                line=dict(width=1.5, dash='solid' if deg == 2 else 'dot'),
                name=f'Degree {deg}'
            ), row=1, col=1)

        # AIC/BIC comparison
        degrees = [r[0] for r in results]
        aic_vals = [r[3] for r in results]
        bic_vals = [r[4] for r in results]

        fig.add_trace(go.Bar(
            x=[d - 0.2 for d in degrees], y=aic_vals, width=0.35,
            name='AIC', marker_color='blue', opacity=0.7
        ), row=1, col=2)

        fig.add_trace(go.Bar(
            x=[d + 0.2 for d in degrees], y=bic_vals, width=0.35,
            name='BIC', marker_color='orange', opacity=0.7
        ), row=1, col=2)

        fig.update_xaxes(title_text="x", row=1, col=1)
        fig.update_yaxes(title_text="y", row=1, col=1)
        fig.update_xaxes(title_text="Polynomial Degree", row=1, col=2)
        fig.update_yaxes(title_text="Information Criterion", row=1, col=2)

        fig.update_layout(height=500, width=1200, showlegend=True)
        fig.write_html("static_model_selection.html")
        print("\n  [Plot saved to static_model_selection.html]")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("# PyTCL Static Estimation Example")
    print("#" * 70)

    # Least squares methods
    demo_ordinary_least_squares()
    demo_weighted_least_squares()
    demo_total_least_squares()
    demo_recursive_least_squares()
    demo_ridge_regression()

    # Robust estimation
    demo_robust_estimation()
    demo_ransac()

    # MLE and model selection
    demo_mle_and_fisher()
    demo_model_selection()

    print("\n" + "=" * 70)
    print("Example complete!")
    if SHOW_PLOTS:
        print("Plots saved: static_ols.html, static_robust_estimation.html,")
        print("             static_ransac.html, static_model_selection.html")
    print("=" * 70)


if __name__ == "__main__":
    main()
