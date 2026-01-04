"""
Robust Estimation Tutorial
==========================

This tutorial demonstrates robust estimation methods for handling outliers
and non-Gaussian noise.

Topics covered:
  - RANSAC (Random Sample Consensus)
  - Iteratively Reweighted Least Squares (IRLS)
  - Huber loss function
  - Outlier detection and removal
"""

import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Output directory for plots
OUTPUT_DIR = Path("../_static/images/tutorials")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SHOW_PLOTS = False


def robust_estimation_tutorial():
    """Run complete robust estimation tutorial with visualizations."""
    
    print("\n" + "="*70)
    print("ROBUST ESTIMATION TUTORIAL")
    print("="*70)
    
    # Step 1: Generate data with outliers
    print("\nStep 1: Generate Synthetic Data with Outliers")
    print("-" * 70)
    
    np.random.seed(42)
    n_data = 100
    n_outliers = 15
    
    # True model: y = 2*x + 1
    x_data = np.linspace(0, 10, n_data)
    y_true = 2 * x_data + 1
    
    # Add Gaussian noise
    y_data = y_true + np.random.randn(n_data) * 0.5
    
    # Add outliers
    outlier_indices = np.random.choice(n_data, n_outliers, replace=False)
    y_data[outlier_indices] += np.random.uniform(-5, 5, n_outliers)
    
    print(f"Generated {n_data} data points with {n_outliers} outliers")
    print("True model: y = 2*x + 1")
    
    # Step 2: Ordinary Least Squares (OLS)
    print("\nStep 2: Run Ordinary Least Squares")
    print("-" * 70)
    
    A_ols = np.vstack([x_data, np.ones(n_data)]).T
    params_ols = np.linalg.lstsq(A_ols, y_data, rcond=None)[0]
    y_ols = params_ols[0] * x_data + params_ols[1]
    
    residuals_ols = y_data - y_ols
    rmse_ols = np.sqrt(np.mean(residuals_ols**2))
    
    print(f"OLS: y = {params_ols[0]:.3f}*x + {params_ols[1]:.3f}")
    print(f"OLS RMSE: {rmse_ols:.4f}")
    
    # Step 3: RANSAC
    print("\nStep 3: Run RANSAC (Random Sample Consensus)")
    print("-" * 70)
    
    n_iter = 100
    threshold = 1.0
    best_inliers = 0
    best_params = None
    
    for _ in range(n_iter):
        # Random sample of 2 points
        sample_idx = np.random.choice(n_data, 2, replace=False)
        x_sample = x_data[sample_idx]
        y_sample = y_data[sample_idx]
        
        # Fit line through samples
        A = np.vstack([x_sample, np.ones(2)]).T
        params = np.linalg.lstsq(A, y_sample, rcond=None)[0]
        
        # Count inliers
        y_pred = params[0] * x_data + params[1]
        residuals = np.abs(y_data - y_pred)
        inliers = np.sum(residuals < threshold)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_params = params
            best_inlier_mask = residuals < threshold
    
    y_ransac = best_params[0] * x_data + best_params[1]
    residuals_ransac = y_data - y_ransac
    rmse_ransac = np.sqrt(np.mean(residuals_ransac[best_inlier_mask]**2))
    
    print(f"RANSAC: y = {best_params[0]:.3f}*x + {best_params[1]:.3f}")
    print(f"RANSAC inliers: {best_inliers}/{n_data}")
    print(f"RANSAC RMSE (inliers only): {rmse_ransac:.4f}")
    
    # Step 4: Iteratively Reweighted Least Squares (IRLS)
    print("\nStep 4: Run Iteratively Reweighted Least Squares (IRLS)")
    print("-" * 70)
    
    params_irls = np.copy(params_ols)
    
    for iteration in range(10):
        # Compute residuals with current estimate
        y_pred = params_irls[0] * x_data + params_irls[1]
        residuals = y_data - y_pred
        
        # Huber loss weights
        c = 1.0  # threshold
        weights = np.where(
            np.abs(residuals) <= c,
            np.ones(n_data),
            c / np.abs(residuals)
        )
        
        # Weighted least squares
        W = np.diag(weights)
        A = np.vstack([x_data, np.ones(n_data)]).T
        params_irls = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ y_data, rcond=None)[0]
    
    y_irls = params_irls[0] * x_data + params_irls[1]
    residuals_irls = y_data - y_irls
    rmse_irls = np.sqrt(np.mean(residuals_irls**2))
    
    print(f"IRLS: y = {params_irls[0]:.3f}*x + {params_irls[1]:.3f}")
    print(f"IRLS RMSE: {rmse_irls:.4f}")
    
    # Step 5: Create visualizations
    print("\nStep 5: Create Visualizations")
    print("-" * 70)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "OLS Regression (Affected by Outliers)",
            "RANSAC (Outlier-Robust)",
            "IRLS with Huber Loss",
            "Residual Distributions"
        ),
        specs=[[{}, {}], [{}, {"type": "histogram"}]]
    )
    
    # Define colors for inliers/outliers
    colors = np.where(best_inlier_mask, "blue", "red")
    
    # Plot 1: OLS
    fig.add_trace(
        go.Scatter(x=x_data, y=y_data, mode="markers",
                   name="Data", marker=dict(color="lightblue", size=6, opacity=0.6)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_ols, mode="lines",
                   name="OLS Fit", line=dict(color="blue", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_true, mode="lines",
                   name="True", line=dict(color="black", width=2, dash="dash")),
        row=1, col=1
    )
    
    # Plot 2: RANSAC
    fig.add_trace(
        go.Scatter(x=x_data[best_inlier_mask], y=y_data[best_inlier_mask],
                   mode="markers", name="Inliers",
                   marker=dict(color="blue", size=6)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_data[~best_inlier_mask], y=y_data[~best_inlier_mask],
                   mode="markers", name="Outliers",
                   marker=dict(color="red", size=6)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_ransac, mode="lines",
                   name="RANSAC Fit", line=dict(color="green", width=2)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_true, mode="lines",
                   name="True", line=dict(color="black", width=2, dash="dash"),
                   showlegend=False),
        row=1, col=2
    )
    
    # Plot 3: IRLS
    fig.add_trace(
        go.Scatter(x=x_data, y=y_data, mode="markers",
                   name="Data", marker=dict(color="lightblue", size=6, opacity=0.6),
                   showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_irls, mode="lines",
                   name="IRLS Fit", line=dict(color="purple", width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_data, y=y_true, mode="lines",
                   name="True", line=dict(color="black", width=2, dash="dash"),
                   showlegend=False),
        row=2, col=1
    )
    
    # Plot 4: Residual distributions
    fig.add_trace(
        go.Histogram(x=residuals_ols, name="OLS Residuals", nbinsx=20,
                     marker=dict(color="blue"), opacity=0.6),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=residuals_ransac, name="RANSAC Residuals", nbinsx=20,
                     marker=dict(color="green"), opacity=0.6),
        row=2, col=2
    )
    fig.add_trace(
        go.Histogram(x=residuals_irls, name="IRLS Residuals", nbinsx=20,
                     marker=dict(color="purple"), opacity=0.6),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="X", row=1, col=1)
    fig.update_xaxes(title_text="X", row=1, col=2)
    fig.update_xaxes(title_text="X", row=2, col=1)
    fig.update_xaxes(title_text="Residual (m)", row=2, col=2)
    
    fig.update_yaxes(title_text="Y", row=1, col=1)
    fig.update_yaxes(title_text="Y", row=1, col=2)
    fig.update_yaxes(title_text="Y", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=2)
    
    # Add annotations with RMSE values
    fig.add_annotation(
        text=f"RMSE: {rmse_ols:.3f}", xref="paper", yref="paper",
        x=0.25, y=0.95, showarrow=False, xanchor="center"
    )
    fig.add_annotation(
        text=f"RMSE: {rmse_ransac:.3f}", xref="paper", yref="paper",
        x=0.75, y=0.95, showarrow=False, xanchor="center"
    )
    fig.add_annotation(
        text=f"RMSE: {rmse_irls:.3f}", xref="paper", yref="paper",
        x=0.25, y=0.45, showarrow=False, xanchor="center"
    )
    
    fig.update_layout(
        title="Robust Estimation Tutorial - OLS vs RANSAC vs IRLS",
        height=700,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)",
        barmode="overlay"
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "robust_estimation.html"))
    
    print("✓ Robust estimation visualization complete")
    print("\n" + "="*70)


if __name__ == "__main__":
    robust_estimation_tutorial()
