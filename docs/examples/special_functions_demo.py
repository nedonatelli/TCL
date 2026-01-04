"""
Demonstration of special functions and mathematical operations.

This example shows:
- Bessel function computation and visualization
- Special function properties and characteristics
- Performance characteristics
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.mathematical_functions.special_functions import (
    bessel_zeros,
    besselj,
    bessely,
)

SHOW_PLOTS = True
OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def demo_bessel_functions() -> None:
    """Demonstrate Bessel functions of the first and second kind."""
    print("\n" + "=" * 60)
    print("Bessel Functions Demonstration")
    print("=" * 60)

    # Compute Bessel functions over a range
    x = np.linspace(0.1, 10, 100)

    # First kind - J_0 and J_1
    j0 = besselj(0, x)
    j1 = besselj(1, x)

    # Second kind - Y_0 and Y_1
    y0 = bessely(0, x)
    y1 = bessely(1, x)

    print(f"\nBessel Functions J_n and Y_n:")
    print(f"  J_0(1) = {besselj(0, 1.0):.6f}")
    print(f"  J_1(1) = {besselj(1, 1.0):.6f}")
    print(f"  Y_0(1) = {bessely(0, 1.0):.6f}")
    print(f"  Y_1(1) = {bessely(1, 1.0):.6f}")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Bessel Functions (First Kind)",
            "Bessel Functions (Second Kind)",
        ),
    )

    # First kind
    fig.add_trace(
        go.Scatter(
            x=x,
            y=j0,
            mode="lines",
            name="J₀(x)",
            line=dict(color="blue", width=2),
            hovertemplate="<b>J₀(x)</b><br>x: %{x:.3f}<br>J₀(x): %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=j1,
            mode="lines",
            name="J₁(x)",
            line=dict(color="red", width=2),
            hovertemplate="<b>J₁(x)</b><br>x: %{x:.3f}<br>J₁(x): %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Second kind
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y0,
            mode="lines",
            name="Y₀(x)",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Y₀(x)</b><br>x: %{x:.3f}<br>Y₀(x): %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y1,
            mode="lines",
            name="Y₁(x)",
            line=dict(color="red", width=2),
            hovertemplate="<b>Y₁(x)</b><br>x: %{x:.3f}<br>Y₁(x): %{y:.6f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_yaxes(title_text="Function Value", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="Function Value", row=1, col=2)

    fig.update_layout(
        height=500,
        title_text="Bessel Functions of the First and Second Kind",
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)",
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "special_functions_demo.html"))


def demo_higher_order_bessel() -> None:
    """Demonstrate higher order Bessel functions."""
    print("\n" + "=" * 60)
    print("Higher Order Bessel Functions")
    print("=" * 60)

    x = np.linspace(0.1, 10, 100)
    x_val = 5.0

    print(f"\nBessel functions at x={x_val}:")
    for n in range(5):
        j_n = besselj(n, x_val)
        y_n = bessely(n, x_val)
        print(f"  J_{n}({x_val}) = {j_n:.6f}, Y_{n}({x_val}) = {y_n:.6f}")

    # Plot multiple orders
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for n in range(5):
        jn_vals = besselj(n, x)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=jn_vals,
                mode="lines",
                name=f"J_{n}(x)",
                line=dict(width=2.5, color=colors[n]),
                hovertemplate=f"<b>J_{n}(x)</b><br>x: %{{x:.3f}}<br>J_{n}(x): %{{y:.6f}}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Bessel Functions of Different Orders",
        xaxis_title="x",
        yaxis_title="J_n(x)",
        height=500,
        hovermode="x unified",
        plot_bgcolor="rgba(240,240,240,0.5)",
        showlegend=True,
        legend=dict(x=0.65, y=0.95),
    )
    
    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "special_functions_demo_higher_order.html"))


def demo_bessel_zeros() -> None:
    """Demonstrate Bessel function zeros and roots."""
    print("\n" + "=" * 60)
    print("Bessel Function Zeros")
    print("=" * 60)

    print(f"\nZeros of Bessel functions are important for:")
    print(f"  - Circular drum vibrations")
    print(f"  - Cylindrical waveguides")
    print(f"  - Bessel filter design")
    print(f"  - Boundary value problems")

    # Get zeros of J_0
    zeros = bessel_zeros(0, 5)
    print(f"\nFirst 5 zeros of J_0(x): {zeros}")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Mathematical Special Functions Demonstration")
    print("=" * 60)

    demo_bessel_functions()
    demo_higher_order_bessel()
    demo_bessel_zeros()

    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()

OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHOW_PLOTS = True
