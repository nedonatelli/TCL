"""Magnetism module demonstration with geomagnetic models.

This example demonstrates the pytcl.magnetism module capabilities, including
World Magnetic Model (WMM2020) coefficients, dipole moment calculations, and
geomagnetic field properties.

Functions demonstrated:
- create_wmm2020_coefficients(): Create WMM2020 magnetic coefficients
- dipole_moment(): Calculate Earth's magnetic dipole moment
- dipole_axis(): Calculate dipole axis orientation
"""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pytcl.magnetism import (
    create_wmm2020_coefficients,
    dipole_axis,
    dipole_moment,
)

# Controls for visualization
SHOW_PLOTS = False


def demo_wmm2020_coefficients() -> None:
    """Demonstrate WMM2020 magnetic coefficients."""
    print("\n" + "=" * 60)
    print("WMM2020 Magnetic Coefficients")
    print("=" * 60)

    # Create WMM2020 coefficients
    coeffs = create_wmm2020_coefficients()

    print(f"\nCoefficients created for epoch: {coeffs.epoch}")
    print(f"Maximum order (n_max): {coeffs.n_max}")
    print(f"G coefficients shape (Gauss): {coeffs.g.shape}")
    print(f"H coefficients shape (Gauss): {coeffs.h.shape}")
    print(f"G-dot coefficients shape (Gauss/year): {coeffs.g_dot.shape}")
    print(f"H-dot coefficients shape (Gauss/year): {coeffs.h_dot.shape}")

    # Show some sample coefficients
    print("\nSample Gauss coefficients (g) [nT]:")
    print(f"  g[1,0] = {coeffs.g[1, 0]:.2f}")
    print(f"  g[1,1] = {coeffs.g[1, 1]:.2f}")
    print(f"  g[2,0] = {coeffs.g[2, 0]:.2f}")

    print("\nSample Schmidt coefficients (h) [nT]:")
    print(f"  h[1,1] = {coeffs.h[1, 1]:.2f}")
    print(f"  h[2,1] = {coeffs.h[2, 1]:.2f}")
    print(f"  h[2,2] = {coeffs.h[2, 2]:.2f}")

    # Show secular variation rates
    print("\nSample secular variation rates [nT/year]:")
    print(f"  g_dot[1,0] = {coeffs.g_dot[1, 0]:.2f}")
    print(f"  h_dot[1,1] = {coeffs.h_dot[1, 1]:.2f}")

    # Visualization: Coefficient magnitudes by order
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("G Coefficients by Order", "H Coefficients by Order"),
        vertical_spacing=0.12,
    )

    orders = np.arange(coeffs.n_max + 1)
    g_means = [np.mean(np.abs(coeffs.g[n, :])) for n in orders]
    h_means = [np.mean(np.abs(coeffs.h[n, :])) for n in orders]

    fig.add_trace(
        go.Bar(x=orders, y=g_means, name="G coefficients", marker_color="steelblue"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=orders, y=h_means, name="H coefficients", marker_color="coral"),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Order (n)", row=1, col=1)
    fig.update_xaxes(title_text="Order (n)", row=2, col=1)
    fig.update_yaxes(title_text="Mean |Coefficient| [nT]", row=1, col=1)
    fig.update_yaxes(title_text="Mean |Coefficient| [nT]", row=2, col=1)
    fig.update_layout(height=500, showlegend=True)

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "magnetism_demo.html"))


def demo_dipole_moment() -> None:
    """Demonstrate dipole moment calculation."""
    print("\n" + "=" * 60)
    print("Earth's Magnetic Dipole Moment")
    print("=" * 60)

    # Get WMM2020 coefficients
    coeffs = create_wmm2020_coefficients()

    # Calculate dipole moment
    moment = dipole_moment(coeffs)
    print(f"\nMagnetic dipole moment: {moment:.4e} A·m²")
    print(f"Equivalent magnitude: {moment:.2e} Tesla·m³")

    # Extract individual dipole components from coefficients
    g10 = coeffs.g[1, 0]  # Axial dipole coefficient
    g11 = coeffs.g[1, 1]  # Equatorial dipole component
    h11 = coeffs.h[1, 1]  # Equatorial dipole component

    print(f"\nDipole components:")
    print(f"  g10 (axial): {g10:.2f} nT")
    print(f"  g11: {g11:.2f} nT")
    print(f"  h11: {h11:.2f} nT")

    # Calculate dipole direction and magnitude
    dipole_magnitude = np.sqrt(g10**2 + g11**2 + h11**2)
    print(f"\nDipole magnitude from coefficients: {dipole_magnitude:.2f} nT")

    # Visualization: Dipole strength over harmonic orders
    fig = go.Figure()

    orders = np.arange(1, coeffs.n_max + 1)
    order_moments = []

    for n in orders:
        order_coeffs = np.sqrt(
            np.sum(coeffs.g[n, :] ** 2) + np.sum(coeffs.h[n, :] ** 2)
        )
        order_moments.append(order_coeffs)

    fig.add_trace(
        go.Scatter(
            x=orders,
            y=order_moments,
            mode="lines+markers",
            name="Harmonic strength",
            line=dict(color="darkblue", width=2),
            marker=dict(size=8),
        )
    )

    fig.update_layout(
        title="Magnetic Field Harmonic Strength by Order",
        xaxis_title="Harmonic Order (n)",
        yaxis_title="RMS Strength [nT]",
        height=400,
        hovermode="x unified",
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "magnetism_demo.html"))


def demo_dipole_axis() -> None:
    """Demonstrate dipole axis calculation."""
    print("\n" + "=" * 60)
    print("Magnetic Dipole Axis")
    print("=" * 60)

    # Get WMM2020 coefficients
    coeffs = create_wmm2020_coefficients()

    # Calculate dipole axis
    dipole_axis_result = dipole_axis(coeffs)
    print(f"\nDipole axis: {dipole_axis_result}")

    # Calculate poles from coefficients
    # Magnetic poles are where field lines are vertical
    g10 = coeffs.g[1, 0]
    g11 = coeffs.g[1, 1]
    h11 = coeffs.h[1, 1]

    # Calculate magnetic inclination (dip angle) at equator
    # At magnetic equator, inclination = arctan(2 * g10 / sqrt(g11^2 + h11^2))
    equatorial_dip = 2 * g10 / np.sqrt(g11**2 + h11**2)
    inclination_deg = np.degrees(np.arctan(equatorial_dip))

    print(f"\nDipole field properties:")
    print(f"  Axial dipole ratio: {g10 / np.sqrt(g11**2 + h11**2):.4f}")
    print(f"  Field inclination estimate: {inclination_deg:.2f}°")

    # Visualization: Dipole offset from center
    # Geographic pole vs Magnetic pole
    fig = go.Figure()

    # Add geographic pole
    fig.add_trace(
        go.Scattergeo(
            lon=[0],
            lat=[90],
            mode="markers",
            marker=dict(size=12, color="blue", symbol="star"),
            name="Geographic North Pole",
        )
    )

    # Add magnetic dipole approximation
    # Magnetic declination and inclination affect pole position
    mag_lat_offset = np.degrees(np.arctan2(h11, g11))
    mag_lon_offset = np.degrees(np.arctan2(g11, g10))

    fig.add_trace(
        go.Scattergeo(
            lon=[mag_lon_offset],
            lat=[90 - np.degrees(np.arctan2(np.sqrt(g11**2 + h11**2), g10))],
            mode="markers",
            marker=dict(size=12, color="red", symbol="x"),
            name="Magnetic Dipole Location",
        )
    )

    fig.update_layout(
        title="Magnetic Dipole Axis Orientation (Approximate)",
        geo=dict(projection_type="orthographic"),
        height=500,
        showlegend=True,
    )

    if SHOW_PLOTS:
        fig.show()
    else:
        fig.write_html(str(OUTPUT_DIR / "magnetism_demo.html"))


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("Magnetism Module Demonstration")
    print("=" * 60)

    demo_wmm2020_coefficients()
    demo_dipole_moment()
    demo_dipole_axis()

    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)


OUTPUT_DIR = Path("docs/_static/images/examples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    main()
