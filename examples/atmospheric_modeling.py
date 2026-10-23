"""
Atmospheric Modeling and Orbital Decay Analysis

Demonstrates the NRLMSISE-00 empirical atmosphere (the reference C
implementation, compiled into pytcl) and drag calculations for
analyzing satellite orbital decay and atmospheric interactions.

Key scenarios:
1. NRLMSISE-00 density and composition, quiet vs geomagnetic storm
2. ISS altitude profile across different solar activity levels
3. Satellite drag coefficient database
4. Orbit decay simulation (LEO satellite)
5. Temperature profile comparison across altitude range
"""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from pytcl.atmosphere import (
    nrlmsise00,
    us_standard_atmosphere_1976,
    uses_compiled_backend,
)


def _nrlmsise00_profile(
    altitudes_km, f107, f107a, ap, lat_deg=0.0, lon_deg=0.0, doy=1, sec=0.0
):
    """Evaluate NRLMSISE-00 over altitudes; returns (d, t) arrays, one row each."""
    lst = (sec / 3600.0 + lon_deg / 15.0) % 24.0
    outs = [
        nrlmsise00(doy, sec, float(alt), lat_deg, lon_deg, lst, f107a, f107, ap)
        for alt in altitudes_km
    ]
    return np.array([o.d for o in outs]), np.array([o.t for o in outs])


def plot_nrlmsise00_profiles():
    """
    Plot NRLMSISE-00 mass density and constituents vs altitude.

    Compares a quiet day against a geomagnetic storm (7-element Ap
    history), using the same reference C code the MATLAB TCL wraps.
    """
    backend = "compiled C extension" if uses_compiled_backend() else "pure Python"
    print(f"NRLMSISE-00 backend: {backend}")

    altitudes_km = np.linspace(100, 1000, 90)
    quiet_rho = []
    storm_rho = []
    quiet_t = []
    storm_t = []
    storm_ap = [148.8, 160.0, 139.0, 127.0, 118.0, 122.5, 115.4]
    for alt in altitudes_km:
        q = nrlmsise00(310, 50000.0, float(alt), 55.0, 10.0, 16.0, 140.0, 180.0, 4.0)
        st = nrlmsise00(
            310,
            50000.0,
            float(alt),
            55.0,
            10.0,
            16.0,
            140.0,
            180.0,
            48.8,
            ap_array=storm_ap,
        )
        quiet_rho.append(q.d[5])
        storm_rho.append(st.d[5])
        quiet_t.append(q.t[1])
        storm_t.append(st.t[1])

    fig = sp.make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Total mass density", "Temperature"),
    )
    fig.add_trace(
        go.Scatter(x=quiet_rho, y=altitudes_km, name="Quiet (Ap 4)"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=storm_rho, y=altitudes_km, name="Storm (Ap 48.8)"), row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=quiet_t, y=altitudes_km, name="Quiet T", showlegend=False),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=storm_t, y=altitudes_km, name="Storm T", showlegend=False),
        row=1,
        col=2,
    )
    fig.update_xaxes(type="log", title_text="Density (kg/m^3)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (K)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (km)", row=1, col=1)
    fig.update_layout(title="NRLMSISE-00: quiet vs geomagnetic storm")

    ratio_400 = None
    for alt, qd, sd in zip(altitudes_km, quiet_rho, storm_rho):
        if abs(alt - 400.0) < 6:
            ratio_400 = sd / qd
            break
    if ratio_400 is not None:
        print(f"Storm/quiet density ratio near 400 km: {ratio_400:.2f}x")
    return fig


def plot_density_vs_altitude():
    """
    Plot NRLMSISE-00 density across the altitude range.

    Compares quiet and active solar activity conditions.
    """
    # Altitude range from sea level to 1000 km
    altitudes_km = np.concatenate(
        [
            np.linspace(0, 100, 50),  # Lower atmosphere (detailed)
            np.linspace(100, 1000, 100),  # Upper atmosphere
        ]
    )

    # Quiet solar activity (F107=70, Ap=0)
    d_quiet, _ = _nrlmsise00_profile(
        altitudes_km, 70, 70, 0, lat_deg=45.0, lon_deg=-75.0, doy=100, sec=43200.0
    )

    # Active solar activity (F107=200, Ap=50)
    d_active, _ = _nrlmsise00_profile(
        altitudes_km, 200, 200, 50, lat_deg=45.0, lon_deg=-75.0, doy=100, sec=43200.0
    )

    # US Standard Atmosphere for comparison (up to 85 km only)
    altitudes_short = altitudes_km[altitudes_km <= 85]
    output_us76 = np.array(
        [us_standard_atmosphere_1976(h * 1000).density for h in altitudes_short]
    )

    fig = sp.make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # Log-scale density plot
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=np.maximum(d_quiet[:, 5], 1e-20),
            name="Quiet (F107=70, Ap=0)",
            mode="lines",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Density (Quiet)</b><br>Alt: %{x:.1f} km<br>rho: %{y:.2e} kg/m³",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=np.maximum(d_active[:, 5], 1e-20),
            name="Active (F107=200, Ap=50)",
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="<b>Density (Active)</b><br>Alt: %{x:.1f} km<br>rho: %{y:.2e} kg/m³",
        ),
        secondary_y=False,
    )

    # US 76 for reference
    fig.add_trace(
        go.Scatter(
            x=altitudes_short,
            y=np.maximum(output_us76, 1e-20),
            name="US Standard 1976",
            mode="lines",
            line=dict(color="green", width=2, dash="dot"),
            hovertemplate="<b>Density (US76)</b><br>Alt: %{x:.1f} km<br>rho: %{y:.2e} kg/m³",
        ),
        secondary_y=False,
    )

    fig.update_yaxes(type="log", title_text="Density (kg/m³)", secondary_y=False)

    fig.update_xaxes(title_text="Altitude (km)", range=[0, 1000])

    fig.update_layout(
        title="Atmospheric Density vs. Altitude<br><sub>NRLMSISE-00 (Quiet vs. Active Solar Activity)</sub>",
        hovermode="x unified",
        height=600,
        template="plotly_white",
    )

    return fig


def plot_composition_profile():
    """
    Plot atmospheric composition (species densities) vs. altitude.
    """
    # Altitude range
    altitudes_km = np.linspace(80, 500, 200)
    d, _ = _nrlmsise00_profile(altitudes_km, 150, 150, 5)

    fig = go.Figure()

    # Plot each species
    species = [
        ("N2", d[:, 2], "blue"),
        ("O2", d[:, 3], "green"),
        ("O", d[:, 1], "red"),
        ("He", d[:, 0], "purple"),
        ("H", d[:, 6], "orange"),
        ("Ar", d[:, 4], "brown"),
        ("N", d[:, 7], "pink"),
    ]

    for name, density, color in species:
        fig.add_trace(
            go.Scatter(
                x=altitudes_km,
                y=np.maximum(density, 1e8),
                name=name,
                mode="lines",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{name}</b><br>Alt: %{{x:.1f}} km<br>n: %{{y:.2e}} m⁻³",
            )
        )

    fig.update_yaxes(type="log", title_text="Number Density (m^-³)")

    fig.update_xaxes(title_text="Altitude (km)")

    fig.update_layout(
        title="Atmospheric Composition vs. Altitude<br><sub>NRLMSISE-00 at F107=150, Ap=5</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_temperature_profile():
    """
    Plot temperature vs. altitude across full range.
    """
    altitudes_km = np.concatenate(
        [np.linspace(0, 100, 50), np.linspace(100, 1000, 100)]
    )

    # Different solar activity levels
    conditions = [
        ("Quiet (F107=70)", 70, 70, 0, "blue"),
        ("Moderate (F107=150)", 150, 150, 5, "green"),
        ("Active (F107=200)", 200, 200, 50, "orange"),
        ("Storm (F107=150, Ap=300)", 150, 150, 300, "red"),
    ]

    fig = go.Figure()

    for label, f107, f107a, ap, color in conditions:
        _, t = _nrlmsise00_profile(altitudes_km, f107, f107a, ap)

        fig.add_trace(
            go.Scatter(
                x=t[:, 1],
                y=altitudes_km,
                name=label,
                mode="lines",
                line=dict(color=color, width=2.5),
                hovertemplate="<b>"
                + label
                + "</b><br>T: %{x:.0f} K<br>Alt: %{y:.1f} km",
            )
        )

    fig.update_yaxes(title_text="Altitude (km)", range=[0, 500])

    fig.update_xaxes(title_text="Temperature (K)", range=[150, 1100])

    fig.update_layout(
        title="Temperature Profile vs. Altitude<br><sub>NRLMSISE-00 Under Various Solar Activity Levels</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_solar_activity_effect():
    """
    Plot density response to varying solar activity index (F107).
    """
    # Fixed altitude ISS orbit
    iss_altitude = 408.0  # km

    # Vary F107 from quiet to stormy
    f107_range = np.linspace(50, 300, 50)

    densities = []
    temperatures = []

    for f107 in f107_range:
        output = nrlmsise00(
            1, 0.0, iss_altitude, 51.6, 0.0, 0.0, float(f107), float(f107), 5.0
        )  # latitude 51.6 deg: ISS inclination
        densities.append(output.d[5])
        temperatures.append(output.t[1])

    fig = sp.make_subplots(
        rows=1, cols=2, specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Density vs F107
    fig.add_trace(
        go.Scatter(
            x=f107_range,
            y=densities,
            name="Density",
            mode="lines+markers",
            line=dict(color="blue", width=2),
            marker=dict(size=4),
            hovertemplate="F107: %{x:.0f} SFU<br>rho: %{y:.2e} kg/m³",
        ),
        row=1,
        col=1,
    )

    # Temperature vs F107
    fig.add_trace(
        go.Scatter(
            x=f107_range,
            y=temperatures,
            name="Temperature",
            mode="lines+markers",
            line=dict(color="red", width=2),
            marker=dict(size=4),
            hovertemplate="F107: %{x:.0f} SFU<br>T: %{y:.0f} K",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Solar Flux F107 (SFU)", row=1, col=1)
    fig.update_xaxes(title_text="Solar Flux F107 (SFU)", row=1, col=2)
    fig.update_yaxes(title_text="Density at 408 km (kg/m³)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Temperature at 408 km (K)", row=1, col=2)

    fig.update_layout(
        title_text="Effect of Solar Activity on Atmospheric Conditions at ISS Altitude",
        height=500,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


def plot_composition_transitions():
    """
    Plot composition transition from molecular to atomic atmosphere.
    """
    altitudes_km = np.linspace(70, 300, 300)
    d, _ = _nrlmsise00_profile(altitudes_km, 150, 150, 5)

    # Calculate composition percentages
    total_dens = d[:, [0, 1, 2, 3, 4, 6, 7]].sum(axis=1)

    fig = go.Figure()

    # Stacked area chart (as percentages)
    species_data = [
        ("N2", d[:, 2] / total_dens * 100, "blue"),
        ("O2", d[:, 3] / total_dens * 100, "green"),
        ("O", d[:, 1] / total_dens * 100, "red"),
        ("He", d[:, 0] / total_dens * 100, "purple"),
        ("H", d[:, 6] / total_dens * 100, "orange"),
        ("Other", (d[:, 4] + d[:, 7]) / total_dens * 100, "gray"),
    ]

    for name, fractions, color in species_data:
        fig.add_trace(
            go.Scatter(
                x=altitudes_km,
                y=fractions,
                name=name,
                mode="lines",
                line=dict(width=0.5, color=color),
                fillcolor=color,
                fill="tonexty" if name != "N2" else "tozeroy",
                hovertemplate=f"<b>{name}</b><br>Alt: %{{x:.1f}} km<br>Fraction: %{{y:.1f}}%",
            )
        )

    fig.update_yaxes(title_text="Composition (%)", range=[0, 100])

    fig.update_xaxes(title_text="Altitude (km)")

    fig.update_layout(
        title="Atmospheric Composition Transition<br><sub>Molecular -> Atomic Atmosphere (70-300 km)</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white",
    )

    return fig


if __name__ == "__main__":
    print("Generating thermosphere modeling visualizations...")

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots
    fig0 = plot_nrlmsise00_profiles()
    output_path0 = os.path.join(output_dir, "nrlmsise00_profiles.html")
    fig0.write_html(
        output_path0, include_plotlyjs="cdn", div_id=Path(output_path0).stem
    )
    print(f"OK Saved: {output_path0}")

    fig1 = plot_density_vs_altitude()
    output_path1 = os.path.join(output_dir, "thermosphere_density.html")
    fig1.write_html(
        output_path1, include_plotlyjs="cdn", div_id=Path(output_path1).stem
    )
    print(f"OK Saved: {output_path1}")

    fig2 = plot_composition_profile()
    output_path2 = os.path.join(output_dir, "thermosphere_composition.html")
    fig2.write_html(
        output_path2, include_plotlyjs="cdn", div_id=Path(output_path2).stem
    )
    print(f"OK Saved: {output_path2}")

    fig3 = plot_temperature_profile()
    output_path3 = os.path.join(output_dir, "thermosphere_temperature.html")
    fig3.write_html(
        output_path3, include_plotlyjs="cdn", div_id=Path(output_path3).stem
    )
    print(f"OK Saved: {output_path3}")

    fig4 = plot_solar_activity_effect()
    output_path4 = os.path.join(output_dir, "thermosphere_solar_activity.html")
    fig4.write_html(
        output_path4, include_plotlyjs="cdn", div_id=Path(output_path4).stem
    )
    print(f"OK Saved: {output_path4}")

    fig5 = plot_composition_transitions()
    output_path5 = os.path.join(output_dir, "thermosphere_composition_transition.html")
    fig5.write_html(
        output_path5, include_plotlyjs="cdn", div_id=Path(output_path5).stem
    )
    print(f"OK Saved: {output_path5}")

    print("\nAll visualizations complete!")
    print("View the HTML files in a browser to interact with the plots.")
