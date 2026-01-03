"""
Atmospheric Modeling and Orbital Decay Analysis

Demonstrates NRLMSISE-00 high-fidelity atmosphere model and drag calculations
for analyzing satellite orbital decay and atmospheric interactions.

Key scenarios:
1. ISS altitude profile across different solar activity levels
2. Satellite drag coefficient database
3. Orbit decay simulation (LEO satellite)
4. Temperature profile comparison across altitude range
"""

import os

import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp

from pytcl.atmosphere import (
    NRLMSISE00,
    us_standard_atmosphere_1976,
)


def plot_density_vs_altitude():
    """
    Plot atmospheric density from NRLMSISE-00 across altitude range.
    
    Compares quiet and active solar activity conditions.
    """
    # Altitude range from sea level to 1000 km
    altitudes_km = np.concatenate([
        np.linspace(0, 100, 50),       # Lower atmosphere (detailed)
        np.linspace(100, 1000, 100)    # Upper atmosphere
    ])
    altitudes_m = altitudes_km * 1000
    
    # Quiet solar activity (F107=70, Ap=0)
    model = NRLMSISE00()
    output_quiet = model(
        latitude=np.radians(45) * np.ones_like(altitudes_m),
        longitude=np.radians(-75) * np.ones_like(altitudes_m),
        altitude=altitudes_m,
        year=2024,
        day_of_year=100,
        seconds_in_day=43200,
        f107=70,
        f107a=70,
        ap=0
    )
    
    # Active solar activity (F107=200, Ap=50)
    output_active = model(
        latitude=np.radians(45) * np.ones_like(altitudes_m),
        longitude=np.radians(-75) * np.ones_like(altitudes_m),
        altitude=altitudes_m,
        year=2024,
        day_of_year=100,
        seconds_in_day=43200,
        f107=200,
        f107a=200,
        ap=50
    )
    
    # US Standard Atmosphere for comparison (up to 85 km only)
    altitudes_short = altitudes_km[altitudes_km <= 85]
    output_us76 = np.array([
        us_standard_atmosphere_1976(h * 1000).density
        for h in altitudes_short
    ])
    
    fig = sp.make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]]
    )
    
    # Log-scale density plot
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=np.maximum(output_quiet.density, 1e-20),
            name="Quiet (F107=70, Ap=0)",
            mode="lines",
            line=dict(color="blue", width=2),
            hovertemplate="<b>Density (Quiet)</b><br>Alt: %{x:.1f} km<br>ρ: %{y:.2e} kg/m³"
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=altitudes_km,
            y=np.maximum(output_active.density, 1e-20),
            name="Active (F107=200, Ap=50)",
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            hovertemplate="<b>Density (Active)</b><br>Alt: %{x:.1f} km<br>ρ: %{y:.2e} kg/m³"
        ),
        secondary_y=False
    )
    
    # US 76 for reference
    fig.add_trace(
        go.Scatter(
            x=altitudes_short,
            y=np.maximum(output_us76, 1e-20),
            name="US Standard 1976",
            mode="lines",
            line=dict(color="green", width=2, dash="dot"),
            hovertemplate="<b>Density (US76)</b><br>Alt: %{x:.1f} km<br>ρ: %{y:.2e} kg/m³"
        ),
        secondary_y=False
    )
    
    fig.update_yaxes(
        type="log",
        title_text="Density (kg/m³)",
        secondary_y=False
    )
    
    fig.update_xaxes(
        title_text="Altitude (km)",
        range=[0, 1000]
    )
    
    fig.update_layout(
        title="Atmospheric Density vs. Altitude<br><sub>NRLMSISE-00 Model (Quiet vs. Active Solar Activity)</sub>",
        hovermode="x unified",
        height=600,
        template="plotly_white"
    )
    
    return fig


def plot_composition_profile():
    """
    Plot atmospheric composition (species densities) vs. altitude.
    """
    # Altitude range
    altitudes_km = np.linspace(80, 500, 200)
    altitudes_m = altitudes_km * 1000
    
    model = NRLMSISE00()
    output = model(
        latitude=np.zeros_like(altitudes_m),
        longitude=np.zeros_like(altitudes_m),
        altitude=altitudes_m,
        year=2024,
        day_of_year=1,
        seconds_in_day=0,
        f107=150,
        f107a=150,
        ap=5
    )
    
    fig = go.Figure()
    
    # Plot each species
    species = [
        ("N2", output.n2_density, "blue"),
        ("O2", output.o2_density, "green"),
        ("O", output.o_density, "red"),
        ("He", output.he_density, "purple"),
        ("H", output.h_density, "orange"),
        ("Ar", output.ar_density, "brown"),
        ("N", output.n_density, "pink"),
    ]
    
    for name, density, color in species:
        fig.add_trace(
            go.Scatter(
                x=altitudes_km,
                y=np.maximum(density, 1e8),
                name=name,
                mode="lines",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{name}</b><br>Alt: %{{x:.1f}} km<br>n: %{{y:.2e}} m⁻³"
            )
        )
    
    fig.update_yaxes(
        type="log",
        title_text="Number Density (m⁻³)"
    )
    
    fig.update_xaxes(
        title_text="Altitude (km)"
    )
    
    fig.update_layout(
        title="Atmospheric Composition vs. Altitude<br><sub>NRLMSISE-00 at F107=150, Ap=5</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def plot_temperature_profile():
    """
    Plot temperature vs. altitude across full range.
    """
    altitudes_km = np.concatenate([
        np.linspace(0, 100, 50),
        np.linspace(100, 1000, 100)
    ])
    altitudes_m = altitudes_km * 1000
    
    model = NRLMSISE00()
    
    # Different solar activity levels
    conditions = [
        ("Quiet (F107=70)", 70, 70, 0, "blue"),
        ("Moderate (F107=150)", 150, 150, 5, "green"),
        ("Active (F107=200)", 200, 200, 50, "orange"),
        ("Storm (F107=150, Ap=300)", 150, 150, 300, "red"),
    ]
    
    fig = go.Figure()
    
    for label, f107, f107a, ap, color in conditions:
        output = model(
            latitude=np.zeros_like(altitudes_m),
            longitude=np.zeros_like(altitudes_m),
            altitude=altitudes_m,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=f107,
            f107a=f107a,
            ap=ap
        )
        
        fig.add_trace(
            go.Scatter(
                x=output.temperature,
                y=altitudes_km,
                name=label,
                mode="lines",
                line=dict(color=color, width=2.5),
                hovertemplate="<b>" + label + "</b><br>T: %{x:.0f} K<br>Alt: %{y:.1f} km"
            )
        )
    
    fig.update_yaxes(
        title_text="Altitude (km)",
        range=[0, 500]
    )
    
    fig.update_xaxes(
        title_text="Temperature (K)",
        range=[150, 1100]
    )
    
    fig.update_layout(
        title="Temperature Profile vs. Altitude<br><sub>NRLMSISE-00 Under Various Solar Activity Levels</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def plot_solar_activity_effect():
    """
    Plot density response to varying solar activity index (F107).
    """
    # Fixed altitude ISS orbit
    iss_altitude = 408_000  # meters
    
    # Vary F107 from quiet to stormy
    f107_range = np.linspace(50, 300, 50)
    
    model = NRLMSISE00()
    densities = []
    temperatures = []
    
    for f107 in f107_range:
        output = model(
            latitude=np.radians(51.6),  # ISS inclination
            longitude=np.radians(0),
            altitude=iss_altitude,
            year=2024,
            day_of_year=1,
            seconds_in_day=0,
            f107=f107,
            f107a=f107,
            ap=5
        )
        densities.append(output.density)
        temperatures.append(output.temperature)
    
    fig = sp.make_subplots(
        rows=1, cols=2,
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
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
            hovertemplate="F107: %{x:.0f} SFU<br>ρ: %{y:.2e} kg/m³"
        ),
        row=1, col=1
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
            hovertemplate="F107: %{x:.0f} SFU<br>T: %{y:.0f} K"
        ),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Solar Flux F107 (SFU)", row=1, col=1)
    fig.update_xaxes(title_text="Solar Flux F107 (SFU)", row=1, col=2)
    fig.update_yaxes(title_text="Density at 408 km (kg/m³)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Temperature at 408 km (K)", row=1, col=2)
    
    fig.update_layout(
        title_text="Effect of Solar Activity on Atmospheric Conditions at ISS Altitude",
        height=500,
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


def plot_composition_transitions():
    """
    Plot composition transition from molecular to atomic atmosphere.
    """
    altitudes_km = np.linspace(70, 300, 300)
    altitudes_m = altitudes_km * 1000
    
    model = NRLMSISE00()
    output = model(
        latitude=np.zeros_like(altitudes_m),
        longitude=np.zeros_like(altitudes_m),
        altitude=altitudes_m,
        year=2024,
        day_of_year=1,
        seconds_in_day=0,
        f107=150,
        f107a=150,
        ap=5
    )
    
    # Calculate composition percentages
    total_dens = (
        output.n2_density + output.o2_density + output.o_density +
        output.he_density + output.h_density + output.ar_density + output.n_density
    )
    
    fig = go.Figure()
    
    # Stacked area chart (as percentages)
    species_data = [
        ("N2", output.n2_density / total_dens * 100, "blue"),
        ("O2", output.o2_density / total_dens * 100, "green"),
        ("O", output.o_density / total_dens * 100, "red"),
        ("He", output.he_density / total_dens * 100, "purple"),
        ("H", output.h_density / total_dens * 100, "orange"),
        ("Other", (output.ar_density + output.n_density) / total_dens * 100, "gray"),
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
                hovertemplate=f"<b>{name}</b><br>Alt: %{{x:.1f}} km<br>Fraction: %{{y:.1f}}%"
            )
        )
    
    fig.update_yaxes(
        title_text="Composition (%)",
        range=[0, 100]
    )
    
    fig.update_xaxes(
        title_text="Altitude (km)"
    )
    
    fig.update_layout(
        title="Atmospheric Composition Transition<br><sub>Molecular → Atomic Atmosphere (70-300 km)</sub>",
        height=600,
        hovermode="x unified",
        template="plotly_white"
    )
    
    return fig


if __name__ == "__main__":
    print("Generating NRLMSISE-00 atmospheric modeling visualizations...")
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate all plots
    fig1 = plot_density_vs_altitude()
    output_path1 = os.path.join(output_dir, "nrlmsise00_density.html")
    fig1.write_html(output_path1)
    print(f"✓ Saved: {output_path1}")
    
    fig2 = plot_composition_profile()
    output_path2 = os.path.join(output_dir, "nrlmsise00_composition.html")
    fig2.write_html(output_path2)
    print(f"✓ Saved: {output_path2}")
    
    fig3 = plot_temperature_profile()
    output_path3 = os.path.join(output_dir, "nrlmsise00_temperature.html")
    fig3.write_html(output_path3)
    print(f"✓ Saved: {output_path3}")
    
    fig4 = plot_solar_activity_effect()
    output_path4 = os.path.join(output_dir, "nrlmsise00_solar_activity.html")
    fig4.write_html(output_path4)
    print(f"✓ Saved: {output_path4}")
    
    fig5 = plot_composition_transitions()
    fig5.write_html("nrlmsise00_composition_transition.html")
    print("✓ Saved: nrlmsise00_composition_transition.html")
    
    print("\nAll visualizations complete!")
    print("View the HTML files in a browser to interact with the plots.")
