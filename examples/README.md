# Examples

This directory contains 29 comprehensive example scripts demonstrating the Tracker Component Library's major features and capabilities.

**Status**: ✅ **All 29 examples validated** - 100% execution success rate

## Quick Start

Run any example directly:

```bash
python examples/kalman_filter_comparison.py
python examples/multi_target_tracking.py
python examples/orbital_mechanics.py
```

## Examples by Category

### Tracking & Estimation (8 examples)

- **[kalman_filter_comparison.py](kalman_filter_comparison.py)** - Compare EKF, UKF, CKF, and H-infinity filters
- **[particle_filters.py](particle_filters.py)** - Particle filter implementations for nonlinear estimation
- **[smoothers_information_filters.py](smoothers_information_filters.py)** - RTS smoother and information filter techniques
- **[multi_target_tracking.py](multi_target_tracking.py)** - Global Nearest Neighbor and JPDA data association
- **[tracking_containers.py](tracking_containers.py)** - Track and detection container data structures
- **[tracking_3d.py](tracking_3d.py)** - 3D tracking scenarios with multiple sensors
- **[assignment_algorithms.py](assignment_algorithms.py)** - Hungarian algorithm, k-best, and 3D assignment
- **[performance_evaluation.py](performance_evaluation.py)** - OSPA, track metrics, and evaluation measures

### Coordinate Systems & Transforms (4 examples)

- **[coordinate_systems.py](coordinate_systems.py)** - Cartesian, spherical, geodetic conversions
- **[coordinate_visualization.py](coordinate_visualization.py)** - Visual demonstrations of coordinate transforms
- **[transforms.py](transforms.py)** - Frame transformations and rotation handling
- **[reference_frame_advanced.py](reference_frame_advanced.py)** - GCRF, ITRF, and reference frame chains

### Dynamics & Motion Models (3 examples)

- **[dynamic_models_demo.py](dynamic_models_demo.py)** - Constant velocity, coordinated turn, and other motion models
- **[orbital_mechanics.py](orbital_mechanics.py)** - Kepler orbit propagation, Lambert solution
- **[special_functions_demo.py](special_functions_demo.py)** - Mathematical and statistical functions

### Navigation & Geodesy (4 examples)

- **[navigation_geodesy.py](navigation_geodesy.py)** - WGS84, geodetic calculations, map projections
- **[ins_gnss_navigation.py](ins_gnss_navigation.py)** - INS mechanization and GNSS integration
- **[geophysical_models.py](geophysical_models.py)** - Gravity, magnetism, atmosphere models
- **[relativistic_demo.py](relativity_demo.py)** - Relativistic corrections for high-speed objects

### Astronomy & Ephemerides (3 examples)

- **[ephemeris_demo.py](ephemeris_demo.py)** - Sun, moon, planet positions at any epoch
- **[magnetic_field_demo.py](magnetism_demo.py)** - WMM and IGRF magnetic field models
- **[atmospheric_modeling.py](atmospheric_modeling.py)** - Atmosphere density, refraction effects

### Filtering & Signal Processing (2 examples)

- **[signal_processing.py](signal_processing.py)** - Digital filters, FFT, matched filtering
- **[filter_uncertainty_visualization.py](filter_uncertainty_visualization.py)** - Kalman filter covariance visualization

### Terrain & Spatial Data (3 examples)

- **[terrain_demo.py](terrain_demo.py)** - Digital elevation models, terrain analysis
- **[spatial_data_structures.py](spatial_data_structures.py)** - KD-trees, spatial indexing
- **[gaussian_mixtures.py](gaussian_mixtures.py)** - Gaussian mixture models and clustering

### Advanced Filters (2 examples)

- **[advanced_filters_comparison.py](advanced_filters_comparison.py)** - Advanced filtering techniques
- **[static_estimation.py](static_estimation.py)** - Maximum likelihood and least squares estimation

## Running All Examples

Run the comprehensive test suite:

```bash
# Python batch test with timeout
python3 << 'EOF'
import subprocess
import sys
from pathlib import Path

examples_dir = Path("examples")
example_files = sorted([f for f in examples_dir.glob("*.py") if f.name != "__init__.py"])

for example_file in example_files:
    print(f"Running {example_file.name}...", end=" ")
    try:
        result = subprocess.run(
            [sys.executable, str(example_file)],
            capture_output=True,
            timeout=60,
            text=True
        )
        if result.returncode == 0:
            print("✅ PASS")
        else:
            print(f"❌ FAIL: {result.stderr[:50]}")
    except subprocess.TimeoutExpired:
        print("⏱️ TIMEOUT")

EOF
```

## Example Structure

Each example typically follows this pattern:

```python
"""Module demonstration with feature overview.

Functions demonstrated:
- function1(): Description
- function2(): Description
"""

# Standard imports
import numpy as np
import plotly.graph_objects as go
from pytcl.<module> import Function1, Function2

# Optional: Data visualization
SHOW_PLOTS = False  # Set True to display interactive Plotly figures

def demo_feature() -> None:
    """Demonstrate a specific feature."""
    print("\nFeature Demonstration")
    print("=" * 60)
    
    # Feature implementation
    result = Function1(data)
    
    # Output results
    print(f"Result: {result}")
    
    # Optional: Create visualization
    if not SKIP_VISUALIZATIONS:
        fig = go.Figure(...)
        if SHOW_PLOTS:
            fig.show()

def main() -> None:
    """Run all demonstrations."""
    demo_feature()

if __name__ == "__main__":
    main()
```

## Performance Notes

- **terrain_demo.py**: ~24 seconds (large grid computations with heatmap visualizations)
- **multi_target_tracking.py**: ~5 seconds
- **kalman_filter_comparison.py**: ~7 seconds
- Most others: <5 seconds

Total batch execution time: ~2-3 minutes for all 29 examples

## Visualization

Examples use Plotly for interactive visualization:

- Set `SHOW_PLOTS = True` to display interactive figures in browser
- Set `SHOW_PLOTS = False` to save HTML files to `docs/_static/images/examples/`
- Set `SKIP_VISUALIZATIONS = True` to skip rendering (computational results only)

## Dependencies

All examples require:
- NumPy, SciPy
- Plotly (for visualization)
- pytcl (installed package)

Optional dependencies:
- Skyfield (for astronomy examples)
- Other specialized modules as needed

## Testing & Validation

All examples are validated to run without errors. Test results:

```
✅ PASS:    29/29
❌ FAIL:    0/29
⏱️  TIMEOUT: 0/29
```

**Last validated**: January 4, 2026

## Contributing

When adding new examples:

1. Follow the standard structure above
2. Add docstring with module description and function list
3. Include progress indicators (print statements)
4. Make visualizations optional (SKIP_VISUALIZATIONS flag)
5. Save outputs to `docs/_static/images/examples/`
6. Test execution: `python examples/your_example.py`
7. Add entry to this README

## See Also

- [Tutorials](../docs/tutorials/) - Interactive learning modules
- [API Documentation](https://pytcl.readthedocs.io/)
- [User Guides](../docs/user_guide/)
