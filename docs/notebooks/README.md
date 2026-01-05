# pyTCL Interactive Tutorials

This directory contains Jupyter notebooks providing hands-on tutorials for the
pyTCL (Tracker Component Library).

## Launch in the Cloud

Run these notebooks directly in your browser without any local installation:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks)

## Notebooks

| # | Topic | Description |
|---|-------|-------------|
| 01 | [Kalman Filters](01_kalman_filters.ipynb) | From basics to advanced: KF, EKF, UKF, filter tuning |
| 02 | [Particle Filters](02_particle_filters.ipynb) | Bootstrap PF, resampling strategies, ESS monitoring |
| 03 | [Multi-Target Tracking](03_multi_target_tracking.ipynb) | Data association, JPDA, track management |
| 04 | [Coordinate Systems](04_coordinate_systems.ipynb) | Geodetic/ECEF, ENU/NED, quaternions, projections |
| 05 | [GPU Acceleration](05_gpu_acceleration.ipynb) | CuPy basics, batch processing, memory optimization |
| 06 | [Network Flow Solver](06_network_flow.ipynb) | Assignment problems, successive shortest paths |
| 07 | [INS/GNSS Integration](07_ins_gnss_integration.ipynb) | Strapdown INS, loosely-coupled integration, DOP |
| 08 | [Performance Optimization](08_performance_optimization.ipynb) | Profiling, Numba JIT, vectorization, caching |

## Local Installation

To run notebooks locally:

```bash
# Install pyTCL with visualization dependencies
pip install nrl-tracker[visualization]

# Install Jupyter
pip install jupyter

# Launch JupyterLab
jupyter lab docs/notebooks/
```

## Prerequisites

Each notebook lists specific prerequisites at the top. General requirements:

- Python 3.10+
- numpy, scipy, matplotlib
- nrl-tracker (pyTCL)

## Sample Data

Notebooks use sample datasets from `examples/data/`. To regenerate:

```bash
python examples/data/generate_datasets.py
```

## Contributing

Notebooks are configured with `nbstripout` to remove outputs before committing.
This keeps the repository clean and avoids merge conflicts.

To set up nbstripout locally:

```bash
pip install nbstripout
nbstripout --install
```

## Validation

Notebooks are validated in CI using pytest-nbval:

```bash
# Run notebook validation locally
pytest --nbval-lax docs/notebooks/ -v
```
