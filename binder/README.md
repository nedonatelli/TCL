# Binder Configuration

This directory contains configuration files for [mybinder.org](https://mybinder.org),
enabling users to run pyTCL Jupyter notebooks in the cloud without local installation.

## Launch Binder

Click the badge below to launch the notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks)

## Files

- `environment.yml` - Conda environment specification
- `postBuild` - Post-installation script for generating sample data

## Direct Notebook Links

Launch specific notebooks directly:

| Notebook | Launch |
|----------|--------|
| Kalman Filters | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F01_kalman_filters.ipynb) |
| Particle Filters | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F02_particle_filters.ipynb) |
| Multi-Target Tracking | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F03_multi_target_tracking.ipynb) |
| Coordinate Systems | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F04_coordinate_systems.ipynb) |
| GPU Acceleration | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F05_gpu_acceleration.ipynb) |
| Network Flow Solver | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F06_network_flow.ipynb) |
| INS/GNSS Integration | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F07_ins_gnss_integration.ipynb) |
| Performance Optimization | [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nedonatelli/TCL/main?labpath=docs%2Fnotebooks%2F08_performance_optimization.ipynb) |

## Local Development

To test the Binder configuration locally:

```bash
# Install repo2docker
pip install jupyter-repo2docker

# Build and run locally
repo2docker .
```

## Notes

- First launch may take a few minutes to build the environment
- GPU notebooks will run in CPU-fallback mode on Binder (no GPU available)
- Sample datasets are generated automatically on first launch
