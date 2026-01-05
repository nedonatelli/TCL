# Sample Data for pyTCL Examples

This directory contains sample datasets for use with the Jupyter tutorial notebooks
and example scripts. All datasets are synthetic or derived from public domain sources.

## Datasets

### tracking_scenarios.npz

Synthetic multi-target tracking scenarios with:
- `scenario_1`: 3 targets, linear motion, low clutter
- `scenario_2`: 5 targets, maneuvering, crossing paths
- `scenario_3`: 10 targets, dense clutter environment

**Format:** NumPy compressed archive (.npz)
**Fields per scenario:**
- `true_states`: Ground truth states (N_targets x N_timesteps x state_dim)
- `measurements`: Noisy measurements (N_timesteps x N_meas x meas_dim)
- `timestamps`: Time values (N_timesteps,)

### navigation_trajectory.npz

INS/GNSS integration test trajectory:
- Simulated aircraft flight path
- IMU measurements (accelerometer, gyroscope)
- GNSS position fixes with outages

**Format:** NumPy compressed archive (.npz)
**Fields:**
- `imu_time`: IMU timestamps (N_imu,)
- `accel`: Accelerometer readings (N_imu, 3)
- `gyro`: Gyroscope readings (N_imu, 3)
- `gnss_time`: GNSS fix timestamps (N_gnss,)
- `gnss_pos`: GNSS positions LLA (N_gnss, 3)
- `true_pos`: Ground truth positions (N_imu, 3)
- `true_vel`: Ground truth velocities (N_imu, 3)

### satellite_tle_samples.txt

Sample Two-Line Element (TLE) sets for orbital mechanics examples:
- ISS (ZARYA)
- GPS satellites
- LEO observation satellites

**Format:** Standard TLE text format

## Generating Datasets

To regenerate or customize datasets, run:

```bash
python examples/data/generate_datasets.py
```

## Usage in Notebooks

```python
import numpy as np
from pathlib import Path

# Load tracking scenario
data_dir = Path(__file__).parent / "data"
scenario = np.load(data_dir / "tracking_scenarios.npz")
true_states = scenario["scenario_1_states"]
measurements = scenario["scenario_1_measurements"]
```

## License

All sample data in this directory is released to the public domain.
