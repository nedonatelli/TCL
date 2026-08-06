# Tutorial Modules Index

Collection of 10 interactive tutorial scripts in `docs/tutorials/` covering the core tracking and estimation algorithms the Tracker Component Library implements. Each script is self-contained (NumPy, SciPy, and Plotly only) and generates an interactive HTML visualization. Six of the tutorials (Kalman filtering, nonlinear filtering, signal processing, radar detection, INS-GNSS integration, and multi-target tracking) have companion `.rst` pages in the same directory showing the equivalent `pytcl` API calls.

## Available Tutorials

### Core Filtering & Estimation

#### 1. **Kalman Filtering** (`kalman_filtering.py`)
- **Topics**: Linear Kalman filter, state-space models, trajectory estimation
- **Key Concepts**:
  - Filter initialization and covariance propagation
  - Measurement update and state prediction
  - Performance metrics (RMSE)
- **Example Output**: 2D trajectory tracking with position and velocity estimation
- **Files**: `kalman_filtering.html`

#### 2. **Nonlinear Filtering** (`nonlinear_filtering.py`)
- **Topics**: Extended Kalman Filter (EKF), Unscented Kalman Filter (UKF), nonlinear systems
- **Key Concepts**:
  - Jacobian-based linearization (EKF)
  - Sigma-point transforms (UKF)
  - Measurement nonlinearity (polar to Cartesian)
- **Example Output**: Comparison of EKF vs UKF for nonlinear measurement model
- **Files**: `nonlinear_filtering.html`

#### 3. **Particle Filters** (`particle_filters.py`)
- **Topics**: Sequential Monte Carlo, bootstrap particle filter, resampling
- **Key Concepts**:
  - Particle weight updating via likelihood
  - Systematic resampling (ESS criterion)
  - Nonlinear, non-Gaussian state estimation
- **Example Output**: Bootstrap particle filter vs EKF trajectory and error comparison
- **Files**: `particle_filters.html`

#### 4. **Smoothing Algorithms** (`smoothing_algorithms.py`)
- **Topics**: Rauch-Tung-Striebel (RTS) smoother, backward smoothing pass
- **Key Concepts**:
  - Forward filter pass (standard Kalman)
  - Backward smoother gain computation
  - Uncertainty reduction from future measurements
- **Example Output**: 66% improvement in RMSE through smoothing
- **Files**: `smoothing_algorithms.html`

### Signal & Data Processing

#### 5. **Signal Processing** (`signal_processing.py`)
- **Topics**: FFT analysis, Butterworth filtering, time-frequency analysis
- **Key Concepts**:
  - Frequency domain representation
  - IIR filter design and application
  - Spectrogram computation via STFT
- **Example Output**: Multi-component signal with filtering and spectrogram
- **Files**: `signal_processing.html`

#### 6. **Robust Estimation** (`robust_estimation.py`)
- **Topics**: RANSAC, IRLS with Huber loss, outlier rejection
- **Key Concepts**:
  - Random sample consensus for outlier-robust fitting
  - Iterative reweighting with robust loss functions
  - Comparison of OLS, RANSAC, and IRLS
- **Example Output**: Line fitting with ~15% outliers effectively handled
- **Files**: `robust_estimation.html`

### Radar & Navigation

#### 7. **Radar Detection** (`radar_detection.py`)
- **Topics**: OS-CFAR detection, Range-Doppler processing
- **Key Concepts**:
  - Order-Statistic CFAR algorithm
  - Adaptive thresholding
  - Range and Doppler profile extraction
- **Example Output**: Synthetic radar data with target detection map
- **Files**: `radar_detection.html`

#### 8. **INS-GNSS Integration** (`ins_gnss_integration.py`)
- **Topics**: Sensor fusion, inertial navigation, GNSS integration
- **Key Concepts**:
  - INS drift accumulation modeling
  - Sparse GNSS measurement fusion
  - Kalman filter fusion architecture
- **Example Output**: Navigation accuracy improvement with sensor fusion
- **Files**: `ins_gnss_integration.html`

### Tracking & Data Association

#### 9. **Multi-Target Tracking** (`multi_target_tracking.py`)
- **Topics**: Global Nearest Neighbor association, track management
- **Key Concepts**:
  - GNN greedy matching
  - Track initiation, confirmation, deletion
  - Measurement gating
- **Example Output**: 4 targets with track confirmation and false alarm handling
- **Files**: `multi_target_tracking.html`

#### 10. **Data Association** (`data_association.py`)
- **Topics**: GNN vs optimal assignment (Hungarian algorithm), assignment costs
- **Key Concepts**:
  - Cost matrix formulation
  - Global Nearest Neighbor (greedy)
  - Optimal assignment via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)
  - Track management with age/confidence logic
- **Example Output**: Comparison of GNN vs Hungarian algorithms
- **Files**: `data_association.html`

## Tutorial Statistics

| Category | Tutorials | Focus |
|----------|-----------|-------|
| Filtering | 4 | Linear, nonlinear, particle-based, smoothing |
| Processing | 2 | Signal analysis, robust estimation |
| Radar/Navigation | 2 | Radar detection, sensor fusion |
| Tracking | 2 | Multi-target, data association |
| **Total** | **10** | Core tracking and estimation algorithms |

## Generated Visualizations

All tutorials generate interactive Plotly HTML visualizations saved to:
```
docs/tutorials/output/
```

Files generated (one per tutorial, ~4.9-5.7 MB each):
- `kalman_filtering.html`
- `nonlinear_filtering.html`
- `particle_filters.html`
- `smoothing_algorithms.html`
- `signal_processing.html`
- `robust_estimation.html`
- `radar_detection.html`
- `ins_gnss_integration.html`
- `multi_target_tracking.html`
- `data_association.html`

**Total size**: ~50 MB of interactive visualizations

Four of these (`kalman_filtering`, `nonlinear_filtering`, `signal_processing`, `multi_target_tracking`) are also copied to `docs/_static/images/tutorials/` so the Sphinx tutorial pages can embed them.

## Running Tutorials

The scripts require Plotly (`pip install nrl-tracker[visualization]`).

Run individual tutorial:
```bash
cd docs/tutorials/
python kalman_filtering.py
```

Run all tutorials:
```bash
cd docs/tutorials/
for script in *.py; do python "$script"; done
```

## Features

All tutorials include:
- Step-by-step algorithm explanations
- Performance metrics (RMSE, detection counts, etc.)
- Interactive Plotly visualizations
- Synthetic data generation
- Algorithm comparison where applicable
- Complete source code with comments

## Learning Path

Recommended progression for learning TCL:

1. **Start**: Kalman Filtering -> understand basic concepts
2. **Extend**: Nonlinear Filtering -> handle real-world nonlinearity
3. **Advanced**: Particle Filters -> non-Gaussian systems
4. **Refinement**: Smoothing Algorithms -> improve estimates with future data
5. **Robustness**: Robust Estimation -> handle outliers
6. **Applications**:
   - Signal Processing -> pre/post-processing
   - Radar Detection -> specific application
   - INS-GNSS Integration -> multi-sensor fusion
   - Multi-Target Tracking -> complex scenarios
   - Data Association -> measurement-to-track matching

## Integration with Documentation

The `.rst` tutorial pages embed the visualizations (copied to `docs/_static/images/tutorials/`) via:
```rst
.. raw:: html

   <div class="plotly-container aspect-wide">
       <iframe class="plotly-iframe" src="../_static/images/tutorials/kalman_filtering.html"></iframe>
   </div>
```

## Resources

- **Tutorial Scripts**: `docs/tutorials/`
- **HTML Visualizations**: `docs/tutorials/output/`
- **Main Documentation**: `docs/index.rst`
- **API Reference**: TCL library documentation
- **Examples**: `examples/` directory

---
Last reviewed: August 6, 2026 (v2.0.0)
