# Tutorial Modules Index

Comprehensive collection of 10 interactive tutorial modules demonstrating the Tracker Component Library's capabilities.

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
- **Example Output**: Particle filter tracking with particle cloud visualization
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
  - Jonkeers-Volgenant/Hungarian optimal algorithm
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
| **Total** | **10** | Comprehensive TCL coverage |

## Generated Visualizations

All tutorials generate interactive Plotly HTML visualizations saved to:
```
docs/_static/images/tutorials/
```

Files generated:
- `kalman_filtering.html` (4.5 MB)
- `nonlinear_filtering.html` (4.5 MB)
- `particle_filters.html` (4.5 MB)
- `smoothing_algorithms.html` (4.5 MB)
- `signal_processing.html` (5.1 MB)
- `robust_estimation.html` (4.5 MB)
- `radar_detection.html` (5.2 MB)
- `ins_gnss_integration.html` (4.6 MB)
- `multi_target_tracking.html` (4.5 MB)
- `data_association.html` (4.4 MB)

**Total size**: ~46 MB of interactive visualizations

## Running Tutorials

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
- ✅ Step-by-step algorithm explanations
- ✅ Performance metrics (RMSE, timing, etc.)
- ✅ Interactive Plotly visualizations
- ✅ Synthetic data generation
- ✅ Algorithm comparison where applicable
- ✅ Real-world scenario simulation
- ✅ Complete source code with comments
- ✅ Configurable parameters

## Learning Path

Recommended progression for learning TCL:

1. **Start**: Kalman Filtering → understand basic concepts
2. **Extend**: Nonlinear Filtering → handle real-world nonlinearity
3. **Advanced**: Particle Filters → non-Gaussian systems
4. **Refinement**: Smoothing Algorithms → improve estimates with future data
5. **Robustness**: Robust Estimation → handle outliers
6. **Applications**:
   - Signal Processing → pre/post-processing
   - Radar Detection → specific application
   - INS-GNSS Integration → multi-sensor fusion
   - Multi-Target Tracking → complex scenarios
   - Data Association → measurement-to-track matching

## Integration with Documentation

These tutorials are designed to be embedded in Sphinx documentation via:
```rst
.. raw:: html

   <iframe src="../_static/images/tutorials/kalman_filtering.html"
           width="100%" height="800" frameborder="0"></iframe>
```

## Resources

- **Tutorial Scripts**: `docs/tutorials/`
- **HTML Visualizations**: `docs/_static/images/tutorials/`
- **Main Documentation**: `docs/index.rst`
- **API Reference**: TCL library documentation
- **Examples**: `examples/` directory

---
Generated: January 4, 2026
TCL Version: 1.7.1
