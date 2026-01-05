Interactive Notebooks
=====================

These Jupyter notebooks provide hands-on tutorials covering key pytcl functionality.
Each notebook is self-contained with explanations, code examples, and visualizations.

.. note::

   These notebooks are rendered statically in the documentation. To run them interactively,
   clone the repository and open them in Jupyter:

   .. code-block:: bash

      git clone https://github.com/nedonatelli/TCL.git
      cd TCL
      pip install -e ".[dev]"
      jupyter notebook docs/notebooks/

Filtering & Estimation
----------------------

.. toctree::
   :maxdepth: 1

   01_kalman_filters
   02_particle_filters

These notebooks cover state estimation fundamentals from linear Kalman filtering
to particle filters for nonlinear/non-Gaussian systems.

Multi-Target Tracking
---------------------

.. toctree::
   :maxdepth: 1

   03_multi_target_tracking

Learn data association algorithms including GNN, JPDA, and track management
for tracking multiple targets simultaneously.

Coordinate Systems
------------------

.. toctree::
   :maxdepth: 1

   04_coordinate_systems

Comprehensive coverage of geodetic, ECEF, ENU/NED frames, rotations,
quaternions, and map projections.

Navigation
----------

.. toctree::
   :maxdepth: 1

   07_ins_gnss_integration

INS/GNSS integration covering strapdown mechanization, loosely-coupled
fusion, and GNSS outage handling.

Advanced Topics
---------------

.. toctree::
   :maxdepth: 1

   05_gpu_acceleration
   06_network_flow
   08_performance_optimization

GPU acceleration with CuPy, network flow algorithms for assignment problems,
and performance optimization techniques.

Notebook Summary
----------------

.. list-table::
   :widths: 10 30 60
   :header-rows: 1

   * - #
     - Notebook
     - Description
   * - 1
     - :doc:`01_kalman_filters`
     - Linear KF, Extended KF, Unscented KF, Cubature KF, IMM estimator
   * - 2
     - :doc:`02_particle_filters`
     - Bootstrap PF, resampling strategies, effective sample size, degeneracy
   * - 3
     - :doc:`03_multi_target_tracking`
     - Data association, GNN, JPDA, track management, OSPA metrics
   * - 4
     - :doc:`04_coordinate_systems`
     - Geodetic/ECEF/ENU/NED, rotations, quaternions, map projections
   * - 5
     - :doc:`05_gpu_acceleration`
     - CuPy basics, batch processing, particle filter GPU acceleration
   * - 6
     - :doc:`06_network_flow`
     - Min-cost flow for assignment, successive shortest paths, Hungarian comparison
   * - 7
     - :doc:`07_ins_gnss_integration`
     - INS mechanization, GNSS DOP, loosely-coupled integration, outage handling
   * - 8
     - :doc:`08_performance_optimization`
     - Profiling, Numba JIT, vectorization, caching, memory optimization

Prerequisites
-------------

Most notebooks require only the core pytcl package:

.. code-block:: bash

   pip install nrl-tracker matplotlib

Some advanced notebooks have additional requirements:

- **GPU Acceleration**: ``pip install cupy-cuda12x`` (NVIDIA GPU with CUDA)
- **Network Flow**: No additional requirements
- **Performance**: ``pip install numba line_profiler memory_profiler``

Getting Started
---------------

1. Start with **Kalman Filters** (01) if you're new to state estimation
2. Move to **Particle Filters** (02) for nonlinear problems
3. Explore **Multi-Target Tracking** (03) for data association
4. Study **Coordinate Systems** (04) for navigation applications
5. Advanced users: GPU acceleration, network flow, and optimization
