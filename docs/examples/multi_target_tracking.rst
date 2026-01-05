Multi-Target Tracking
=====================

This example demonstrates GNN-based multi-target tracking with track management.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/multi_target_tracking.html"></iframe>
   </div>

Overview
--------

Multi-target tracking (MTT) addresses:

- **Data association**: Matching measurements to tracks
- **Track initiation**: Detecting new targets
- **Track maintenance**: Updating confirmed tracks
- **Track termination**: Removing lost targets

Key Concepts
------------

- **Global Nearest Neighbor (GNN)**: Optimal measurement-to-track assignment
- **Gating**: Reducing assignment candidates using statistical tests
- **Track scoring**: M/N logic and likelihood-based confirmation
- **Clutter modeling**: False alarm rate estimation

Code Highlights
---------------

The example demonstrates:

- Track initialization from unassigned measurements
- GNN assignment using Hungarian algorithm
- Kalman filter updates for each track
- Track state machine (tentative, confirmed, deleted)
- OSPA metric computation for performance evaluation

Source Code
-----------

.. literalinclude:: ../../examples/multi_target_tracking.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/multi_target_tracking.py

See Also
--------

- :doc:`assignment_algorithms` - Assignment algorithm details
- :doc:`performance_evaluation` - OSPA and tracking metrics
- :doc:`tracking_3d` - 3D target tracking
- :doc:`tracking_containers` - Track and measurement data structures
