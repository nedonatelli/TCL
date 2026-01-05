Tracking Containers
===================

This example demonstrates track and measurement container data structures.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/filter_viz_tracking_ellipses.html"></iframe>
   </div>

Overview
--------

Efficient tracking systems require organized data structures:

- **TrackList**: Collection of tracks with spatial queries
- **MeasurementSet**: Organized measurement storage
- **Track state**: Position, velocity, covariance, metadata

Key Concepts
------------

- **Track ID management**: Unique identifiers for each track
- **Temporal indexing**: Accessing data by time step
- **Spatial queries**: Finding tracks in a region
- **Track history**: Storing past states for smoothing

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/spatial_kdtree.html"></iframe>
   </div>

**Spatial Indexing**: KD-trees enable efficient nearest-neighbor queries for track-to-measurement association.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/spatial_rtree.html"></iframe>
   </div>

**Range Queries**: R-trees support efficient rectangular range queries for gating operations.

Code Highlights
---------------

The example demonstrates:

- Creating and populating TrackList containers
- Adding tracks with state and covariance
- Querying tracks by ID, time, or spatial region
- Iterating over tracks for batch processing

Source Code
-----------

.. literalinclude:: ../../../examples/tracking_containers.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/tracking_containers.py

See Also
--------

- :doc:`multi_target_tracking` - Using containers in MTT
- :doc:`spatial_data_structures` - Spatial indexing
