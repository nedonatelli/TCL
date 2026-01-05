Spatial Data Structures
=======================

This example demonstrates spatial data structures for efficient queries.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/spatial_kdtree.html"></iframe>
   </div>

Overview
--------

Spatial data structures enable fast nearest-neighbor and range queries:

- **KD-Tree**: k-dimensional binary search tree
- **R-Tree**: Rectangle tree for bounding box queries
- **Ball Tree**: Metric tree for arbitrary metrics
- **Cover Tree**: Efficient for intrinsic dimensionality

Key Concepts
------------

- **Nearest neighbor queries**: Find k closest points
- **Range queries**: Find all points within radius
- **Bulk loading**: Efficient tree construction
- **Metric spaces**: Distance-based operations

Data Structures
---------------

**KD-Tree**
   - Best for low-dimensional data (d < 20)
   - O(log n) average query time
   - Standard Euclidean distance

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/spatial_rtree.html"></iframe>
   </div>

**R-Tree**
   - Designed for spatial indexing
   - Handles bounding boxes well
   - Good for GIS applications

**Ball Tree**
   - Works in any metric space
   - Better for high dimensions
   - Supports custom distance functions

Code Highlights
---------------

The example demonstrates:

- Building KD-tree with ``KDTree()``
- Nearest neighbor queries with ``query()``
- Range queries with ``query_radius()``
- Bulk operations for efficiency

Source Code
-----------

.. literalinclude:: ../../../examples/spatial_data_structures.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/spatial_data_structures.py

See Also
--------

- :doc:`gaussian_mixtures` - Clustering with spatial trees
- :doc:`multi_target_tracking` - Gating with spatial queries
