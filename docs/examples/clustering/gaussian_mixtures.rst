Gaussian Mixtures and Clustering
=================================

This example demonstrates Gaussian mixture operations and clustering algorithms.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/clustering_comparison.html"></iframe>
   </div>

Overview
--------

Gaussian mixtures and clustering are essential for:

- **Multi-hypothesis tracking**: Representing multiple target hypotheses
- **Mixture reduction**: Limiting computational complexity
- **Data clustering**: Grouping measurements or tracks

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/gaussian_mixtures.html"></iframe>
   </div>

Clustering Algorithms
---------------------

**K-Means**
   - Partitions data into k clusters
   - Minimizes within-cluster variance
   - Fast and scalable

**DBSCAN**
   - Density-based clustering
   - Handles arbitrary cluster shapes
   - Identifies outliers automatically

**Hierarchical**
   - Builds cluster tree (dendrogram)
   - Multiple linkage options
   - Flexible number of clusters

Gaussian Mixture Operations
---------------------------

- **Moment matching**: Reducing mixture to single Gaussian
- **Runnalls' algorithm**: Optimal mixture reduction
- **West's algorithm**: Alternative reduction method
- **Splitting/merging**: Dynamic mixture management

Code Highlights
---------------

The example demonstrates:

- K-means clustering with ``kmeans()``
- DBSCAN with ``dbscan()``
- Hierarchical clustering with ``hierarchical_cluster()``
- Gaussian mixture reduction with ``reduce_mixture_runnalls()``

Source Code
-----------

.. literalinclude:: ../../../examples/gaussian_mixtures.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/gaussian_mixtures.py

See Also
--------

- :doc:`spatial_data_structures` - KD-trees for clustering
- :doc:`advanced_filters_comparison` - Gaussian sum filters
