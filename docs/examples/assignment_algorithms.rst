Assignment Algorithms
=====================

This example demonstrates various assignment algorithms for data association.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/assignment_algorithms.html"></iframe>
   </div>

Overview
--------

Assignment algorithms solve the measurement-to-track association problem:

- **2D Assignment**: One-to-one matching (Hungarian, Auction)
- **Multi-dimensional**: S-D assignment for multi-sensor fusion
- **K-best**: Finding multiple good assignments (Murty's algorithm)

Algorithms
----------

**Hungarian Algorithm (Kuhn-Munkres)**
   - Optimal O(n³) solution for 2D assignment
   - Guaranteed minimum cost assignment

**Auction Algorithm**
   - Bertsekas' auction-based approach
   - Good for sparse cost matrices

**Global Nearest Neighbor (GNN)**
   - Fast suboptimal assignment
   - Uses gating for efficiency

**JPDA**
   - Joint Probabilistic Data Association
   - Maintains association probabilities

**Murty's K-best**
   - Finds k best assignments
   - Used in MHT hypothesis management

Key Concepts
------------

- **Cost matrix**: Negative log-likelihood of associations
- **Gating**: Statistical test to reduce candidates
- **Dummy assignments**: Handling missed detections and clutter

Code Highlights
---------------

The example demonstrates:

- Constructing cost matrices from track-measurement distances
- Using ``hungarian()`` for optimal assignment
- ``auction_algorithm()`` for iterative bidding
- ``murty_kbest()`` for ranked assignments
- ``jpda()`` for association probabilities

Source Code
-----------

.. literalinclude:: ../../examples/assignment_algorithms.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/assignment_algorithms.py

See Also
--------

- :doc:`multi_target_tracking` - Using assignment in MTT
- :doc:`performance_evaluation` - Evaluating association quality
