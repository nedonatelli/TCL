Assignment Algorithms
=====================

Algorithms for data association and assignment problems in multi-target tracking.

.. automodule:: pytcl.assignment_algorithms
   :members:
   :undoc-members:
   :show-inheritance:

2D Assignment
-------------

Optimal assignment algorithms for bipartite matching problems.

Hungarian Algorithm
^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.assignment_algorithms.assignment2d
   :members:
   :undoc-members:
   :show-inheritance:

Gating
------

Validation region (gating) functions for measurement association.

.. automodule:: pytcl.assignment_algorithms.gating
   :members:
   :undoc-members:
   :show-inheritance:

Data Association
----------------

Track-to-measurement association algorithms.

Global Nearest Neighbor (GNN)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: pytcl.assignment_algorithms.data_association
   :members:
   :undoc-members:
   :show-inheritance:

Joint Probabilistic Data Association (JPDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JPDA algorithm computes association probabilities for all feasible
track-measurement pairings and updates each track with a weighted combination
of innovations.

.. automodule:: pytcl.assignment_algorithms.jpda
   :members:
   :undoc-members:
   :show-inheritance:
