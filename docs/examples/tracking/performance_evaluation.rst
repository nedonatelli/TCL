Performance Evaluation
======================

This example demonstrates tracking performance metrics and evaluation.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../../_static/images/examples/performance_evaluation.html"></iframe>
   </div>

Overview
--------

Evaluating tracker performance requires multiple metrics:

- **OSPA/GOSPA**: Optimal Sub-Pattern Assignment distance
- **RMSE**: Root Mean Square Error for localization
- **Track statistics**: Purity, fragmentation, switches
- **Detection metrics**: Probability of detection, false alarm rate

OSPA Metric
-----------

OSPA combines localization error and cardinality error:

- **Localization**: Distance between matched targets
- **Cardinality**: Penalty for missed/false targets
- **Order parameter (p)**: Controls metric sensitivity
- **Cutoff (c)**: Maximum localization error

Key Concepts
------------

- **Track-to-truth assignment**: Matching estimated tracks to ground truth
- **Track purity**: Fraction of time track follows same target
- **Track fragmentation**: Number of track breaks per target
- **ID switches**: Number of times track switches targets

Code Highlights
---------------

The example demonstrates:

- Computing OSPA at each time step with ``ospa()``
- OSPA over time with ``ospa_over_time()``
- NEES/NIS consistency metrics
- Track-to-truth assignment and purity calculation
- ROC curves for detection performance

Source Code
-----------

.. literalinclude:: ../../../examples/performance_evaluation.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/performance_evaluation.py

See Also
--------

- :doc:`multi_target_tracking` - Tracker to evaluate
- :doc:`assignment_algorithms` - Assignment for track-truth matching
