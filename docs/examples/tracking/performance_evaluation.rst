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

- **OSPA**: Optimal Sub-Pattern Assignment distance
- **RMSE**: Root Mean Square Error for localization
- **Consistency**: NEES and NIS statistics for filter tuning
- **Monte Carlo**: Averaging performance over repeated runs

OSPA Metric
-----------

OSPA combines localization error and cardinality error:

- **Localization**: Distance between matched targets
- **Cardinality**: Penalty for missed/false targets
- **Order parameter (p)**: Controls metric sensitivity
- **Cutoff (c)**: Maximum localization error

Key Concepts
------------

- **Localization vs cardinality**: OSPA separates position error from
  missed/false target penalties
- **Filter consistency**: NEES compares state error against the filter's
  own covariance
- **Innovation consistency**: NIS checks measurement residuals
- **Tuning diagnosis**: Optimistic and conservative filters show up as
  NEES above or below the chi-squared bounds

Code Highlights
---------------

The example demonstrates:

- Computing OSPA with ``ospa()``, including its localization and
  cardinality components
- OSPA history over a scenario, computed scan by scan
- NEES consistency for correctly, optimistically, and conservatively
  tuned filters
- Monte Carlo evaluation of RMSE, NEES, and NIS

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
