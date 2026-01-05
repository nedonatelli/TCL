Smoothers and Information Filters
==================================

This example demonstrates fixed-interval smoothing and information filter formulations.

.. raw:: html

   <div class="plotly-container">
       <iframe class="plotly-iframe" src="../_static/images/examples/smoothers_information_filters_result.html"></iframe>
   </div>

Overview
--------

**Smoothers** use future measurements to improve past estimates:

- **RTS Smoother** - Rauch-Tung-Striebel fixed-interval smoother
- **Fixed-lag smoother** - Bounded delay for real-time applications

**Information filters** work in information (inverse covariance) space:

- More stable for high-dimensional states
- Natural for multi-sensor fusion
- Efficient for sparse measurements

Key Concepts
------------

- **Forward-backward passes**: Combining forward filter with backward pass
- **Information matrix**: Inverse of covariance matrix
- **Information vector**: Information-weighted state
- **Fusion**: Combining information from multiple sources

Code Highlights
---------------

The example demonstrates:

- RTS smoother implementation with backward recursion
- Information filter predict and update
- Converting between covariance and information forms
- Comparing filter vs smoother estimates

Source Code
-----------

.. literalinclude:: ../../examples/smoothers_information_filters.py
   :language: python
   :linenos:

Running the Example
-------------------

.. code-block:: bash

   python examples/smoothers_information_filters.py

See Also
--------

- :doc:`kalman_filter_comparison` - Standard Kalman filters
- :doc:`multi_target_tracking` - Multi-target tracking with smoothing
