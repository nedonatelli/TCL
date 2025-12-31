Tutorials
=========

Step-by-step tutorials for common tracking and estimation tasks.

These tutorials provide hands-on examples with complete, runnable code.
Each tutorial builds on concepts progressively, starting from basic
implementations and advancing to more complex scenarios.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   kalman_filtering
   nonlinear_filtering

Signal Processing
-----------------

.. toctree::
   :maxdepth: 1

   signal_processing
   radar_detection

Navigation
----------

.. toctree::
   :maxdepth: 1

   ins_gnss_integration

Multi-Target Tracking
---------------------

.. toctree::
   :maxdepth: 1

   multi_target_tracking

Prerequisites
-------------

All tutorials assume you have pytcl installed::

   pip install pytcl

For visualization examples, matplotlib is also required::

   pip install matplotlib

For wavelet tutorials, install the signal extra::

   pip install pytcl[signal]
