Examples
========

Standalone example scripts demonstrating pytcl functionality.

These examples are complete, runnable Python scripts that you can use
as starting points for your own applications.

.. toctree::
   :maxdepth: 2

   filtering/index
   tracking/index
   clustering/index
   signal_processing/index
   coordinates/index
   orbital/index
   geophysical/index
   dynamics/index


Running Examples
----------------

All examples can be run directly from the repository root::

   python examples/kalman_filter_comparison.py
   python examples/multi_target_tracking.py

Or from the examples directory::

   cd examples
   python kalman_filter_comparison.py

Requirements
------------

Examples require pytcl to be installed::

   pip install -e .

Some examples require additional dependencies for visualization::

   pip install plotly kaleido  # For interactive and static plots

Generating Documentation Images
-------------------------------

To regenerate the static images shown in this documentation::

   python scripts/generate_example_plots.py

This will create PNG images in ``docs/_static/images/examples/``.
