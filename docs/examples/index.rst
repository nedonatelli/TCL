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

Running these scripts already requires a checkout of this repository (the
``examples/`` directory lives in it, and the "Running Examples" commands
above assume you are standing in it), so install pytcl from that checkout in
editable mode::

   pip install -e .

If you use `uv <https://docs.astral.sh/uv/>`_ for development on this repo,
``uv sync`` (see CONTRIBUTING.md) does the same thing and additionally pins
the exact dependency versions the repo is tested against.

Some examples require additional dependencies for visualization::

   pip install plotly kaleido  # For interactive and static plots

Generating Documentation Images
-------------------------------

To regenerate the static images shown in this documentation::

   python scripts/generate_example_plots.py

This will create PNG images in ``docs/_static/images/examples/``.
