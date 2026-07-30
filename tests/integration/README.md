# Integration tests

More than one subsystem, composed the way a caller would compose them.

`test_end_to_end_pipeline.py` is the reference example: truth to polar
measurement, Cartesian conversion with covariance, gating, association,
filtering, track management, HDF5 persistence, and OSPA/NEES scoring. It exists
because everything else in this suite checks one function against a reference,
and nothing checked that the pieces fit together -- which is how the library
reached 4000 passing tests while its own examples called an API it did not have.
