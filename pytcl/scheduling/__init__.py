"""
Scheduling algorithms.

Ports of the MATLAB TCL's ``Scheduling`` directory: interval
scheduling by greedy and dynamic-programming methods, as used for
sensor time-allocation problems.
"""

from pytcl.scheduling.intervals import (
    partition_intervals,
    schedule_intervals,
    schedule_min_lateness_dense,
    schedule_weighted_intervals,
)

__all__ = [
    "schedule_intervals",
    "partition_intervals",
    "schedule_min_lateness_dense",
    "schedule_weighted_intervals",
]
