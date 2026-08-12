"""Hypothesis configuration for pytcl's property tests.

CI runs derandomized so a red build is always reproducible from the commit
alone; local runs explore with a larger budget and the example database on.
Deadlines are disabled in both: numba JIT warm-up on the first call to a
compiled kernel makes per-example deadlines flaky in a way that says
nothing about correctness.
"""

import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "dev",
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci" if os.environ.get("CI") else "dev")
