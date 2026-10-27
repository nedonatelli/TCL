"""Shared numba compilation settings."""

# fastmath=True implies nnan and ninf, under which LLVM may compile a float
# comparison as if neither operand were NaN or infinite, and deletes NaN
# tests outright. On x86 ``det != 0.0`` then lost its unordered check (a NaN
# covariance gated as distance 0.0), and the GO-CFAR max() dropped a NaN
# average. Kernels that compare floats use this set: every fastmath flag
# except nnan and ninf. numba accepts only a ``set`` here, not a frozenset.
NAN_SAFE_FASTMATH = {"nsz", "arcp", "contract", "afn", "reassoc"}
