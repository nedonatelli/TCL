"""
Compute-backend dispatch for the GPU package.

The batch filter implementations are written once against the small operation
surface defined here; this module maps that surface onto CuPy (NVIDIA CUDA) or
MLX (Apple Silicon). Two MLX properties drive the design:

- **Linear algebra runs on the CPU stream.** ``inv``, ``cholesky``, ``solve``,
  ``eigh``, and ``slogdet`` are not implemented for the MLX GPU stream, so this
  layer routes them to ``stream=mx.cpu``. On Apple Silicon that is cheap:
  unified memory means no host/device copy, only a scheduling change.
- **GPU compute is float32.** ``float64`` raises on the first MLX GPU
  operation, so the MLX backend uses float32 throughout. Results are therefore
  precision-limited relative to the CuPy backend; see
  :func:`Backend.float_dtype` and the package documentation.

Backends are selected by :func:`get_compute_backend`, which prefers CuPy when
importable (it supports float64) and otherwise uses MLX.

See Also
--------
pytcl.gpu.utils : Backend detection and host/device array transfer.
"""

from typing import Any, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from pytcl.core.exceptions import DependencyError

__all__ = [
    "Backend",
    "CuPyBackend",
    "MLXBackend",
    "get_compute_backend",
    "compute_backend_available",
]


class Backend:
    """Uniform operation surface used by the batch filter implementations.

    Subclasses adapt a specific array library. Every method mirrors the NumPy
    function of the same name unless documented otherwise.
    """

    name: str = "abstract"

    @property
    def xp(self) -> Any:
        """The underlying array module."""
        raise NotImplementedError

    @property
    def float_dtype(self) -> Any:
        """Floating dtype used for all computation on this backend."""
        raise NotImplementedError

    @property
    def supports_float64(self) -> bool:
        """Whether the backend computes in double precision."""
        raise NotImplementedError

    def asarray(self, a: Any) -> Any:
        """Convert to a backend array of :attr:`float_dtype`."""
        raise NotImplementedError

    def to_numpy(self, a: Any) -> NDArray[np.floating]:
        """Convert a backend array back to NumPy."""
        raise NotImplementedError

    def evaluate(self, *arrays: Any) -> None:
        """Force evaluation of lazily-computed arrays (no-op when eager)."""

    # -- array creation ---------------------------------------------------
    def zeros(self, shape: Any) -> Any:
        raise NotImplementedError

    def ones(self, shape: Any) -> Any:
        raise NotImplementedError

    def eye(self, n: int) -> Any:
        raise NotImplementedError

    def arange(self, n: int) -> Any:
        raise NotImplementedError

    def uniform(self, shape: Any, key: Optional[int] = None) -> Any:
        """Uniform [0, 1) samples of the given shape."""
        raise NotImplementedError

    # -- shape manipulation -----------------------------------------------
    def einsum(self, subscripts: str, *operands: Any) -> Any:
        raise NotImplementedError

    def broadcast_to(self, a: Any, shape: Any) -> Any:
        raise NotImplementedError

    def swapaxes(self, a: Any, axis1: int, axis2: int) -> Any:
        raise NotImplementedError

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        raise NotImplementedError

    def reshape(self, a: Any, shape: Any) -> Any:
        raise NotImplementedError

    def take_along_axis(self, a: Any, indices: Any, axis: int) -> Any:
        raise NotImplementedError

    # -- reductions and elementwise ---------------------------------------
    def sum(self, a: Any, axis: Optional[int] = None) -> Any:
        raise NotImplementedError

    def max(self, a: Any, axis: Optional[int] = None) -> Any:
        raise NotImplementedError

    def exp(self, a: Any) -> Any:
        raise NotImplementedError

    def log(self, a: Any) -> Any:
        raise NotImplementedError

    def sqrt(self, a: Any) -> Any:
        raise NotImplementedError

    def clip(self, a: Any, lo: Any, hi: Any) -> Any:
        raise NotImplementedError

    def maximum(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def cumsum(self, a: Any, axis: int = -1) -> Any:
        raise NotImplementedError

    def searchsorted(self, sorted_a: Any, values: Any) -> Any:
        """Indices where ``values`` would be inserted to keep ``sorted_a`` sorted.

        Matches ``numpy.searchsorted(..., side="right")`` for 1-D inputs.
        """
        raise NotImplementedError

    # -- linear algebra (CPU stream on MLX) --------------------------------
    def inv(self, a: Any) -> Any:
        raise NotImplementedError

    def cholesky(self, a: Any) -> Any:
        """Lower-triangular Cholesky factor."""
        raise NotImplementedError

    def solve(self, a: Any, b: Any) -> Any:
        raise NotImplementedError

    def qr(self, a: Any, mode: str = "reduced") -> Tuple[Any, Any]:
        """QR factorization; ``mode`` is ``'reduced'`` or ``'complete'``."""
        raise NotImplementedError

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        raise NotImplementedError

    def slogdet(self, a: Any) -> Tuple[Any, Any]:
        raise NotImplementedError


class CuPyBackend(Backend):
    """CuPy backend (NVIDIA CUDA), double precision."""

    name = "cupy"

    def __init__(self) -> None:
        import cupy as cp

        self._cp = cp

    @property
    def xp(self) -> Any:
        return self._cp

    @property
    def float_dtype(self) -> Any:
        return self._cp.float64

    @property
    def supports_float64(self) -> bool:
        return True

    def asarray(self, a: Any) -> Any:
        return self._cp.asarray(a, dtype=self._cp.float64)

    def to_numpy(self, a: Any) -> NDArray[np.floating]:
        return np.asarray(self._cp.asnumpy(a))

    def zeros(self, shape: Any) -> Any:
        return self._cp.zeros(shape, dtype=self._cp.float64)

    def ones(self, shape: Any) -> Any:
        return self._cp.ones(shape, dtype=self._cp.float64)

    def eye(self, n: int) -> Any:
        return self._cp.eye(n, dtype=self._cp.float64)

    def arange(self, n: int) -> Any:
        return self._cp.arange(n)

    def uniform(self, shape: Any, key: Optional[int] = None) -> Any:
        if key is not None:
            self._cp.random.seed(key)
        return self._cp.random.uniform(size=shape).astype(self._cp.float64)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self._cp.einsum(subscripts, *operands)

    def broadcast_to(self, a: Any, shape: Any) -> Any:
        return self._cp.broadcast_to(a, shape)

    def swapaxes(self, a: Any, axis1: int, axis2: int) -> Any:
        return self._cp.swapaxes(a, axis1, axis2)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self._cp.stack(arrays, axis=axis)

    def reshape(self, a: Any, shape: Any) -> Any:
        return self._cp.reshape(a, shape)

    def take_along_axis(self, a: Any, indices: Any, axis: int) -> Any:
        return self._cp.take_along_axis(a, indices, axis)

    def sum(self, a: Any, axis: Optional[int] = None) -> Any:
        return self._cp.sum(a, axis=axis)

    def max(self, a: Any, axis: Optional[int] = None) -> Any:
        return self._cp.max(a, axis=axis)

    def exp(self, a: Any) -> Any:
        return self._cp.exp(a)

    def log(self, a: Any) -> Any:
        return self._cp.log(a)

    def sqrt(self, a: Any) -> Any:
        return self._cp.sqrt(a)

    def clip(self, a: Any, lo: Any, hi: Any) -> Any:
        return self._cp.clip(a, lo, hi)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._cp.maximum(a, b)

    def cumsum(self, a: Any, axis: int = -1) -> Any:
        return self._cp.cumsum(a, axis=axis)

    def searchsorted(self, sorted_a: Any, values: Any) -> Any:
        return self._cp.searchsorted(sorted_a, values, side="right")

    def inv(self, a: Any) -> Any:
        return self._cp.linalg.inv(a)

    def cholesky(self, a: Any) -> Any:
        return self._cp.linalg.cholesky(a)

    def solve(self, a: Any, b: Any) -> Any:
        return self._cp.linalg.solve(a, b)

    def qr(self, a: Any, mode: str = "reduced") -> Tuple[Any, Any]:
        q, r = self._cp.linalg.qr(a, mode=mode)
        return q, r

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        return self._cp.linalg.eigh(a)

    def slogdet(self, a: Any) -> Tuple[Any, Any]:
        return self._cp.linalg.slogdet(a)


class MLXBackend(Backend):
    """MLX backend (Apple Silicon), single precision.

    Linear-algebra operations are dispatched to the CPU stream because MLX does
    not implement them for the GPU stream. Unified memory makes this a
    scheduling change rather than a data transfer.
    """

    name = "mlx"

    def __init__(self) -> None:
        import mlx.core as mx

        self._mx = mx
        self._cpu = mx.cpu

    @property
    def xp(self) -> Any:
        return self._mx

    @property
    def float_dtype(self) -> Any:
        return self._mx.float32

    @property
    def supports_float64(self) -> bool:
        return False

    def asarray(self, a: Any) -> Any:
        if isinstance(a, self._mx.array):
            return a.astype(self._mx.float32)
        return self._mx.array(np.asarray(a, dtype=np.float32))

    def to_numpy(self, a: Any) -> NDArray[np.floating]:
        self._mx.eval(a)
        return np.asarray(a, dtype=np.float64)

    def evaluate(self, *arrays: Any) -> None:
        self._mx.eval(*arrays)

    def zeros(self, shape: Any) -> Any:
        return self._mx.zeros(shape, dtype=self._mx.float32)

    def ones(self, shape: Any) -> Any:
        return self._mx.ones(shape, dtype=self._mx.float32)

    def eye(self, n: int) -> Any:
        return self._mx.eye(n, dtype=self._mx.float32)

    def arange(self, n: int) -> Any:
        return self._mx.arange(n)

    def uniform(self, shape: Any, key: Optional[int] = None) -> Any:
        if key is not None:
            return self._mx.random.uniform(
                shape=shape, dtype=self._mx.float32, key=self._mx.random.key(key)
            )
        return self._mx.random.uniform(shape=shape, dtype=self._mx.float32)

    def einsum(self, subscripts: str, *operands: Any) -> Any:
        return self._mx.einsum(subscripts, *operands)

    def broadcast_to(self, a: Any, shape: Any) -> Any:
        return self._mx.broadcast_to(a, shape)

    def swapaxes(self, a: Any, axis1: int, axis2: int) -> Any:
        return self._mx.swapaxes(a, axis1, axis2)

    def stack(self, arrays: Any, axis: int = 0) -> Any:
        return self._mx.stack(list(arrays), axis=axis)

    def reshape(self, a: Any, shape: Any) -> Any:
        return self._mx.reshape(a, shape)

    def take_along_axis(self, a: Any, indices: Any, axis: int) -> Any:
        return self._mx.take_along_axis(a, indices, axis=axis)

    def sum(self, a: Any, axis: Optional[int] = None) -> Any:
        return self._mx.sum(a, axis=axis)

    def max(self, a: Any, axis: Optional[int] = None) -> Any:
        return self._mx.max(a, axis=axis)

    def exp(self, a: Any) -> Any:
        return self._mx.exp(a)

    def log(self, a: Any) -> Any:
        return self._mx.log(a)

    def sqrt(self, a: Any) -> Any:
        return self._mx.sqrt(a)

    def clip(self, a: Any, lo: Any, hi: Any) -> Any:
        return self._mx.clip(a, lo, hi)

    def maximum(self, a: Any, b: Any) -> Any:
        return self._mx.maximum(a, b)

    def cumsum(self, a: Any, axis: int = -1) -> Any:
        return self._mx.cumsum(a, axis=axis)

    def searchsorted(self, sorted_a: Any, values: Any) -> Any:
        # MLX has no searchsorted. A comparison matrix would be O(n*m) memory
        # (10k particles -> 1e8 elements), so use a vectorized binary search:
        # O(m) memory and ceil(log2(n)) gather steps.
        mx = self._mx
        n = int(sorted_a.shape[0])
        if n == 0:
            return mx.zeros(values.shape, dtype=mx.int32)

        lo = mx.zeros(values.shape, dtype=mx.int32)
        hi = mx.full(values.shape, n, dtype=mx.int32)
        for _ in range(int(np.ceil(np.log2(n))) + 1):
            active = lo < hi
            mid = (lo + hi) // 2
            # Clamp only for the gather; inactive lanes are masked out below.
            probe = mx.take(sorted_a, mx.minimum(mid, n - 1))
            go_right = mx.logical_and(active, probe <= values)
            go_left = mx.logical_and(active, probe > values)
            lo = mx.where(go_right, mid + 1, lo)
            hi = mx.where(go_left, mid, hi)
        return lo

    def inv(self, a: Any) -> Any:
        return self._mx.linalg.inv(a, stream=self._cpu)

    def cholesky(self, a: Any) -> Any:
        return self._mx.linalg.cholesky(a, stream=self._cpu)

    def solve(self, a: Any, b: Any) -> Any:
        return self._mx.linalg.solve(a, b, stream=self._cpu)

    def qr(self, a: Any, mode: str = "reduced") -> Tuple[Any, Any]:
        if mode not in ("reduced", "complete"):
            raise ValueError(f"mode must be 'reduced' or 'complete', got {mode!r}")
        mx = self._mx
        q, r = mx.linalg.qr(a, stream=self._cpu)
        m, n = a.shape[-2], a.shape[-1]
        if mode == "reduced" or m <= n:
            # For m <= n the reduced and complete factorizations coincide.
            return q, r
        # MLX has no 'complete' mode. Extend Q's columns with an orthonormal
        # basis of the orthogonal complement of its range: the eigenvectors of
        # the projector I - Q Q^T with eigenvalue 1. eigh returns eigenvalues in
        # ascending order, so those are the trailing m - n columns.
        eye = mx.broadcast_to(mx.eye(m, dtype=a.dtype), a.shape[:-2] + (m, m))
        projector = eye - q @ mx.swapaxes(q, -2, -1)
        _, basis = mx.linalg.eigh(projector, stream=self._cpu)
        q_full = mx.concatenate([q, basis[..., :, n:]], axis=-1)
        pad = mx.zeros(a.shape[:-2] + (m - n, n), dtype=a.dtype)
        r_full = mx.concatenate([r, pad], axis=-2)
        return q_full, r_full

    def eigh(self, a: Any) -> Tuple[Any, Any]:
        return self._mx.linalg.eigh(a, stream=self._cpu)

    def slogdet(self, a: Any) -> Tuple[Any, Any]:
        return self._mx.linalg.slogdet(a, stream=self._cpu)


def compute_backend_available() -> bool:
    """Whether any GPU compute backend can be constructed."""
    try:
        import cupy  # noqa: F401

        return True
    except Exception:
        pass
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


def get_compute_backend(prefer: Optional[str] = None) -> Backend:
    """Return the active GPU compute backend.

    Parameters
    ----------
    prefer : {'cupy', 'mlx'}, optional
        Force a specific backend. Raises :class:`DependencyError` if the
        requested backend is unavailable.

    Returns
    -------
    Backend
        CuPy when importable (double precision), otherwise MLX (single
        precision on Apple Silicon).

    Raises
    ------
    DependencyError
        If no GPU compute backend is installed.
    """
    if prefer == "cupy":
        try:
            return CuPyBackend()
        except Exception as exc:
            raise DependencyError(
                "cupy is required for the requested GPU backend. "
                "Install with: pip install nrl-tracker[gpu]",
                package="cupy",
                feature="GPU compute backend",
                install_command="pip install nrl-tracker[gpu]",
            ) from exc
    if prefer == "mlx":
        try:
            return MLXBackend()
        except Exception as exc:
            raise DependencyError(
                "mlx is required for the requested GPU backend. "
                "Install with: pip install nrl-tracker[gpu-apple]",
                package="mlx",
                feature="GPU compute backend",
                install_command="pip install nrl-tracker[gpu-apple]",
            ) from exc

    try:
        return CuPyBackend()
    except Exception:
        pass
    try:
        return MLXBackend()
    except Exception as exc:
        raise DependencyError(
            "A GPU compute backend is required: install cupy (NVIDIA) with "
            "pip install nrl-tracker[gpu] or mlx (Apple Silicon) with "
            "pip install nrl-tracker[gpu-apple]",
            package="cupy or mlx",
            feature="GPU compute backend",
            install_command="pip install nrl-tracker[gpu]",
        ) from exc
