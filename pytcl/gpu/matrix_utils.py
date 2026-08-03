"""
GPU matrix utilities for numerical linear algebra.

This module provides GPU-accelerated matrix operations commonly used in
tracking algorithms, including:
- Cholesky decomposition
- QR factorization
- Matrix inversion and solving
- Memory pool management

The operations run on either GPU backend -- CuPy (NVIDIA CUDA, double
precision) or MLX (Apple Silicon, single precision) -- through the dispatch
layer in :mod:`pytcl.gpu._backend`. The backend is selected automatically.

Notes
-----
On the MLX backend all computation is float32 (MLX does not support float64
on the GPU), so results agree with the CPU implementations to roughly float32
precision rather than to machine epsilon. MLX also routes linear algebra to
the CPU stream, which on Apple Silicon is a scheduling change rather than a
data transfer.

Examples
--------
>>> from pytcl.gpu.matrix_utils import gpu_cholesky, gpu_solve
>>> import numpy as np
>>>
>>> # Compute Cholesky decomposition on GPU
>>> A = np.eye(4) + np.random.randn(4, 4) * 0.1
>>> A = A @ A.T  # Make positive definite
>>> L = gpu_cholesky(A)
>>>
>>> # Solve linear system
>>> b = np.random.randn(4)
>>> x = gpu_solve(A, b)

See Also
--------
pytcl.gpu._backend : Compute-backend dispatch layer.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import is_available
from pytcl.gpu._backend import Backend, get_compute_backend

# Module logger
_logger = logging.getLogger("pytcl.gpu.matrix_utils")


def _cholesky_succeeded(b: Backend, L: Any) -> bool:
    """
    Check whether a Cholesky factor is genuine.

    Neither backend reliably signals a non-positive-definite input, so
    exception handling is not a portable failure detector:

    - CuPy's ``cholesky`` can return an array containing NaN instead of
      raising ``LinAlgError``.
    - MLX's ``cholesky`` returns the *partial* factorization computed before
      the failing pivot, with no NaN and no exception.

    Both failure modes are caught by validating the factor directly: every
    entry must be finite and every diagonal entry strictly positive. The
    diagonal test is exact, because a factorization stops at the first pivot
    whose value is non-positive, leaving that value on the diagonal. A zero
    pivot means the input is positive *semi*-definite (singular), which is
    also not a successful Cholesky decomposition -- LAPACK, and therefore
    NumPy, treats it as a failure too.

    Parameters
    ----------
    b : Backend
        Active compute backend.
    L : array
        Candidate lower-triangular factor, shape (n, n) or batch (k, n, n).

    Returns
    -------
    bool
        True if ``L`` is a valid Cholesky factor.
    """
    L_host = b.to_numpy(L)
    if not np.isfinite(L_host).all():
        return False
    diagonal = np.diagonal(L_host, axis1=-2, axis2=-1)
    return bool((diagonal > 0.0).all())


def _try_cholesky(b: Backend, A_gpu: Any) -> Optional[Any]:
    """
    Attempt a Cholesky factorization, returning None on failure.

    Parameters
    ----------
    b : Backend
        Active compute backend.
    A_gpu : array
        Symmetric matrix on the device, shape (n, n) or batch (k, n, n).

    Returns
    -------
    array or None
        Lower-triangular factor, or None if ``A_gpu`` is not positive
        definite.
    """
    try:
        L = b.cholesky(A_gpu)
        b.evaluate(L)
    except np.linalg.LinAlgError:
        # CuPy raises this for some non-positive-definite inputs.
        return None
    return L if _cholesky_succeeded(b, L) else None


def _nearest_psd(b: Backend, A_gpu: Any, floor: float) -> Any:
    """
    Project a symmetric matrix onto the positive definite cone.

    Eigenvalues below ``floor`` (and below the level at which the working
    precision can resolve them relative to the largest eigenvalue) are raised
    to it, then the matrix is reassembled. Unlike diagonal regularization this
    repairs indefinite matrices, not just near-singular ones.

    Parameters
    ----------
    b : Backend
        Active compute backend.
    A_gpu : array
        Symmetric matrix on the device, shape (n, n) or batch (k, n, n).
    floor : float
        Minimum eigenvalue to allow.

    Returns
    -------
    array
        Positive definite matrix with the same shape as ``A_gpu``.
    """
    eigvals, eigvecs = b.eigh(A_gpu)
    eps = float(np.finfo(np.float64 if b.supports_float64 else np.float32).eps)
    # eigh returns ascending eigenvalues, so [..., -1:] is the largest per
    # matrix. Flooring relative to it keeps the result resolvably definite at
    # the working precision even when `floor` alone is below eps.
    resolvable = b.maximum(10.0 * eps * eigvals[..., -1:], floor)
    eigvals = b.maximum(eigvals, resolvable)
    return b.einsum("...ij,...j,...kj->...ik", eigvecs, eigvals, eigvecs)


def gpu_cholesky(A: ArrayLike, lower: bool = True) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated Cholesky decomposition.

    Computes L such that A = L @ L.T (lower=True) or A = U.T @ U (lower=False).

    Parameters
    ----------
    A : array_like
        Symmetric positive definite matrix, shape (n, n) or batch (k, n, n).
    lower : bool
        If True, return lower triangular. If False, return upper triangular.

    Returns
    -------
    L : ndarray
        Cholesky factor, same shape as A.

    Raises
    ------
    numpy.linalg.LinAlgError
        If matrix is not positive definite.

    Notes
    -----
    The factor is validated on the host before it is returned, because neither
    GPU backend raises reliably for non-positive-definite input. This forces a
    device synchronization.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_cholesky
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.array([[4, 2], [2, 3]])
    >>> L = to_cpu(gpu_cholesky(A))
    >>> np.allclose(L @ L.T, A)
    True
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)

    L = _try_cholesky(b, A_gpu)
    if L is None:
        raise np.linalg.LinAlgError("Matrix is not positive definite")

    if not lower:
        L = b.swapaxes(L, -2, -1)

    return L


def gpu_cholesky_safe(
    A: ArrayLike,
    lower: bool = True,
    regularization: float = 1e-10,
) -> Tuple[NDArray[np.floating[Any]], bool]:
    """
    GPU Cholesky decomposition with fallback for non-positive-definite matrices.

    If standard Cholesky fails, adds regularization to diagonal and retries.

    Parameters
    ----------
    A : array_like
        Symmetric matrix, shape (n, n) or batch (k, n, n).
    lower : bool
        Return lower (True) or upper (False) triangular factor.
    regularization : float
        Amount to add to diagonal if matrix is not positive definite.

    Returns
    -------
    L : ndarray
        Cholesky factor.
    success : bool
        True if succeeded without regularization.

    Notes
    -----
    Diagonal regularization only repairs a matrix that is positive
    semi-definite but singular; it cannot repair an indefinite one. When the
    regularized retry also fails, the factor is instead computed for the
    nearest positive definite matrix, obtained by flooring the eigenvalues of
    ``A`` at ``regularization``. This function therefore always returns a
    factor and never raises for a non-positive-definite input; ``success``
    reports whether any repair was needed.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_cholesky_safe
    >>> A = np.array([[1, 2], [2, 1]])  # Not positive definite
    >>> L, success = gpu_cholesky_safe(A)
    >>> success
    False
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)

    L = _try_cholesky(b, A_gpu)
    success = L is not None

    if L is None:
        # Add regularization
        eye = b.eye(A_gpu.shape[-1])
        L = _try_cholesky(b, A_gpu + regularization * eye)

        if L is None:
            _logger.warning(
                "Cholesky decomposition failed after regularization; "
                "using the nearest positive definite matrix"
            )
            L = b.cholesky(_nearest_psd(b, A_gpu, regularization))
        else:
            _logger.warning("Cholesky decomposition required regularization")

    if not lower:
        L = b.swapaxes(L, -2, -1)

    return L, success


def gpu_qr(
    A: ArrayLike, mode: str = "reduced"
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    GPU-accelerated QR decomposition.

    Computes A = Q @ R where Q is orthogonal and R is upper triangular.

    Parameters
    ----------
    A : array_like
        Matrix to decompose, shape (m, n) or batch (k, m, n).
    mode : str
        'reduced' (default) or 'complete'.

    Returns
    -------
    Q : ndarray
        Orthogonal matrix.
    R : ndarray
        Upper triangular matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_qr
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.random.randn(4, 3)
    >>> Q, R = gpu_qr(A)
    >>> np.allclose(to_cpu(Q) @ to_cpu(R), A)
    True
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)
    Q, R = b.qr(A_gpu, mode=mode)

    return Q, R


def gpu_solve(A: ArrayLike, b: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated linear system solve.

    Solves A @ x = b for x.

    Parameters
    ----------
    A : array_like
        Coefficient matrix, shape (n, n) or batch (k, n, n).
    b : array_like
        Right-hand side, shape (n,) or (n, m) or batch (k, n).

    Returns
    -------
    x : ndarray
        Solution vector/matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_solve
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.array([[3, 1], [1, 2]])
    >>> b = np.array([9, 8])
    >>> x = to_cpu(gpu_solve(A, b))
    >>> np.allclose(A @ x, b)
    True
    """
    backend = get_compute_backend()

    A_gpu = backend.asarray(A)
    b_gpu = backend.asarray(b)

    x = backend.solve(A_gpu, b_gpu)

    return x


def gpu_inv(A: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated matrix inversion.

    Parameters
    ----------
    A : array_like
        Matrix to invert, shape (n, n) or batch (k, n, n).

    Returns
    -------
    A_inv : ndarray
        Inverse matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_inv
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_inv = to_cpu(gpu_inv(A))
    >>> np.allclose(A @ A_inv, np.eye(2))
    True
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)
    A_inv = b.inv(A_gpu)

    return A_inv


def gpu_eigh(
    A: ArrayLike,
) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    GPU-accelerated eigendecomposition for symmetric matrices.

    Computes eigenvalues and eigenvectors of symmetric matrix A.

    Parameters
    ----------
    A : array_like
        Symmetric matrix, shape (n, n) or batch (k, n, n).

    Returns
    -------
    eigenvalues : ndarray
        Eigenvalues in ascending order.
    eigenvectors : ndarray
        Corresponding eigenvectors as columns.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_eigh
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.array([[2, 1], [1, 2]])
    >>> eigvals, eigvecs = gpu_eigh(A)
    >>> bool(np.allclose(np.asarray(to_cpu(eigvals)), [1.0, 3.0]))
    True
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)
    eigvals, eigvecs = b.eigh(A_gpu)

    return eigvals, eigvecs


def gpu_matrix_sqrt(A: ArrayLike) -> NDArray[np.floating[Any]]:
    """
    GPU-accelerated matrix square root for positive definite matrices.

    Computes S such that S @ S = A using eigendecomposition.

    Parameters
    ----------
    A : array_like
        Symmetric positive definite matrix.

    Returns
    -------
    S : ndarray
        Matrix square root.

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_matrix_sqrt
    >>> from pytcl.gpu.utils import to_cpu
    >>> A = np.array([[4, 0], [0, 9]])
    >>> S = to_cpu(gpu_matrix_sqrt(A))
    >>> np.allclose(S @ S, A)
    True
    """
    b = get_compute_backend()

    A_gpu = b.asarray(A)

    # Eigendecomposition
    eigvals, eigvecs = b.eigh(A_gpu)

    # Ensure non-negative eigenvalues
    eigvals = b.maximum(eigvals, 0.0)

    # Compute sqrt
    sqrt_eigvals = b.sqrt(eigvals)

    # Reconstruct: S = V @ diag(sqrt(lambda)) @ V'. The ellipsis covers both
    # the single-matrix and the batched case.
    S = b.einsum("...ij,...j,...kj->...ik", eigvecs, sqrt_eigvals, eigvecs)

    return S


class MemoryPool:
    """
    GPU memory pool manager for efficient memory allocation.

    Wraps CuPy's memory pool on NVIDIA GPUs and MLX's allocator on Apple
    Silicon, adding monitoring and limit management. With no GPU backend
    installed every method is a no-op.

    Examples
    --------
    >>> from pytcl.gpu.matrix_utils import MemoryPool
    >>> pool = MemoryPool()
    >>> stats = pool.get_stats()
    >>> sorted(stats)
    ['device_total', 'free', 'total', 'used']
    >>>
    >>> # Free cached memory
    >>> pool.free_all()
    """

    def __init__(self) -> None:
        """Initialize memory pool manager."""
        self._pool: Any = None
        self._pinned_pool: Any = None
        self._mx: Any = None
        self._default_memory_limit = 0

        if is_available("cupy"):
            import cupy as cp

            self._pool = cp.get_default_memory_pool()
            self._pinned_pool = cp.get_default_pinned_memory_pool()
        elif is_available("mlx"):
            import mlx.core as mx

            self._mx = mx
            # MLX has no getter for the limit, only a setter that returns the
            # previous value; probe and restore to capture the default.
            default_limit = mx.set_memory_limit(1 << 62)
            mx.set_memory_limit(default_limit)
            self._default_memory_limit = default_limit
        else:
            _logger.warning("No GPU backend available, MemoryPool is a no-op")

    def get_stats(self) -> dict[str, int]:
        """
        Get memory pool statistics.

        Returns
        -------
        stats : dict
            Dictionary with 'used', 'total', 'free', and 'device_total' bytes.

        Examples
        --------
        >>> from pytcl.gpu.matrix_utils import get_memory_pool
        >>> pool = get_memory_pool()
        >>> stats = pool.get_stats()
        >>> sorted(stats)
        ['device_total', 'free', 'total', 'used']
        >>> stats['used'] >= 0
        True
        """
        if self._pool is not None:
            import cupy as cp

            free, total = cp.cuda.Device().mem_info

            return {
                "used": self._pool.used_bytes(),
                "total": self._pool.total_bytes(),
                "free": free,
                "device_total": total,
            }

        if self._mx is not None:
            mx = self._mx
            active = mx.get_active_memory()
            cached = mx.get_cache_memory()
            device_total = int(mx.device_info()["memory_size"])

            return {
                "used": active,
                "total": active + cached,
                "free": device_total - active,
                "device_total": device_total,
            }

        return {"used": 0, "total": 0, "free": 0, "device_total": 0}

    def free_all(self) -> None:
        """
        Free all cached memory blocks.

        Clears the memory pool cache, which can help free up GPU memory
        when operations are complete.

        Examples
        --------
        >>> from pytcl.gpu.matrix_utils import get_memory_pool
        >>> pool = get_memory_pool()
        >>> # After allocations
        >>> pool.free_all()  # Clear cached blocks
        """
        if self._pool is not None:
            self._pool.free_all_blocks()
        if self._pinned_pool is not None:
            self._pinned_pool.free_all_blocks()
        if self._mx is not None:
            self._mx.clear_cache()

    def set_limit(self, limit: Optional[int] = None) -> None:
        """
        Set memory pool limit.

        Parameters
        ----------
        limit : int or None
            Maximum bytes to allocate. None restores the backend default
            (unlimited on CuPy).

        Examples
        --------
        >>> from pytcl.gpu.matrix_utils import get_memory_pool
        >>> pool = get_memory_pool()
        >>> # Limit to 2 GB
        >>> pool.set_limit(2 * 1024**3)
        >>> # Reset to the backend default
        >>> pool.set_limit(None)
        """
        if self._pool is not None:
            if limit is None:
                self._pool.set_limit(size=0)  # 0 means no limit
            else:
                self._pool.set_limit(size=int(limit))
        if self._mx is not None:
            if limit is None:
                self._mx.set_memory_limit(self._default_memory_limit)
            else:
                self._mx.set_memory_limit(int(limit))

    @contextmanager
    def limit_memory(self, max_bytes: int) -> Generator[None, None, None]:
        """
        Context manager for temporary memory limit.

        Parameters
        ----------
        max_bytes : int
            Maximum bytes allowed during context.

        Examples
        --------
        >>> pool = MemoryPool()
        >>> with pool.limit_memory(10**9):  # 1GB limit
        ...     # Operations here have limited memory
        ...     pass
        """
        if self._pool is not None:
            old_limit = self._pool.get_limit()
            self._pool.set_limit(size=int(max_bytes))
            try:
                yield
            finally:
                self._pool.set_limit(size=old_limit)
            return

        if self._mx is not None:
            old_limit = self._mx.set_memory_limit(int(max_bytes))
            try:
                yield
            finally:
                self._mx.set_memory_limit(old_limit)
            return

        yield


# Global memory pool instance
_memory_pool: Optional[MemoryPool] = None


def get_memory_pool() -> MemoryPool:
    """
    Get the global GPU memory pool manager.

    Returns
    -------
    pool : MemoryPool
        Global memory pool instance.

    Examples
    --------
    >>> from pytcl.gpu.matrix_utils import get_memory_pool
    >>> pool = get_memory_pool()
    >>> stats = pool.get_stats()
    >>> "used" in stats
    True
    >>> pool.set_limit(1024**3)  # 1 GB limit
    >>> pool.free_all()
    """
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = MemoryPool()
    return _memory_pool


__all__ = [
    "gpu_cholesky",
    "gpu_cholesky_safe",
    "gpu_qr",
    "gpu_solve",
    "gpu_inv",
    "gpu_eigh",
    "gpu_matrix_sqrt",
    "MemoryPool",
    "get_memory_pool",
]
