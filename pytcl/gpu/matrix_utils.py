"""
GPU matrix utilities for numerical linear algebra.

This module provides GPU-accelerated matrix operations commonly used in
tracking algorithms, including:
- Cholesky decomposition
- QR factorization
- Matrix inversion and solving
- Memory pool management

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
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional, Tuple

from numpy.typing import ArrayLike, NDArray

from pytcl.core.optional_deps import import_optional, is_available, requires
from pytcl.gpu.utils import ensure_gpu_array

# Module logger
_logger = logging.getLogger("pytcl.gpu.matrix_utils")


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_cholesky(A: ArrayLike, lower: bool = True) -> NDArray:
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

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_cholesky
    >>> A = np.array([[4, 2], [2, 3]])
    >>> L = gpu_cholesky(A)
    >>> np.allclose(L @ L.T, A)
    True
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)

    L = cp.linalg.cholesky(A_gpu)

    if not lower:
        if A_gpu.ndim == 2:
            L = L.T
        else:
            L = cp.swapaxes(L, -2, -1)

    return L


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_cholesky_safe(
    A: ArrayLike,
    lower: bool = True,
    regularization: float = 1e-10,
) -> Tuple[NDArray, bool]:
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

    Examples
    --------
    >>> import numpy as np
    >>> from pytcl.gpu.matrix_utils import gpu_cholesky_safe
    >>> A = np.array([[1, 2], [2, 1]])  # Not positive definite
    >>> L, success = gpu_cholesky_safe(A)
    >>> success
    False
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)

    try:
        L = cp.linalg.cholesky(A_gpu)
        success = True
    except cp.linalg.LinAlgError:
        # Add regularization
        if A_gpu.ndim == 2:
            A_reg = A_gpu + regularization * cp.eye(A_gpu.shape[0], dtype=cp.float64)
        else:
            # Batch case
            n = A_gpu.shape[-1]
            eye = cp.eye(n, dtype=cp.float64)
            A_reg = A_gpu + regularization * eye

        L = cp.linalg.cholesky(A_reg)
        success = False
        _logger.warning("Cholesky decomposition required regularization")

    if not lower:
        if A_gpu.ndim == 2:
            L = L.T
        else:
            L = cp.swapaxes(L, -2, -1)

    return L, success


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_qr(A: ArrayLike, mode: str = "reduced") -> Tuple[NDArray, NDArray]:
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
    >>> A = np.random.randn(4, 3)
    >>> Q, R = gpu_qr(A)
    >>> np.allclose(Q @ R, A)
    True
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)
    Q, R = cp.linalg.qr(A_gpu, mode=mode)

    return Q, R


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_solve(A: ArrayLike, b: ArrayLike) -> NDArray:
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
    >>> A = np.array([[3, 1], [1, 2]])
    >>> b = np.array([9, 8])
    >>> x = gpu_solve(A, b)
    >>> np.allclose(A @ x, b)
    True
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)
    b_gpu = ensure_gpu_array(b, dtype=cp.float64)

    x = cp.linalg.solve(A_gpu, b_gpu)

    return x


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_inv(A: ArrayLike) -> NDArray:
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
    >>> A = np.array([[1, 2], [3, 4]])
    >>> A_inv = gpu_inv(A)
    >>> np.allclose(A @ A_inv, np.eye(2))
    True
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)
    A_inv = cp.linalg.inv(A_gpu)

    return A_inv


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_eigh(A: ArrayLike) -> Tuple[NDArray, NDArray]:
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
    >>> A = np.array([[2, 1], [1, 2]])
    >>> eigvals, eigvecs = gpu_eigh(A)
    >>> eigvals
    array([1., 3.])
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)
    eigvals, eigvecs = cp.linalg.eigh(A_gpu)

    return eigvals, eigvecs


@requires("cupy", extra="gpu", feature="GPU matrix utilities")
def gpu_matrix_sqrt(A: ArrayLike) -> NDArray:
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
    >>> A = np.array([[4, 0], [0, 9]])
    >>> S = gpu_matrix_sqrt(A)
    >>> np.allclose(S @ S, A)
    True
    """
    cp = import_optional("cupy", extra="gpu", feature="GPU matrix utilities")

    A_gpu = ensure_gpu_array(A, dtype=cp.float64)

    # Eigendecomposition
    eigvals, eigvecs = cp.linalg.eigh(A_gpu)

    # Ensure non-negative eigenvalues
    eigvals = cp.maximum(eigvals, 0)

    # Compute sqrt
    sqrt_eigvals = cp.sqrt(eigvals)

    # Reconstruct: S = V @ diag(sqrt(lambda)) @ V'
    if A_gpu.ndim == 2:
        S = eigvecs @ cp.diag(sqrt_eigvals) @ eigvecs.T
    else:
        # Batch case
        S = cp.einsum("...ij,...j,...kj->...ik", eigvecs, sqrt_eigvals, eigvecs)

    return S


class MemoryPool:
    """
    GPU memory pool manager for efficient memory allocation.

    This class provides convenient access to CuPy's memory pool
    with additional monitoring and management utilities.

    Examples
    --------
    >>> from pytcl.gpu.matrix_utils import MemoryPool
    >>> pool = MemoryPool()
    >>> print(pool.get_stats())
    {'used': 0, 'total': 0, 'free': ...}
    >>>
    >>> # Allocate some arrays
    >>> import cupy as cp
    >>> x = cp.zeros((1000, 1000))
    >>> print(pool.get_stats())
    {'used': 8000000, ...}
    >>>
    >>> # Free cached memory
    >>> pool.free_all()
    """

    def __init__(self):
        """Initialize memory pool manager."""
        if not is_available("cupy"):
            _logger.warning("CuPy not available, MemoryPool is a no-op")
            self._pool = None
            self._pinned_pool = None
        else:
            import cupy as cp

            self._pool = cp.get_default_memory_pool()
            self._pinned_pool = cp.get_default_pinned_memory_pool()

    def get_stats(self) -> dict[str, int]:
        """
        Get memory pool statistics.

        Returns
        -------
        stats : dict
            Dictionary with 'used', 'total', and 'free' bytes.
        """
        if self._pool is None:
            return {"used": 0, "total": 0, "free": 0}

        import cupy as cp

        free, total = cp.cuda.Device().mem_info

        return {
            "used": self._pool.used_bytes(),
            "total": self._pool.total_bytes(),
            "free": free,
            "device_total": total,
        }

    def free_all(self) -> None:
        """Free all cached memory blocks."""
        if self._pool is not None:
            self._pool.free_all_blocks()
        if self._pinned_pool is not None:
            self._pinned_pool.free_all_blocks()

    def set_limit(self, limit: Optional[int] = None) -> None:
        """
        Set memory pool limit.

        Parameters
        ----------
        limit : int or None
            Maximum bytes to allocate. None for no limit.
        """
        if self._pool is not None:
            if limit is None:
                self._pool.set_limit(size=0)  # 0 means no limit
            else:
                self._pool.set_limit(size=limit)

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
        >>> with pool.limit_memory(1e9):  # 1GB limit
        ...     # Operations here have limited memory
        ...     pass
        """
        if self._pool is None:
            yield
            return

        old_limit = self._pool.get_limit()
        self._pool.set_limit(size=max_bytes)
        try:
            yield
        finally:
            self._pool.set_limit(size=old_limit)


# Global memory pool instance
_memory_pool: Optional[MemoryPool] = None


def get_memory_pool() -> MemoryPool:
    """
    Get the global GPU memory pool manager.

    Returns
    -------
    pool : MemoryPool
        Global memory pool instance.
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
