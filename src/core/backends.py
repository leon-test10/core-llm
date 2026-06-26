"""
Linear algebra backend abstraction for CPU and GPU execution.

Provides a unified interface for sparse matrix assembly and solving,
with automatic backend selection based on available hardware.

Backends:
  - cpu: numpy + scipy.sparse (always available)
  - gpu: cupy + cupyx.scipy.sparse (requires NVIDIA GPU + CuPy)

Usage:
    backend = get_backend("auto")  # auto-detect
    A = backend.csr_matrix(data, indices, indptr, shape)
    x = backend.spsolve(A, b)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# ── Optional GPU imports ────────────────────────────────────────────────

_GPU_AVAILABLE = False
_GPU_INFO = "not available"

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsparse
    from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve
    _GPU_AVAILABLE = True
    _GPU_INFO = f"CuPy {cp.__version__}, {cp.cuda.runtime.getDeviceCount()} device(s)"
except ImportError:
    cp = None
    cpsparse = None
    cp_spsolve = None
    _GPU_INFO = "CuPy not installed"

# ── Backend interface ───────────────────────────────────────────────────

class Backend(ABC):
    """Abstract linear algebra backend."""

    name: str = "abstract"

    @abstractmethod
    def array(self, data, dtype=None) -> Any:
        """Create a dense array."""
        ...

    @abstractmethod
    def zeros(self, shape, dtype=None) -> Any:
        """Create a zero array."""
        ...

    @abstractmethod
    def ones(self, shape, dtype=None) -> Any:
        """Create an array of ones."""
        ...

    @abstractmethod
    def asnumpy(self, arr) -> np.ndarray:
        """Convert array to numpy (no-op for CPU)."""
        ...

    @abstractmethod
    def lil_matrix(self, shape) -> Any:
        """Create a LIL sparse matrix."""
        ...

    @abstractmethod
    def csr_matrix(self, lil_mat) -> Any:
        """Convert LIL to CSR."""
        ...

    @abstractmethod
    def spsolve(self, A, b) -> Any:
        """Solve sparse linear system A x = b."""
        ...

    @abstractmethod
    def dot(self, A, x) -> Any:
        """Sparse matrix-vector product."""
        ...

    @abstractmethod
    def norm(self, x, ord=2) -> float:
        """Vector norm."""
        ...

    @abstractmethod
    def sum(self, x) -> float:
        """Sum of array elements."""
        ...


# ── CPU Backend ─────────────────────────────────────────────────────────

class CPUBackend(Backend):
    """NumPy + SciPy sparse backend."""

    name = "cpu"

    def array(self, data, dtype=None):
        return np.array(data, dtype=dtype or float)

    def zeros(self, shape, dtype=None):
        return np.zeros(shape, dtype=dtype or float)

    def ones(self, shape, dtype=None):
        return np.ones(shape, dtype=dtype or float)

    def asnumpy(self, arr) -> np.ndarray:
        return np.asarray(arr)

    def lil_matrix(self, shape):
        from scipy import sparse
        return sparse.lil_matrix(shape)

    def csr_matrix(self, lil_mat):
        return lil_mat.tocsr()

    def spsolve(self, A, b):
        from scipy.sparse.linalg import spsolve as _spsolve
        return _spsolve(A, b)

    def dot(self, A, x):
        return A.dot(x)

    def norm(self, x, ord=2):
        return float(np.linalg.norm(np.asarray(x), ord=ord))

    def sum(self, x):
        return float(np.asarray(x).sum())


# ── GPU Backend ─────────────────────────────────────────────────────────

class GPUBackend(Backend):
    """CuPy GPU backend.

    Requires: pip install cupy-cuda12x (or appropriate CUDA version)

    Falls back to CPU if GPU not available. All operations run on GPU;
    results are transferred back to CPU via asnumpy().
    """

    name = "gpu"

    def __init__(self):
        if not _GPU_AVAILABLE:
            raise RuntimeError(
                "GPU backend requested but CuPy not available. "
                "Install with: pip install cupy-cuda12x"
            )

    def array(self, data, dtype=None):
        return cp.array(data, dtype=dtype or float)

    def zeros(self, shape, dtype=None):
        return cp.zeros(shape, dtype=dtype or float)

    def ones(self, shape, dtype=None):
        return cp.ones(shape, dtype=dtype or float)

    def asnumpy(self, arr) -> np.ndarray:
        return cp.asnumpy(arr)

    def lil_matrix(self, shape):
        return cpsparse.lil_matrix(shape)

    def csr_matrix(self, lil_mat):
        return lil_mat.tocsr()

    def spsolve(self, A, b):
        return cp_spsolve(A, b)

    def dot(self, A, x):
        return A.dot(x)

    def norm(self, x, ord=2):
        return float(cp.linalg.norm(cp.asarray(x), ord=ord).get())

    def sum(self, x):
        return float(cp.asarray(x).sum().get())


# ── Backend factory ─────────────────────────────────────────────────────

def get_backend(backend: str = "auto") -> Backend:
    """Get the appropriate linear algebra backend.

    Args:
        backend: "cpu", "gpu", or "auto" (selects GPU if available)

    Returns:
        Backend instance
    """
    if backend == "cpu":
        return CPUBackend()
    elif backend == "gpu":
        return GPUBackend()
    elif backend == "auto":
        if _GPU_AVAILABLE:
            return GPUBackend()
        return CPUBackend()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'cpu', 'gpu', or 'auto'.")


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def gpu_info() -> str:
    """Human-readable GPU availability info."""
    return _GPU_INFO
