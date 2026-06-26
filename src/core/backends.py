"""
Linear algebra backend abstraction for CPU and GPU execution.

Provides a unified interface for sparse matrix assembly and solving,
with automatic backend selection based on available hardware.

Backends:
  - cpu: numpy + scipy.sparse (always available)
  - cuda: CuPy + NVIDIA CUDA (requires NVIDIA GPU + pip install cupy-cuda12x)
  - rocm: CuPy + AMD ROCm (requires AMD GPU + pip install cupy-rocm-*)
  - gpu: auto-select CUDA or ROCm if available
  - auto: GPU if available, else CPU

Usage:
    backend = get_backend("auto")
    A = backend.csr_matrix(data, indices, indptr, shape)
    x = backend.spsolve(A, b)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np

# ── GPU Detection ──────────────────────────────────────────────────────

_GPU_VENDOR = None       # "nvidia", "amd", or None
_GPU_INFO = "not available"
_GPU_AVAILABLE = False

def _detect_gpu():
    """Detect available GPU hardware and compatible libraries."""
    global _GPU_VENDOR, _GPU_INFO, _GPU_AVAILABLE

    # Try NVIDIA CuPy first
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        _GPU_VENDOR = "nvidia"
        _GPU_INFO = f"NVIDIA CUDA via CuPy {cp.__version__}, {cp.cuda.runtime.getDeviceCount()} device(s)"
        _GPU_AVAILABLE = True
        return
    except (ImportError, Exception):
        pass

    # Try AMD ROCm CuPy
    try:
        import cupy as cp
        # ROCm CuPy reports devices through hip
        cp.cuda.runtime.getDeviceCount()
        _GPU_VENDOR = "amd"
        _GPU_INFO = f"AMD ROCm via CuPy {cp.__version__}, {cp.cuda.runtime.getDeviceCount()} device(s)"
        _GPU_AVAILABLE = True
        return
    except (ImportError, Exception):
        pass

    # Try PyTorch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            _GPU_VENDOR = "nvidia"
            _GPU_INFO = f"NVIDIA CUDA via PyTorch {torch.__version__}, {torch.cuda.device_count()} device(s) [no sparse solver]"
            _GPU_AVAILABLE = False  # PyTorch sparse solve not as efficient
            return
    except ImportError:
        pass

    # Try PyTorch ROCm (AMD)
    try:
        import torch
        # ROCm PyTorch also reports cuda.is_available() as True
        if torch.cuda.is_available():
            # Check if it's ROCm by looking at the device name
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ""
            if "AMD" in dev_name.upper() or "RADEON" in dev_name.upper() or "MI" in dev_name.upper():
                _GPU_VENDOR = "amd"
                _GPU_INFO = f"AMD ROCm via PyTorch {torch.__version__}, device: {dev_name} [no sparse solver]"
                _GPU_AVAILABLE = False
                return
    except ImportError:
        pass

    _GPU_INFO = "not available"
    _GPU_AVAILABLE = False

_detect_gpu()

# Re-import cupy if available (after detection)
cp = None
cpsparse = None
cp_spsolve = None
if _GPU_AVAILABLE:
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpsparse
        from cupyx.scipy.sparse.linalg import spsolve as cp_spsolve
    except ImportError:
        _GPU_AVAILABLE = False
        _GPU_INFO = "GPU library import failed"


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
    """GPU backend via CuPy (supports both NVIDIA CUDA and AMD ROCm).

    Installation:
      NVIDIA: pip install cupy-cuda12x
      AMD:    pip install cupy-rocm-*-12x  (match your ROCm version)

    Falls back to CPU if GPU not available.
    All operations run on GPU; results transferred via asnumpy().
    """

    name = "gpu"
    vendor: str = "unknown"

    def __init__(self):
        if not _GPU_AVAILABLE:
            raise RuntimeError(
                "GPU backend requested but no GPU library available.\n"
                "NVIDIA GPU: pip install cupy-cuda12x\n"
                "AMD GPU:    pip install cupy-rocm-*-12x (e.g. cupy-rocm-6-0)\n"
                "Current: " + _GPU_INFO
            )
        self.vendor = _GPU_VENDOR or "unknown"

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
        backend:
          - "cpu": always CPU
          - "cuda": NVIDIA GPU only (error if not available)
          - "rocm": AMD GPU only (error if not available)
          - "gpu": any GPU (CUDA or ROCm, error if neither)
          - "auto": GPU if available, else CPU

    Returns:
        Backend instance
    """
    if backend == "cpu":
        return CPUBackend()
    elif backend == "cuda":
        if _GPU_VENDOR != "nvidia":
            raise RuntimeError(
                "NVIDIA CUDA backend requested but not available.\n"
                "Install: pip install cupy-cuda12x\n"
                "Current: " + _GPU_INFO
            )
        return GPUBackend()
    elif backend == "rocm":
        if _GPU_VENDOR != "amd":
            raise RuntimeError(
                "AMD ROCm backend requested but not available.\n"
                "Install: pip install cupy-rocm-*-12x\n"
                "Current: " + _GPU_INFO
            )
        return GPUBackend()
    elif backend in ("gpu", "auto"):
        if _GPU_AVAILABLE:
            return GPUBackend()
        if backend == "gpu":
            raise RuntimeError(
                "GPU backend requested but no GPU library found.\n"
                "NVIDIA: pip install cupy-cuda12x\n"
                "AMD:    pip install cupy-rocm-*-12x\n"
                "Current: " + _GPU_INFO
            )
        return CPUBackend()
    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Use 'cpu', 'cuda', 'rocm', 'gpu', or 'auto'."
        )


def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return _GPU_AVAILABLE


def gpu_vendor() -> str | None:
    """Return GPU vendor: 'nvidia', 'amd', or None."""
    return _GPU_VENDOR


def gpu_info() -> str:
    """Human-readable GPU availability info."""
    return _GPU_INFO
