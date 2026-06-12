"""
Ply Backend — Pluggable array backend (NumPy / CuPy / Metal via mlx).
Auto-detects GPU and selects the fastest available backend.

Usage:
    from backend import xp  # numpy or cupy
    xp.array([1, 2, 3])     # works on both CPU and GPU
"""
import os
from typing import Any


class Backend:
    """Abstract array backend. All tensor ops go through this."""

    def __init__(self, module, name: str, device: str):
        self._mod = module
        self.name = name
        self.device = device

    def __getattr__(self, name: str) -> Any:
        return getattr(self._mod, name)

    @property
    def is_gpu(self) -> bool:
        return self.device != 'cpu'

    def array(self, data, dtype=None):
        arr = self._mod.array(data, dtype=dtype or self.float32)
        return arr

    @property
    def float32(self):
        return self._mod.float32

    @property
    def float64(self):
        return self._mod.float64


def _detect_backend() -> Backend:
    """Auto-detect the best available backend."""

    # Check for CuPy (NVIDIA GPU)
    if not os.environ.get('PLY_FORCE_CPU'):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.cuda.Device(0).use()
                return Backend(cp, 'cupy', f'cuda:0')
        except (ImportError, Exception):
            pass

    # Check for MLX (Apple Silicon)
    if not os.environ.get('PLY_FORCE_CPU'):
        try:
            import mlx.core as mx
            return Backend(mx, 'mlx', 'mps')
        except ImportError:
            pass

    # Fall back to NumPy
    import numpy as np
    return Backend(np, 'numpy', 'cpu')


# Global backend instance
_xp: Backend = None


def get_backend() -> Backend:
    global _xp
    if _xp is None:
        _xp = _detect_backend()
    return _xp


def set_backend(name: str):
    """Force a specific backend: 'numpy', 'cupy', or 'mlx'."""
    global _xp
    if name == 'cupy':
        import cupy as cp
        cp.cuda.Device(0).use()
        _xp = Backend(cp, 'cupy', 'cuda:0')
    elif name == 'mlx':
        import mlx.core as mx
        _xp = Backend(mx, 'mlx', 'mps')
    else:
        import numpy as np
        _xp = Backend(np, 'numpy', 'cpu')


# Convenience: direct access for inline use
def xp():
    return get_backend()._mod


# Cache friendly aliases for common ops
def _max(tensor, axis=None, keepdims=False):
    return get_backend()._mod.max(tensor, axis=axis, keepdims=keepdims)


def _sum(tensor, axis=None, keepdims=False):
    return get_backend()._mod.sum(tensor, axis=axis, keepdims=keepdims)


def _exp(tensor):
    return get_backend()._mod.exp(tensor)


def _log(tensor):
    return get_backend()._mod.log(tensor)


def _sqrt(tensor):
    return get_backend()._mod.sqrt(tensor)


def _abs(tensor):
    return get_backend()._mod.abs(tensor)


def _sin(tensor):
    return get_backend()._mod.sin(tensor)


def _cos(tensor):
    return get_backend()._mod.cos(tensor)


def _tanh(tensor):
    return get_backend()._mod.tanh(tensor)


def _maximum(a, b):
    return get_backend()._mod.maximum(a, b)


def _minimum(a, b):
    return get_backend()._mod.minimum(a, b)


def _where(cond, a, b):
    return get_backend()._mod.where(cond, a, b)


def _clip(tensor, lo, hi):
    return get_backend()._mod.clip(tensor, lo, hi)


def _sign(tensor):
    return get_backend()._mod.sign(tensor)


def _concatenate(tensors, axis=0):
    return get_backend()._mod.concatenate(tensors, axis=axis)


def _stack(tensors, axis=0):
    return get_backend()._mod.stack(tensors, axis=axis)


def _broadcast_to(tensor, shape):
    return get_backend()._mod.broadcast_to(tensor, shape)


def _transpose(tensor, axes=None):
    return get_backend()._mod.transpose(tensor, axes)


def _reshape(tensor, shape):
    return get_backend()._mod.reshape(tensor, shape)


def _einsum(spec, *operands):
    return get_backend()._mod.einsum(spec, *operands)


def _zeros(*shape, dtype=None):
    return get_backend()._mod.zeros(shape, dtype=dtype or get_backend().float32)


def _ones(*shape, dtype=None):
    return get_backend()._mod.ones(shape, dtype=dtype or get_backend().float32)


def _full(shape, value, dtype=None):
    return get_backend()._mod.full(shape, value, dtype=dtype or get_backend().float32)


def _eye(n, dtype=None):
    return get_backend()._mod.eye(n, dtype=dtype or get_backend().float32)


def _arange(*args, dtype=None):
    return get_backend()._mod.arange(*args, dtype=dtype or get_backend().float32)


def _randn(*shape):
    return get_backend()._mod.random.randn(*shape).astype(get_backend().float32)


def _zeros_like(tensor):
    return get_backend()._mod.zeros_like(tensor)


def _ones_like(tensor):
    return get_backend()._mod.ones_like(tensor)


def _asarray(data, dtype=None):
    return get_backend()._mod.asarray(data, dtype=dtype or get_backend().float32)


def _expand_dims(tensor, axis):
    return get_backend()._mod.expand_dims(tensor, axis=axis)


def _put_along_axis(arr, indices, values, axis):
    return get_backend()._mod.put_along_axis(arr, indices, values, axis)


def _argmax(tensor, axis=None):
    return get_backend()._mod.argmax(tensor, axis=axis)


def _argmin(tensor, axis=None):
    return get_backend()._mod.argmin(tensor, axis=axis)


def _argsort(arr):
    return get_backend()._mod.argsort(arr)


def _swapaxes(tensor, a, b):
    return get_backend()._mod.swapaxes(tensor, a, b)
