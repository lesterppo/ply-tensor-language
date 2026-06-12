"""
Ply Tensor — Tensor class with reverse-mode autodiff + GPU backend.
Uses backend.py for pluggable array backends (NumPy/CuPy/MLX).
Every tensor is a node in a static computation graph.
"""
import numpy as np
from typing import List, Optional, Callable, Any, Tuple
from backend import (_asarray, _randn, _zeros, _ones, _zeros_like, _ones_like,
                      _maximum, _where, _broadcast_to, _concatenate, _stack,
                      _transpose, _reshape, _einsum, _expand_dims,
                      _put_along_axis, _argmax, _argmin, _argsort, _swapaxes)


class Tensor:
    __slots__ = ('data', 'grad', '_backward', '_inputs', '_op', '_ctx', '_shape')

    def __init__(self, data, _inputs=(), _op='', _backward=None, _ctx=None):
        self.data = _asarray(data, dtype=np.float32)
        self.grad: Optional[Any] = None
        self._backward: Optional[Callable[[], None]] = _backward
        self._inputs: List[Tensor] = list(_inputs)
        self._op: str = _op
        self._ctx: Any = _ctx
        self._shape: Tuple[int, ...] = self.data.shape

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.permute(*reversed(range(self.ndim)))

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op})"

    # ── Gradient flow ─────────────────────────────────────

    def zero_grad(self):
        self.grad = None
        for inp in self._inputs:
            inp.zero_grad()

    @staticmethod
    def _acc_grad(tensor, value):
        g = tensor.ensure_grad()
        if g.ndim == 0:
            tensor.grad = g + value
        else:
            g += value

    @staticmethod
    def _broadcast_grad(grad, target_shape: Tuple[int, ...]):
        g = grad
        while g.ndim > len(target_shape):
            g = g.sum(axis=0)
        for axis, (gs, ts) in enumerate(zip(g.shape, target_shape)):
            if ts == 1 and gs > 1:
                g = g.sum(axis=axis, keepdims=True)
        return g

    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            tid = id(t)
            if tid not in visited:
                visited.add(tid)
                for inp in t._inputs:
                    build_topo(inp)
                topo.append(t)

        build_topo(self)
        self.grad = _ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None and t.grad is not None:
                t._backward()

    def ensure_grad(self):
        if self.grad is None:
            self.grad = _zeros_like(self.data)
        return self.grad

    # ── Operator overloads ─────────────────────────────────

    def __add__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        def _bw(): Tensor._acc_grad(self, self._broadcast_grad(out.grad, self.shape)); Tensor._acc_grad(other, self._broadcast_grad(out.grad, other.shape))
        out._backward = _bw; return out

    def __radd__(self, other): return _ensure_tensor(other).__add__(self)

    def __sub__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(self.data - other.data, (self, other), '-')
        def _bw(): Tensor._acc_grad(self, self._broadcast_grad(out.grad, self.shape)); Tensor._acc_grad(other, -self._broadcast_grad(out.grad, other.shape))
        out._backward = _bw; return out

    def __rsub__(self, other): return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        def _bw(): Tensor._acc_grad(self, self._broadcast_grad(out.grad * other.data, self.shape)); Tensor._acc_grad(other, self._broadcast_grad(out.grad * self.data, other.shape))
        out._backward = _bw; return out

    def __rmul__(self, other): return _ensure_tensor(other).__mul__(self)

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(self.data / other.data, (self, other), '/')
        def _bw():
            Tensor._acc_grad(self, self._broadcast_grad(out.grad / other.data, self.shape))
            Tensor._acc_grad(other, -self._broadcast_grad(out.grad * self.data / (other.data * other.data), other.shape))
        out._backward = _bw; return out

    def __rtruediv__(self, other): return _ensure_tensor(other).__truediv__(self)

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')
        def _bw():
            g = out.grad
            if self.ndim == 2 and other.ndim == 2:
                Tensor._acc_grad(self, g @ other.data.T)
                Tensor._acc_grad(other, self.data.T @ g)
            else:
                Tensor._acc_grad(self, g @ _swapaxes(other.data, -1, -2))
                Tensor._acc_grad(other, _swapaxes(self.data, -1, -2) @ g)
        out._backward = _bw; return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')
        def _bw(): Tensor._acc_grad(self, -out.grad)
        out._backward = _bw; return out

    # Comparison ops (no gradient)
    def __gt__(self, other):
        return Tensor((self.data > _ensure_tensor(other).data).astype(np.float32), (self, other), '>', _backward=lambda: None)
    def __lt__(self, other):
        return Tensor((self.data < _ensure_tensor(other).data).astype(np.float32), (self, other), '<', _backward=lambda: None)
    def __ge__(self, other):
        return Tensor((self.data >= _ensure_tensor(other).data).astype(np.float32), (self, other), '>=', _backward=lambda: None)
    def __le__(self, other):
        return Tensor((self.data <= _ensure_tensor(other).data).astype(np.float32), (self, other), '<=', _backward=lambda: None)
    def __eq__(self, other):
        return Tensor((self.data == _ensure_tensor(other).data).astype(np.float32), (self, other), '==', _backward=lambda: None)
    def __ne__(self, other):
        return Tensor((self.data != _ensure_tensor(other).data).astype(np.float32), (self, other), '!=', _backward=lambda: None)
    def __and__(self, other):
        other = _ensure_tensor(other)
        return Tensor((self.data.astype(bool) & other.data.astype(bool)).astype(np.float32), (self, other), '&', _backward=lambda: None)
    def __or__(self, other):
        other = _ensure_tensor(other)
        return Tensor((self.data.astype(bool) | other.data.astype(bool)).astype(np.float32), (self, other), '|', _backward=lambda: None)
    def __invert__(self):
        return Tensor((~self.data.astype(bool)).astype(np.float32), (self,), '~', _backward=lambda: None)

    # ── Shape manipulation ────────────────────────────────

    def reshape(self, *shape):
        out = Tensor(_reshape(self.data, shape), (self,), 'reshape', _ctx=self.shape)
        def _bw(): Tensor._acc_grad(self, _reshape(out.grad, out._ctx))
        out._backward = _bw; return out

    def permute(self, *axes):
        out = Tensor(_transpose(self.data, axes), (self,), 'permute', _ctx=axes)
        def _bw(): Tensor._acc_grad(self, _transpose(out.grad, _argsort(axes)))
        out._backward = _bw; return out

    def sum(self, dim=None):
        if dim is not None:
            out_data = self.data.sum(axis=dim)
        else:
            out_data = self.data.sum()
        out = Tensor(out_data, (self,), 'sum', _ctx=(dim, self.shape))
        def _bw():
            grad = out.grad
            if out._ctx[0] is not None:
                grad = _expand_dims(grad, out._ctx[0])
            grad = _broadcast_to(grad, out._ctx[1])
            Tensor._acc_grad(self, grad)
        out._backward = _bw; return out

    def mean(self, dim=None):
        if dim is not None:
            n = self.shape[dim]
            return (self.sum(dim=dim)).__truediv__(_ensure_tensor(float(n)))
        else:
            return (self.sum()).__truediv__(_ensure_tensor(float(self.size)))

    def max(self, dim=None):
        if dim is not None:
            out_data = self.data.max(axis=dim)
            out = Tensor(out_data, (self,), 'max', _ctx=(dim, self.data))
            def _bw():
                am = _argmax(out._ctx[1], axis=out._ctx[0])
                mask = _zeros_like(out._ctx[1])
                _put_along_axis(mask, _expand_dims(am, out._ctx[0]), 1.0, out._ctx[0])
                Tensor._acc_grad(self, mask * _broadcast_to(_expand_dims(out.grad, out._ctx[0]), out._ctx[1].shape))
            out._backward = _bw; return out
        else:
            return Tensor(self.data.max(), (self,), 'max_scalar', _backward=lambda: None)

    def min(self, dim=None):
        if dim is not None:
            out_data = self.data.min(axis=dim)
            out = Tensor(out_data, (self,), 'min', _ctx=(dim, self.data))
            def _bw():
                am = _argmin(out._ctx[1], axis=out._ctx[0])
                mask = _zeros_like(out._ctx[1])
                _put_along_axis(mask, _expand_dims(am, out._ctx[0]), 1.0, out._ctx[0])
                Tensor._acc_grad(self, mask * _broadcast_to(_expand_dims(out.grad, out._ctx[0]), out._ctx[1].shape))
            out._backward = _bw; return out
        else:
            return Tensor(self.data.min(), (self,), 'min_scalar', _backward=lambda: None)

    def std(self, dim=None):
        if dim is not None:
            out_data = self.data.std(axis=dim)
        else:
            out_data = self.data.std()
        out = Tensor(out_data, (self,), 'std', _ctx=(dim, self.data))
        def _bw():
            x = out._ctx[1]; dim = out._ctx[0]; s = out.data
            if dim is not None:
                mu = x.mean(axis=dim, keepdims=True); N = x.shape[dim]
                g = _expand_dims(out.grad, dim)
            else:
                mu = x.mean(); N = x.size; g = out.grad
            Tensor._acc_grad(self, g * (x - mu) / (_maximum(s, 1e-8) * N))
        out._backward = _bw; return out

    def __getitem__(self, key):
        out = Tensor(self.data[key], (self,), 'slice', _ctx=(key, self.shape))
        def _bw():
            g = _zeros(out._ctx[1])
            g[key] = out.grad
            Tensor._acc_grad(self, g)
        out._backward = _bw; return out


def _ensure_tensor(x) -> Tensor:
    if isinstance(x, Tensor):
        return x
    return Tensor(x)


# ── Activation function gradients ────────────────────────

def tensor_relu(t: Tensor) -> Tensor:
    out = Tensor(_maximum(t.data, 0), (t,), 'relu')
    def _bw(): Tensor._acc_grad(t, out.grad * (t.data > 0).astype(np.float32))
    out._backward = _bw; return out


def tensor_gelu(t: Tensor) -> Tensor:
    x = t.data
    sqrt_2_pi = np.float32(0.7978845608)  # sqrt(2/pi)
    inner = sqrt_2_pi * (x + np.float32(0.044715) * x ** 3)
    tanh_inner = np.tanh(inner)
    out = Tensor(np.float32(0.5) * x * (1 + tanh_inner), (t,), 'gelu', _ctx=(x, tanh_inner))
    def _bw():
        x, th = out._ctx
        d_inner = sqrt_2_pi * (1 + 3 * np.float32(0.044715) * x ** 2)
        dgelu = np.float32(0.5) * (1 + th) + np.float32(0.5) * x * (1 - th ** 2) * d_inner
        Tensor._acc_grad(t, out.grad * dgelu)
    out._backward = _bw; return out


def tensor_sigmoid(t: Tensor) -> Tensor:
    s = 1 / (1 + np.exp(-t.data))
    out = Tensor(s, (t,), 'sigmoid')
    def _bw(): Tensor._acc_grad(t, out.grad * s * (1 - s))
    out._backward = _bw; return out


def tensor_tanh(t: Tensor) -> Tensor:
    th = np.tanh(t.data)
    out = Tensor(th, (t,), 'tanh')
    def _bw(): Tensor._acc_grad(t, out.grad * (1 - th ** 2))
    out._backward = _bw; return out


def tensor_softmax(t: Tensor, dim: int = -1) -> Tensor:
    x = t.data
    x_max = x.max(axis=dim, keepdims=True)
    e = np.exp(x - x_max)
    s = e / e.sum(axis=dim, keepdims=True)
    out = Tensor(s, (t,), 'softmax', _ctx=(dim, s))
    def _bw():
        dim, s = out._ctx
        sg = out.grad * s
        Tensor._acc_grad(t, sg - s * sg.sum(axis=dim, keepdims=True))
    out._backward = _bw; return out


def tensor_layernorm(t: Tensor) -> Tensor:
    x = t.data
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = np.sqrt(var + 1e-5)
    normalized = (x - mean) / std
    out = Tensor(normalized, (t,), 'layernorm', _ctx=(std, normalized))
    def _bw():
        std, norm = out._ctx; N = x.shape[-1]; g = out.grad
        g_sum = g.sum(axis=-1, keepdims=True)
        g_norm_sum = (g * norm).sum(axis=-1, keepdims=True)
        Tensor._acc_grad(t, (g - g_sum / N - norm * g_norm_sum / N) / std)
    out._backward = _bw; return out


def tensor_exp(t: Tensor) -> Tensor:
    e = np.exp(t.data)
    out = Tensor(e, (t,), 'exp')
    def _bw(): Tensor._acc_grad(t, out.grad * e)
    out._backward = _bw; return out


def tensor_log(t: Tensor) -> Tensor:
    safe = _maximum(t.data, 1e-10)
    out = Tensor(np.log(safe), (t,), 'log')
    def _bw(): Tensor._acc_grad(t, out.grad / safe)
    out._backward = _bw; return out


def tensor_sqrt(t: Tensor) -> Tensor:
    safe = _maximum(t.data, 0)
    s = np.sqrt(safe)
    out = Tensor(s, (t,), 'sqrt')
    def _bw(): Tensor._acc_grad(t, out.grad / (2 * _maximum(s, 1e-10)))
    out._backward = _bw; return out


def tensor_abs(t: Tensor) -> Tensor:
    out = Tensor(np.abs(t.data), (t,), 'abs')
    def _bw(): Tensor._acc_grad(t, out.grad * np.sign(t.data))
    out._backward = _bw; return out


def tensor_sin(t: Tensor) -> Tensor:
    out = Tensor(np.sin(t.data), (t,), 'sin')
    def _bw(): Tensor._acc_grad(t, out.grad * np.cos(t.data))
    out._backward = _bw; return out


def tensor_cos(t: Tensor) -> Tensor:
    out = Tensor(np.cos(t.data), (t,), 'cos')
    def _bw(): Tensor._acc_grad(t, -out.grad * np.sin(t.data))
    out._backward = _bw; return out


def tensor_clip(t: Tensor, lo: float, hi: float) -> Tensor:
    out = Tensor(np.clip(t.data, lo, hi), (t,), 'clip', _ctx=(t.data, lo, hi))
    def _bw():
        x, lo, hi = out._ctx
        mask = ((x >= lo) & (x <= hi)).astype(np.float32)
        Tensor._acc_grad(t, out.grad * mask)
    out._backward = _bw; return out


def tensor_where(cond: Tensor, a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(_where(cond.data, a.data, b.data), (cond, a, b), 'where')
    def _bw():
        c = cond.data.astype(bool)
        Tensor._acc_grad(a, Tensor._broadcast_grad(out.grad * c.astype(np.float32), a.shape))
        Tensor._acc_grad(b, Tensor._broadcast_grad(out.grad * (~c).astype(np.float32), b.shape))
    out._backward = _bw; return out


def tensor_concat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    data = _concatenate([t.data for t in tensors], axis=dim)
    out = Tensor(data, tuple(tensors), 'concat', _ctx=(dim, [t.shape[dim] for t in tensors]))
    def _bw():
        dim, sizes = out._ctx; offset = 0
        for i, t in enumerate(tensors):
            sl = [slice(None)] * out.ndim; sl[dim] = slice(offset, offset + sizes[i])
            Tensor._acc_grad(t, out.grad[tuple(sl)]); offset += sizes[i]
    out._backward = _bw; return out


def tensor_stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    data = _stack([t.data for t in tensors], axis=dim)
    out = Tensor(data, tuple(tensors), 'stack', _ctx=dim)
    def _bw():
        for i, t in enumerate(tensors):
            sl = [slice(None)] * out.ndim; sl[out._ctx] = i
            Tensor._acc_grad(t, out.grad[tuple(sl)])
    out._backward = _bw; return out


def tensor_einsum(spec: str, *tensors: Tensor) -> Tensor:
    arrs = [t.data for t in tensors]
    out = Tensor(_einsum(spec, *arrs), tensors, 'einsum', _ctx=(spec, arrs, [t.shape for t in tensors]))
    def _bw():
        spec, arrs, shapes = out._ctx; g = out.grad
        if '->' in spec:
            inputs_spec, output_spec = spec.split('->')
        else:
            inputs_spec = spec; output_spec = ''
        input_specs = inputs_spec.split(',')
        for i, t in enumerate(tensors):
            other_specs = input_specs.copy()
            other_specs[i] = output_spec
            other_arrs = arrs.copy()
            other_arrs[i] = g
            grad_spec = ','.join(other_specs) + '->' + input_specs[i]
            try:
                Tensor._acc_grad(t, _einsum(grad_spec, *other_arrs))
            except (ValueError, Exception):
                pass
    out._backward = _bw; return out
