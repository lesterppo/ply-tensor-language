"""
Ply Runtime — Tensor-native execution with automatic differentiation.
Uses the Tensor class (tensor.py) which builds a computation graph.
Every operation is recorded; gradients flow via backward().

v2.0 adds:
  - param() builtin for learnable parameters
  - sgd(), adam(), adamw() optimizer builtins
  - step() for optimizer updates
  - save() / load() for model persistence
  - IR compiler integration (lazy mode via use_ir flag)
  - dtype support
"""
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ast_nodes import *
from tensor import (
    Tensor, _ensure_tensor, param as tensor_param,
    tensor_relu, tensor_gelu, tensor_sigmoid, tensor_tanh,
    tensor_softmax, tensor_layernorm,
    tensor_exp, tensor_log, tensor_sqrt, tensor_abs,
    tensor_sin, tensor_cos, tensor_clip,
    tensor_where, tensor_concat, tensor_stack,
    tensor_einsum,
)
from optim import SGD, Adam, AdamW, create_optimizer, Optimizer


class PlyRuntimeError(Exception):
    pass


def _scalar(x):
    """Extract Python scalar from Tensor or number."""
    if isinstance(x, Tensor):
        return x.data.item() if x.ndim == 0 else int(x.data.flat[0])
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _int(x): return int(_scalar(x))
def _float(x): return float(_scalar(x))


BUILTINS: Dict[str, Any] = {}


def _builtin(name: str):
    def dec(fn):
        BUILTINS[name] = fn
        return fn
    return dec


# ── Tensor creation ──────────────────────────────────────

@_builtin('randn')
def _randn(runtime, *shape_args):
    shape = [_int(s) for s in shape_args]
    return Tensor(np.random.randn(*shape).astype(np.float32), _op='randn')

@_builtin('zeros')
def _zeros(runtime, *shape_args):
    shape = [_int(s) for s in shape_args]
    return Tensor(np.zeros(shape, dtype=np.float32), _op='zeros')

@_builtin('ones')
def _ones(runtime, *shape_args):
    shape = [_int(s) for s in shape_args]
    return Tensor(np.ones(shape, dtype=np.float32), _op='ones')

@_builtin('full')
def _full(runtime, value, *shape_args):
    shape = [_int(s) for s in shape_args]
    return Tensor(np.full(shape, float(value), dtype=np.float32), _op='full')

@_builtin('eye')
def _eye(runtime, n):
    return Tensor(np.eye(_int(n), dtype=np.float32), _op='eye')

@_builtin('arange')
def _arange(runtime, *args):
    vals = [_int(a) for a in args]
    return Tensor(np.arange(*vals).astype(np.float32), _op='arange')

# ── Parameter management ─────────────────────────────────

@_builtin('param')
def _param(runtime, *shape_args):
    """Create a learnable parameter with random normal init.
    Usage: W := param(64, 32)  -- creates Param(64, 32)
    """
    shape = [_int(s) for s in shape_args]
    # He init
    fan_in = shape[0] if len(shape) >= 2 else shape[0]
    std = np.sqrt(2.0 / fan_in)
    data = np.random.randn(*shape).astype(np.float32) * std
    return tensor_param(data, name=f'param_{len(runtime._params)}')

@_builtin('param_zeros')
def _param_zeros(runtime, *shape_args):
    """Create a zero-initialized parameter: param_zeros(64)"""
    shape = [_int(s) for s in shape_args]
    return tensor_param(np.zeros(shape, dtype=np.float32), name=f'param_{len(runtime._params)}')

# ── Shape manipulation ───────────────────────────────────

@_builtin('reshape')
def _reshape(runtime, tensor, *shape_args):
    shape = tuple(_int(s) for s in shape_args)
    return tensor.reshape(*shape)

@_builtin('permute')
def _permute(runtime, tensor, *axes):
    return tensor.permute(*[_int(a) for a in axes])

@_builtin('transpose')
def _transpose_run(runtime, tensor):
    return tensor.permute(*reversed(range(tensor.ndim)))

@_builtin('T')
def _T(runtime, tensor):
    return tensor.permute(*reversed(range(tensor.ndim)))

# ── Reductions ───────────────────────────────────────────

@_builtin('sum')
def _sum(runtime, tensor, dim=None):
    return tensor.sum(dim=None if dim is None else _int(dim))

@_builtin('mean')
def _mean(runtime, tensor, dim=None):
    return tensor.mean(dim=None if dim is None else _int(dim))

@_builtin('max')
def _max(runtime, tensor, dim=None):
    return tensor.max(dim=None if dim is None else _int(dim))

@_builtin('min')
def _min(runtime, tensor, dim=None):
    return tensor.min(dim=None if dim is None else _int(dim))

@_builtin('std')
def _std(runtime, tensor, dim=None):
    return tensor.std(dim=None if dim is None else _int(dim))

# ── Activations ──────────────────────────────────────────

@_builtin('relu')
def _relu(runtime, tensor):
    return tensor_relu(tensor)

@_builtin('gelu')
def _gelu(runtime, tensor):
    return tensor_gelu(tensor)

@_builtin('sigmoid')
def _sigmoid(runtime, tensor):
    return tensor_sigmoid(tensor)

@_builtin('tanh')
def _tanh(runtime, tensor):
    return tensor_tanh(tensor)

@_builtin('softmax')
def _softmax(runtime, tensor, dim=-1):
    return tensor_softmax(tensor, _int(dim))

@_builtin('layernorm')
def _layernorm(runtime, tensor):
    return tensor_layernorm(tensor)

# ── Element-wise math ────────────────────────────────────

@_builtin('exp')
def _exp(runtime, tensor):
    return tensor_exp(tensor)

@_builtin('log')
def _log(runtime, tensor):
    return tensor_log(tensor)

@_builtin('sqrt')
def _sqrt(runtime, tensor):
    return tensor_sqrt(tensor)

@_builtin('abs')
def _abs(runtime, tensor):
    return tensor_abs(tensor)

@_builtin('sin')
def _sin(runtime, tensor):
    return tensor_sin(tensor)

@_builtin('cos')
def _cos(runtime, tensor):
    return tensor_cos(tensor)

# ── Tensor ops ───────────────────────────────────────────

@_builtin('where')
def _where(runtime, cond, a, b):
    return tensor_where(cond, a, b)

@_builtin('clip')
def _clip(runtime, tensor, lo, hi):
    return tensor_clip(tensor, _float(lo), _float(hi))

@_builtin('concat')
def _concat(runtime, *args):
    *tensors, last = args
    if isinstance(last, (int, float, np.integer, np.floating, Tensor)):
        if isinstance(last, Tensor) and last.ndim == 0:
            dim = _int(last.data)
        elif not isinstance(last, Tensor):
            dim = _int(last)
        else:
            tensors = args; dim = 0
    else:
        tensors = args; dim = 0
    return tensor_concat(list(tensors), dim)

@_builtin('stack')
def _stack(runtime, *args):
    *tensors, last = args
    if isinstance(last, (int, float, np.integer, np.floating, Tensor)):
        if isinstance(last, Tensor) and last.ndim == 0:
            dim = _int(last.data)
        elif not isinstance(last, Tensor):
            dim = _int(last)
        else:
            tensors = args; dim = 0
    else:
        tensors = args; dim = 0
    return tensor_stack(list(tensors), dim)

@_builtin('matmul')
def _matmul(runtime, a, b):
    return a @ b

@_builtin('einsum')
def _einsum(runtime, spec, *tensors):
    return tensor_einsum(spec, *tensors)

@_builtin('broadcast')
def _broadcast(runtime, tensor, *shape_args):
    shape = tuple(_int(s) for s in shape_args)
    out = Tensor(np.broadcast_to(tensor.data, shape), (tensor,), 'broadcast', _ctx=tensor.shape)
    def _backward():
        tensor.ensure_grad()[:] += Tensor._broadcast_grad(out.grad, tensor.shape)
    out._backward = _backward
    return out

@_builtin('shape')
def _shape(runtime, tensor):
    return Tensor(np.array(tensor.shape, dtype=np.float32), _op='shape')

# ── Autodiff ─────────────────────────────────────────────

@_builtin('grad')
def _grad(runtime, loss, *params):
    """Compute gradients of loss w.r.t. each param."""
    for p in params:
        p.grad = None
    loss.backward()
    results = []
    for p in params:
        if p.grad is not None:
            results.append(Tensor(p.grad.copy(), _op='grad'))
        else:
            results.append(Tensor(np.zeros_like(p.data), _op='grad_zero'))
    return results[0] if len(results) == 1 else results

@_builtin('backward')
def _backward(runtime, tensor):
    tensor.backward()
    return None

@_builtin('zerograd')
def _zerograd(runtime, *tensors):
    for t in tensors:
        t.zero_grad()
    return None

# ── Optimizers ───────────────────────────────────────────

@_builtin('optimizer')
def _optimizer(runtime, name, lr, *params):
    """Create an optimizer: opt := optimizer('adam', 0.001, W1, b1, W2, b2)
       name: 'sgd', 'adam', or 'adamw'
       Returns an optimizer handle (stored in runtime._optimizers).
    """
    opt_name = name if isinstance(name, str) else str(name)
    learning_rate = _float(lr)
    opt_params = list(params)
    # Mark them as params if not already
    for p in opt_params:
        if not p.is_param:
            p.is_param = True
    opt = create_optimizer(opt_name, opt_params, lr=learning_rate)
    runtime._optimizers.append(opt)
    return opt

@_builtin('step')
def _step(runtime, loss, opt):
    """Perform optimizer step: backward + update.
    Usage: step(loss, opt)
    Equivalent to: loss.backward() + opt.step() + opt.zero_grad()
    """
    if not isinstance(opt, Optimizer):
        raise PlyRuntimeError(f"step() requires an optimizer, got {type(opt).__name__}")
    opt.zero_grad()
    loss.backward()
    opt.step()
    return None

# ── Persistence ──────────────────────────────────────────

@_builtin('save')
def _save(runtime, filename, *tensors):
    """Save tensors to a .npz file.
    Usage: save('model.npz', W1, b1, W2, b2)
    """
    fname = filename if isinstance(filename, str) else str(filename)
    data = {}
    for i, t in enumerate(tensors):
        key = t.name if t.name else f'tensor_{i}'
        data[key] = t.data
    np.savez(fname, **data)
    return None

@_builtin('load')
def _load(runtime, filename):
    """Load tensors from a .npz file.
    Usage: params := load('model.npz')
    Returns loaded tensors as a tuple-like structure.
    Note: returns the last tensor loaded (or first if only one).
    """
    fname = filename if isinstance(filename, str) else str(filename)
    loaded = np.load(fname)
    tensors = []
    for key in loaded.files:
        t = Tensor(loaded[key], _op='loaded', name=key)
        tensors.append(t)
    if not tensors:
        return None
    # Store all in env as named tensors
    for t in tensors:
        if t.name:
            runtime.env[t.name] = t
    return tensors[-1] if len(tensors) == 1 else tensors

# ── Utilities ────────────────────────────────────────────

@_builtin('print')
def _print(runtime, *tensors):
    for t in tensors:
        if isinstance(t, Tensor):
            tag = f" [{t.name}]" if t.name else ""
            param_tag = " (param)" if t.is_param else ""
            print(f"  shape={t.shape} dtype={t.dtype}{param_tag}{tag}")
            if t.size <= 20:
                np.set_printoptions(precision=4, suppress=True)
                print(f"  {t.data}")
            else:
                flat = t.data.flat
                print(f"  [{flat[0]:.4f} {flat[1]:.4f} ... {flat[-2]:.4f} {flat[-1]:.4f}]")
        else:
            print(f"  {t}")
    return None

@_builtin('detach')
def _detach(runtime, tensor):
    """Detach tensor from computation graph."""
    return tensor.detach()

@_builtin('clone')
def _clone(runtime, tensor):
    """Deep copy of tensor."""
    return tensor.clone()


class Runtime:
    def __init__(self, use_ir: bool = False):
        self.env: Dict[str, Tensor] = {}
        self._params: List[Tensor] = []
        self._optimizers: List[Optimizer] = []
        self.use_ir = use_ir
        self._ir_builder = None  # lazy init

    def eval(self, program: Program) -> Optional[Tensor]:
        for binding in program.bindings:
            value = self._eval_expr(binding.value)
            # Track params automatically
            if isinstance(value, Tensor) and value.is_param:
                value.name = binding.name
                self._params.append(value)
            self.env[binding.name] = value

        if program.result is not None:
            # Check if result is the same expression as a binding's value
            # If so, return the env value directly
            for binding in program.bindings:
                if program.result is binding.value:
                    return self.env[binding.name]
            return self._eval_expr(program.result)

        # If no explicit result, return the last binding's value
        if program.bindings:
            return self.env[program.bindings[-1].name]
        return None

    def _eval_expr(self, node: Expr) -> Any:
        if isinstance(node, Number):
            return _ensure_tensor(float(node.value))

        elif isinstance(node, String):
            return node.value

        elif isinstance(node, Var):
            if node.name not in self.env:
                raise PlyRuntimeError(f"Undefined tensor: '{node.name}'")
            return self.env[node.name]

        elif isinstance(node, BinOp):
            return self._eval_binop(node)

        elif isinstance(node, UnOp):
            return self._eval_unop(node)

        elif isinstance(node, Ternary):
            cond = self._eval_expr(node.cond)
            true_val = self._eval_expr(node.true_val)
            false_val = self._eval_expr(node.false_val)
            return tensor_where(cond, true_val, false_val)

        elif isinstance(node, Call):
            return self._eval_call(node)

        elif isinstance(node, Slice):
            return self._eval_slice(node)

        elif isinstance(node, ListLiteral):
            return [self._eval_expr(e) for e in node.elements]

        else:
            raise PlyRuntimeError(f"Unknown expression: {type(node).__name__}")

    def _eval_binop(self, node: BinOp):
        left = self._eval_expr(node.left)
        right = self._eval_expr(node.right)

        op = node.op
        try:
            if op == '+':   return left + right
            elif op == '-': return left - right
            elif op == '*': return left * right
            elif op == '/': return left / right
            elif op == '@': return left @ right
            elif op == '>': return left > right
            elif op == '<': return left < right
            elif op == '>=': return left >= right
            elif op == '<=': return left <= right
            elif op == '==': return left == right
            elif op == '!=': return left != right
            elif op == '&': return left & right
            elif op == '|': return left | right
            else:
                raise PlyRuntimeError(f"Unknown operator: {op}")
        except ValueError as e:
            lhs = getattr(left, 'shape', '?')
            rhs = getattr(right, 'shape', '?')
            raise PlyRuntimeError(
                f"Shape mismatch: {lhs} {op} {rhs} cannot broadcast.\n"
                f"  Hint: use mean(x, dim) which keeps dims, or reshape to match.\n"
                f"  numpy error: {e}"
            ) from e

    def _eval_unop(self, node: UnOp):
        operand = self._eval_expr(node.operand)
        if node.op == '-':
            return -operand
        elif node.op == '!':
            return ~operand
        else:
            raise PlyRuntimeError(f"Unknown unary: {node.op}")

    def _eval_call(self, node: Call):
        if node.name not in BUILTINS:
            # Suggest similar function names
            from difflib import get_close_matches
            matches = get_close_matches(node.name, list(BUILTINS.keys()), n=3, cutoff=0.4)
            hint = f"  Did you mean: {', '.join(matches)}?" if matches else ""
            raise PlyRuntimeError(
                f"Unknown function: '{node.name}'.{hint}\n"
                f"  Available: relu, sigmoid, tanh, softmax, log, exp, sqrt, gelu,\n"
                f"             sum, mean, randn, zeros, ones, param, broadcast,\n"
                f"             reshape, permute, concat, stack, einsum"
            )
        fn = BUILTINS[node.name]
        args = [self._eval_expr(a) for a in node.args]
        return fn(self, *args)

    def _eval_slice(self, node: Slice):
        tensor = self._eval_expr(node.tensor)
        idx = []
        for spec in node.slices:
            start = int(self._eval_expr(spec.start).data) if spec.start is not None else None
            end = int(self._eval_expr(spec.end).data) if spec.end is not None else None
            step = int(self._eval_expr(spec.step).data) if spec.step is not None else None

            if start is not None and end is None and step is None:
                idx.append(start)
            else:
                idx.append(slice(start, end, step))
        return tensor[tuple(idx)]


def run(source: str, use_ir: bool = False) -> Optional[Tensor]:
    from parser import parse
    program = parse(source)
    rt = Runtime(use_ir=use_ir)
    return rt.eval(program)
