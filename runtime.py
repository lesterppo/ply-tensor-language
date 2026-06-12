"""
Ply Runtime — Tensor-native execution with automatic differentiation.
Uses the Tensor class (tensor.py) which builds a computation graph.
Every operation is recorded; gradients flow via backward().
"""
import numpy as np
from typing import Dict, Any, List, Optional
from ast_nodes import *
from tensor import (
    Tensor, _ensure_tensor,
    tensor_relu, tensor_gelu, tensor_sigmoid, tensor_tanh,
    tensor_softmax, tensor_layernorm,
    tensor_exp, tensor_log, tensor_sqrt, tensor_abs,
    tensor_sin, tensor_cos, tensor_clip,
    tensor_where, tensor_concat, tensor_stack,
    tensor_einsum,
)


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

@_builtin('reshape')
def _reshape(runtime, tensor, *shape_args):
    shape = tuple(_int(s) for s in shape_args)
    return tensor.reshape(*shape)

@_builtin('permute')
def _permute(runtime, tensor, *axes):
    return tensor.permute(*[_int(a) for a in axes])

@_builtin('transpose')
def _transpose(runtime, tensor):
    return tensor.permute(*reversed(range(tensor.ndim)))

@_builtin('T')
def _T(runtime, tensor):
    return tensor.permute(*reversed(range(tensor.ndim)))

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
            tensors = args
            dim = 0
    else:
        tensors = args
        dim = 0
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
            tensors = args
            dim = 0
    else:
        tensors = args
        dim = 0
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

@_builtin('grad')
def _grad(runtime, loss, *params):
    """Compute gradients of loss w.r.t. each param. Returns a tuple/list."""
    # Zero all parameter gradients first
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
    """Trigger backpropagation from tensor. Returns None (side-effect only)."""
    tensor.backward()
    return None

@_builtin('zerograd')
def _zerograd(runtime, *tensors):
    for t in tensors:
        t.zero_grad()
    return None

@_builtin('print')
def _print(runtime, *tensors):
    for t in tensors:
        if isinstance(t, Tensor):
            print(f"  shape={t.shape} dtype={t.dtype}")
            if t.size <= 20:
                np.set_printoptions(precision=4, suppress=True)
                print(f"  {t.data}")
            else:
                flat = t.data.flat
                print(f"  [{flat[0]:.4f} {flat[1]:.4f} ... {flat[-2]:.4f} {flat[-1]:.4f}]")
        else:
            print(f"  {t}")
    return None


class Runtime:
    def __init__(self):
        self.env: Dict[str, Tensor] = {}

    def eval(self, program: Program) -> Optional[Tensor]:
        for binding in program.bindings:
            value = self._eval_expr(binding.value)
            self.env[binding.name] = value

        if program.result and program.result not in [b.value for b in program.bindings]:
            return self._eval_expr(program.result)

        return self.env.get(program.bindings[-1].name) if program.bindings else None

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

        if node.op == '+':   return left + right
        elif node.op == '-': return left - right
        elif node.op == '*': return left * right
        elif node.op == '/': return left / right
        elif node.op == '@': return left @ right
        elif node.op == '>': return left > right
        elif node.op == '<': return left < right
        elif node.op == '>=': return left >= right
        elif node.op == '<=': return left <= right
        elif node.op == '==': return left == right
        elif node.op == '!=': return left != right
        elif node.op == '&': return left & right
        elif node.op == '|': return left | right
        else:
            raise PlyRuntimeError(f"Unknown operator: {node.op}")

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
            raise PlyRuntimeError(f"Unknown function: '{node.name}'")
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


def run(source: str) -> Optional[Tensor]:
    from parser import parse
    program = parse(source)
    rt = Runtime()
    return rt.eval(program)
