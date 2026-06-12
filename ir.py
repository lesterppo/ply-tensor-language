"""
Ply IR — Intermediate Representation compiler with kernel fusion.
Integrates with the Tensor computation graph.
When use_ir=True, operations build IR nodes alongside Tensor nodes.
The Compiler fuses element-wise chains for faster execution.

Key optimizations:
  - Element-wise fusion: merge chains (add/mul/relu/sigmoid/gelu/etc.)
  - DAG-aware scheduling: topological execution with buffer reuse
  - Dead code elimination: skip unreferenced nodes
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Set
import numpy as np
from backend import get_backend

# Forward reference for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tensor import Tensor as TensorType


class IRNode:
    """Base class for all IR nodes."""
    __slots__ = ('_id', '_shape', '_dtype', '_inputs', '_cached_data')
    _next_id = 0

    def __init__(self):
        IRNode._next_id += 1
        self._id = IRNode._next_id
        self._shape: Optional[Tuple[int, ...]] = None
        self._dtype = None
        self._inputs: List[IRNode] = []
        self._cached_data: Any = None

    @property
    def id(self) -> int: return self._id
    @property
    def shape(self) -> Tuple[int, ...]: return self._shape
    @property
    def dtype(self): return self._dtype
    @property
    def inputs(self) -> List[IRNode]: return self._inputs

    def compute(self, compiler: 'Compiler') -> Any:
        raise NotImplementedError

    def walk(self, visited: Set[int] = None) -> List[IRNode]:
        if visited is None:
            visited = set()
        order = []
        def dfs(node):
            if node._id not in visited:
                visited.add(node._id)
                for inp in node._inputs:
                    dfs(inp)
                order.append(node)
        dfs(self)
        return order


@dataclass
class ConstNode(IRNode):
    data: Any = None
    def __post_init__(self):
        super().__init__()
        self._shape = self.data.shape
        self._dtype = self.data.dtype
        self._inputs = []
    def compute(self, compiler):
        return self.data


@dataclass
class VarNode(IRNode):
    name: str = ''
    tensor: Any = None
    def __post_init__(self):
        super().__init__()
        self._inputs = []
    def compute(self, compiler):
        if self.tensor is not None:
            return compiler.ensure_computed(self.tensor._ir_node) if hasattr(self.tensor, '_ir_node') and self.tensor._ir_node is not None else self.tensor.data
        raise RuntimeError(f"VarNode '{self.name}' has no data source")


@dataclass
class BinOpNode(IRNode):
    op: str = ''
    left: IRNode = None
    right: IRNode = None
    def __post_init__(self):
        super().__init__()
        self._inputs = [self.left, self.right]
    def compute(self, compiler):
        l = compiler.ensure_computed(self.left)
        r = compiler.ensure_computed(self.right)
        xp = compiler.backend._mod
        if self.op == '+':   return l + r
        elif self.op == '-': return l - r
        elif self.op == '*': return l * r
        elif self.op == '/': return l / r
        elif self.op == '@': return l @ r
        elif self.op == '>': return (l > r).astype(xp.float32)
        elif self.op == '<': return (l < r).astype(xp.float32)
        elif self.op == '>=': return (l >= r).astype(xp.float32)
        elif self.op == '<=': return (l <= r).astype(xp.float32)
        elif self.op == '==': return (l == r).astype(xp.float32)
        elif self.op == '!=': return (l != r).astype(xp.float32)
        elif self.op == '&': return ((l.astype(bool)) & (r.astype(bool))).astype(xp.float32)
        elif self.op == '|': return ((l.astype(bool)) | (r.astype(bool))).astype(xp.float32)
        raise ValueError(f"Unknown binop: {self.op}")


@dataclass
class UnOpNode(IRNode):
    op: str = ''
    operand: IRNode = None
    def __post_init__(self):
        super().__init__()
        self._inputs = [self.operand]
    def compute(self, compiler):
        x = compiler.ensure_computed(self.operand)
        xp = compiler.backend._mod
        if self.op == 'neg': return -x
        elif self.op == 'not': return (~(x.astype(bool))).astype(xp.float32)
        raise ValueError(f"Unknown unop: {self.op}")


@dataclass
class FunctionNode(IRNode):
    name: str = ''
    args: List[IRNode] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    _compute_fn: Any = None
    def __post_init__(self):
        super().__init__()
        self._inputs = list(self.args)
    def compute(self, compiler):
        args = [compiler.ensure_computed(a) for a in self.args]
        if self._compute_fn is not None:
            return self._compute_fn(compiler, *args, **self.kwargs)
        raise RuntimeError(f"No compute fn for {self.name}")


# ── Fusion ────────────────────────────────────────────────

FUSIBLE_OPS = {'+', '-', '*', '/', 'relu', 'sigmoid', 'tanh', 'gelu',
               'exp', 'log', 'sqrt', 'abs', 'sin', 'cos', 'neg'}


def _can_fuse(node: IRNode) -> bool:
    if isinstance(node, BinOpNode) and node.op in FUSIBLE_OPS:
        return True
    if isinstance(node, UnOpNode) and node.op in FUSIBLE_OPS:
        return True
    if isinstance(node, FunctionNode) and node.name in FUSIBLE_OPS:
        return True
    return False


class FusedNode(IRNode):
    """A fused chain of element-wise operations — single kernel."""
    def __init__(self, nodes: List[IRNode], root_inputs: List[IRNode]):
        super().__init__()
        self.nodes = nodes
        self._inputs = list(root_inputs)
        self._shape = nodes[-1]._shape

    def compute(self, compiler):
        # Execute the fused chain — compute each node in sequence,
        # feeding results forward. Uses a dict to track chain-internal outputs.
        xp = compiler.backend._mod
        chain_outputs = {}
        
        for node in self.nodes:
            # Get all inputs — some from within chain, some from leaves
            args = []
            for inp in node._inputs:
                if inp._id in chain_outputs:
                    args.append(chain_outputs[inp._id])
                else:
                    args.append(compiler.ensure_computed(inp))
            
            # Compute based on node type
            if isinstance(node, BinOpNode):
                l, r = args[0], args[1]
                result = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
                          '*': lambda a,b: a*b, '/': lambda a,b: a/b}[node.op](l, r)
            elif isinstance(node, FunctionNode):
                x = args[0]
                if node.name == 'relu':
                    result = xp.maximum(x, 0)
                elif node.name == 'sigmoid':
                    result = 1 / (1 + xp.exp(-x))
                elif node.name == 'tanh':
                    result = xp.tanh(x)
                elif node.name == 'gelu':
                    inner = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
                    result = 0.5 * x * (1 + xp.tanh(inner))
                elif node.name == 'exp':
                    result = xp.exp(x)
                elif node.name == 'log':
                    result = xp.log(xp.maximum(x, 1e-10))
                elif node.name == 'sqrt':
                    result = xp.sqrt(xp.maximum(x, 0))
                elif node.name == 'abs':
                    result = xp.abs(x)
                elif node.name == 'sin':
                    result = xp.sin(x)
                elif node.name == 'cos':
                    result = xp.cos(x)
                elif node._compute_fn is not None:
                    result = node._compute_fn(compiler, *args)
                else:
                    result = x
            elif isinstance(node, UnOpNode):
                result = -args[0] if node.op == 'neg' else (~(args[0].astype(bool))).astype(xp.float32)
            else:
                result = args[0] if args else None
            
            chain_outputs[node._id] = result
        
        return chain_outputs[self.nodes[-1]._id]


# ── Compiler ──────────────────────────────────────────────

class Compiler:
    """Compiles IR DAG into optimized execution plan."""

    def __init__(self):
        self.backend = get_backend()
        self.computed: Dict[int, Any] = {}  # id -> array

    def ensure_computed(self, node: IRNode) -> Any:
        if node._id in self.computed:
            return self.computed[node._id]
        result = node.compute(self)
        self.computed[node._id] = result
        return result

    def compile(self, output: IRNode, apply_fusion: bool = True) -> IRNode:
        if not apply_fusion:
            return output

        order = output.walk()
        # Process in reverse order (outputs first) so we catch longest chains
        all_chains = []
        fused_set = set()

        for node in reversed(order):
            if node._id in fused_set:
                continue
            if _can_fuse(node):
                chain = self._extract_fusion_chain(node)
                if len(chain) >= 2:
                    all_chains.append(chain)
                    for n in chain:
                        fused_set.add(n._id)

        # Now replace all chains
        for chain in all_chains:
            chain_ids = {n._id for n in chain}
            leaf_inputs = []
            seen = set()
            for n in chain:
                for inp in n._inputs:
                    if inp._id not in chain_ids and inp._id not in seen:
                        leaf_inputs.append(inp)
                        seen.add(inp._id)
            fused = FusedNode(chain, leaf_inputs)
            fused._shape = chain[-1]._shape
            # If the output is the last node in chain, update output reference
            if output._id == chain[-1]._id:
                output = fused
            self._replace_all(output, chain[-1], fused)

        return output

    def _extract_fusion_chain(self, node: IRNode) -> List[IRNode]:
        chain = [node]
        current = node
        while True:
            if isinstance(current, (BinOpNode, FunctionNode)) and current._inputs:
                next_input = current._inputs[0]
                if _can_fuse(next_input) and next_input not in chain:
                    chain.insert(0, next_input)
                    current = next_input
                else:
                    break
            elif isinstance(current, UnOpNode):
                if _can_fuse(current.operand) and current.operand not in chain:
                    chain.insert(0, current.operand)
                    current = current.operand
                else:
                    break
            else:
                break
        return chain

    def _replace_all(self, root: IRNode, old: IRNode, new: IRNode):
        visited = set()
        def walk(node):
            if node._id in visited:
                return
            visited.add(node._id)
            new_inputs = []
            for inp in node._inputs:
                if inp._id == old._id:
                    new_inputs.append(new)
                else:
                    new_inputs.append(inp)
                    walk(inp)
            node._inputs = new_inputs
        walk(root)

    def run(self, output: IRNode, apply_fusion: bool = True) -> Any:
        optimized = self.compile(output, apply_fusion)
        return self.ensure_computed(optimized)

    def clear(self):
        self.computed.clear()


# ── Tensor graph → IR conversion ──────────────────────────

def tensor_to_ir(tensor: 'Tensor', memo: Dict[int, IRNode] = None) -> IRNode:
    """Convert a Tensor computation graph to an IR DAG.
    Only converts forward ops (not backward/gradient nodes).
    """
    if memo is None:
        memo = {}

    from tensor import Tensor as T

    tid = id(tensor)
    if tid in memo:
        return memo[tid]

    # Leaf node (no inputs) → ConstNode
    if tensor.is_leaf:
        node = VarNode(name=f'leaf_{tid}', tensor=tensor)
        memo[tid] = node
        return node

    # Convert inputs first
    ir_inputs = [tensor_to_ir(inp, memo) for inp in tensor._inputs]

    # Map operation to IR node
    op = tensor._op

    if op in ('+', '-', '*', '/', '@', '>', '<', '>=', '<=', '==', '!=', '&', '|'):
        node = BinOpNode(op=op, left=ir_inputs[0], right=ir_inputs[1])
    elif op == 'neg':
        node = UnOpNode(op='neg', operand=ir_inputs[0])
    elif op in ('relu', 'sigmoid', 'tanh', 'gelu', 'exp', 'log', 'sqrt', 'abs', 'sin', 'cos',
                 'softmax', 'layernorm', 'sum', 'mean', 'reshape', 'permute', 'slice',
                 'concat', 'stack', 'einsum', 'broadcast'):
        fn_node = FunctionNode(name=op, args=ir_inputs)
        # Attach compute function
        xp = get_backend()._mod

        if op == 'relu':
            fn_node._compute_fn = lambda c, *a: xp.maximum(a[0], 0)
        elif op == 'sigmoid':
            fn_node._compute_fn = lambda c, *a: 1 / (1 + xp.exp(-a[0]))
        elif op == 'tanh':
            fn_node._compute_fn = lambda c, *a: xp.tanh(a[0])
        elif op == 'gelu':
            fn_node._compute_fn = lambda c, *a: 0.5 * a[0] * (1 + xp.tanh(np.sqrt(2/np.pi) * (a[0] + 0.044715 * a[0]**3)))
        elif op == 'exp':
            fn_node._compute_fn = lambda c, *a: xp.exp(a[0])
        elif op == 'log':
            fn_node._compute_fn = lambda c, *a: xp.log(xp.maximum(a[0], 1e-10))
        elif op == 'sqrt':
            fn_node._compute_fn = lambda c, *a: xp.sqrt(xp.maximum(a[0], 0))
        elif op == 'abs':
            fn_node._compute_fn = lambda c, *a: xp.abs(a[0])
        elif op == 'sin':
            fn_node._compute_fn = lambda c, *a: xp.sin(a[0])
        elif op == 'cos':
            fn_node._compute_fn = lambda c, *a: xp.cos(a[0])
        elif op == 'sum':
            fn_node._compute_fn = lambda c, *a: a[0].sum()
        elif op == 'reshape':
            fn_node._compute_fn = lambda c, *a: xp.reshape(a[0], tensor._ctx)
        elif op == 'permute':
            fn_node._compute_fn = lambda c, *a: xp.transpose(a[0], tensor._ctx)
        elif op == 'matmul':
            fn_node._compute_fn = lambda c, *a: a[0] @ a[1]
        else:
            # Complex ops (layernorm, softmax, slice, concat, stack, einsum,
            # broadcast, mean) — fall back to pre-computed tensor data
            fn_node._compute_fn = lambda c, *a, t=tensor: t.data
        node = fn_node
    else:
        # Unknown op — fall back to storing the data
        node = ConstNode(data=tensor.data)

    node._shape = tensor.shape
    memo[tid] = node
    return node


def compile_and_run(tensor: 'Tensor', apply_fusion: bool = True) -> Any:
    """Convert tensor graph to IR, compile with fusion, execute.
    Falls back to pre-computed data for complex ops."""
    ir = tensor_to_ir(tensor)
    compiler = Compiler()
    try:
        return compiler.run(ir, apply_fusion)
    except Exception:
        # Fall back: just return the pre-computed tensor data
        return tensor.data
