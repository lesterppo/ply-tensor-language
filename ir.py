"""
Ply IR — Intermediate Representation for lazy tensor computation.
Operations build a DAG instead of executing immediately.
The Compiler schedules, fuses, and executes the graph.

Key optimizations:
  - Element-wise fusion: merge chains of add/mul/relu/sigmoid/etc.
  - DAG-aware scheduling: execute dependencies in order, reuse buffers
  - Dead code elimination: skip unreferenced nodes
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Dict, Set
import numpy as np
from backend import get_backend


# ── IR Node types ─────────────────────────────────────────

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
    def id(self) -> int:
        return self._id

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def inputs(self) -> List[IRNode]:
        return self._inputs

    def compute(self, backend) -> Any:
        """Execute this node. Subclasses override."""
        raise NotImplementedError

    def walk(self, visited: Set[int] = None) -> List[IRNode]:
        """Topological walk of the DAG rooted at this node."""
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

    def compute(self, backend):
        return self.data


@dataclass
class VarNode(IRNode):
    """Reference to a named tensor (lazy or pre-computed)."""
    name: str = ''
    tensor: Any = None  # The actual Tensor object (for .data access)

    def __post_init__(self):
        super().__init__()
        self._inputs = []

    def compute(self, backend):
        if self.tensor is not None and self.tensor._ir is not None:
            return self.tensor._ir.compute(backend)
        if self.tensor is not None and self.tensor.data is not None:
            return self.tensor.data
        raise RuntimeError(f"VarNode '{self.name}' has no data source")


@dataclass
class BinOpNode(IRNode):
    op: str = ''       # + - * / @ > < >= <= == != & |
    left: IRNode = None
    right: IRNode = None

    def __post_init__(self):
        super().__init__()
        self._inputs = [self.left, self.right]

    def compute(self, backend):
        l = backend.ensure_computed(self.left)
        r = backend.ensure_computed(self.right)
        xp = backend.module
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
    op: str = ''       # neg, not
    operand: IRNode = None

    def __post_init__(self):
        super().__init__()
        self._inputs = [self.operand]

    def compute(self, backend):
        x = backend.ensure_computed(self.operand)
        xp = backend.module
        if self.op == 'neg': return -x
        elif self.op == 'not': return (~(x.astype(bool))).astype(xp.float32)
        raise ValueError(f"Unknown unop: {self.op}")


@dataclass
class FunctionNode(IRNode):
    """Generic function call: relu, sigmoid, reshape, sum, etc."""
    name: str = ''
    args: List[IRNode] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    _compute_fn: Any = None  # Custom compute function

    def __post_init__(self):
        super().__init__()
        self._inputs = list(self.args)

    def compute(self, backend):
        args = [backend.ensure_computed(a) for a in self.args]
        if self._compute_fn is not None:
            return self._compute_fn(backend, *args, **self.kwargs)
        raise RuntimeError(f"No compute fn for {self.name}")


# ── Fusion ────────────────────────────────────────────────

FUSIBLE_OPS = {'+', '-', '*', '/', 'relu', 'sigmoid', 'tanh', 'gelu', 'exp', 'log',
                'sqrt', 'abs', 'sin', 'cos', 'neg'}

def _can_fuse(node: IRNode) -> bool:
    """Check if node can be fused into an element-wise chain."""
    if isinstance(node, BinOpNode) and node.op in FUSIBLE_OPS:
        return True
    if isinstance(node, UnOpNode) and node.op in FUSIBLE_OPS:
        return True
    if isinstance(node, FunctionNode) and node.name in FUSIBLE_OPS:
        return True
    return False


class FusedNode(IRNode):
    """A fused chain of element-wise operations."""
    def __init__(self, nodes: List[IRNode], root_inputs: List[IRNode]):
        super().__init__()
        self.nodes = nodes
        self._inputs = list(root_inputs)
        self._shape = nodes[-1]._shape

    def compute(self, backend):
        # Execute the fused chain without intermediate allocations
        # For CPU: execute sequentially (future: JIT compile)
        result = None
        for node in self.nodes:
            if isinstance(node, BinOpNode):
                if result is None:
                    l = backend.ensure_computed(node.left)
                    r = backend.ensure_computed(node.right)
                else:
                    # result is one input, get the other
                    if node.left._id == self._inputs[0]._id or any(
                        node.left._id == ri._id for ri in self._inputs
                    ):
                        l = result
                        r = backend.ensure_computed(node.right)
                    else:
                        l = backend.ensure_computed(node.left)
                        r = result
                xp = backend.module
                result = {'+': lambda a,b: a+b, '-': lambda a,b: a-b,
                          '*': lambda a,b: a*b, '/': lambda a,b: a/b}[node.op](l, r)
            elif isinstance(node, FunctionNode):
                x = result if result is not None else backend.ensure_computed(node.args[0])
                if node.name == 'relu':
                    result = backend.module.maximum(x, 0)
                elif node.name == 'sigmoid':
                    result = 1 / (1 + backend.module.exp(-x))
                elif node.name == 'tanh':
                    result = backend.module.tanh(x)
                elif node.name == 'gelu':
                    inner = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
                    result = 0.5 * x * (1 + backend.module.tanh(inner))
                elif node.name == 'exp':
                    result = backend.module.exp(x)
                elif node.name == 'log':
                    result = backend.module.log(backend.module.maximum(x, 1e-10))
                elif node.name == 'sqrt':
                    result = backend.module.sqrt(backend.module.maximum(x, 0))
                elif node.name == 'abs':
                    result = backend.module.abs(x)
                elif node.name == 'sin':
                    result = backend.module.sin(x)
                elif node.name == 'cos':
                    result = backend.module.cos(x)
        return result


# ── Compiler ──────────────────────────────────────────────

class Compiler:
    """Compiles IR DAG into an optimized execution plan and runs it."""

    def __init__(self):
        self.module = get_backend()._mod
        self.computed: Dict[int, Any] = {}  # id -> array

    def ensure_computed(self, node: IRNode) -> Any:
        """Get or compute the value for a node."""
        if node._id in self.computed:
            return self.computed[node._id]
        result = node.compute(self)
        self.computed[node._id] = result
        return result

    def compile(self, output: IRNode, apply_fusion: bool = True) -> IRNode:
        """Optimize the IR DAG. Returns the optimized output node."""
        if not apply_fusion:
            return output

        # Walk the DAG
        order = output.walk()

        # Find fusible chains
        fused_set = set()

        for node in order:
            if node._id in fused_set:
                continue
            if _can_fuse(node):
                chain = self._extract_fusion_chain(node)
                if len(chain) >= 2:
                    # Collect leaf inputs (inputs not in the chain)
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
                    # Replace references
                    self._replace_all(output, chain[-1], fused)
                    for n in chain:
                        fused_set.add(n._id)

        return output

    def _extract_fusion_chain(self, node: IRNode) -> List[IRNode]:
        """Extract a chain of fusible ops ending at node."""
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
        """Replace all references to old with new in the DAG."""
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
        """Compile and execute the graph. Returns the computed result."""
        optimized = self.compile(output, apply_fusion)
        return self.ensure_computed(optimized)

    def clear(self):
        self.computed.clear()
