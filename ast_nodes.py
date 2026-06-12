"""
Ply AST — Abstract Syntax Tree for the tensor-native language.
Every program is a static DAG of tensor operations — no control flow.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Expressions ──────────────────────────────────────────────

@dataclass
class Number:
    value: float

@dataclass
class String:
    value: str

@dataclass
class Var:
    name: str

@dataclass
class BinOp:
    op: str           # + - * / @ > < >= <= == != & |
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnOp:
    op: str           # - ! (negate, logical not)
    operand: 'Expr'

@dataclass
class Ternary:
    """cond ? true_val : false_val — the only 'conditional' in Ply"""
    cond: 'Expr'
    true_val: 'Expr'
    false_val: 'Expr'

@dataclass
class Call:
    name: str
    args: List['Expr']

@dataclass
class Slice:
    tensor: 'Expr'
    slices: List['SliceSpec']

@dataclass
class SliceSpec:
    start: Optional['Expr'] = None
    end: Optional['Expr'] = None
    step: Optional['Expr'] = None

@dataclass
class ListLiteral:
    elements: List['Expr']

# ── Top-level ────────────────────────────────────────────────

@dataclass
class Binding:
    name: str
    value: 'Expr'

@dataclass
class Program:
    bindings: List[Binding]
    result: Optional['Expr'] = None  # final expression (optional)

# Union type
Expr = Number | String | Var | BinOp | UnOp | Ternary | Call | Slice | SliceSpec | ListLiteral
