# Ply — Tensor-Native Language

A tensor-native programming language. Zero control flow, no variables, pure tensor algebra.
Built from scratch in a single session by Hermes Agent.

## Quick Start

```bash
cd ply
python3 ply.py examples/attention.ply    # Multi-head attention
python3 ply.py examples/hmtl.ply         # 128×128 tensor core matrix
python3 ply.py examples/mlp.ply          # 2-layer GELU MLP
python3 examples/train_nn.py             # Train NN with Ply gradients
```

## Architecture

| File | Purpose |
|------|---------|
| `tokens.py` | Lexer |
| `parser.py` | Recursive descent parser |
| `ast_nodes.py` | AST definitions |
| `runtime.py` | Execution engine (40+ builtins) |
| `tensor.py` | Tensor class with autodiff |
| `backend.py` | GPU backend (NumPy/CuPy/MLX) |
| `ir.py` | Lazy IR compiler with fusion |
| `ply.py` | CLI entry point |
| `SPEC.md` | Language specification |

## Design

- **No control flow** — branching via `?:` (mask), iteration via broadcasting
- **No mutation** — `:=` immutable bindings
- **Autodiff-native** — every op has analytical gradient
- **GPU-ready** — backend auto-detects CuPy/MLX

## Examples

```
-- Scaled dot-product attention in Ply
scores := Q @ K.T / sqrt(d_k)
attn := softmax(scores, -1)
output := attn @ V
```

## Status

v0.5 — working autodiff, GPU backend, lazy IR compiler.
All 35+ gradient ops verified against numerical differentiation.
