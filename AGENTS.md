# Ply v2.0 — Tensor-Native Language

A complete tensor-native programming language. Zero control flow, no variables, pure tensor algebra.
Built from scratch with reverse-mode autodiff, GPU backend, kernel fusion compiler, and optimizer infrastructure.

## Quick Start

```bash
cd ply-tensor-language
python3 ply.py examples/attention.ply    # Multi-head attention (Transformer core)
python3 ply.py --ir examples/mlp.ply     # MLP with IR kernel fusion
python3 ply.py examples/hmtl.ply         # 128×128 tensor core dispatch matrix
python3 examples/train_v2.py             # Train NN with Adam optimizer
python3 ply_tests.py                     # 17-test suite
```

## Architecture (v2.0)

| File | Lines | Role |
|------|-------|------|
| `tokens.py` | 119 | Regex lexer — 41 token types, zero keywords |
| `parser.py` | 288 | Recursive descent parser with full precedence climbing |
| `ast_nodes.py` | 74 | AST — Number, Var, BinOp, Ternary, Call, Slice |
| `runtime.py` | 410 | 50+ builtins: tensor ops, autodiff, optimizers, save/load |
| `tensor.py` | 426 | Tensor class with reverse-mode autodiff — analytical gradients for every op |
| `optim.py` | 177 | SGD, Adam, AdamW optimizers with momentum/weight decay |
| `backend.py` | 236 | Pluggable GPU backend — auto-detects CuPy/MLX/NumPy |
| `ir.py` | 403 | Lazy IR compiler with kernel fusion (5 ops → 1 kernel) |
| `ply.py` | 120 | CLI with `--ir` flag, REPL mode |

## v2.0 Features

### Training Infrastructure
- `param(d0, d1, ...)` — create learnable parameters with He init
- `optimizer('adam', lr, W1, b1, ...)` — create SGD/Adam/AdamW optimizer
- `step(loss, opt)` — backward + optimizer update in one call
- In-place weight updates with automatic graph detachment

### IR Compiler with Kernel Fusion
- Element-wise ops (add, mul, relu, sigmoid, gelu, exp, log, sqrt, abs, sin, cos, tanh, neg) fused into single kernels
- DAG-aware topological scheduling
- `--ir` flag on CLI for IR-compiled execution

### Persistence
- `save('model.npz', W1, b1, ...)` — save parameters
- `load('model.npz')` — load parameters

### Graph Management
- `detach()` — strip backward graph, return leaf
- `clone()` — deep copy with graph intact
- `is_param` flag for automatic parameter tracking
- `data_add_()`, `data_sub_()`, `data_mul_()` — in-place updates with graph detachment

## Design

- **No control flow** — branching via `?:` (mask), iteration via broadcasting
- **No mutation** — `:=` immutable bindings
- **Autodiff-native** — every op has analytical gradient
- **GPU-ready** — backend auto-detects CuPy/MLX
- **Kernel fusion** — element-wise chain fusion at IR level

## Examples

### Complete training loop (Python + Ply)
```python
W1 = param(np.random.randn(1, 16) * sqrt(2.0), name='W1')
opt = Adam([W1, b1, W2, b2], lr=0.01)
for epoch in range(500):
    pred = relu(X @ W1 + b1) @ W2 + b2
    loss = ((pred - Y) ** 2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
```

### Scaled dot-product attention (pure Ply)
```ply
scores := Q @ K.T / sqrt(d_k)
attn := softmax(scores, -1)
output := attn @ V
```

## Testing

```bash
python3 ply_tests.py  # 17 tests: tensor ops, autodiff, optimizers, parser, IR, save/load
```

## Status

v2.0 — complete tensor-native language with training infrastructure, optimizer API, IR kernel fusion compiler, save/load, and comprehensive test suite. 17/17 tests passing.
