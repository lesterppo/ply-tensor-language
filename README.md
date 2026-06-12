# Ply — Tensor-Native Language

**A complete tensor-native programming language in 3,600 lines of Python.**
Zero control flow. No variables. Pure tensor algebra. Built with reverse-mode autodiff, GPU backend, kernel fusion compiler, and optimizer infrastructure.

```ply
-- Scaled dot-product attention — the core of every Transformer
scores := Q @ K.T / sqrt(d_k)
attn := softmax(scores, -1)
output := attn @ V
```

## Quick Start

```bash
git clone https://github.com/lesterppo/ply-tensor-language.git
cd ply-tensor-language

# Run examples
python3 ply.py examples/attention.ply    # Multi-head attention
python3 ply.py examples/mlp.ply          # 2-layer GELU MLP + residual
python3 ply.py examples/hmtl.ply         # 128×128 tensor core matrix
python3 ply.py --ir examples/mlp.ply     # With kernel fusion

# Train a neural network
python3 examples/train_v2.py             # Adam optimizer, R²=0.94

# Run tests
python3 ply_tests.py                     # 17/17 passing
```

## What Makes Ply Different

| | PyTorch | JAX | Ply |
|---|---|---|---|
| Core concepts | ~50 | ~10 | **3** (`:=`, expression, call) |
| Codebase | ~2M lines | ~200K lines | **3,600 lines** |
| Control flow | `if`/`for`/`while` | `lax.cond`/`lax.scan` | **`? :` (mask-based)** |
| Variables | Mutable tensors | Immutable arrays | **Immutable bindings** |
| Learning curve | Months | Weeks | **Hours** |

### Design Philosophy

- **No control flow** — `?:` ternary is the only conditional (mask-based, no short-circuit)
- **No variables** — `:=` creates immutable bindings, assigned once
- **No side effects** — every expression is pure
- **Autodiff-native** — every operation has an analytical gradient
- **Static DAG** — every program is a static computation graph, directly mappable to hardware

### What Ply Does NOT Have

- ❌ `if`/`else` statements (use `mask ? true_val : false_val`)
- ❌ `for`/`while` loops (use broadcasting + reduction)
- ❌ Functions or procedures (use bindings for sub-expressions)
- ❌ Classes or objects
- ❌ Mutation or side effects
- ❌ I/O (except `print` for debugging)

## Architecture

```
source.ply  →  tokens.py  →  parser.py  →  ast_nodes.py  →  runtime.py
                 119L           288L            74L             410L
                                                              ↓
                                                          tensor.py (426L)
                                                         reverse-mode autodiff
                                                              ↓
                                              ┌───────────────┼───────────────┐
                                          backend.py        optim.py         ir.py
                                         CuPy/MLX/NumPy   SGD/Adam/AdamW   kernel fusion
                                             236L             177L            403L
```

## v2.0 Features

### Training Infrastructure

```python
# Create parameters with He initialization
W1 = param(64, 32)          # in .ply:  W1 := param(64, 32)
b1 = param_zeros(32)

# Choose optimizer
opt = optimizer('adam', 0.001, W1, b1, W2, b2)

# One-call backward + update
step(loss, opt)
```

### Kernel Fusion Compiler

```bash
$ python3 ply.py --ir examples/mlp.ply
[IR compiled with fusion in 1.4ms]
```

Element-wise chains (add, mul, relu, sigmoid, gelu, exp, log, sqrt, sin, cos) automatically fused into single kernels.

### Persistence

```ply
save('model.npz', W1, b1, W2, b2)
params := load('model.npz')
```

## Example: Complete Training Loop

```python
from tensor import Tensor, param, tensor_relu
from optim import Adam

# Generate data: y = sin(x) + noise
X = Tensor(np.random.randn(200, 1).astype(np.float32) * 2)
Y = Tensor(np.sin(X_data) + 0.1 * np.random.randn(200, 1))

# Create parameters
W1 = param(np.random.randn(1, 16) * sqrt(2.0), name='W1')
b1 = param(np.zeros(16), name='b1')
W2 = param(np.random.randn(16, 1) * sqrt(2.0/16), name='W2')
b2 = param(np.zeros(1), name='b2')

opt = Adam([W1, b1, W2, b2], lr=0.01)

for epoch in range(500):
    # Forward pass — builds computation graph
    h = X @ W1 + b1
    a = tensor_relu(h)
    pred = a @ W2 + b2
    diff = pred - Y
    loss = (diff * diff).mean()

    # Backward + update
    opt.zero_grad()
    loss.backward()
    opt.step()

# R² = 0.94 after 500 epochs
```

## Built-in Functions

### Tensor Creation
`randn` `zeros` `ones` `full` `eye` `arange` `param` `param_zeros`

### Shape Manipulation
`reshape` `permute` `transpose` `concat` `stack` `broadcast`

### Reductions
`sum` `mean` `max` `min` `std`

### Activations
`relu` `gelu` `sigmoid` `tanh` `softmax` `layernorm`

### Math
`exp` `log` `sqrt` `abs` `sin` `cos` `clip` `where`

### Tensor Ops
`matmul` `einsum` `shape`

### Autodiff & Training
`grad` `backward` `zerograd` `optimizer` `step` `detach` `clone`

### Persistence
`save` `load` `print`

## Use Cases

- **Teaching deep learning** — read the 426-line `tensor.py` and understand autodiff completely
- **Edge deployment** — pure NumPy backend, <700 lines for inference
- **Research prototyping** — test new attention variants in 22 lines
- **Compiler research** — 400-line IR with kernel fusion as case study
- **Hardware bringup** — write a 200-line backend, run all Ply programs

## Requirements

- Python 3.10+
- NumPy
- Optional: CuPy (NVIDIA GPU), MLX (Apple Silicon)

## License

MIT
