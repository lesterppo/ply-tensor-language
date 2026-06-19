# Tensor Language — Ply + CMTIP

**A tensor-native programming language and cross-model communication protocol.**

Ply: compute in pure tensor algebra. CMTIP: transmit results as semantic tensor
vectors between AI models. Together they form an end-to-end pipeline where
tensor computations are communicated tensor-natively — no text tokens needed.

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

# Ply: tensor-native programming language
echo 'softmax(randn(1, 10), 1)' | python3 ply.py
python3 ply_tests.py                     # 17/17

# CMTIP: cross-model tensor communication
python3 cmtip/cmtip_chat.py              # 5-turn demo (needs free Nvidia API key)
python3 cmtip/ply_cmtip_bridge.py        # Full pipeline: Ply → tensor → LLM decode
python3 cmtip/cmtip_repl.py              # Interactive tensor REPL
```

## The Full Pipeline

```
Ply Program          CMTIP Tensor         LLM Decode
───────────         ─────────────         ──────────
softmax(x, 1)  ──→  softmax:0.42        "10-class classification
                    sigmoid:0.47          with softmax output"
                    broadcast:0.44
```

1. **Compute** in Ply (tensor-native language, 3,600 lines of Python)
2. **Encode** the result as a 72-concept semantic tensor via CMTIP
3. **Transmit** to a different AI model (Llama ↔ Gemma, free Nvidia API)
4. **Decode** — the receiving model reconstructs what was computed

## What Makes Ply Different

| | PyTorch | JAX | Ply |
|---|---|---|---|
| Core concepts | ~50 | ~10 | **3** (`:=`, expression, call) |
| Codebase | ~2M lines | ~200K lines | **3,600 lines** |
| Control flow | `if`/`for`/`while` | `lax.cond`/`lax.scan` | **`? :` (mask-based)** |
| Variables | Mutable tensors | Immutable arrays | **Immutable bindings** |
| Learning curve | Months | Weeks | **Hours** |

### Design Philosophy

- **No control flow** — `?:` ternary is the only conditional (mask-based)
- **No variables** — `:=` creates immutable bindings, assigned once
- **No side effects** — every expression is pure
- **Autodiff-native** — every operation has an analytical gradient
- **Static DAG** — every program is a static computation graph

## CMTIP: Cross-Model Tensor Communication

Two AI models converse via concept tensors instead of text. Messages are
encoded as weighted concept vectors (72 concepts across 8 domains), then
the receiving model reconstructs the meaning.

```bash
# Watch Llama-3.1-8b and Gemma-2-2b have a 5-turn conversation
$ python3 cmtip/cmtip_chat.py

Turn 1 [A] "The database is throwing connection pool errors."
    → connection_pool(0.72), timeout(0.68), latency(0.63), overload(0.63)
    [B] decoded: "The database connection pool is overloaded with timeouts..."
    fidelity: 0.432
```

**Performance (Nvidia free API, Jun 2026):**
| Metric | Value |
|--------|-------|
| CMTIP dialogue fidelity | 0.43 avg (72 concepts) |
| Ply→CMTIP decode fidelity | 0.49 avg (4 ML ops) |
| Cross-model agreement | 0.90 (Llama ↔ Gemma concept definitions) |
| Best single-message decoder | Gemma-2-2b (0.62) |

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

cmtip/
  cmtip.py              Core protocol: TensorPacket, CmtipBus, CCA adapter
  cmtip_chat.py         Working chat CLI (0.43 avg fidelity)
  ply_cmtip_bridge.py   Ply→CMTIP end-to-end pipeline
  cmtip_repl.py         Interactive REPL with concept memory
  cmtip_server.py       gRPC server (585 msg/s)
  cmtip_learn.py        Self-improving adapter training
  cmtip_memory.py       Semantic memory (conversation compression)
```

## v2.0 Features

### Training Infrastructure
```python
W1 = param(64, 32)
opt = optimizer('adam', 0.001, W1, b1, W2, b2)
step(loss, opt)  # one-call backward + update
```

### Kernel Fusion Compiler
```bash
$ python3 ply.py --ir examples/mlp.ply
[IR compiled with fusion in 1.4ms]
```

### Concept Memory (CMTIP)
The REPL accumulates concept tensors across turns, enabling multi-turn
conversations where later reconstructions reference earlier context:

```
Turn 3: "Thanks!" → remembers "increasing database connections" from Turn 2
```

## Built-in Functions

**Tensor Creation:** `randn` `zeros` `ones` `param` `eye` `arange`
**Shape:** `reshape` `permute` `transpose` `concat` `stack` `broadcast`
**Reductions:** `sum` `mean` `max` `min` `std`
**Activations:** `relu` `gelu` `sigmoid` `tanh` `softmax` `layernorm`
**Math:** `exp` `log` `sqrt` `abs` `sin` `cos` `clip` `where`
**Tensor Ops:** `matmul` `einsum`
**Training:** `grad` `backward` `zerograd` `optimizer` `step`
**I/O:** `save` `load` `print`

## Requirements

- Python 3.10+
- NumPy
- Optional: CuPy (NVIDIA GPU), MLX (Apple Silicon)
- CMTIP tools: Nvidia API key (free tier, set `NVIDIA_API_KEY` env var)

## Related

- [hmtl-kernel](https://github.com/lesterppo/hmtl-kernel) — Hardware-mapped tensor kernel (Rust)
- [CMTIP paper](cmtip/SPEC.md) — Protocol specification
- [Ply language spec](SPEC.md) — Full language reference

## License

MIT
