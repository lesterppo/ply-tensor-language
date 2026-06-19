# AGENTS.md — Ply Tensor-Native Language + CMTIP

## What This Is
Two systems that form one pipeline:
1. **Ply** — tensor-native programming language (3,600 lines Python)
2. **CMTIP** — cross-model tensor communication protocol

Together: compute in pure tensor algebra, then transmit results as concept
vectors between AI models — no text tokens needed.

## Quick Start
```bash
git clone https://github.com/lesterppo/ply-tensor-language
cd ply-tensor-language

# Ply
echo 'softmax(randn(1, 10), 1)' | python3 ply.py
python3 ply_tests.py                    # 17/17

# CMTIP (needs NVIDIA_API_KEY env var, free tier)
python3 cmtip/cmtip_chat.py             # 5-turn demo
python3 cmtip/ply_cmtip_bridge.py       # Full pipeline
python3 cmtip/cmtip_repl.py             # Interactive REPL
```

## Architecture
```
Ply (compute):
  source.ply → tokens.py → parser.py → ast_nodes.py → runtime.py
                                                  ↓
                              tensor.py (autodiff) → backend.py (NumPy/CuPy/MLX)
                                                  → optim.py (SGD/Adam/AdamW)
                                                  → ir.py (kernel fusion)

CMTIP (communicate):
  cmtip/
    cmtip.py              Core protocol (~1200 lines)
    cmtip_chat.py         Working chat CLI (0.43 avg fidelity, 72 concepts)
    ply_cmtip_bridge.py   Ply→CMTIP end-to-end pipeline
    cmtip_repl.py         Interactive REPL with concept memory
    cmtip_server.py       gRPC server (585 msg/s)
    cmtip_learn.py        Self-improving adapter training
    cmtip_memory.py       Semantic memory
```

## Key Invariants

### Ply
- Bindings (`:=`) are immutable — assigned once, never mutated
- No control flow — branching via `mask ? a : b`
- No loops — iteration via broadcasting + reduction
- Every operation has an analytical gradient
- `mean(x, dim)` keeps dims (returns same rank, fixed Jun 2026)
- 17/17 tests must pass before any PR

### CMTIP
- Cross-family paraphrase ceiling: cos_sim ~0.51
- Same-family ceiling: cos_sim ~0.87
- Concept-label projection (CCA): 0.64-0.89 depending on rank
- "Intercepted message" prompt framing produces best decode results
- Gemma-2-2b is strongest decoder on Nvidia free tier (0.62 single-message)
- CCA rank r=8 is Goldilocks for gamified cross-model comms

## Testing
```bash
python3 ply_tests.py                    # Ply core (17 tests)
python3 cmtip/cmtip.py                  # CMTIP core self-tests
python3 cmtip/cmtip_chat.py             # Integration test (needs API key)
```

## Pitfalls
- `=` is NOT `:=` — use `:=` for all Ply bindings
- `matmul(A,B)` is the builtin, but prefer `A @ B`
- nv-embed-v1 is coarse (poor inter-category separation)
- Matrix transpose: `W_target.T` required in CCA `project()` for cross-family
- CCA rank < 4 collapses concepts into 2D plane
- Only 2 of 5 tested models available on Nvidia free tier
