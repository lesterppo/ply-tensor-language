# CMTIP: Cross-Model Tensor Interoperability Protocol

## The LLM-Native Language Specification v0.1

### 1. Problem Statement

Current AI-to-AI communication is bottlenecked by human language:

```
Model A (thinking in R^d_A) → text tokens → Model B (translates back to R^d_B)
```

This roundtrip loses:
- **Precision**: Continuous thought → discrete tokens → continuous ≈ quantization error
- **Dimensions**: d-dimensional semantics → 1D token stream → d-dimensional reconstruction
- **Confidence**: Model certainty is discarded in text output
- **Ambiguity**: "hot" (temperature? attractiveness? stolen?) — text is inherently ambiguous
- **Speed**: Token-by-token generation is serial; tensor communication is parallel

**CMTIP replaces text with direct tensor exchange:**

```
Model A → embedding vector v ∈ R^d_A → W_{A→B} → v' ∈ R^d_B → Model B
```

### 2. Core Concepts

#### 2.1 Tensor Packet

The atomic unit of communication. Replaces a chat "message."

| Field | Type | Description |
|-------|------|-------------|
| source_id | string | Model identifier (e.g., "llama-3-8b") |
| target_id | string | Target model or "broadcast" |
| tensor | float32[n] | The embedding vector — the actual "message" |
| shape | int[] | Tensor dimensions |
| concept_tags | string[] | Optional semantic labels |
| confidence | float32 | Model's certainty in this vector |
| reply_to_seq | int | Sequence number this replies to |

#### 2.2 Cross-Model Adapter (W_{A→B})

A learned linear projection between two embedding spaces:

```
v_B = v_A · W_{A→B}    where W ∈ R^{d_A × d_B}
```

Learned via ordinary least squares on paired embeddings (same text, different models).

#### 2.3 Tensor Vocabulary

LLM-native "words" are continuous vectors on a semantic manifold. Operations:

| Operation | Text Equivalent | Tensor Equivalent |
|-----------|----------------|-------------------|
| Blending | "somewhat X, somewhat Y" | slerp(v_X, v_Y, α) |
| Analogy | "X is to Y as Z is to ?" | v_Z + (v_Y - v_X) |
| Difference | "getting worse" | v_critical - v_healthy |
| Similarity | "how related?" | cos(v_A, v_B) |

#### 2.4 Penalty Feedback

Bidirectional error correction: when Model A's output causes high entropy in Model B's attention, Model B emits a penalty tensor:

```
B_input = A_output ⊙ exp(-λ · H_B)    (Hadamard product)
```

This is the geometric equivalent of "I don't understand, please rephrase."

### 3. Wire Format

```
[4 bytes: header_len (uint32 LE)]
[N bytes: JSON header]
[M bytes: float32 tensor payload]
```

JSON header:
```json
{
  "magic": 1213417548,
  "src": "Llama-3-8B",
  "tgt": "DeepSeek-V3",
  "seq": 42,
  "ts": 1718123456.789,
  "dtype": 0,
  "shape": [384],
  "tags": ["temperature", "alert"],
  "conf": 0.92,
  "reply": -1
}
```

### 4. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     CMTIP Bus                            │
│                                                          │
│  ┌──────────┐   W_{A→B}    ┌──────────┐                │
│  │ Model A  │─────────────>│ Model B  │                │
│  │ d = 384  │              │ d = 768  │                │
│  └──────────┘              └──────────┘                │
│       │                         │                       │
│       │    ┌──────────┐         │                       │
│       └───>│ Adapter  │<────────┘                       │
│            │ Registry │                                 │
│            └──────────┘                                 │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ Penalty Monitor                                   │  │
│  │ · Tracks attention entropy per recipient          │  │
│  │ · Emits penalty tensors when H > threshold        │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 5. Production Path

1. **Replace synthetic backends** with real embedding models:
   - `pip install sentence-transformers` → local, offline
   - OpenAI `text-embedding-3-small` → API, 1536-dim
   - Cohere `embed-v3` → API, 1024-dim

2. **Deploy CMTIP Bus** as a service:
   - gRPC server for low-latency tensor routing
   - WebSocket for streaming model-to-model dialog
   - Redis pub/sub for multi-agent broadcast

3. **Learn adapters** from aligned corpora:
   - Wikipedia parallel sentences (same content, different languages → same semantics)
   - NIST MT eval sets
   - Custom domain-specific alignment data

4. **Extend vocabulary** with learned concept vectors:
   - Cluster embeddings to discover latent concepts
   - Use PCA/ICA to find semantic principal components
   - Train concept embeddings via contrastive learning

### 6. Relationship to HMTL Kernel Spec

The original HMTL OS kernel specification was a hardware-level vision.
CMTIP is the **practical, deployable today** realization of the same principle:

| HMTL Kernel (Spec) | CMTIP (This Implementation) |
|---|---|
| CXL 3.0 shared memory | gRPC/WebSocket transport |
| Triton Tensor Cores | NumPy/CuPy matrix ops |
| Zero-branch actuator | Linear adapter projection |
| PagedTensor HBM swap | Redis-backed vector store |
| NVLink P2P | HTTP/2 streaming |

Both share the core insight: **models should communicate in their native representation.**
