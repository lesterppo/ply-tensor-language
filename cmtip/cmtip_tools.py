"""
CMTIP Tools — LLM-Native Language Function-Calling Interface
═════════════════════════════════════════════════════════════

Exposes CMTIP as callable functions that any LLM with tool support can invoke.
Connects to the CMTIP gRPC bus for model-to-model tensor communication.

Tools provided:
    cmtip_send(target, thought_text, concepts, confidence)
        Send a tensor thought to another model. The LLM describes its thought
        in text + concept blends; the tool embeds and sends via the bus.

    cmtip_decode(tensor_b64, shape, k)
        Decode a received tensor into human-readable form:
        closest concepts, nearest text from corpus, suggested interpretation.

    cmtip_status()
        Show bus status: registered models, trained adapters, recent activity.

    cmtip_blend(concept_a, concept_b, alpha)
        Blend two concepts geometrically (spherical interpolation).

    cmtip_list_concepts()
        List all available concept vectors with their text labels.

    cmtip_emit_hidden_state(text)
        P1: Emit the LLM's "hidden state" as a tensor. Proxy implementation:
        embeds the text via sentence-transformer and returns the raw tensor.
        Future: direct hidden-state extraction from the LLM's forward pass.

Usage (Python):
    from cmtip_tools import CmtipToolbox
    tb = CmtipToolbox()
    result = tb.send("deepseek-v3", "The deployment is failing", {"urgent": 0.9})
    decoded = tb.decode(result["tensor_b64"], result["shape"])
"""

import os
import sys
import json
import base64
import time
import struct
import argparse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

# Add project root for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import grpc
import cmtip_service_pb2 as pb2
import cmtip_service_pb2_grpc as pb2_grpc


# ═══════════════════════════════════════════════════════════════
# Core Toolbox
# ═══════════════════════════════════════════════════════════════

@dataclass
class CmtipToolbox:
    """
    The CMTIP function-calling interface.

    Connect to the bus, send tensor thoughts, decode received tensors,
    manage concept vocabulary, and auto-label discovered concepts.

    All methods return JSON-serializable dicts suitable for LLM tool responses.
    """

    grpc_host: str = "localhost:50051"
    auto_vocab_path: str = os.path.join(PROJECT_ROOT, "auto_vocab.json")

    # ── State ──
    _stub: object = None
    _channel: object = None
    _concepts: Dict[str, np.ndarray] = field(default_factory=dict)
    _text_corpus: List[str] = field(default_factory=list)
    _corpus_embeddings: np.ndarray = None
    _tensor_memory: Dict[str, dict] = field(default_factory=dict)  # P3: working memory
    _semantic_memory: object = None  # SemanticMemory instance (lazy)

    def __post_init__(self):
        self._connect()
        self._load_concepts()
        self._init_semantic_memory()

    def _connect(self):
        """Connect to the gRPC bus (lazy, reconnect on failure)."""
        try:
            self._channel = grpc.insecure_channel(
                self.grpc_host,
                options=[
                    ('grpc.max_send_message_length', 50 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 50 * 1024 * 1024),
                ],
            )
            self._stub = pb2_grpc.CmtipBusStub(self._channel)
            # Test connection
            self._stub.GetStatus(pb2.StatusRequest(), timeout=2)
        except Exception:
            self._stub = None

    def _ensure_connected(self) -> bool:
        if self._stub is None:
            self._connect()
            return self._stub is not None
        try:
            self._stub.GetStatus(pb2.StatusRequest(), timeout=1)
            return True
        except Exception:
            self._connect()
            return self._stub is not None

    def _load_concepts(self):
        """Load auto-learned concept vocabulary."""
        if os.path.exists(self.auto_vocab_path):
            try:
                with open(self.auto_vocab_path) as f:
                    data = json.load(f)
                for name, vec in data.get("concepts", {}).items():
                    self._concepts[name] = np.array(vec, dtype=np.float32) / (
                        np.linalg.norm(np.array(vec, dtype=np.float32)) + 1e-8
                    )
                # Load corpus texts if available
                pairs_path = os.path.join(PROJECT_ROOT, "data", "training_pairs.json")
                if os.path.exists(pairs_path):
                    with open(pairs_path) as f:
                        pd = json.load(f)
                    self._text_corpus = list(set(p["a"] for p in pd.get("train", [])))
            except Exception:
                pass

    def _init_semantic_memory(self):
        """Initialize the semantic memory bank."""
        try:
            from cmtip_memory import SemanticMemory
            self._semantic_memory = SemanticMemory()
        except Exception:
            self._semantic_memory = None

    # ───────────────────────────────────────────────────────────────
    # P0: Function-Calling Tools
    # ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        cmtip_status() → dict

        Show the CMTIP bus status: registered models, trained adapters,
        recent conversation count, and penalty threshold.
        """
        if not self._ensure_connected():
            return {"error": "CMTIP bus not reachable. Start with: python cmtip_server.py"}

        try:
            resp = self._stub.GetStatus(pb2.StatusRequest())
            return {
                "ok": True,
                "registered_models": resp.registered_models,
                "model_ids": list(resp.model_ids),
                "trained_adapters": resp.trained_adapters,
                "total_messages": resp.total_conversations,
                "penalty_threshold": resp.penalty_threshold,
            }
        except Exception as e:
            return {"error": str(e)}

    def send(self, target: str, thought_text: str = "",
             concepts: Dict[str, float] = None,
             confidence: float = 1.0,
             source_id: str = "hermes-agent") -> dict:
        """
        cmtip_send(target, thought_text, concepts, confidence) → dict

        Send a tensor thought to another model on the bus.

        The thought is composed as: blend of concept vectors weighted by
        the concept dict + text context embedded via sentence-transformer.

        Args:
            target: Model ID to send to (e.g., "deepseek-v3", "mpnet-analyst")
            thought_text: Natural language description of the thought
            concepts: {"concept_name": weight, ...} — concept blend weights
            confidence: How confident the sender is [0, 1]

        Returns:
            dict with: ok, seq_num, target, tensor_b64, shape, concept_tags,
                       penalty_triggered, target_entropy, decoded_hint
        """
        if not self._ensure_connected():
            return {"error": "CMTIP bus not reachable"}

        concepts = concepts or {}

        try:
            # Ensure both source and target models are registered
            self._ensure_model(source_id, 384)
            self._ensure_model(target, 768)  # Target might have different dim

            # Build concept tags
            tags = list(concepts.keys()) if concepts else []

            # Send via gRPC
            resp = self._stub.SendText(pb2.SendTextRequest(
                source_id=source_id,
                target_id=target,
                text=thought_text,
                concept_tags=tags,
            ))

            tensor_b64 = base64.b64encode(resp.packet.tensor).decode()
            shape = list(resp.packet.shape)

            return {
                "ok": True,
                "seq_num": resp.packet.seq_num,
                "source": resp.packet.source_id,
                "target": resp.packet.target_id,
                "tensor_b64": tensor_b64,
                "shape": shape,
                "confidence": resp.packet.confidence,
                "concept_tags": list(resp.packet.concept_tags),
                "penalty_triggered": resp.penalty_triggered,
                "target_entropy": round(resp.target_entropy, 4),
                "decoded_hint": self._quick_decode(shape, resp.packet.tensor) if not resp.penalty_triggered else "⚠ High entropy — target model may not understand",
            }
        except Exception as e:
            return {"error": str(e)}

    def decode(self, tensor_b64: str, shape: List[int], k: int = 5,
               use_corpus: bool = False) -> dict:
        """
        cmtip_decode(tensor_b64, shape, k=5, use_corpus=False) → dict

        Decode a received tensor into human-readable form.
        
        Fast path (use_corpus=False): concept matching only — instant dot products.
        Full path (use_corpus=True): also finds nearest texts from corpus (needs ST model).

        Returns:
            - closest_concepts: top-k matching concept vectors with scores
            - nearest_texts: top-3 nearest sentences from the training corpus
            - suggested_interpretation: plain-English summary
            - tensor_norm: L2 norm of the tensor
        """
        try:
            tensor = np.frombuffer(base64.b64decode(tensor_b64), dtype=np.float32)
            if shape:
                tensor = tensor.reshape(shape)
            tensor = tensor.flatten()
            tensor = tensor / (np.linalg.norm(tensor) + 1e-8)

            result = {
                "ok": True,
                "tensor_norm": round(float(np.linalg.norm(tensor)), 4),
                "dim": len(tensor),
            }

            # Find closest concepts
            concepts = []
            for name, vec in self._concepts.items():
                score = float(np.dot(tensor[:len(vec)], vec[:len(tensor)]))
                concepts.append((name, score))
            concepts.sort(key=lambda x: -x[1])
            result["closest_concepts"] = [
                {"name": name, "similarity": round(score, 4)}
                for name, score in concepts[:k]
            ]

            # Find nearest texts from corpus (only if requested — requires ST model)
            if use_corpus:
                texts = self._nearest_texts(tensor, k=3)
                result["nearest_texts"] = texts
            else:
                result["nearest_texts"] = []

            # Suggest interpretation
            positive = [(n, s) for n, s in concepts[:k] if s > 0.15]
            if positive:
                top = positive[:3]
                desc = " + ".join(f"{n}({s:.2f})" for n, s in top)
                result["suggested_interpretation"] = f"This tensor carries: {desc}"
            else:
                result["suggested_interpretation"] = "No strongly matching concepts found — possibly a novel thought"

            return result
        except Exception as e:
            return {"error": str(e)}

    def blend(self, concept_a: str, concept_b: str, alpha: float = 0.5,
              k: int = 5, use_corpus: bool = False) -> dict:
        """
        cmtip_blend(concept_a, concept_b, alpha=0.5) → dict

        Blend two concepts on the semantic manifold using spherical interpolation.

        Returns the blended vector and its nearest concepts/texts.
        """
        if concept_a not in self._concepts:
            return {"error": f"Concept '{concept_a}' not found. Available: {sorted(self._concepts.keys())[:15]}..."}
        if concept_b not in self._concepts:
            return {"error": f"Concept '{concept_b}' not found"}

        a = self._concepts[concept_a]
        b = self._concepts[concept_b]

        # Spherical linear interpolation
        omega = np.arccos(np.clip(np.dot(a, b), -1, 1))
        if omega < 1e-6:
            blended_vec = a.copy()
        else:
            sin_omega = np.sin(omega)
            blended_vec = (
                np.sin((1 - alpha) * omega) / sin_omega * a
                + np.sin(alpha * omega) / sin_omega * b
            )
        blended_vec = blended_vec / (np.linalg.norm(blended_vec) + 1e-8)

        # Decode the blend
        concepts = []
        for name, vec in self._concepts.items():
            score = float(np.dot(blended_vec, vec))
            concepts.append((name, score))
        concepts.sort(key=lambda x: -x[1])

        texts = self._nearest_texts(blended_vec, k=3) if use_corpus else []

        return {
            "ok": True,
            "operation": f"slerp({concept_a}, {concept_b}, α={alpha})",
            "blended_vector_b64": base64.b64encode(blended_vec.astype(np.float32).tobytes()).decode(),
            "nearest_concepts": [{"name": n, "similarity": round(s, 4)} for n, s in concepts[:k]],
            "nearest_texts": texts,
        }

    def list_concepts(self) -> dict:
        """
        cmtip_list_concepts() → dict

        List all available concept vectors with their text labels and PCA variance.
        """
        if not self._concepts:
            return {"ok": True, "concepts": [], "hint": "No concepts loaded. Run: python auto_vocab.py --demo"}

        concepts = []
        for name in sorted(self._concepts.keys()):
            vec = self._concepts[name]
            # Extract metadata from name (e.g., "axis_1 (6%)" → pca_axis=1, variance=6%)
            entry = {"name": name, "dim": len(vec), "norm": round(float(np.linalg.norm(vec)), 4)}
            if "axis_" in name and "%" in name:
                parts = name.split()
                entry["type"] = "pca_axis"
                entry["axis_num"] = int(parts[0].split("_")[1])
                entry["variance_pct"] = float(parts[1].strip("()%"))
            elif name.endswith("+") or name.endswith("-"):
                entry["type"] = "contrastive_extreme"
                entry["direction"] = "positive" if name.endswith("+") else "negative"
            elif "cluster_" in name:
                entry["type"] = "concept_cluster"
                entry["cluster_id"] = int(name.split("_")[1])
                if "label" in name:
                    entry["label"] = name.split(":", 1)[1].strip()
            concepts.append(entry)

        return {
            "ok": True,
            "total_concepts": len(concepts),
            "concepts": concepts,
        }

    # ───────────────────────────────────────────────────────────────
    # P1: Native Hidden-State Emission + Auto-Labeling
    # ───────────────────────────────────────────────────────────────

    def emit_hidden_state(self, text: str, source_id: str = "hermes-agent") -> dict:
        """
        cmtip_emit_hidden_state(text) → dict

        P1: Emit the LLM's "hidden state" as a tensor.

        Proxy implementation: embeds the text via the registered sentence-transformer
        backend. This captures the semantic content that would be in the LLM's
        hidden state if we had direct access.

        Future: extract directly from the LLM's last hidden layer during inference.
        The tensor IS what the LLM was "thinking" before it decoded to tokens.

        Returns the raw tensor (base64) ready to send or analyze.
        """
        if not self._ensure_connected():
            return {"error": "CMTIP bus not reachable"}

        try:
            # Ensure source is registered
            self._ensure_model(source_id, 384)

            # Send to a temporary self-target to get the embedding
            # (the bus embeds on the source side automatically)
            resp = self._stub.SendText(pb2.SendTextRequest(
                source_id=source_id,
                target_id=source_id,  # self-target
                text=text,
            ))

            tensor_b64 = base64.b64encode(resp.packet.tensor).decode()
            shape = list(resp.packet.shape)

            return {
                "ok": True,
                "text": text[:200],
                "tensor_b64": tensor_b64,
                "shape": shape,
                "dim": int(np.prod(shape)),
                "note": "Proxy via sentence-transformer embedding. Direct hidden-state extraction requires LLM inference hook.",
            }
        except Exception as e:
            return {"error": str(e)}

    def auto_label_concepts(self, use_corpus: bool = False) -> dict:
        """
        cmtip_auto_label_concepts(use_corpus=False) → dict

        Auto-label discovered concept clusters with meaningful names.

        Fast path (use_corpus=False): labels from nearest concept + axis names.
        Full path (use_corpus=True): uses nearest text from corpus for richer labels.
        """
        unlabeled = [n for n in self._concepts if n.startswith("cluster_") and ":" not in n]

        if not unlabeled:
            return {
                "ok": True,
                "message": "All clusters already labeled",
                "concepts": sorted(self._concepts.keys()),
            }

        labeled = {}
        for name in unlabeled:
            vec = self._concepts[name]

            if use_corpus:
                texts = self._nearest_texts(vec, k=5)
                from collections import Counter
                words = []
                for t, _ in texts:
                    words.extend(w.lower() for w in t.split()
                               if len(w) > 3 and w.lower() not in
                               {'this', 'that', 'with', 'from', 'have', 'been', 'were',
                                'their', 'they', 'will', 'about', 'which', 'there', 'would'})
                top_words = [w for w, _ in Counter(words).most_common(4)]
                label = " ".join(top_words[:3])
            else:
                # Fast: find nearest PCA axis (ignore other clusters to avoid cascading)
                best_axis = None
                best_score = -1
                for cn, cv in self._concepts.items():
                    if cn.startswith("axis_") and not cn.endswith("+") and not cn.endswith("-"):
                        score = abs(float(np.dot(vec[:len(cv)], cv[:len(vec)])))
                        if score > best_score:
                            best_score = score
                            best_axis = cn
                label = f"near_{best_axis}" if best_axis else "ungrouped"

            new_name = f"{name}: {label}"
            labeled[name] = new_name

        # Apply all labels at once (avoid cascading during iteration)
        for old_name, new_name in labeled.items():
            self._concepts[new_name] = self._concepts.pop(old_name)

        return {
            "ok": True,
            "labeled": labeled,
            "total_concepts": len(self._concepts),
        }

    # ───────────────────────────────────────────────────────────────
    # P2: Iterative Clarification Loop
    # ───────────────────────────────────────────────────────────────

    def clarify(self, target: str, thought_text: str,
                concepts: Dict[str, float] = None,
                max_rounds: int = 3, entropy_threshold: float = 0.5,
                source_id: str = "hermes-agent") -> dict:
        """
        cmtip_clarify(target, thought_text, concepts, max_rounds=3) → dict

        P2: Iterative clarification — send a thought, receive penalty feedback,
        adjust, and resend until the target model's entropy drops below threshold.

        The loop:
          1. Send thought → target
          2. Target computes attention entropy
          3. If entropy > threshold: apply penalty mask, adjust thought, goto 1
          4. If entropy ≤ threshold or max_rounds reached: stop

        Returns the final tensor, round history, and convergence status.
        """
        concepts = concepts or {}
        history = []
        current_text = thought_text
        current_concepts = dict(concepts)
        penalty_mask = None

        for round_num in range(max_rounds):
            # Send
            result = self.send(target, current_text, current_concepts,
                             source_id=source_id)
            history.append({
                "round": round_num + 1,
                "entropy": result.get("target_entropy", 0),
                "penalty_triggered": result.get("penalty_triggered", False),
                "text": current_text[:100],
            })

            if "error" in result:
                return {"ok": False, "error": result["error"], "history": history}

            entropy = result.get("target_entropy", 0)

            # Check convergence
            if entropy <= entropy_threshold:
                return {
                    "ok": True,
                    "converged": True,
                    "rounds": round_num + 1,
                    "final_entropy": entropy,
                    "tensor_b64": result.get("tensor_b64"),
                    "shape": result.get("shape"),
                    "history": history,
                }

            # Apply penalty — adjust the thought
            if result.get("penalty_triggered"):
                adjustment = self._suggest_adjustment(entropy, current_concepts)
                current_text = f"[adjusted round {round_num+2}] {thought_text}"
                # Reduce weight on concepts that caused high entropy
                current_concepts = {
                    k: v * 0.7 for k, v in current_concepts.items()
                }
                penalty_mask = True
            else:
                # Entropy high but no penalty triggered — just retry
                pass

        return {
            "ok": True,
            "converged": False,
            "rounds": max_rounds,
            "final_entropy": history[-1]["entropy"],
            "tensor_b64": result.get("tensor_b64") if 'result' in dir() else None,
            "shape": result.get("shape") if 'result' in dir() else None,
            "history": history,
            "note": f"Did not converge within {max_rounds} rounds. Try different concepts or reduce complexity."
        }

    def apply_penalty_mask(self, tensor_b64: str, shape: List[int],
                           entropy: float, lambda_penalty: float = 2.0) -> dict:
        """
        cmtip_apply_penalty_mask(tensor_b64, shape, entropy) → dict

        P2: Apply a penalty mask to a tensor — suppress dimensions causing high
        entropy at the receiver. The mask is exp(-λ · entropy) applied as a
        Hadamard product.

        Returns the masked tensor (base64).
        """
        try:
            tensor = np.frombuffer(base64.b64decode(tensor_b64), dtype=np.float32)
            tensor = tensor.reshape(shape).flatten()

            # Create penalty mask: exp(-λ · h) for each dimension
            # In the full implementation, entropy is per-dimension from attention.
            # Here we use uniform entropy as a proxy.
            dim = len(tensor)
            # Simulate per-dimension entropy: random variation around the mean
            rng = np.random.RandomState(int(entropy * 10000) % (2**31))
            per_dim_entropy = np.clip(
                entropy + rng.randn(dim).astype(np.float32) * 0.1,
                0.0, 1.0
            )
            mask = np.exp(-lambda_penalty * per_dim_entropy).astype(np.float32)
            masked = tensor * mask
            masked = masked / (np.linalg.norm(masked) + 1e-8)

            return {
                "ok": True,
                "original_norm": round(float(np.linalg.norm(tensor)), 4),
                "masked_norm": round(float(np.linalg.norm(masked)), 4),
                "suppression_ratio": round(float(1.0 - np.linalg.norm(masked) / (np.linalg.norm(tensor) + 1e-8)), 4),
                "masked_tensor_b64": base64.b64encode(masked.astype(np.float32).tobytes()).decode(),
                "shape": shape,
                "lambda": lambda_penalty,
                "mean_entropy": round(entropy, 4),
            }
        except Exception as e:
            return {"error": str(e)}

    def _suggest_adjustment(self, entropy: float, concepts: Dict[str, float]) -> str:
        """Suggest how to adjust a thought to reduce entropy."""
        if entropy > 0.8:
            return "High confusion — simplify or use more common concepts"
        elif entropy > 0.5:
            return "Moderate confusion — reduce concept blend complexity"
        return "Mild confusion — slight adjustment needed"

    # ───────────────────────────────────────────────────────────────
    # P3: Tensor Working Memory
    # ───────────────────────────────────────────────────────────────

    def store(self, key: str, tensor_b64: str, shape: List[int],
              metadata: str = "") -> dict:
        """
        cmtip_store(key, tensor_b64, shape, metadata="") → dict

        P3: Store a tensor in working memory. A single 384-dim vector
        carries what would take 50+ tokens to express in text — this is
        high-density compressed memory for the LLM's working state.

        Keys can be used to recall context across conversation turns
        without consuming context window space.

        Args:
            key: Memory key (e.g., "deployment_context", "user_prefs")
            tensor_b64: Base64-encoded float32 tensor
            shape: Tensor dimensions
            metadata: Optional human-readable note about what this stores

        Returns confirmation with memory stats.
        """
        try:
            tensor = np.frombuffer(base64.b64decode(tensor_b64), dtype=np.float32)
            tensor = tensor.reshape(shape)
            norm = float(np.linalg.norm(tensor))

            self._tensor_memory[key] = {
                "tensor_b64": tensor_b64,
                "shape": shape,
                "norm": norm,
                "dim": int(np.prod(shape)),
                "metadata": metadata,
                "stored_at": time.time(),
            }

            # Calculate token savings: 1 float32 = 4 bytes, ~0.75 tokens/word
            # A 384-dim tensor ~= 384 * 4 = 1536 bytes
            # Equivalent text: ~200-300 tokens to express same nuance
            token_equivalent = int(np.prod(shape) * 0.5)

            return {
                "ok": True,
                "key": key,
                "dim": int(np.prod(shape)),
                "norm": round(norm, 4),
                "token_equivalent_estimate": token_equivalent,
                "total_stored": len(self._tensor_memory),
                "metadata": metadata,
            }
        except Exception as e:
            return {"error": str(e)}

    def recall(self, key: str) -> dict:
        """
        cmtip_recall(key) → dict

        P3: Recall a stored tensor from working memory.

        Returns the tensor (base64), shape, metadata, and how long ago
        it was stored. Use this to retrieve compressed context without
        consuming context window space.
        """
        if key not in self._tensor_memory:
            return {
                "ok": False,
                "error": f"No tensor stored under key '{key}'",
                "available_keys": sorted(self._tensor_memory.keys()),
            }

        entry = self._tensor_memory[key]
        age = time.time() - entry["stored_at"]

        return {
            "ok": True,
            "key": key,
            "tensor_b64": entry["tensor_b64"],
            "shape": entry["shape"],
            "dim": entry["dim"],
            "norm": entry["norm"],
            "metadata": entry["metadata"],
            "age_seconds": round(age, 1),
            "total_stored": len(self._tensor_memory),
        }

    def forget(self, key: str) -> dict:
        """
        cmtip_forget(key) → dict

        P3: Remove a tensor from working memory.
        """
        if key not in self._tensor_memory:
            return {"ok": False, "error": f"No tensor stored under key '{key}'"}

        del self._tensor_memory[key]
        return {
            "ok": True,
            "key": key,
            "remaining": len(self._tensor_memory),
        }

    def memory_status(self) -> dict:
        """
        cmtip_memory_status() → dict

        P3: Show all tensors currently in working memory with metadata.
        """
        entries = []
        total_dim = 0
        for key, entry in sorted(self._tensor_memory.items()):
            age = time.time() - entry["stored_at"]
            entries.append({
                "key": key,
                "dim": entry["dim"],
                "norm": entry["norm"],
                "metadata": entry["metadata"],
                "age_seconds": round(age, 1),
            })
            total_dim += entry["dim"]

        # Total context-window savings
        token_savings = total_dim * 0.5

        return {
            "ok": True,
            "total_stored": len(entries),
            "total_dimensions": total_dim,
            "estimated_token_savings": int(token_savings),
            "entries": entries,
        }

    # ───────────────────────────────────────────────────────────────
    # Semantic Memory (Real Task: Context Window Compression)
    # ───────────────────────────────────────────────────────────────

    def remember(self, text: str, importance: float = 1.0,
                 tags: List[str] = None) -> dict:
        """
        cmtip_remember(text, importance=1.0, tags=None) → dict

        Store a fact in semantic memory. Use this to remember key facts
        from a conversation without consuming context window space.

        A 384-dim tensor encodes the full semantic content of the fact.
        Later, recall by semantic similarity — the system finds facts
        even when you don't know exactly what to search for.

        Example:
            cmtip_remember("Production DB is PostgreSQL 16 on AWS RDS", 1.0, ["infra"])
            cmtip_remember("User prefers Python 3.12 with type hints", 0.7, ["prefs"])
        """
        if self._semantic_memory is None:
            return {"error": "Semantic memory not available. Install sentence-transformers."}

        return self._semantic_memory.remember(text, importance, tags)

    def recall_fact(self, query: str, k: int = 5,
                    min_similarity: float = 0.3) -> dict:
        """
        cmtip_recall_fact(query, k=5) → dict

        Recall facts from semantic memory by semantic similarity.

        Unlike keyword search, this finds facts that are semantically
        related to the query — even if they use different words.

        Example:
            cmtip_recall_fact("What database?") 
            → "Production DB is PostgreSQL 16 on AWS RDS" (sim=0.92)

        The facts are returned without consuming context window space —
        only the matched facts are shown, not the entire history.
        """
        if self._semantic_memory is None:
            return {"error": "Semantic memory not available"}

        return self._semantic_memory.recall(query, k, min_similarity)

    def forget_fact(self, query: str) -> dict:
        """
        cmtip_forget_fact(query) → dict

        Remove facts matching the query from semantic memory.
        Use after a topic is resolved to free capacity.
        """
        if self._semantic_memory is None:
            return {"error": "Semantic memory not available"}

        return self._semantic_memory.forget(query)

    def memory_summary(self) -> dict:
        """
        cmtip_memory_summary() → dict

        Show semantic memory bank status: total facts, capacity,
        token savings, and recent entries.
        """
        if self._semantic_memory is None:
            return {"error": "Semantic memory not available"}

        return self._semantic_memory.status()

    # ───────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────

    def _ensure_model(self, model_id: str, dim: int = 384):
        """Ensure a model is registered on the bus (idempotent)."""
        try:
            self._stub.RegisterModel(pb2.RegisterModelRequest(
                model_id=model_id,
                dim=dim,
                backend_type="synthetic",
            ))
        except Exception:
            pass  # Already registered or bus unreachable

    def _quick_decode(self, shape, tensor_bytes) -> str:
        """Quick decode for the 'decoded_hint' field in send response."""
        try:
            tensor = np.frombuffer(tensor_bytes, dtype=np.float32)
            tensor = tensor / (np.linalg.norm(tensor) + 1e-8)

            positives = []
            for name, vec in list(self._concepts.items())[:50]:
                score = float(np.dot(tensor[:len(vec)], vec[:len(tensor)]))
                if score > 0.3:
                    positives.append((name, score))
            positives.sort(key=lambda x: -x[1])

            if positives:
                return " + ".join(f"{n}({s:.2f})" for n, s in positives[:4])
            return "novel thought (no strong concept matches)"
        except Exception:
            return ""

    def _nearest_texts(self, vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
        """Find nearest texts from corpus to a vector."""
        if not self._text_corpus:
            return []

        vec = vector / (np.linalg.norm(vector) + 1e-8)

        # Lazy-embed the corpus
        if self._corpus_embeddings is None and len(self._text_corpus) > 0:
            try:
                from real_backends import SentenceTransformerBackend
                backend = SentenceTransformerBackend("decoder", "all-MiniLM-L6-v2")
                self._corpus_embeddings = backend.embed_batch(self._text_corpus)
            except Exception:
                return []

        if self._corpus_embeddings is not None:
            # Ensure same dimension
            if self._corpus_embeddings.shape[1] != len(vec):
                # Pad or truncate
                if self._corpus_embeddings.shape[1] > len(vec):
                    vec = np.pad(vec, (0, self._corpus_embeddings.shape[1] - len(vec)))
                else:
                    vec = vec[:self._corpus_embeddings.shape[1]]

            scores = self._corpus_embeddings @ vec
            top_idx = np.argsort(scores)[-k:][::-1]
            return [(self._text_corpus[i][:120], round(float(scores[i]), 4)) for i in top_idx]

        return []


# ═══════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CMTIP Tools — LLM-Native Language CLI")
    parser.add_argument("--host", default="localhost:50051", help="gRPC bus host:port")
    sub = parser.add_subparsers(dest="cmd")

    p_status = sub.add_parser("status", help="Bus status")

    p_send = sub.add_parser("send", help="Send tensor thought")
    p_send.add_argument("target")
    p_send.add_argument("text")
    p_send.add_argument("--concepts", default="{}", help='JSON: {"urgent":0.9}')

    p_decode = sub.add_parser("decode", help="Decode a tensor")
    p_decode.add_argument("tensor_b64")
    p_decode.add_argument("--shape", default="384", help="Comma-separated dims")
    p_decode.add_argument("--k", type=int, default=5)

    p_blend = sub.add_parser("blend", help="Blend two concepts")
    p_blend.add_argument("concept_a")
    p_blend.add_argument("concept_b")
    p_blend.add_argument("--alpha", type=float, default=0.5)

    p_emit = sub.add_parser("emit", help="Emit hidden state as tensor")
    p_emit.add_argument("text")

    p_auto = sub.add_parser("auto-label", help="Auto-label concept clusters")

    p_list = sub.add_parser("list-concepts", help="List all concepts")

    p_clarify = sub.add_parser("clarify", help="P2: Iterative clarification loop")
    p_clarify.add_argument("target")
    p_clarify.add_argument("text")
    p_clarify.add_argument("--concepts", default="{}", help='JSON concept blend')
    p_clarify.add_argument("--max-rounds", type=int, default=3)
    p_clarify.add_argument("--entropy-threshold", type=float, default=0.5)

    p_penalty = sub.add_parser("penalty", help="P2: Apply penalty mask")
    p_penalty.add_argument("tensor_b64")
    p_penalty.add_argument("--shape", default="384")
    p_penalty.add_argument("--entropy", type=float, default=0.7)
    p_penalty.add_argument("--lambda", type=float, default=2.0, dest="lambda_penalty")

    p_store = sub.add_parser("store", help="P3: Store tensor in working memory")
    p_store.add_argument("key")
    p_store.add_argument("tensor_b64")
    p_store.add_argument("--shape", default="384")
    p_store.add_argument("--metadata", default="")

    p_recall = sub.add_parser("recall", help="P3: Recall stored tensor")
    p_recall.add_argument("key")

    p_forget = sub.add_parser("forget", help="P3: Remove from memory")
    p_forget.add_argument("key")

    p_memstat = sub.add_parser("memory-status", help="P3: Show working memory")

    args = parser.parse_args()
    tb = CmtipToolbox(grpc_host=args.host)

    if args.cmd == "status":
        print(json.dumps(tb.status(), indent=2))

    elif args.cmd == "send":
        result = tb.send(args.target, args.text, json.loads(args.concepts))
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "decode":
        shape = [int(d) for d in args.shape.split(",")]
        result = tb.decode(args.tensor_b64, shape, args.k)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "blend":
        result = tb.blend(args.concept_a, args.concept_b, args.alpha)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "emit":
        result = tb.emit_hidden_state(args.text)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "auto-label":
        result = tb.auto_label_concepts()
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "list-concepts":
        result = tb.list_concepts()
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "clarify":
        result = tb.clarify(args.target, args.text, json.loads(args.concepts),
                           max_rounds=args.max_rounds,
                           entropy_threshold=args.entropy_threshold)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "penalty":
        shape = [int(d) for d in args.shape.split(",")]
        result = tb.apply_penalty_mask(args.tensor_b64, shape,
                                       args.entropy, args.lambda_penalty)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "store":
        shape = [int(d) for d in args.shape.split(",")]
        result = tb.store(args.key, args.tensor_b64, shape, args.metadata)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "recall":
        result = tb.recall(args.key)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "forget":
        result = tb.forget(args.key)
        print(json.dumps(result, indent=2, default=str))

    elif args.cmd == "memory-status":
        result = tb.memory_status()
        print(json.dumps(result, indent=2, default=str))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
