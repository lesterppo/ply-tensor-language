"""
CMTIP Self-Improving Language
══════════════════════════════

The LLM-native language that learns from every exchange.

Instead of static adapters trained once offline, the self-improving bus:
  1. Updates adapter weights online from every successful exchange
  2. Reinforces concepts that work, decays unused ones  
  3. Learns routing strategies from success/failure patterns
  4. Adjusts from penalty feedback — learns what causes confusion

This turns CMTIP from a protocol into a learning system.

Architecture:
  ┌──────────────────────────────────────────────────────────────┐
  │                   SelfImprovingBus                            │
  │                                                               │
  │  Exchange: A → tensor → B                                     │
  │       │                                                       │
  │       ├── Success (low entropy): reinforce adapter weights    │
  │       │   + reinforce concepts used                           │
  │       │   + update routing success score                      │
  │       │                                                       │
  │       └── Failure (high entropy / penalty):                   │
  │           + identify confusing dimensions                     │
  │           + dampen adapter weights for those dimensions       │
  │           + reduce concept weights that triggered confusion   │
  │           + update routing failure score                      │
  │                                                               │
  │  Over time:                                                   │
  │    adapter improves   → higher cos_sim, lower entropy         │
  │    concepts sharpen   → used ones strengthen, bad ones fade   │
  │    routing optimizes  → auto-selects best target model        │
  └──────────────────────────────────────────────────────────────┘

Usage:
    bus = SelfImprovingBus()
    bus.register_model(model_a)
    bus.register_model(model_b)
    
    # Each exchange improves the system:
    result = bus.exchange("MiniLM", "MPNet", "urgent deployment issue",
                          concepts={"urgent": 0.9, "technical": 0.7})
    
    # The adapter, concepts, and routing all improve from this exchange.
    bus.stats()  # Show learning progress
"""

import numpy as np
import time
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import (
    CmtipBus, TensorPacket, CrossModelAdapter,
    SyntheticEmbeddingBackend, TensorVocabulary,
    compute_attention_entropy, create_penalty_tensor,
)


# ═══════════════════════════════════════════════════════════════
# Exchange Record — memory of every interaction
# ═══════════════════════════════════════════════════════════════

@dataclass
class ExchangeRecord:
    """Complete record of one tensor exchange for learning."""
    source_id: str
    target_id: str
    seq_num: int
    timestamp: float
    source_tensor: np.ndarray      # Before projection (in source space)
    projected_tensor: np.ndarray   # After projection (in target space)
    concept_tags: List[str]
    concepts_used: Dict[str, float]
    target_entropy: float
    penalty_triggered: bool
    success: bool                   # entropy <= threshold
    adapter_weights_snapshot: np.ndarray = None

    def is_useful_for_training(self) -> bool:
        """Was this exchange informative enough for online learning?"""
        return not self.penalty_triggered and self.target_entropy < 0.6


# ═══════════════════════════════════════════════════════════════
# Concept Reinforcement System
# ═══════════════════════════════════════════════════════════════

@dataclass 
class ConceptReinforcement:
    """
    Tracks concept usage and success. Concepts that lead to successful
    exchanges get reinforced; unused concepts decay; confusing concepts
    get penalized.
    """
    # Per-concept stats
    usage_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    success_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    confusion_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Decay parameters
    reinforcement_rate: float = 0.1     # How much to strengthen on success
    decay_rate: float = 0.01            # How much unused concepts decay per step
    penalty_strength: float = 0.3       # How much to penalize on confusion

    def record_success(self, concepts: Dict[str, float]):
        """Record that these concept weights led to a successful exchange."""
        for name, weight in concepts.items():
            self.usage_count[name] += 1
            self.success_count[name] += 1

    def record_failure(self, concepts: Dict[str, float]):
        """Record that these concept weights led to confusion."""
        for name, weight in concepts.items():
            self.usage_count[name] += 1
            self.confusion_count[name] += 1

    def get_adjusted_weight(self, name: str, base_weight: float) -> float:
        """
        Get a reinforcement-adjusted weight for a concept.
        
        Successful concepts get boosted, confusing ones get dampened.
        """
        if name not in self.usage_count or self.usage_count[name] == 0:
            return base_weight

        success_rate = self.success_count[name] / max(1, self.usage_count[name])
        confusion_rate = self.confusion_count[name] / max(1, self.usage_count[name])

        # Boost successful concepts, penalize confusing ones
        adjusted = base_weight * (1.0 + self.reinforcement_rate * success_rate 
                                  - self.penalty_strength * confusion_rate)
        return max(0.01, min(2.0, adjusted))

    def get_best_concepts(self, k: int = 10) -> List[Tuple[str, float]]:
        """Return the k best-performing concepts by success rate."""
        scored = []
        for name in self.usage_count:
            if self.usage_count[name] >= 3:  # Minimum usage for reliability
                success_rate = self.success_count[name] / self.usage_count[name]
                scored.append((name, success_rate))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def decay_unused(self, active_concepts: set):
        """Decay concepts that weren't used in this exchange."""
        for name in list(self.usage_count.keys()):
            if name not in active_concepts:
                self.usage_count[name] = max(0, self.usage_count[name] - self.decay_rate)

    def stats(self) -> dict:
        """Return reinforcement statistics."""
        if not self.usage_count:
            return {"total_concepts": 0}

        success_rates = []
        for name in self.usage_count:
            if self.usage_count[name] > 0:
                success_rates.append(self.success_count[name] / self.usage_count[name])

        return {
            "total_concepts": len(self.usage_count),
            "avg_success_rate": round(np.mean(success_rates), 3) if success_rates else 0,
            "top_performers": self.get_best_concepts(5),
            "most_confusing": sorted(
                [(n, self.confusion_count[n] / max(1, self.usage_count[n]))
                 for n in self.usage_count if self.confusion_count[n] > 0],
                key=lambda x: -x[1]
            )[:3],
        }


# ═══════════════════════════════════════════════════════════════
# Routing Optimizer
# ═══════════════════════════════════════════════════════════════

@dataclass
class RoutingOptimizer:
    """
    Learns which target model is best for which type of query.
    
    Tracks success/failure per (source, target) pair and per concept domain.
    Auto-routes to the model with the best track record for similar queries.
    """
    # Per-pair success tracking
    pair_successes: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    pair_failures: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    
    # Per-concept-domain tracking (which target handles which concepts best)
    concept_target_scores: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(lambda: 0.5)
    )

    def record(self, source: str, target: str, success: bool, concepts: List[str]):
        """Record an exchange outcome for routing optimization."""
        key = (source, target)
        if success:
            self.pair_successes[key] += 1
            # Boost concept-target association
            for c in concepts:
                self.concept_target_scores[(c, target)] = min(
                    1.0, self.concept_target_scores[(c, target)] + 0.05
                )
        else:
            self.pair_failures[key] += 1
            for c in concepts:
                self.concept_target_scores[(c, target)] = max(
                    0.1, self.concept_target_scores[(c, target)] - 0.05
                )

    def best_target(self, source: str, candidates: List[str],
                    concepts: List[str] = None) -> str:
        """
        Select the best target model for this source and concepts.
        
        Uses a combination of pair success rate and concept-domain fit.
        """
        best = None
        best_score = -1.0

        for target in candidates:
            key = (source, target)
            total = self.pair_successes[key] + self.pair_failures[key]
            
            if total == 0:
                pair_score = 0.5  # Unknown, neutral
            else:
                pair_score = self.pair_successes[key] / total

            # Add concept-domain score
            if concepts:
                concept_score = np.mean([
                    self.concept_target_scores[(c, target)]
                    for c in concepts
                ])
                score = 0.6 * pair_score + 0.4 * concept_score
            else:
                score = pair_score

            if score > best_score:
                best_score = score
                best = target

        return best

    def stats(self) -> dict:
        """Return routing statistics."""
        pairs = []
        for (src, tgt), succ in self.pair_successes.items():
            total = succ + self.pair_failures[(src, tgt)]
            if total >= 3:
                pairs.append({
                    "pair": f"{src}→{tgt}",
                    "successes": succ,
                    "failures": self.pair_failures[(src, tgt)],
                    "rate": round(succ / total, 3),
                })

        pairs.sort(key=lambda x: -x["rate"])
        return {
            "pairs_tracked": len(self.pair_successes),
            "best_pairs": pairs[:5],
        }


# ═══════════════════════════════════════════════════════════════
# Self-Improving Bus — the core
# ═══════════════════════════════════════════════════════════════

class SelfImprovingBus(CmtipBus):
    """
    CMTIP bus that learns from every exchange.
    
    Online learning mechanisms:
      1. Adapter weights updated via gradient descent on each exchange
      2. Concepts reinforced/penalized based on outcome
      3. Routing optimized from success/failure patterns
      4. Penalty signals used to dampen confusing dimensions
    """

    def __init__(self, learning_rate: float = 0.001, history_size: int = 100):
        super().__init__()
        self.learning_rate = learning_rate
        self.history_size = history_size

        # Self-improvement subsystems
        self.exchange_history: List[ExchangeRecord] = []
        self.concept_reinforcement = ConceptReinforcement()
        self.routing = RoutingOptimizer()
        
        # Learning stats
        self.total_exchanges = 0
        self.successful_exchanges = 0
        self.online_updates = 0

    def _measure_alignment(self, source_id: str, target_id: str,
                           source_tensor: np.ndarray,
                           projected_tensor: np.ndarray,
                           text: str = None) -> float:
        """
        Measure cross-model alignment quality.
        
        Better than batch entropy: directly measures how well the projected
        tensor preserves the original semantics in the target space.
        
        If text is available: embed text with target model, compare.
        If not: use roundtrip fidelity (project there and back).
        
        Returns a score in [0, 1] where higher = better alignment.
        """
        if text is not None and target_id in self.models:
            # Gold standard: how close is projection to target's own embedding?
            target_embed = self.models[target_id].embed(text)
            target_embed = target_embed / (np.linalg.norm(target_embed) + 1e-8)
            proj_norm = projected_tensor / (np.linalg.norm(projected_tensor) + 1e-8)
            # Pad to same dimensions if needed
            min_dim = min(len(target_embed), len(proj_norm))
            alignment = float(np.dot(target_embed[:min_dim], proj_norm[:min_dim]))
        else:
            # Fallback: roundtrip fidelity
            key = (source_id, target_id)
            if key in self.adapters:
                # Project back: target → source
                back_key = (target_id, source_id)
                if back_key in self.adapters:
                    back_proj = self.adapters[back_key].project(projected_tensor)
                    back_proj = back_proj / (np.linalg.norm(back_proj) + 1e-8)
                    src_norm = source_tensor / (np.linalg.norm(source_tensor) + 1e-8)
                    min_dim = min(len(back_proj), len(src_norm))
                    alignment = float(np.dot(back_proj[:min_dim], src_norm[:min_dim]))
                else:
                    alignment = 0.5  # Unknown
            else:
                alignment = 0.5

        # Clamp to [0, 1]
        return max(0.0, min(1.0, alignment))

    # ─── Core Exchange ────────────────────────────────────────────────

    def exchange(self, source_id: str, target_id: str,
                 text: str = None, tensor: np.ndarray = None,
                 concepts: Dict[str, float] = None) -> dict:
        """
        Send a tensor thought AND learn from the result.
        
        This is the main entry point. Every call:
          1. Sends the tensor (normal CMTIP routing)
          2. Observes the result (entropy, penalty)
          3. Updates adapter, concepts, and routing
        """
        concepts = concepts or {}

        # Apply concept reinforcement — adjust weights based on history
        adjusted_concepts = {
            name: self.concept_reinforcement.get_adjusted_weight(name, weight)
            for name, weight in concepts.items()
        }

        # Auto-route if no explicit target
        if target_id == "auto":
            candidates = [m for m in self.models if m != source_id]
            if candidates:
                target_id = self.routing.best_target(
                    source_id, candidates, list(concepts.keys())
                )

        # Save source tensor before projection (for online learning)
        if text is not None:
            source_tensor = self.models[source_id].embed(text)
        elif tensor is not None:
            source_tensor = tensor.copy()
        else:
            return {"error": "Must provide text or tensor"}

        # Send via parent (normal CMTIP routing)
        if text is not None:
            packet = self.send(source_id, target_id, text=text,
                              concept_tags=list(concepts.keys()))
        else:
            packet = self.send(source_id, target_id, tensor=tensor,
                              concept_tags=list(concepts.keys()))

        # Measure alignment quality (better than batch entropy)
        alignment = self._measure_alignment(
            source_id, target_id, source_tensor, packet.tensor, text
        )
        entropy = 1.0 - alignment  # Invert: high alignment = low "entropy"
        penalty_triggered = alignment < 0.3  # Below 0.3 alignment = penalty
        success = alignment > 0.5  # Above 0.5 alignment = success

        # Record exchange
        record = ExchangeRecord(
            source_id=source_id,
            target_id=target_id,
            seq_num=packet.seq_num,
            timestamp=time.time(),
            source_tensor=source_tensor,
            projected_tensor=packet.tensor.copy(),
            concept_tags=packet.concept_tags,
            concepts_used=adjusted_concepts,
            target_entropy=entropy,
            penalty_triggered=penalty_triggered,
            success=success,
        )
        self.exchange_history.append(record)

        # Trim history
        if len(self.exchange_history) > self.history_size:
            self.exchange_history = self.exchange_history[-self.history_size:]

        # ─── Learn from this exchange ─────────────────────────────────

        # 1. Online adapter update
        if record.is_useful_for_training():
            self._online_adapter_update(record)

        # 2. Concept reinforcement
        if success:
            self.concept_reinforcement.record_success(adjusted_concepts)
        elif penalty_triggered:
            self.concept_reinforcement.record_failure(adjusted_concepts)

        # Decay unused concepts
        self.concept_reinforcement.decay_unused(set(concepts.keys()))

        # 3. Routing optimization
        self.routing.record(source_id, target_id, success, list(concepts.keys()))

        # 4. Penalty-driven learning
        if penalty_triggered:
            self._learn_from_penalty(record)

        # Update counters
        self.total_exchanges += 1
        if success:
            self.successful_exchanges += 1

        return {
            "ok": True,
            "seq_num": packet.seq_num,
            "target": target_id,
            "entropy": round(entropy, 4),
            "penalty_triggered": penalty_triggered,
            "success": success,
            "success_rate": round(self.successful_exchanges / max(1, self.total_exchanges), 3),
            "tensor_b64": None,  # Would be base64 in production
            "shape": list(packet.shape),
            "concepts_used": adjusted_concepts,
            "learning": {
                "online_updates": self.online_updates,
                "adapter_improving": self.online_updates > 0,
            },
        }

    # ─── Online Adapter Update ────────────────────────────────────────

    def _online_adapter_update(self, record: ExchangeRecord):
        """
        Update adapter weights via online gradient descent.
        
        For each successful exchange, we have:
          v_source (before projection) — what we sent
          v_projected (after W projection) — what arrived
        
        The target model's internal representation of the same text
        would be target_model.embed(decoded_text). But we don't have
        the decoded text. Instead, we use a self-supervised proxy:
        
        Loss = || v_source @ W - v_target ||²
        where v_target ≈ v_projected (if the exchange was successful,
        the projection is close to what the target would produce).
        
        Gradient: dW = -2 * v_source^T @ (v_projected - v_projected_after_W)
                   = 0 (if already perfect)
        
        Better approach: use the penalty mask to identify which dimensions
        of the adapter weight matrix contribute to confusion.
        """
        source = record.source_tensor.reshape(1, -1).astype(np.float32)
        projected = record.projected_tensor.reshape(1, -1).astype(np.float32)

        # Get adapter
        key = (record.source_id, record.target_id)
        if key not in self.adapters:
            return
        adapter = self.adapters[key]

        # Simple Hebbian-like update:
        # If the exchange was successful, reinforce the weights
        # that map source features to projected features.
        # ΔW = η · v_source^T @ v_projected  (outer product)
        hebbian_update = self.learning_rate * (source.T @ projected)

        # Apply to adapter weight
        # Pad/trim to match weight dimensions
        w = adapter.weight
        min_rows = min(hebbian_update.shape[0], w.shape[0])
        min_cols = min(hebbian_update.shape[1], w.shape[1])
        w[:min_rows, :min_cols] += hebbian_update[:min_rows, :min_cols]

        # Normalize to prevent weight explosion
        w_norm = np.linalg.norm(w)
        if w_norm > 10.0:
            w *= 10.0 / w_norm

        adapter.weight = w
        self.online_updates += 1

    # ─── Penalty-Driven Learning ──────────────────────────────────────

    def _learn_from_penalty(self, record: ExchangeRecord):
        """
        When an exchange causes confusion, learn what went wrong.
        
        1. Identify which source dimensions mapped to high-entropy dimensions
        2. Dampen those adapter weights
        3. Reduce future reliance on concepts that triggered confusion
        """
        key = (record.source_id, record.target_id)
        if key not in self.adapters:
            return
        adapter = self.adapters[key]

        # Identify high-entropy dimensions in the projected tensor
        # (in production: use per-dimension entropy from attention)
        # Here: use variance across recent history as proxy for confusion
        if len(self.exchange_history) >= 5:
            recent_projected = np.array([
                e.projected_tensor for e in self.exchange_history[-5:]
                if len(e.projected_tensor) == len(record.projected_tensor)
            ])
            if len(recent_projected) >= 3:
                dim_variance = recent_projected.var(axis=0)
                high_var_dims = np.where(dim_variance > dim_variance.mean())[0]
                
                # Dampen adapter weights for high-variance dimensions
                dampen_factor = 0.95
                for dim in high_var_dims:
                    if dim < adapter.weight.shape[1]:
                        adapter.weight[:, dim] *= dampen_factor

    # ─── Auto-Route ───────────────────────────────────────────────────

    def auto_send(self, source_id: str, text: str = None,
                  concepts: Dict[str, float] = None) -> dict:
        """
        Send to the best target model automatically.
        
        Uses the RoutingOptimizer to select the model with the
        best track record for this type of query.
        """
        candidates = [m for m in self.models if m != source_id]
        if not candidates:
            return {"error": "No other models registered"}

        target_id = self.routing.best_target(
            source_id, candidates, list((concepts or {}).keys())
        )
        return self.exchange(source_id, target_id, text=text, concepts=concepts)

    # ─── Statistics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """Comprehensive self-improvement statistics."""
        return {
            "exchanges": {
                "total": self.total_exchanges,
                "successful": self.successful_exchanges,
                "success_rate": round(
                    self.successful_exchanges / max(1, self.total_exchanges), 3
                ),
            },
            "learning": {
                "online_updates": self.online_updates,
                "adapter_pairs": len(self.adapters),
                "learning_rate": self.learning_rate,
            },
            "concepts": self.concept_reinforcement.stats(),
            "routing": self.routing.stats(),
            "history_size": len(self.exchange_history),
            "penalty_threshold": self.penalty_threshold,
        }

    def learning_curve(self) -> dict:
        """
        Show how success rate evolved over exchanges.
        """
        if not self.exchange_history:
            return {"exchanges": 0}

        window = 10
        success_trend = []
        for i in range(0, len(self.exchange_history), window):
            chunk = self.exchange_history[i:i + window]
            rate = sum(1 for e in chunk if e.success) / len(chunk)
            success_trend.append({
                "from_exchange": i + 1,
                "to_exchange": min(i + window, len(self.exchange_history)),
                "success_rate": round(rate, 3),
            })

        return {
            "exchanges": self.total_exchanges,
            "window_size": window,
            "trend": success_trend,
            "improving": (
                len(success_trend) >= 2 and
                success_trend[-1]["success_rate"] > success_trend[0]["success_rate"]
            ),
        }


# ═══════════════════════════════════════════════════════════════
# Self-Improvement Demo
# ═══════════════════════════════════════════════════════════════

def demo_self_improvement():
    """
    Demonstrate self-improvement over simulated conversations.
    
    Shows:
      1. Success rate improving over time
      2. Concepts being reinforced/penalized
      3. Routing optimization
      4. Adapter weights evolving
    """
    print("=" * 64)
    print("  CMTIP — Self-Improving Language Demo")
    print("=" * 64)

    # Setup
    bus = SelfImprovingBus(learning_rate=0.005)
    bus.register_model(SyntheticEmbeddingBackend("agent-ops", 384, 1))
    bus.register_model(SyntheticEmbeddingBackend("agent-security", 768, 2))
    bus.register_model(SyntheticEmbeddingBackend("agent-research", 512, 3))

    # Initial adapter training (minimal — 10 pairs)
    print("\n─── Initial training (10 pairs) ───")
    paired = [(f"train {i}", f"train {i}") for i in range(10)]
    for src in ["agent-ops", "agent-research"]:
        bus.train_adapter(src, "agent-security", paired)
        if src != "agent-ops":
            bus.train_adapter("agent-security", src, paired)
    print(f"  Initial adapters: {len(bus.adapters)}")

    # ─── Phase 1: Many exchanges with the same concept patterns ─────
    print(f"\n─── Phase 1: 30 exchanges (should learn which concepts work) ───")

    domains = [
        # (text, concepts, success_likely)
        ("urgent security breach in production", {"urgent": 0.9, "technical": 0.7}, True),
        ("routine database maintenance scheduled", {"calm": 0.8, "technical": 0.5}, True),
        ("critical deployment failure — rollback needed", {"urgent": 1.0, "cautious": 0.6}, True),
        ("research findings on transformer architecture", {"abstract": 0.8, "creative": 0.6}, True),
        ("performance metrics analysis request", {"precise": 0.9, "technical": 0.7}, True),
    ]

    for i in range(30):
        text, concepts, _ = domains[i % len(domains)]
        result = bus.auto_send("agent-ops", text=text, concepts=concepts)

        if i % 5 == 0:
            status = "✓" if result["success"] else "✗"
            print(f"  [{i+1:2d}] {status} {text[:50]}... "
                  f"entropy={result['entropy']:.3f} "
                  f"rate={result['success_rate']:.2f} "
                  f"target={result['target']}")

    # ─── Stats after Phase 1 ────────────────────────────────────────
    print(f"\n─── After {bus.total_exchanges} exchanges ───")
    s = bus.stats()
    print(f"  Success rate: {s['exchanges']['success_rate']}")
    print(f"  Online updates: {s['learning']['online_updates']}")
    print(f"  Top concepts: {s['concepts']['top_performers']}")
    print(f"  Best pairs: {s['routing']['best_pairs'][:3]}")

    # ─── Phase 2: Learning curve ────────────────────────────────────
    print(f"\n─── Learning Curve ───")
    lc = bus.learning_curve()
    print(f"  {'Window':<15s} {'Success Rate':>12s}")
    print(f"  {'─'*27}")
    for point in lc["trend"]:
        bar = "█" * int(point["success_rate"] * 20)
        print(f"  {point['from_exchange']:>3d}-{point['to_exchange']:<3d}       "
              f"{point['success_rate']:.3f}  {bar}")

    if lc["improving"]:
        first_rate = lc["trend"][0]["success_rate"]
        last_rate = lc["trend"][-1]["success_rate"]
        improvement = (last_rate - first_rate) / max(0.01, first_rate) * 100
        print(f"\n  ★ Self-improvement: {improvement:+.0f}% improvement in success rate")
        print(f"    {first_rate:.3f} → {last_rate:.3f}")
    else:
        print(f"\n  No clear improvement trend yet — need more exchanges")

    # ─── Phase 3: Demonstrate concept learning ──────────────────────
    print(f"\n─── Concept Learning ───")
    cr = bus.concept_reinforcement
    print(f"  Concepts tracked: {len(cr.usage_count)}")
    print(f"  Avg success rate: {cr.stats()['avg_success_rate']}")

    # Show how a specific concept weight is adjusted
    if "urgent" in cr.usage_count:
        base = 0.9
        adjusted = cr.get_adjusted_weight("urgent", base)
        diff = adjusted - base
        print(f"  'urgent' weight: {base:.2f} → {adjusted:.3f} ({diff:+.3f} from learning)")

    # ─── Final Comprehensive Stats ──────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  FINAL SELF-IMPROVEMENT REPORT")
    print(f"{'='*64}")
    final = bus.stats()
    for section, data in final.items():
        if isinstance(data, dict):
            print(f"\n  [{section}]")
            for k, v in data.items():
                if isinstance(v, (int, float, str)):
                    print(f"    {k}: {v}")
                elif isinstance(v, list) and len(v) > 0:
                    print(f"    {k}: {v[:3]}")

    print(f"\n{'='*64}")
    print(f"  Self-improvement mechanisms active:")
    print(f"    1. Online adapter updates: {final['learning']['online_updates']} applied")
    print(f"    2. Concept reinforcement: {final['concepts']['total_concepts']} tracked")
    print(f"    3. Routing optimization: {final['routing']['pairs_tracked']} pairs")
    print(f"    4. Penalty-driven dampening")
    print(f"{'='*64}")


if __name__ == "__main__":
    demo_self_improvement()
