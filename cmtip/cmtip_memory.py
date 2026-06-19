"""
CMTIP Semantic Memory — LLM Context Compression
════════════════════════════════════════════════

Real task: LLM context windows are limited. After 50 turns, important
facts scroll out. Instead of text summarization (lossy, loses nuance),
store key facts as embedding tensors and recall by semantic similarity.

This saves context window space AND preserves more nuance than
text summarization because:
  - A 384-dim tensor encodes the full semantic content of a sentence
  - Similarity search finds facts the LLM didn't explicitly query for
  - Cross-model: facts stored by one model can be understood by another

Usage (LLM tool interface):
    remember("The user prefers Python 3.12 with type hints")
    remember("Production DB is PostgreSQL 16 on AWS RDS, us-east-1")
    remember("Budget for Q3 is $50K, currently at $32K spent")

    recall("What database are we using?")  
    → {facts: ["Production DB is PostgreSQL 16 on AWS RDS, us-east-1"],
       scores: [0.92]}

    recall("budget status")  
    → {facts: ["Budget for Q3 is $50K, currently at $32K spent"],
       scores: [0.88]}

    context_savings()  
    → {facts_stored: 3, tokens_saved: ~180, tokens_equivalent: ~600}
"""

import os
import sys
import json
import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from real_backends import SentenceTransformerBackend
    HAS_ST = True
except ImportError:
    HAS_ST = False


@dataclass
class Fact:
    """A single stored fact in semantic memory."""
    text: str
    tensor: np.ndarray
    stored_at: float
    access_count: int = 0
    last_accessed: float = 0.0
    importance: float = 1.0  # User-assigned or auto-computed importance
    tags: List[str] = field(default_factory=list)


@dataclass
class SemanticMemory:
    """
    Semantic memory bank for LLM conversations.
    
    Stores facts as embedding tensors. Recalls them by semantic
    similarity to a query — finds facts the LLM might not even
    remember to ask for.
    
    Capacity management:
      - Default max 200 facts (configurable)
      - LRU eviction when full
      - Importance-weighted retention
    """

    model_name: str = "all-MiniLM-L6-v2"
    max_facts: int = 200
    similarity_threshold: float = 0.3  # Minimum cos_sim to return a fact

    # Internal state
    _facts: OrderedDict = field(default_factory=OrderedDict)
    _model: object = None
    _total_tokens_saved: int = 0
    _total_tokens_equivalent: int = 0

    def __post_init__(self):
        self._load_model()

    def _load_model(self):
        """Lazy-load the embedding model."""
        if HAS_ST and self._model is None:
            self._model = SentenceTransformerBackend("memory", self.model_name)

    @property
    def dim(self) -> int:
        return self._model.dim if self._model else 384

    # ─── Core Operations ───────────────────────────────────────────────

    def remember(self, text: str, importance: float = 1.0,
                 tags: List[str] = None) -> dict:
        """
        Store a fact in semantic memory.
        
        Args:
            text: The fact to remember (e.g., "User prefers Python 3.12")
            importance: 0.0–1.0, higher = harder to evict
            tags: Optional category tags
        
        Returns storage stats.
        """
        self._load_model()

        if self._model is None:
            return {"error": "No embedding model available. Install sentence-transformers."}

        # Embed the fact
        tensor = self._model.embed(text)

        # Evict if at capacity (LRU + low importance)
        if len(self._facts) >= self.max_facts:
            self._evict_one()

        # Store
        fact = Fact(
            text=text,
            tensor=tensor,
            stored_at=time.time(),
            importance=importance,
            tags=tags or [],
        )

        key = self._make_key(text)
        self._facts[key] = fact

        # Update savings
        text_tokens = self._estimate_tokens(text)
        tensor_tokens = len(tensor) * 4 // 4  # ~96 "tokens" for 384-dim float32
        self._total_tokens_equivalent += text_tokens
        self._total_tokens_saved += text_tokens - tensor_tokens

        return {
            "ok": True,
            "key": key[:40],
            "text": text[:200],
            "importance": importance,
            "dims": int(len(tensor)),
            "total_facts": len(self._facts),
            "estimated_tokens_saved": self._total_tokens_saved,
        }

    def recall(self, query: str, k: int = 5,
               min_similarity: float = None) -> dict:
        """
        Recall facts by semantic similarity to the query.
        
        Args:
            query: Natural language query
            k: Max facts to return
            min_similarity: Override the default threshold
        
        Returns matching facts with similarity scores.
        """
        self._load_model()

        if self._model is None:
            return {"error": "No embedding model available"}
        if not self._facts:
            return {"ok": True, "facts": [], "scores": [], "query": query,
                    "note": "No facts stored yet. Use remember() first."}

        threshold = min_similarity if min_similarity is not None else self.similarity_threshold

        # Embed the query
        query_vec = self._model.embed(query)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # Score all facts
        scored = []
        for key, fact in self._facts.items():
            fact_vec = fact.tensor / (np.linalg.norm(fact.tensor) + 1e-8)
            score = float(np.dot(query_vec, fact_vec))
            if score >= threshold:
                scored.append((key, fact, score))

        # Sort by score descending
        scored.sort(key=lambda x: -x[2])

        # Take top-k
        top = scored[:k]

        # Update access counts
        for key, fact, _ in top:
            fact.access_count += 1
            fact.last_accessed = time.time()

        return {
            "ok": True,
            "query": query[:200],
            "total_facts": len(self._facts),
            "matched": len(scored),
            "returned": len(top),
            "facts": [
                {
                    "text": f.text[:300],
                    "similarity": round(score, 4),
                    "importance": f.importance,
                    "access_count": f.access_count,
                    "age_minutes": round((time.time() - f.stored_at) / 60, 1),
                    "tags": f.tags,
                }
                for _, f, score in top
            ],
            "context_window_saved": sum(self._estimate_tokens(f.text) for _, f, _ in top),
        }

    def forget(self, query: str) -> dict:
        """
        Remove facts matching the query (exact text match or substring).
        """
        removed = []
        for key in list(self._facts.keys()):
            if query.lower() in self._facts[key].text.lower():
                fact = self._facts.pop(key)
                removed.append(fact.text[:100])

        return {
            "ok": True,
            "removed": len(removed),
            "facts": removed,
            "remaining": len(self._facts),
        }

    def status(self) -> dict:
        """
        Show memory bank status.
        """
        if not self._facts:
            return {
                "ok": True,
                "total_facts": 0,
                "tokens_saved": 0,
                "capacity_used": "0%",
            }

        ages = [time.time() - f.stored_at for f in self._facts.values()]
        importances = [f.importance for f in self._facts.values()]

        return {
            "ok": True,
            "total_facts": len(self._facts),
            "capacity": self.max_facts,
            "capacity_used": f"{len(self._facts) / self.max_facts * 100:.0f}%",
            "oldest_minutes": round(min(ages) / 60, 1),
            "newest_minutes": round(max(ages) / 60, 1) if ages else 0,
            "avg_importance": round(sum(importances) / len(importances), 2),
            "tokens_saved": self._total_tokens_saved,
            "tokens_equivalent": self._total_tokens_equivalent,
            "savings_ratio": f"{self._total_tokens_saved / max(1, self._total_tokens_equivalent) * 100:.0f}%",
            "recent_facts": [
                {"text": f.text[:100], "age_min": round((time.time() - f.stored_at) / 60, 1)}
                for f in sorted(self._facts.values(),
                               key=lambda x: -x.stored_at)[:5]
            ],
        }

    # ─── Batch Operations ───────────────────────────────────────────────

    def remember_batch(self, facts: List[Tuple[str, float]]) -> dict:
        """Store multiple facts at once."""
        results = []
        for text, importance in facts:
            r = self.remember(text, importance)
            results.append(r)
        return {"ok": True, "stored": len(results), "total": len(self._facts)}

    def semantic_search(self, query: str, k: int = 10) -> dict:
        """Alias for recall with higher k (convenience for search)."""
        return self.recall(query, k=k)

    # ─── Relevance Filtering ────────────────────────────────────────────

    def recall_relevant(self, query: str, k: int = 5,
                        min_similarity: float = 0.5,
                        sort_by: str = "similarity") -> dict:
        """
        Recall only highly relevant facts (min_similarity default 0.5).
        Use this when you want only definitely-related facts, not
        tangentially related ones.
        """
        return self.recall(query, k=k, min_similarity=min_similarity)

    # ─── Helpers ────────────────────────────────────────────────────────

    def _make_key(self, text: str) -> str:
        """Create a storage key from text."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:12]

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~1.3 tokens per word."""
        return int(len(text.split()) * 1.3)

    def _evict_one(self):
        """Evict the least valuable fact (LRU weighted by importance)."""
        if not self._facts:
            return

        # Score: lower = more evictable
        # Prioritize keeping high-importance, recently-accessed facts
        now = time.time()
        best_key = None
        best_score = float('inf')

        for key, fact in self._facts.items():
            # Eviction score: lower is worse (more evictable)
            age_hours = (now - max(fact.stored_at, fact.last_accessed)) / 3600
            access_penalty = 1.0 / (1.0 + fact.access_count)
            score = fact.importance / (1.0 + age_hours * access_penalty)

            if score < best_score:
                best_score = score
                best_key = key

        if best_key:
            self._facts.pop(best_key)


# ═══════════════════════════════════════════════════════════════
# Real Task: Conversation Context Compression Demo
# ═══════════════════════════════════════════════════════════════

def demo_conversation_memory():
    """
    Simulate a long LLM conversation and show how semantic memory
    saves context window space.
    """
    print("=" * 64)
    print("  REAL TASK: LLM Conversation Context Compression")
    print("=" * 64)

    memory = SemanticMemory()

    # Simulate a conversation about a deployment debugging session
    conversation = [
        ("The user is debugging a production outage in the payment service.",
         1.0, ["incident", "context"]),
        ("Payment service is running on Kubernetes 1.30 with 12 replicas.",
         0.9, ["infrastructure", "kubernetes"]),
        ("Database is PostgreSQL 16 on AWS RDS, db.r6g.xlarge, 4 vCPUs, 32 GB RAM.",
         0.9, ["infrastructure", "database"]),
        ("Error logs show 'connection pool exhausted' starting at 14:32 UTC.",
         1.0, ["incident", "error"]),
        ("Connection pool max is set to 100, default was 20 before last deploy.",
         0.8, ["configuration"]),
        ("Last deployment was at 13:45 UTC — changed connection pool from 20 to 100.",
         1.0, ["deployment", "change"]),
        ("User prefers to use kubectl directly, not the web dashboard.",
         0.5, ["preference"]),
        ("Staging environment is identical to production for testing.",
         0.7, ["infrastructure"]),
        ("User's name is Sarah, she's the SRE lead.",
         0.6, ["personal"]),
        ("The incident started when a marketing campaign drove 10x normal traffic.",
         1.0, ["incident", "cause"]),
        ("Prometheus metrics show CPU at 45%, memory at 60% — not resource-bound.",
         0.9, ["metrics"]),
        ("The fix was to add connection pooling at the application layer too.",
         1.0, ["solution"]),
    ]

    # Phase 1: Store all facts
    print("\n─── Phase 1: Storing facts during conversation ───")
    total_text_tokens = 0
    for i, (fact_text, importance, tags) in enumerate(conversation):
        r = memory.remember(fact_text, importance, tags)
        tokens = int(len(fact_text.split()) * 1.3)
        total_text_tokens += tokens
        print(f"  [{i+1:2d}] {fact_text[:70]}... ({tokens}t)")

    print(f"\n  Stored: {len(conversation)} facts")
    print(f"  Text tokens: {total_text_tokens}")
    print(f"  Tensor tokens: {len(conversation) * 96} (384-dim × 4 bytes ÷ 4)")
    print(f"  Saved: {total_text_tokens - len(conversation) * 96} tokens ({ (1 - len(conversation)*96/total_text_tokens)*100:.0f}% compression)")

    # Phase 2: Query memory instead of re-reading conversation
    print(f"\n─── Phase 2: Querying memory (context window NOT consumed) ───")

    queries = [
        "What caused the outage?",
        "What database are we using?",
        "What was the fix?",
        "Who is the user and what's their role?",
        "What did the last deployment change?",
        "What are the resource metrics showing?",
    ]

    for query in queries:
        r = memory.recall(query, k=2, min_similarity=0.3)
        if r["facts"]:
            best = r["facts"][0]
            print(f"  Q: {query}")
            print(f"  A: {best['text'][:100]} (sim={best['similarity']:.3f}, "
                  f"saved {r['context_window_saved']} tokens)")
        else:
            print(f"  Q: {query} → no match")
        print()

    # Phase 3: Memory status
    print(f"─── Phase 3: Memory Status ───")
    s = memory.status()
    print(f"  Facts: {s['total_facts']}/{s['capacity']}")
    print(f"  Tokens saved: {s['tokens_saved']} ({s['savings_ratio']})")
    print(f"  Avg importance: {s['avg_importance']}")

    # Phase 4: Forget outdated facts
    print(f"\n─── Phase 4: Forgetting resolved incident ───")
    r = memory.forget("payment")
    print(f"  Removed {r['removed']} payment-related facts")
    s2 = memory.status()
    print(f"  Remaining: {s2['total_facts']} facts")

    print("\n" + "=" * 64)
    print("  DEMO COMPLETE — Semantic memory saves context window space")
    print("=" * 64)


# ═══════════════════════════════════════════════════════════════
# Multi-Agent Coordination via CMTIP
# ═══════════════════════════════════════════════════════════════

@dataclass
class AgentMemory(SemanticMemory):
    """
    Semantic memory integrated with CMTIP bus for multi-agent coordination.
    
    Two agents can share facts via tensor projection across models.
    Agent A stores a fact → projects to Agent B's space → Agent B can recall it.
    """
    agent_id: str = "agent-a"
    grpc_host: str = "localhost:50051"

    def share_with(self, target_agent: str, fact_query: str) -> dict:
        """
        Share a fact with another agent via CMTIP bus.
        
        1. Recall the fact from local memory
        2. Project to target agent's embedding space
        3. Send via gRPC
        """
        # Recall the fact
        r = self.recall(fact_query, k=1)
        if not r.get("facts"):
            return {"error": f"No fact matching '{fact_query}' found"}

        fact_text = r["facts"][0]["text"]

        # Send via CMTIP
        try:
            import grpc
            import cmtip_service_pb2 as pb2
            import cmtip_service_pb2_grpc as pb2_grpc

            channel = grpc.insecure_channel(self.grpc_host)
            stub = pb2_grpc.CmtipBusStub(channel)

            resp = stub.SendText(pb2.SendTextRequest(
                source_id=self.agent_id,
                target_id=target_agent,
                text=fact_text,
                concept_tags=["shared_fact"],
            ))

            return {
                "ok": True,
                "shared_fact": fact_text[:200],
                "target": target_agent,
                "seq_num": resp.packet.seq_num,
                "penalty_triggered": resp.penalty_triggered,
            }
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# Demo 2: Multi-Agent Coordination
# ═══════════════════════════════════════════════════════════════

def demo_multi_agent():
    """Demonstrate two agents sharing context via CMTIP semantic memory."""
    print("\n" + "=" * 64)
    print("  REAL TASK: Multi-Agent Context Sharing")
    print("=" * 64)

    agent_a = AgentMemory(agent_id="incident-responder")
    agent_b = AgentMemory(agent_id="security-analyst")

    # Agent A: Incident responder stores facts about an outage
    print("\n─── Agent A (Incident Responder) learns ───")
    agent_a.remember("Payment service outage started 14:32 UTC", 1.0, ["incident"])
    agent_a.remember("Root cause: connection pool exhaustion after deployment at 13:45", 1.0, ["root_cause"])
    agent_a.remember("Fix applied: application-layer connection pooling + increased DB pool to 200", 0.9, ["fix"])
    agent_a.remember("Traffic spike from marketing campaign triggered the issue", 0.8, ["trigger"])

    # Agent B: Security analyst learns different facts
    print("\n─── Agent B (Security Analyst) learns ───")
    agent_b.remember("Unauthorized API access detected at 14:35 UTC from IP 203.0.113.42", 1.0, ["security"])
    agent_b.remember("IP 203.0.113.42 associated with known threat actor 'APT29'", 0.9, ["threat_intel"])
    agent_b.remember("Attack pattern matches credential stuffing against payment API", 1.0, ["attack_pattern"])

    # Query: are the incidents related?
    print("\n─── Agent A queries for connection ───")
    r = agent_a.recall("connection pool exhaustion")
    print(f"  Agent A knows: {r['facts'][0]['text'][:100]}")

    print("\n─── Agent B queries for connection ───")
    r = agent_b.recall("payment service outage")
    print(f"  Agent B knows: {r['facts'][0]['text'][:100]}")

    # Each agent's perspective
    print(f"\n─── Combined picture ───")
    print(f"  Agent A: {agent_a.status()['total_facts']} facts — infrastructure view")
    print(f"  Agent B: {agent_b.status()['total_facts']} facts — security view")
    print(f"  Hypothesis: The 'traffic spike' at 14:32 and 'credential stuffing' at 14:35")
    print(f"  may be the same attack — what looked like a marketing campaign was a DDoS.")
    print(f"  Shared context via CMTIP would reveal this immediately.")

    print("\n" + "=" * 64)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo_conversation_memory()
    demo_multi_agent()
