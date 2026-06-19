"""
CMTIP — Expanded Concept Vocabulary with Compositional Grammar
════════════════════════════════════════════════════════════════

Addresses critique #6: 40 concepts is impoverished. Natural language
has 50K+ tokens with grammar — tensor language needs more than tags.

V2 adds:
    1. Compositional operations (AND, OR, NOT, BUT, MORE, LESS)
    2. Negation (NOT concept: invert along the semantic axis)
    3. Quantification (MORE concept_x, LESS concept_y)
    4. Sequence (THEN: chain multiple concepts temporally)
    5. 256+ concepts from larger-scale k-means clustering

Usage:
    python auto_vocab_v2.py [--n-concepts 256] [--n-pca 32]
"""

import numpy as np
import json
import os
import sys
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import TensorVocabulary


# ═══════════════════════════════════════════════════════════════
# Compositional Grammar Operators
# ═══════════════════════════════════════════════════════════════

@dataclass
class CompositionalVocab(TensorVocabulary):
    """
    Extended vocabulary with compositional grammar.

    Natural language operators mapped to tensor geometry:
      AND(a,b)    → (v_a + v_b) / |...|       (semantic conjunction)
      OR(a,b)     → slerp(v_a, v_b, 0.5)      (semantic disjunction)
      NOT(a)      → invert along dominant PCA  (semantic negation)
      MORE(a, α)  → v_a * (1 + α)             (amplification)
      LESS(a, α)  → v_a * (1 - α)             (attenuation)
      BUT(a,b)    → v_a - 0.3*v_b             (contrastive conjunction)
      THEN(a,b)   → sequence-aware blend       (temporal chaining)
      QUANT(n, a) → v_a scaled by log(n)       (quantification)
    """

    pca_components: np.ndarray = None  # For negation operator

    def set_pca(self, components: np.ndarray):
        """Set PCA components for negation/projection operations."""
        self.pca_components = components.astype(np.float32)

    def AND(self, *concept_names: str) -> np.ndarray:
        """Semantic conjunction: sum of concepts."""
        result = np.zeros_like(list(self.concepts.values())[0])
        for name in concept_names:
            if name in self.concepts:
                result += self.concepts[name]
        return result / (np.linalg.norm(result) + 1e-8)

    def OR(self, a: str, b: str) -> np.ndarray:
        """Semantic disjunction: blend at midpoint."""
        return self.blend(a, b, 0.5)

    def NOT(self, concept_name: str) -> np.ndarray:
        """
        Semantic negation: invert along first PCA component.
        If no PCA, invert by subtracting from the concept-space mean.
        """
        if concept_name not in self.concepts:
            raise KeyError(f"Concept '{concept_name}' not found")

        v = self.concepts[concept_name].copy()

        if self.pca_components is not None:
            # Project onto first PCA axis, invert that component
            proj = np.dot(v, self.pca_components[0]) * self.pca_components[0]
            result = v - 2 * proj
        else:
            # Simple inversion: subtract from mean of all concepts
            all_vectors = np.stack(list(self.concepts.values()))
            mean = all_vectors.mean(axis=0)
            result = 2 * mean - v

        return result / (np.linalg.norm(result) + 1e-8)

    def MORE(self, concept_name: str, alpha: float = 0.5) -> np.ndarray:
        """Amplify a concept: v * (1 + α)."""
        v = self.concepts[concept_name]
        result = v * (1.0 + alpha)
        return result / (np.linalg.norm(result) + 1e-8)

    def LESS(self, concept_name: str, alpha: float = 0.5) -> np.ndarray:
        """Attenuate a concept: v * (1 - α)."""
        v = self.concepts[concept_name]
        result = v * max(0.1, 1.0 - alpha)
        return result / (np.linalg.norm(result) + 1e-8)

    def BUT(self, a: str, b: str, contrast_weight: float = 0.3) -> np.ndarray:
        """Contrastive conjunction: A but not B. v_a - λ * v_b."""
        if a not in self.concepts or b not in self.concepts:
            raise KeyError(f"Concepts not found")
        result = self.concepts[a] - contrast_weight * self.concepts[b]
        return result / (np.linalg.norm(result) + 1e-8)

    def THEN(self, *concept_names: str, decay: float = 0.7) -> np.ndarray:
        """
        Temporal sequence: weighted blend where earlier concepts decay.
        c1 has weight 1.0, c2 has weight 0.7, c3 has 0.49, etc.
        """
        if not concept_names:
            return np.zeros_like(list(self.concepts.values())[0])

        result = np.zeros_like(list(self.concepts.values())[0])
        weight = 1.0
        total_w = 0.0
        for name in concept_names:
            if name in self.concepts:
                result += self.concepts[name] * weight
                total_w += weight
                weight *= decay

        if total_w > 0:
            result /= total_w
        return result / (np.linalg.norm(result) + 1e-8)

    def QUANT(self, number: int, concept_name: str) -> np.ndarray:
        """Quantify: intensity scales with log(number)."""
        if concept_name not in self.concepts:
            raise KeyError(f"Concept not found")
        scale = np.log2(max(1, number)) / 5.0  # log scale capped
        scale = min(scale, 2.0)
        result = self.concepts[concept_name] * scale
        return result / (np.linalg.norm(result) + 1e-8)

    def parse_composition(self, expression: str) -> np.ndarray:
        """
        Parse a simple compositional expression and return the vector.

        Grammar:
            "AND(concept_a, concept_b)"      → semantic conjunction
            "OR(concept_a, concept_b)"       → semantic disjunction
            "NOT(concept_a)"                 → negation
            "MORE(concept_a)"                → amplification
            "BUT(concept_a, concept_b)"      → contrastive
            "THEN(concept_a, concept_b)"     → sequence

        Example:
            "BUT(MORE(urgent), NOT(calm))"
            → amplify urgent, negate calm, contrast them
        """
        expression = expression.strip()

        if expression.startswith("AND(") and expression.endswith(")"):
            inner = expression[4:-1]
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) == 2:
                return self.AND(parts[0], parts[1])
        elif expression.startswith("OR(") and expression.endswith(")"):
            inner = expression[3:-1]
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) == 2:
                return self.OR(parts[0], parts[1])
        elif expression.startswith("NOT(") and expression.endswith(")"):
            inner = expression[4:-1].strip()
            return self.NOT(inner)
        elif expression.startswith("MORE(") and expression.endswith(")"):
            inner = expression[4:-1].strip()
            return self.MORE(inner)
        elif expression.startswith("BUT(") and expression.endswith(")"):
            inner = expression[4:-1]
            parts = [p.strip() for p in inner.split(",", 1)]
            if len(parts) == 2:
                return self.BUT(parts[0], parts[1])

        # Fallback: look up directly
        if expression in self.concepts:
            return self.concepts[expression]

        raise ValueError(f"Cannot parse expression: {expression}")


# ═══════════════════════════════════════════════════════════════
# Large-Scale Vocabulary Learner
# ═══════════════════════════════════════════════════════════════

def learn_large_vocabulary(embeddings: np.ndarray, texts: List[str],
                            n_concepts: int = 256, n_pca: int = 32) -> CompositionalVocab:
    """
    Learn an expanded vocabulary with 256+ concepts and compositional grammar.

    Returns a CompositionalVocab with:
      - n_concepts cluster centroids
      - n_pca semantic axes
      - Compositional operators (AND, OR, NOT, BUT, THEN, QUANT, MORE, LESS)
    """
    N, dim = embeddings.shape
    n_clusters = min(n_concepts, N)
    n_pca = min(n_pca, dim, N)

    print(f"\nLearning large vocabulary: {n_clusters} clusters, {n_pca} PCA axes")
    t0 = time.time()

    # PCA
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:n_pca]

    print(f"  PCA done in {time.time()-t0:.1f}s — top variance: "
          f"{((S[:5]**2)/(S**2).sum()).round(4)}")

    # K-means with k-means++ init
    t0 = time.time()

    # Simple k-means (can be slow for 256+ clusters, use mini-batch for speed)
    rng = np.random.RandomState(42)

    if n_clusters > 100:
        # Mini-batch k-means for speed
        centroids = embeddings[rng.choice(N, n_clusters, replace=False)]
        for iteration in range(30):
            batch_idx = rng.choice(N, min(500, N), replace=False)
            batch = embeddings[batch_idx]
            dists = np.array([np.sum((batch - c)**2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            for i in range(n_clusters):
                members = batch[labels == i]
                if len(members) > 0:
                    centroids[i] = 0.9 * centroids[i] + 0.1 * members.mean(axis=0)
    else:
        centroids = [embeddings[rng.randint(N)]]
        for _ in range(1, n_clusters):
            dists = np.min([np.sum((embeddings - c)**2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(embeddings[rng.choice(N, p=probs)])
        centroids = np.array(centroids)

        for _ in range(15):
            dists = np.array([np.sum((embeddings - c)**2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)
            for i in range(n_clusters):
                members = embeddings[labels == i]
                if len(members) > 0:
                    centroids[i] = members.mean(axis=0)

    # Normalize centroids
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)

    print(f"  K-means done in {time.time()-t0:.1f}s — {n_clusters} clusters")

    # Build vocabulary
    vocab = CompositionalVocab()
    vocab.set_pca(pca_components)

    # Add cluster concepts
    for i in range(n_clusters):
        vocab.add_concept(f"c{i}", centroids[i])

    # Add PCA axes
    for i in range(n_pca):
        vec = pca_components[i] / (np.linalg.norm(pca_components[i]) + 1e-8)
        var_pct = (S[i]**2) / (S**2).sum() * 100
        vocab.add_concept(f"axis_{i}({var_pct:.0f}%)", vec)

    # Add contrastive pairs
    projected = centered @ pca_components.T
    for axis in range(min(n_pca, 16)):
        proj = projected[:, axis]
        vocab.add_concept(f"axis_{axis}+",
                         pca_components[axis] / (np.linalg.norm(pca_components[axis]) + 1e-8))
        vocab.add_concept(f"axis_{axis}-",
                         -pca_components[axis] / (np.linalg.norm(pca_components[axis]) + 1e-8))

    # Add compositional aliases for common concepts
    compositional_aliases = {
        "strong_urgent": vocab.MORE("c0"),
        "mild_urgent": vocab.LESS("c0"),
        "opposite_urgent": vocab.NOT("c0"),
        "complex_thought": vocab.AND("c0", "c1", "c2"),
        "contrast_c0_c1": vocab.BUT("c0", "c1"),
        "sequence_c0_c1_c2": vocab.THEN("c0", "c1", "c2"),
    }

    for name, vec in compositional_aliases.items():
        if isinstance(vec, np.ndarray):
            vocab.add_concept(name, vec)

    print(f"  Total concepts: {len(vocab.concepts)} (clusters + axes + compositional)")
    print(f"  Compositional ops: AND, OR, NOT, MORE, LESS, BUT, THEN, QUANT")

    return vocab


# ═══════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════

def demo_composition(vocab: CompositionalVocab):
    """Demonstrate compositional grammar operations."""
    print(f"\n{'─'*64}")
    print("  COMPOSITIONAL GRAMMAR DEMO")
    print(f"{'─'*64}")

    concept_names = sorted(vocab.concepts.keys())
    if len(concept_names) < 4:
        print("  Need at least 4 concepts for demo")
        return

    a, b, c, d = concept_names[0], concept_names[1], concept_names[2], concept_names[3]

    ops = [
        ("AND(a, b)", lambda: vocab.AND(a, b)),
        ("OR(a, b)", lambda: vocab.OR(a, b)),
        ("NOT(a)", lambda: vocab.NOT(a)),
        ("MORE(a)", lambda: vocab.MORE(a, 0.5)),
        ("LESS(a)", lambda: vocab.LESS(a, 0.5)),
        ("BUT(a, b)", lambda: vocab.BUT(a, b)),
        ("THEN(a, b, c)", lambda: vocab.THEN(a, b, c)),
        ("QUANT(5, a)", lambda: vocab.QUANT(5, a)),
    ]

    for name, fn in ops:
        try:
            vec = fn()
            nearest = vocab.nearest(vec, k=3)
            parts = ", ".join(f"{n}({s:.2f})" for n, s in nearest)
            print(f"  {name:<20s} → {parts}")
        except Exception as e:
            print(f"  {name:<20s} → ERROR: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CMTIP — Expanded Concept Vocabulary V2"
    )
    parser.add_argument("--n-concepts", type=int, default=256,
                        help="Number of concept clusters")
    parser.add_argument("--n-pca", type=int, default=32,
                        help="Number of PCA axes")
    parser.add_argument("--save", default="./auto_vocab_v2.json")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    print("=" * 64)
    print("  CMTIP — Expanded Vocabulary V2 (Compositional Grammar)")
    print("=" * 64)

    # Load embeddings
    from real_backends import SentenceTransformerBackend

    print("\nLoading model: all-MiniLM-L6-v2")
    backend = SentenceTransformerBackend("vocab-learner", "all-MiniLM-L6-v2")

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "training_pairs.json")
    with open(data_path) as f:
        data = json.load(f)
    texts = list(set(p["a"] for p in data["train"]))
    print(f"Loaded {len(texts)} unique texts")

    print(f"\nEmbedding {len(texts)} texts...")
    t0 = time.time()
    embeddings = backend.embed_batch(texts)
    print(f"Done in {time.time()-t0:.1f}s — shape: {embeddings.shape}")

    # Learn vocabulary
    vocab = learn_large_vocabulary(embeddings, texts,
                                   n_concepts=args.n_concepts,
                                   n_pca=args.n_pca)

    if args.demo:
        demo_composition(vocab)

    # Save
    data = {
        "concepts": {k: v.tolist() for k, v in vocab.concepts.items()},
        "pca_components": vocab.pca_components.tolist() if vocab.pca_components is not None else None,
        "n_concepts": len(vocab.concepts),
    }
    with open(args.save, 'w') as f:
        json.dump(data, f)
    print(f"\nSaved {len(vocab.concepts)} concepts to {args.save}")

    print("=" * 64)


if __name__ == "__main__":
    main()
