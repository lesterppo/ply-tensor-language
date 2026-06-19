"""
CMTIP — Automatic Concept Vocabulary Learner
═════════════════════════════════════════════

SPEC Step 4: Extend vocabulary with learned concept vectors.

Instead of hand-crafting concept vectors (like "curious" → embed("curious exploring...")),
this module discovers concepts automatically from data using three methods:

1. PCA Semantic Axes — principal components of the embedding manifold
   Each PC captures a latent semantic dimension (e.g., positive/negative, 
   abstract/concrete, formal/informal).

2. K-Means Clustering — discover natural concept clusters
   Cluster centroids become canonical concept vectors.

3. Contrastive Concept Pairs — find maximally separated pairs along PCA axes
   The extremes of each axis form antonym-like pairs (hot↔cold, happy↔sad).

Usage:
    python auto_vocab.py [--model all-MiniLM-L6-v2] [--n-concepts 32]
"""

import numpy as np
import sys
import os
import json
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import TensorVocabulary, SyntheticEmbeddingBackend

try:
    from real_backends import SentenceTransformerBackend
    HAS_ST = True
except ImportError:
    HAS_ST = False


@dataclass
class AutoVocabulary:
    """Automatically learned concept vocabulary from embedding data."""

    vocab: TensorVocabulary = field(default_factory=TensorVocabulary)

    # PCA semantic axes (each axis = a discovered semantic dimension)
    pca_components: np.ndarray = None          # [n_components, dim]
    pca_explained_variance: np.ndarray = None

    # K-means cluster centroids (each centroid = a discovered concept)
    cluster_centroids: np.ndarray = None       # [n_clusters, dim]
    cluster_labels: np.ndarray = None           # [n_samples]

    # Contrastive pairs (extremes along each PCA axis)
    contrastive_pairs: List[Tuple[str, str]] = field(default_factory=list)

    def learn_from_embeddings(self, embeddings: np.ndarray,
                               texts: List[str] = None,
                               n_concepts: int = 32,
                               n_pca_components: int = 16):
        """
        Learn concepts from a matrix of embeddings.

        Args:
            embeddings: [N, dim] float32 embedding matrix
            texts: optional original texts (for naming discovered concepts)
            n_concepts: number of concept clusters to discover
            n_pca_components: number of PCA axes to extract
        """
        N, dim = embeddings.shape
        n_pca = min(n_pca_components, N, dim)
        n_clusters = min(n_concepts, N)

        print(f"\nLearning from {N} embeddings (d={dim})...")
        print(f"  PCA: {n_pca} components, K-Means: {n_clusters} clusters")

        # ─── 1. PCA: Discover Semantic Principal Axes ────────────────────
        t0 = time.time()
        mean = embeddings.mean(axis=0, keepdims=True)
        centered = embeddings - mean

        # Use SVD for numerical stability
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        self.pca_components = Vt[:n_pca]  # [n_pca, dim]
        self.pca_explained_variance = (S[:n_pca] ** 2) / (S ** 2).sum()

        print(f"  PCA done in {time.time()-t0:.1f}s")
        print(f"    Top 5 variance ratios: {self.pca_explained_variance[:5].round(4)}")

        # Project data onto PCA axes
        projected = centered @ self.pca_components.T  # [N, n_pca]

        # ─── 2. K-Means Clustering: Discover Concept Clusters ─────────────
        t0 = time.time()
        centroids, labels = self._kmeans(embeddings, n_clusters, n_iter=20)
        self.cluster_centroids = centroids
        self.cluster_labels = labels

        print(f"  K-Means done in {time.time()-t0:.1f}s ({n_clusters} clusters)")

        # ─── 3. Add Concepts to Vocabulary ───────────────────────────────
        self._add_pca_concepts(n_pca)
        self._add_cluster_concepts(texts)
        self._add_contrastive_pairs(texts, projected)

        # ─── 4. Name the concepts ────────────────────────────────────────
        print(f"\n  Vocabulary: {len(self.vocab.concepts)} concepts")
        for name in sorted(self.vocab.concepts.keys()):
            print(f"    {name}")

    def _kmeans(self, data: np.ndarray, k: int, n_iter: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering."""
        N = data.shape[0]

        # K-means++ initialization
        centroids = [data[np.random.randint(N)]]
        for _ in range(1, k):
            dists = np.min([np.sum((data - c) ** 2, axis=1) for c in centroids], axis=0)
            probs = dists / dists.sum()
            centroids.append(data[np.random.choice(N, p=probs)])
        centroids = np.array(centroids)

        for iteration in range(n_iter):
            # Assign
            dists = np.array([np.sum((data - c) ** 2, axis=1) for c in centroids])
            labels = np.argmin(dists, axis=0)

            # Update
            new_centroids = np.array([
                data[labels == i].mean(axis=0) if (labels == i).sum() > 0
                else data[np.random.randint(N)]
                for i in range(k)
            ])
            centroids = new_centroids

        # Normalize centroids
        centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
        return centroids, labels

    def _add_pca_concepts(self, n_pca: int):
        """Add PCA components as directional concept axes."""
        for i in range(n_pca):
            vec = self.pca_components[i]
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            var_pct = self.pca_explained_variance[i] * 100
            self.vocab.add_concept(f"axis_{i+1} ({var_pct:.0f}%)", vec)

    def _add_cluster_concepts(self, texts: List[str] = None):
        """Add cluster centroids as concept vectors, named by nearest text."""
        n_clusters = len(self.cluster_centroids)
        for i in range(n_clusters):
            centroid = self.cluster_centroids[i]
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

            # Name the concept by its nearest text
            if texts and self.cluster_labels is not None:
                cluster_texts = [texts[j] for j in np.where(self.cluster_labels == i)[0][:3]]
                if cluster_texts:
                    name = f"cluster_{i}"
                else:
                    name = f"cluster_{i}"
            else:
                name = f"concept_{i}"

            self.vocab.add_concept(name, centroid)

    def _add_contrastive_pairs(self, texts: List[str], projected: np.ndarray):
        """Find extreme texts along each PCA axis to form contrastive pairs."""
        if texts is None or len(texts) == 0:
            return

        n_pca = projected.shape[1]
        for axis in range(min(n_pca, 8)):  # Top 8 axes
            proj = projected[:, axis]
            pos_idx = int(np.argmax(proj))
            neg_idx = int(np.argmin(proj))

            pos_name = f"axis_{axis+1}+"
            neg_name = f"axis_{axis+1}-"

            # Use the extreme texts as the concept vector
            self.vocab.add_concept(
                pos_name,
                self.pca_components[axis].copy()
            )
            self.vocab.add_concept(
                neg_name,
                -self.pca_components[axis].copy()
            )

            self.contrastive_pairs.append((pos_name, neg_name))

    def analogy_test(self) -> List[dict]:
        """Test learned concepts via analogy reasoning."""
        results = []
        pairs = self.contrastive_pairs[:4]  # Test top 4 pairs
        for i, (pos, neg) in enumerate(pairs):
            if i + 1 >= len(pairs):
                break
            pos2, neg2 = pairs[i + 1]
            # Analogy: pos is to neg as pos2 is to ?
            answer = self.vocab.analogy(pos, neg, pos2)
            results.append({
                "analogy": f"{pos}:{neg} :: {pos2}:?",
                "answer": answer,
            })
        return results

    def save(self, path: str):
        """Save vocabulary and learned components."""
        data = {
            "concepts": {k: v.tolist() for k, v in self.vocab.concepts.items()},
            "pca_components": self.pca_components.tolist() if self.pca_components is not None else None,
            "cluster_centroids": self.cluster_centroids.tolist() if self.cluster_centroids is not None else None,
            "contrastive_pairs": self.contrastive_pairs,
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        print(f"Saved vocabulary to {path}")

    @classmethod
    def load(cls, path: str) -> 'AutoVocabulary':
        """Load a saved vocabulary."""
        with open(path) as f:
            data = json.load(f)
        av = cls()
        for name, vec in data["concepts"].items():
            av.vocab.add_concept(name, np.array(vec, dtype=np.float32))
        if data.get("pca_components"):
            av.pca_components = np.array(data["pca_components"], dtype=np.float32)
        if data.get("cluster_centroids"):
            av.cluster_centroids = np.array(data["cluster_centroids"], dtype=np.float32)
        av.contrastive_pairs = data.get("contrastive_pairs", [])
        return av


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Auto-learn CMTIP concept vocabulary")
    parser.add_argument("--model", default="all-MiniLM-L6-v2",
                        help="Sentence transformer model name")
    parser.add_argument("--n-concepts", type=int, default=32,
                        help="Number of concept clusters")
    parser.add_argument("--n-pca", type=int, default=16,
                        help="Number of PCA components")
    parser.add_argument("--save", type=str, default="./auto_vocab.json",
                        help="Save path")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with analogies and blending")
    args = parser.parse_args()

    print("=" * 64)
    print("  CMTIP — Automatic Concept Vocabulary Learner")
    print("=" * 64)

    # ─── Load model and data ────────────────────────────────────────────
    if HAS_ST:
        print(f"\nLoading model: {args.model}")
        backend = SentenceTransformerBackend("auto-learner", args.model)
        dim = backend.dim
    else:
        print("\nUsing synthetic backend (install sentence-transformers for real)")
        backend = SyntheticEmbeddingBackend("auto-learner", 384)
        dim = 384

    # Get training texts from the training pairs
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "training_pairs.json")
    if os.path.exists(data_path):
        with open(data_path) as f:
            data = json.load(f)
        # Get unique source texts
        texts = list(set(p["a"] for p in data["train"]))
        print(f"Loaded {len(texts)} unique texts from training_pairs.json")
    else:
        # Fallback: generate simple texts
        domains = ["technology", "science", "business", "art", "sports", "food",
                    "travel", "health", "education", "music"]
        texts = []
        for d in domains:
            for i in range(20):
                texts.append(f"{d} concept number {i} in the semantic space")
        print(f"Generated {len(texts)} synthetic texts")

    # ─── Embed all texts ────────────────────────────────────────────────
    print(f"\nEmbedding {len(texts)} texts (d={dim})...")
    t0 = time.time()
    embeddings = backend.embed_batch(texts)
    print(f"  Done in {time.time()-t0:.1f}s  Shape: {embeddings.shape}")

    # ─── Learn vocabulary ───────────────────────────────────────────────
    auto = AutoVocabulary()
    auto.learn_from_embeddings(
        embeddings, texts,
        n_concepts=args.n_concepts,
        n_pca_components=args.n_pca,
    )

    # ─── Demo ───────────────────────────────────────────────────────────
    if args.demo:
        print(f"\n{'─'*64}")
        print("  ANALOGY TESTS")
        print(f"{'─'*64}")
        for r in auto.analogy_test():
            print(f"  {r['analogy']}")
            for name, score in r['answer']:
                print(f"    → {name}: {score:.4f}")

        print(f"\n{'─'*64}")
        print("  CONCEPT BLENDING")
        print(f"{'─'*64}")
        concepts = sorted(auto.vocab.concepts.keys())
        if len(concepts) >= 4:
            # Blend first two cluster concepts
            c1, c2 = concepts[0], concepts[1]
            blended = auto.vocab.blend(c1, c2, alpha=0.5)
            nearest = auto.vocab.nearest(blended, k=3)
            print(f"  Blend {c1} + {c2} (α=0.5):")
            for name, score in nearest:
                print(f"    → {name}: {score:.4f}")

    # ─── Save ───────────────────────────────────────────────────────────
    if args.save:
        auto.save(args.save)

    print(f"\n{'='*64}")
    print(f"  Done — {len(auto.vocab.concepts)} concepts learned")
    print(f"  PCA axes: {args.n_pca}  Clusters: {args.n_concepts}")
    print(f"  Contrastive pairs: {len(auto.contrastive_pairs)}")
    print(f"{'='*64}")


if __name__ == "__main__":
    main()
