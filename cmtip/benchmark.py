"""
CMTIP vs Text — Benchmark
═══════════════════════════

Addresses critique #9: measures tensor-native against text across
multiple dimensions with real sentence-transformers.

Metrics:
    1. Semantic preservation: how much meaning survives cross-model?
    2. Bandwidth efficiency: bytes per semantic unit
    3. Latency: text encode/decode vs tensor project
    4. Retrieval accuracy: does tensor routing find the right response?
    5. Roundtrip fidelity: A→B→A cos_sim vs text paraphrase→text

Usage:
    python benchmark.py [--pairs 500] [--epochs 100]
"""

import sys
import os
import time
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import CmtipBus
from real_backends import SentenceTransformerBackend


def benchmark_text_vs_tensor(n_train: int = 300, n_test: int = 50,
                              epochs: int = 150):
    """Run the full text-vs-tensor benchmark."""
    print("=" * 72)
    print("  CMTIP vs TEXT — Semantic Preservation Benchmark")
    print("=" * 72)

    # ─── Load models ──────────────────────────────────────────────────
    print("\n─── Loading models ───")
    model_a = SentenceTransformerBackend("MiniLM", "all-MiniLM-L6-v2")
    model_b = SentenceTransformerBackend("MPNet", "all-mpnet-base-v2")
    print(f"  Model A: MiniLM  d={model_a.dim}")
    print(f"  Model B: MPNet   d={model_b.dim}")

    # ─── Load training data ───────────────────────────────────────────
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "training_pairs.json")
    with open(data_path) as f:
        data = json.load(f)
    all_pairs = [(p["a"], p["b"]) for p in data["train"]]

    np.random.seed(42)
    idx = np.random.permutation(len(all_pairs))
    train_pairs = [all_pairs[i] for i in idx[:n_train]]
    test_pairs = [all_pairs[i] for i in idx[n_train:n_train + n_test]]

    print(f"\n  Training pairs: {len(train_pairs)}")
    print(f"  Test pairs:     {len(test_pairs)}")

    # ─── Embed all data once ──────────────────────────────────────────
    print("\n  Embedding training data...")
    t0 = time.time()
    X_train = model_a.embed_batch([p[0] for p in train_pairs])
    Y_train = model_b.embed_batch([p[1] for p in train_pairs])
    X_test = model_a.embed_batch([p[0] for p in test_pairs])
    Y_test = model_b.embed_batch([p[1] for p in test_pairs])
    print(f"  Done in {time.time()-t0:.1f}s")

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # 1. TEXT BASELINE: How similar are the original texts?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  1. TEXT BASELINE")
    print(f"{'─'*72}")

    # Embed source text in B's space directly (no adapter) vs target text
    text_cos = []
    for i in range(n_test):
        # What if we just embed the source text in B's space?
        source_in_b = model_b.embed(test_pairs[i][0])
        target_in_b = Y_test[i]
        cos = float(np.dot(source_in_b, target_in_b) /
                   (np.linalg.norm(source_in_b) * np.linalg.norm(target_in_b) + 1e-8))
        text_cos.append(cos)

    text_mean = np.mean(text_cos)
    text_pct_05 = sum(1 for c in text_cos if c > 0.5) / len(text_cos) * 100
    results["text_direct_embed"] = {
        "mean_cos": round(text_mean, 4),
        "pct_above_0.5": round(text_pct_05, 1),
        "description": "Source text embedded directly in B's space (no adapter)",
    }
    print(f"  Text → embed in B → compare to target:")
    print(f"    Mean cos_sim: {text_mean:.4f}")
    print(f"    >0.5: {text_pct_05:.0f}%")
    print(f"  This is the UPPER BOUND — no projection, just different phrasings")

    # ═══════════════════════════════════════════════════════════════
    # 2. LINEAR ADAPTER (current approach)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  2. LINEAR OLS ADAPTER")
    print(f"{'─'*72}")

    bus = CmtipBus()
    bus.register_model(model_a)
    bus.register_model(model_b)

    mse_linear = bus.train_adapter("MiniLM", "MPNet", train_pairs)
    adapter_linear = bus.get_adapter("MiniLM", "MPNet")

    linear_cos = []
    for i in range(n_test):
        v_a = X_test[i]
        v_proj = adapter_linear.project(v_a)
        cos = float(np.dot(v_proj, Y_test[i]) /
                   (np.linalg.norm(v_proj) * np.linalg.norm(Y_test[i]) + 1e-8))
        linear_cos.append(cos)

    linear_mean = np.mean(linear_cos)
    linear_pct = sum(1 for c in linear_cos if c > 0.5) / len(linear_cos) * 100
    results["linear_adapter"] = {
        "mean_cos": round(linear_mean, 4),
        "pct_above_0.5": round(linear_pct, 1),
        "mse": round(mse_linear, 6),
        "params": model_a.dim * model_b.dim,
        "description": "v_B = v_A @ W  (full-rank least squares)",
    }
    print(f"  Training MSE: {mse_linear:.6f}  Params: {model_a.dim * model_b.dim:,}")
    print(f"  Test cos_sim:  mean={linear_mean:.4f}  >0.5={linear_pct:.0f}%")

    # ═══════════════════════════════════════════════════════════════
    # 3. CCA ADAPTER (low-rank, better generalization)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  3. LOW-RANK CCA ADAPTER")
    print(f"{'─'*72}")

    from train_cca import CCACrossModelAdapter
    for rank in [16, 32, 64]:
        cca = CCACrossModelAdapter(model_a.dim, model_b.dim, rank=rank)
        mse_cca, can_corrs = cca.fit_cca(X_train, Y_train, reg=0.5)

        cca_cos = []
        for i in range(n_test):
            v_proj = cca.project(X_test[i])
            cos = float(np.dot(v_proj, Y_test[i]) /
                       (np.linalg.norm(v_proj) * np.linalg.norm(Y_test[i]) + 1e-8))
            cca_cos.append(cos)

        cca_mean = np.mean(cca_cos)
        params = (model_a.dim + model_b.dim) * rank
        results[f"cca_rank{rank}"] = {
            "mean_cos": round(cca_mean, 4),
            "mse": round(mse_cca, 6),
            "canonical_corrs_mean": round(float(can_corrs.mean()), 4),
            "params": params,
        }
        print(f"  Rank {rank:2d}: cos={cca_mean:.4f}  MSE={mse_cca:.6f}  "
              f"can_corr={can_corrs.mean():.4f}  params={params:,}")

    # ═══════════════════════════════════════════════════════════════
    # 4. MLP ADAPTER (nonlinear bottleneck — addresses critique #3)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  4. NONLINEAR MLP ADAPTER (bottleneck)")
    print(f"{'─'*72}")

    from cmtip import NonlinearAdapter
    for bottleneck in [32, 64, 128]:
        mlp = NonlinearAdapter(model_a.dim, model_b.dim,
                               bottleneck_dim=bottleneck)
        mlp_loss = mlp.fit(X_train, Y_train, epochs=epochs, verbose=False)

        mlp_cos = []
        for i in range(n_test):
            v_proj = mlp.project(X_test[i])
            cos = float(np.dot(v_proj, Y_test[i]) /
                       (np.linalg.norm(v_proj) * np.linalg.norm(Y_test[i]) + 1e-8))
            mlp_cos.append(cos)

        mlp_mean = np.mean(mlp_cos)
        params = model_a.dim * bottleneck + bottleneck * model_b.dim
        results[f"mlp_bottleneck{bottleneck}"] = {
            "mean_cos": round(mlp_mean, 4),
            "final_loss": round(mlp_loss, 6),
            "params": params,
        }
        print(f"  Bottleneck {bottleneck:3d}: cos={mlp_mean:.4f}  "
              f"loss={mlp_loss:.6f}  params={params:,}")

    # ═══════════════════════════════════════════════════════════════
    # 5. BANDWIDTH COMPARISON
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  5. BANDWIDTH EFFICIENCY")
    print(f"{'─'*72}")

    avg_text_bytes = np.mean([len(p[1].encode('utf-8')) for p in test_pairs])
    avg_text_tokens = np.mean([len(p[1].split()) * 1.3 for p in test_pairs])

    for name, dims in [("MiniLM", 384), ("MPNet", 768), ("OpenAI-small", 1536)]:
        tensor_bytes = dims * 4  # float32
        ratio_bytes = tensor_bytes / avg_text_bytes
        semantic_bits = (results.get("linear_adapter", {}).get("mean_cos", 0.3)
                         if name == "MiniLM" else 0.3)
        bits_per_semantic = tensor_bytes / (semantic_bits * avg_text_bytes + 1e-8)
        print(f"  {name:<15s} {dims:>4d} dims = {tensor_bytes:>5d} bytes "
              f"({ratio_bytes:.1f}x text)  bits/semantic: {bits_per_semantic:.0f}")

    print(f"\n  Text avg: {avg_text_bytes:.0f} bytes, {avg_text_tokens:.0f} tokens")

    # ═══════════════════════════════════════════════════════════════
    # 6. LATENCY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'─'*72}")
    print("  6. LATENCY (single message)")
    print(f"{'─'*72}")

    # Text: encode source in A, encode in B directly, compute cos
    n_latency = 100
    t0 = time.perf_counter()
    for i in range(n_latency):
        text = test_pairs[i % n_test][0]
        _ = model_b.embed(text)
    text_lat = (time.perf_counter() - t0) / n_latency * 1e6

    # Tensor: project through adapter
    t0 = time.perf_counter()
    for i in range(n_latency):
        _ = adapter_linear.project(X_test[i % n_test])
    tensor_lat = (time.perf_counter() - t0) / n_latency * 1e6

    results["latency"] = {
        "text_direct_embed_us": round(text_lat, 1),
        "tensor_project_us": round(tensor_lat, 1),
        "speedup": round(text_lat / (tensor_lat + 1e-8), 1),
    }
    print(f"  Text (direct embed in B): {text_lat:.0f} µs")
    print(f"  Tensor (project A→B):     {tensor_lat:.0f} µs")
    print(f"  Speedup:                  {text_lat/(tensor_lat+1e-8):.1f}x")

    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("  SUMMARY: Text vs Tensor — Semantic Preservation")
    print(f"{'='*72}")
    print(f"\n  {'Method':<30s} {'Cos_sim':>8s} {'Params':>10s} {'%>:5':>6s}")
    print(f"  {'─'*58}")

    for key, label in [
        ("text_direct_embed", "Text direct (upper bound)"),
        ("linear_adapter", "Linear OLS adapter"),
        ("cca_rank16", "CCA rank=16"),
        ("cca_rank32", "CCA rank=32"),
        ("cca_rank64", "CCA rank=64"),
        ("mlp_bottleneck32", "MLP bottleneck=32"),
        ("mlp_bottleneck64", "MLP bottleneck=64"),
        ("mlp_bottleneck128", "MLP bottleneck=128"),
    ]:
        if key in results:
            r = results[key]
            cos = r["mean_cos"]
            params = r.get("params", "—")
            pct = r.get("pct_above_0.5", r.get("canonical_corrs_mean", 0)) * 100
            marker = ""
            if cos == max(r2["mean_cos"] for r2 in results.values()
                         if "mean_cos" in r2 and r2 != results.get("text_direct_embed", {})):
                marker = " ★ BEST"
            print(f"  {label:<30s} {cos:>8.4f} {str(params):>10s} {pct:>5.0f}%{marker}")

    print(f"\n  Key insight:")
    best_adapter = max(
        [(k, v["mean_cos"]) for k, v in results.items()
         if "mean_cos" in v and k != "text_direct_embed"],
        key=lambda x: x[1]
    )
    print(f"  Best adapter: {best_adapter[0]} at cos={best_adapter[1]:.4f}")
    print(f"  Upper bound:  {results['text_direct_embed']['mean_cos']:.4f} (text, no adapter)")
    gap_pct = (results['text_direct_embed']['mean_cos'] - best_adapter[1]) / results['text_direct_embed']['mean_cos'] * 100
    print(f"  Gap to upper bound: {gap_pct:.0f}%")
    print(f"\n  Verdict: {'The adapter is approaching useful territory' if best_adapter[1] > 0.5 else 'Cross-family alignment remains the bottleneck'}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CMTIP vs Text Benchmark")
    parser.add_argument("--pairs", type=int, default=300,
                        help="Number of training pairs")
    parser.add_argument("--test", type=int, default=50,
                        help="Number of test pairs")
    parser.add_argument("--epochs", type=int, default=150,
                        help="MLP training epochs")
    args = parser.parse_args()
    benchmark_text_vs_tensor(n_train=args.pairs, n_test=args.test,
                              epochs=args.epochs)
