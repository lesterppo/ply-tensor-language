"""
CMTIP Ceiling Exploration — #2 (data scale) + #3 (nonlinear architectures)
═══════════════════════════════════════════════════════════════════════════════

Tests two approaches to raise the cross-family ceiling above 0.48:

  #2: Data scale — train with 50, 100, 200, 500, 1000, 3000+ pairs
  #3: Nonlinear architectures — Adam optimizer, 2-layer, 3-layer, residual

All measurements on the SAME 15 held-out pairs for fair comparison.
"""

import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import CmtipBus, NonlinearAdapter, CrossModelAdapter
from real_backends import SentenceTransformerBackend

HELD_OUT = [
    ("The new caching layer reduced latency by 40 percent", "Response times dropped two-fifths after adding the cache"),
    ("Quarterly earnings exceeded analyst expectations by a wide margin", "Financial results far surpassed what experts had predicted"),
    ("The authentication service experienced an outage during peak hours", "Login system went down at the busiest time of day"),
    ("Machine learning models require careful validation before deployment", "AI systems need thorough testing prior to going live"),
    ("Distributed systems must handle partial failures gracefully", "Networked architectures should degrade elegantly when components fail"),
    ("The team completed the migration ahead of schedule", "The move was finished earlier than planned"),
    ("Security audit revealed three critical vulnerabilities", "Security review found three severe weaknesses"),
    ("Customer satisfaction scores improved for the third consecutive quarter", "User happiness metrics rose for the third straight period"),
    ("The new API version introduces breaking changes to the authentication flow", "The updated interface modifies login behavior in incompatible ways"),
    ("Memory utilization spiked after the configuration change was deployed", "RAM usage surged following the settings update rollout"),
    ("The consensus algorithm ensures consistency across all replicas", "The agreement protocol guarantees uniformity among every copy"),
    ("Technical debt accumulated over years finally caused a production incident", "Years of deferred maintenance ultimately triggered a live issue"),
    ("Zero-trust architecture requires continuous authentication of every request", "Never-trust design demands ongoing verification of all queries"),
    ("The observability platform detected anomalous traffic patterns at 3 AM", "Monitoring tools spotted unusual request flows in the early morning"),
    ("Code review identified a race condition in the payment processing pipeline", "Peer inspection found a timing bug in the transaction handling flow"),
]

def measure(bus, src, tgt):
    """Measure alignment on held-out pairs."""
    adapter = bus.get_adapter(src, tgt)
    cos_sims = []
    for a, b in HELD_OUT:
        va = bus.models[src].embed(a); vb = bus.models[tgt].embed(b)
        vp = adapter.project(va)
        cos_sims.append(float(np.dot(vp, vb)/(np.linalg.norm(vp)*np.linalg.norm(vb)+1e-8)))
    return np.mean(cos_sims), cos_sims

def train_ols(bus, src, tgt, pairs):
    """Train OLS adapter."""
    X = np.array([bus.models[src].embed(a) for a,_ in pairs])
    Y = np.array([bus.models[tgt].embed(b) for _,b in pairs])
    adapter = bus.get_adapter(src, tgt)
    mse = adapter.fit(X, Y)
    return mse

def train_mlp_adam(src_dim, tgt_dim, X, Y, hidden=128, epochs=300, lr=0.001):
    """Train nonlinear MLP with Adam optimizer."""
    # Initialize
    rng = np.random.RandomState(42)
    W1 = rng.randn(src_dim, hidden).astype(np.float32) * np.sqrt(2.0/src_dim)
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.randn(hidden, tgt_dim).astype(np.float32) * np.sqrt(2.0/hidden)
    b2 = np.zeros(tgt_dim, dtype=np.float32)
    
    # Adam state
    mW1, vW1 = np.zeros_like(W1), np.zeros_like(W1)
    mb1, vb1 = np.zeros_like(b1), np.zeros_like(b1)
    mW2, vW2 = np.zeros_like(W2), np.zeros_like(W2)
    mb2, vb2 = np.zeros_like(b2), np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t = 0
    
    N = X.shape[0]
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_loss = 0
        for start in range(0, N, 64):
            batch = idx[start:start+64]
            Xb, Yb = X[batch], Y[batch]
            m = len(Xb)
            
            # Forward
            H = np.maximum(0, Xb @ W1 + b1)
            Yp = H @ W2 + b2
            loss = np.mean((Yp - Yb)**2)
            total_loss += loss * m
            
            # Backward
            dY = 2*(Yp - Yb)/m
            dW2 = H.T @ dY; db2 = dY.sum(0)
            dH = dY @ W2.T; dH[H <= 0] = 0
            dW1 = Xb.T @ dH; db1 = dH.sum(0)
            
            # Adam update
            t += 1
            for p, g, mp, vp in [
                (W1, dW1, mW1, vW1), (b1, db1, mb1, vb1),
                (W2, dW2, mW2, vW2), (b2, db2, mb2, vb2)
            ]:
                mp[:] = beta1*mp + (1-beta1)*g
                vp[:] = beta2*vp + (1-beta2)*g**2
                m_hat = mp/(1-beta1**t)
                v_hat = vp/(1-beta2**t)
                p[:] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            
            # Gradient clipping
            for arr in [W1, b1, W2, b2]:
                gnorm = np.sqrt(np.sum(arr**2))
                if gnorm > 10: arr *= 10/gnorm
        
        # LR decay
        lr *= 0.995
    
    # Return projection function
    adapter = NonlinearAdapter(src_dim, tgt_dim, hidden)
    adapter.W1, adapter.b1 = W1, b1
    adapter.W2, adapter.b2 = W2, b2
    return adapter, total_loss/N

def train_deep_adam(src_dim, tgt_dim, X, Y, hidden=128, epochs=300, lr=0.001):
    """3-layer MLP with residual connection and layer norm."""
    rng = np.random.RandomState(42)
    W1 = rng.randn(src_dim, hidden).astype(np.float32) * 0.1
    b1 = np.zeros(hidden, dtype=np.float32)
    W2 = rng.randn(hidden, hidden).astype(np.float32) * 0.1  # Middle layer
    b2 = np.zeros(hidden, dtype=np.float32)
    W3 = rng.randn(hidden, tgt_dim).astype(np.float32) * 0.1
    b3 = np.zeros(tgt_dim, dtype=np.float32)
    
    # Adam states (simplified — track all params)
    params = [W1, b1, W2, b2, W3, b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t_count = 0
    N = X.shape[0]
    
    for epoch in range(epochs):
        idx = np.random.permutation(N)
        total_loss = 0
        for start in range(0, N, 64):
            batch = idx[start:start+64]
            Xb, Yb = X[batch], Y[batch]
            msz = len(Xb)
            
            # Forward: layer1 → ReLU → layer2 → ReLU → layer3
            H1 = np.maximum(0, Xb @ W1 + b1)
            # Layer norm on H1
            h1_mean = H1.mean(axis=1, keepdims=True)
            h1_std = H1.std(axis=1, keepdims=True) + 1e-5
            H1_norm = (H1 - h1_mean) / h1_std
            
            H2_pre = H1_norm @ W2 + b2
            H2 = np.maximum(0, H2_pre)
            # Residual: add back H1_norm (skip connection)
            H2_res = H2 + 0.3 * H1_norm
            
            Yp = H2_res @ W3 + b3
            loss = np.mean((Yp - Yb)**2)
            total_loss += loss * msz
            
            # Backward (manual for 3 layers)
            dY = 2*(Yp - Yb)/msz
            dW3 = H2_res.T @ dY; db3 = dY.sum(0)
            dH2 = dY @ W3.T
            dH2[H2 <= 0] = 0
            dH1 = dH2 @ W2.T + 0.3 * dH2  # Residual gradient
            dW2 = H1_norm.T @ dH2; db2 = dH2.sum(0)
            dH1_raw = dH1 * (1.0 / h1_std)  # Layer norm backward
            dW1 = Xb.T @ dH1_raw; db1 = dH1_raw.sum(0)
            
            grads = [dW1, db1, dW2, db2, dW3, db3]
            
            # Adam + clip
            t_count += 1
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1*m[i] + (1-beta1)*g
                v[i] = beta2*v[i] + (1-beta2)*g**2
                mh = m[i]/(1-beta1**t_count)
                vh = v[i]/(1-beta2**t_count)
                p -= lr * mh / (np.sqrt(vh) + eps)
                gn = np.sqrt(np.sum(p**2))
                if gn > 10: p *= 10/gn
            
        lr *= 0.995
    
    # Return projection
    adapter = NonlinearAdapter(src_dim, tgt_dim, hidden)
    adapter.W1, adapter.b1 = W1, b1
    adapter.W2, adapter.b2 = W3, b3
    # Store deep params for projection override
    adapter._deep_W2 = W2; adapter._deep_b2 = b2
    adapter._deep_H1_norm = None
    
    # Override project for deep
    original_project = adapter.project
    def deep_project(vector):
        v = vector.reshape(1, -1).astype(np.float32)
        h1 = np.maximum(0, v @ W1 + b1)
        h1_m = h1.mean(1, keepdims=True); h1_s = h1.std(1, keepdims=True) + 1e-5
        h1_n = (h1 - h1_m) / h1_s
        h2 = np.maximum(0, h1_n @ W2 + b2) + 0.3 * h1_n
        out = (h2 @ W3 + b3).flatten()
        return out/(np.linalg.norm(out)+1e-8)
    adapter.project = deep_project
    
    return adapter, total_loss/N

def test_adapter_projection(adapter, X_test, Y_test):
    """Measure projection quality."""
    cos_sims = []
    for i in range(len(X_test)):
        vp = adapter.project(X_test[i])
        cos = float(np.dot(vp, Y_test[i])/(np.linalg.norm(vp)*np.linalg.norm(Y_test[i])+1e-8))
        cos_sims.append(cos)
    return np.mean(cos_sims), cos_sims

def main():
    print("="*64)
    print("  CEILING EXPLORATION: #2 Data Scale + #3 Nonlinear")
    print("="*64)
    
    # Load models once
    print("\nLoading models...")
    ma = SentenceTransformerBackend("MiniLM", "all-MiniLM-L6-v2")
    mb = SentenceTransformerBackend("MPNet", "all-mpnet-base-v2")
    print(f"  MiniLM: d={ma.dim}  MPNet: d={mb.dim}")
    
    # Load ALL data
    with open("data/training_pairs.json") as f:
        data = json.load(f)
    all_pairs = [(p["a"], p["b"]) for p in data["train"]]
    np.random.seed(42)
    idx = np.random.permutation(len(all_pairs))
    all_pairs = [all_pairs[i] for i in idx]
    
    print(f"  Total pairs available: {len(all_pairs)}")
    
    results = []
    
    # ═══════════════════════════════════════════════════════════
    # #2: DATA SCALE — measure alignment at increasing pair counts
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*64}")
    print("  #2: DATA SCALE — How many pairs to reach ceiling?")
    print(f"{'─'*64}")
    print(f"  {'Pairs':>6s} {'OLS cos':>10s} {'Best pair':>10s} {'Worst pair':>11s} {'>0.5':>8s}")
    print(f"  {'─'*54}")
    
    pair_counts = [50, 100, 200, 500, 1000, 2000, 3000]
    for n_pairs in pair_counts:
        if n_pairs > len(all_pairs):
            n_pairs = len(all_pairs)
        
        train_pairs = all_pairs[:n_pairs]
        
        bus = CmtipBus()
        bus.register_model(ma); bus.register_model(mb)
        train_ols(bus, "MiniLM", "MPNet", train_pairs)
        
        cos, per_pair = measure(bus, "MiniLM", "MPNet")
        pct_05 = sum(1 for c in per_pair if c > 0.5)/len(per_pair)*100
        
        marker = " ★" if n_pairs >= 500 and cos > 0.5 else ""
        print(f"  {n_pairs:>6d} {cos:>10.4f} {max(per_pair):>10.4f} {min(per_pair):>11.4f} {pct_05:>7.0f}%{marker}")
        
        results.append({"approach": "ols", "pairs": n_pairs, "cos": cos, "per_pair": per_pair})
        
        if cos > 0.8:
            break  # Ceiling reached
    
    # ═══════════════════════════════════════════════════════════
    # #3: NONLINEAR ARCHITECTURES
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'─'*64}")
    print("  #3: NONLINEAR — Adam optimizer, deeper architectures")
    print(f"{'─'*64}")
    
    # Use 1000 pairs for nonlinear training (enough data, not too slow)
    n_nl = min(1000, len(all_pairs))
    train_pairs = all_pairs[:n_nl]
    X = np.array([ma.embed(a) for a,_ in train_pairs])
    Y = np.array([mb.embed(b) for _,b in train_pairs])
    X_test = np.array([ma.embed(a) for a,_ in HELD_OUT])
    Y_test = np.array([mb.embed(b) for _,b in HELD_OUT])
    
    print(f"  Training nonlinear on {n_nl} pairs (384→768)")
    print(f"\n  {'Architecture':<30s} {'Held-out cos':>13s} {'Train loss':>12s} {'Best pair':>10s}")
    print(f"  {'─'*67}")
    
    # OLS baseline
    bus = CmtipBus(); bus.register_model(ma); bus.register_model(mb)
    train_ols(bus, "MiniLM", "MPNet", train_pairs)
    ols_cos, ols_pp = measure(bus, "MiniLM", "MPNet")
    print(f"  {'OLS (baseline)':<30s} {ols_cos:>13.4f} {'n/a':>12s} {max(ols_pp):>10.4f}")
    results.append({"approach": "nonlinear", "arch": "ols_baseline", "cos": ols_cos, "per_pair": ols_pp})
    
    # MLP-128 Adam
    print("  Training MLP-128 Adam...")
    t0 = time.time()
    mlp128, loss128 = train_mlp_adam(384, 768, X, Y, hidden=128, epochs=300)
    cos128, pp128 = test_adapter_projection(mlp128, X_test, Y_test)
    print(f"  {'MLP-128 Adam':<30s} {cos128:>13.4f} {loss128:>12.6f} {max(pp128):>10.4f}  ({time.time()-t0:.0f}s)")
    results.append({"approach": "nonlinear", "arch": "mlp128_adam", "cos": cos128, "per_pair": pp128})
    
    # MLP-64 Adam
    print("  Training MLP-64 Adam...")
    mlp64, loss64 = train_mlp_adam(384, 768, X, Y, hidden=64, epochs=300)
    cos64, pp64 = test_adapter_projection(mlp64, X_test, Y_test)
    print(f"  {'MLP-64 Adam':<30s} {cos64:>13.4f} {loss64:>12.6f} {max(pp64):>10.4f}")
    results.append({"approach": "nonlinear", "arch": "mlp64_adam", "cos": cos64, "per_pair": pp64})
    
    # MLP-256 Adam
    print("  Training MLP-256 Adam...")
    mlp256, loss256 = train_mlp_adam(384, 768, X, Y, hidden=256, epochs=300)
    cos256, pp256 = test_adapter_projection(mlp256, X_test, Y_test)
    print(f"  {'MLP-256 Adam':<30s} {cos256:>13.4f} {loss256:>12.6f} {max(pp256):>10.4f}")
    results.append({"approach": "nonlinear", "arch": "mlp256_adam", "cos": cos256, "per_pair": pp256})
    
    # Deep 3-layer with residual
    print("  Training Deep-3L-Residual Adam...")
    t0 = time.time()
    deep, loss_deep = train_deep_adam(384, 768, X, Y, hidden=128, epochs=300)
    cos_deep, pp_deep = test_adapter_projection(deep, X_test, Y_test)
    print(f"  {'Deep-3L-Residual Adam':<30s} {cos_deep:>13.4f} {loss_deep:>12.6f} {max(pp_deep):>10.4f}  ({time.time()-t0:.0f}s)")
    results.append({"approach": "nonlinear", "arch": "deep3l_residual", "cos": cos_deep, "per_pair": pp_deep})
    
    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*64}")
    print("  CEILING EXPLORATION — SUMMARY")
    print(f"{'='*64}")
    
    print(f"\n  #2 DATA SCALE:")
    best_data = max((r for r in results if r["approach"]=="ols"), key=lambda r: r["cos"])
    print(f"    Best: {best_data['pairs']} pairs → cos={best_data['cos']:.4f}")
    for r in results:
        if r["approach"] == "ols":
            bar = "█" * min(40, int(r["cos"]*50))
            print(f"    {r['pairs']:>5d} pairs  {r['cos']:.4f}  {bar}")
    
    print(f"\n  #3 NONLINEAR:")
    best_nl = max((r for r in results if r["approach"]=="nonlinear"), key=lambda r: r["cos"])
    print(f"    Best: {best_nl['arch']} → cos={best_nl['cos']:.4f}")
    for r in results:
        if r["approach"] == "nonlinear":
            bar = "█" * min(40, int(r["cos"]*50))
            print(f"    {r['arch']:<25s} {r['cos']:.4f}  {bar}")
    
    # Answer
    all_cos = [r["cos"] for r in results]
    best_overall = max(all_cos)
    best_method = [r for r in results if r["cos"] == best_overall][0]
    
    method_desc = best_method.get('arch') or f"{best_method.get('pairs', '?')} pairs OLS"
    print(f"\n  ★ CEILING: {best_overall:.4f}  via {method_desc}")
    print(f"    Improvement over baseline (20 pairs, 0.33): +{best_overall-0.33:.4f}")
    
    if best_method["approach"] == "nonlinear":
        print(f"    Nonlinear approaches matter more than data scale at this ceiling.")
    else:
        print(f"    Data scale matters more than architecture at this ceiling.")
    
    print(f"\n{'='*64}")

if __name__ == "__main__":
    main()
