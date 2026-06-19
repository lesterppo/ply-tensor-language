#!/usr/bin/env python3
"""
cmtip_chat.py — Working Cross-Model Tensor Conversation
Two free Nvidia models talk via concept tensors. Zero setup beyond API key.

Usage:
  python3 cmtip_chat.py
  python3 cmtip_chat.py --models llama-3.1-8b,gemma-2-2b
  python3 cmtip_chat.py --verbose

The two models take turns: one speaks, the other decodes from tensor.
Each message is encoded as a weighted concept vector (50+ concepts),
then the receiver reconstructs the message from those concept weights.
"""
import os, sys, json, math, urllib.request, urllib.error
from collections import OrderedDict

# ── Config ─────────────────────────────────────────────────
BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embed-v1"
MODEL_A = "meta/llama-3.1-8b-instruct"
MODEL_B = "google/gemma-2-2b-it"
TEMPERATURE = 0.1
MAX_TOKENS_DECODE = 100

# ── Expanded concept vocabulary (50+) ──────────────────────
# Grouped by domain so tensor signatures carry domain context
CONCEPTS = OrderedDict({
    # Technical states
    "crash": "system failure or crash",
    "overload": "resource exhaustion or overload",
    "timeout": "connection or request timeout",
    "corruption": "data corruption or integrity loss",
    "latency": "high latency or slowness",
    "healthy": "system running normally",
    "degraded": "degraded but functional",
    "recovery": "recovering from failure",
    "scaling": "scaling up or down",

    # Actions
    "deploy": "deploying or releasing code",
    "rollback": "rolling back a deployment",
    "restart": "restarting a service",
    "configure": "changing configuration",
    "monitor": "monitoring or observing",
    "debug": "debugging or investigating",
    "fix": "fixing a bug or issue",
    "optimize": "optimizing performance",
    "migrate": "migrating data or systems",
    "cleanup": "cleaning up resources",

    # Communication tones
    "urgent": "immediate attention needed",
    "casual": "informal or relaxed tone",
    "formal": "formal or official tone",
    "question": "asking a question",
    "answer": "providing an answer",
    "suggestion": "making a suggestion",
    "agreement": "agreeing or approving",
    "disagreement": "disagreeing or objecting",
    "gratitude": "expressing thanks",

    # Emotional/state
    "confident": "feeling confident or certain",
    "uncertain": "feeling uncertain or unsure",
    "worried": "feeling worried or anxious",
    "relieved": "feeling relieved",
    "frustrated": "feeling frustrated",
    "satisfied": "feeling satisfied",
    "curious": "feeling curious",
    "confused": "feeling confused",

    # Domain: databases
    "database": "relating to databases",
    "connection_pool": "connection pool issues",
    "query": "database query related",
    "index": "database index related",
    "replication": "database replication",
    "backup": "backup or restore",

    # Domain: networking
    "network": "network related",
    "dns": "DNS related",
    "firewall": "firewall related",
    "bandwidth": "bandwidth related",

    # Quantifiers
    "critical": "critically important",
    "minor": "minor or low priority",
    "temporary": "temporary or short-term",
    "permanent": "permanent or long-term",
})
concept_names = list(CONCEPTS.keys())

# ── API helpers ────────────────────────────────────────────
def load_key():
    # Try environment variable first (standard for CI/deployment)
    key = os.environ.get("NVIDIA_API_KEY", "")
    if key:
        return key
    # Fall back to local .env file
    for path in [".env", os.path.expanduser("~/.hermes/.env")]:
        try:
            with open(path) as f:
                for line in f:
                    if "NVIDIA_API_KEY" in line and "=" in line:
                        return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
        except FileNotFoundError:
            continue
    return 

KEY = load_key()
if not KEY:
    print("ERROR: NVIDIA_API_KEY not found. Set in ~/.hermes/.env or environment.")
    sys.exit(1)

HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def api(endpoint, data):
    url = f"{BASE}/{endpoint}"
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=HDR, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"_err": str(e)[:120]}

def chat(model, msg, max_t=100):
    r = api("chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_t, "temperature": TEMPERATURE
    })
    if "_err" in r:
        return f"[API ERROR: {r['_err'][:80]}]"
    return r["choices"][0]["message"]["content"]

def embed_batch(texts):
    r = api("embeddings", {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"})
    if "_err" in r:
        return None
    return [d["embedding"] for d in r["data"]]

def cos(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na*nb > 0 else 0

# ── CMTIP Engine ───────────────────────────────────────────
class CmtipEngine:
    """Encodes/decodes messages as weighted concept tensors."""

    def __init__(self):
        print(f"  Initializing CMTIP engine ({len(concept_names)} concepts)...")
        self.concept_vecs = {}
        # Encode concept definitions (not just names) for better semantics
        descs = [CONCEPTS[c] for c in concept_names]
        e = embed_batch(descs)
        if not e:
            print("  ERROR: Failed to encode concepts")
            sys.exit(1)
        self.concept_vecs = {c: e[i] for i, c in enumerate(concept_names)}
        self.dim = len(e[0])
        print(f"  Ready. {len(concept_names)} concepts, dim={self.dim}")

    def encode(self, text: str, top_n: int = 8) -> list:
        """Encode text as sorted list of (concept, weight) tuples."""
        v = embed_batch([text])
        if not v:
            return []
        v = v[0]
        scored = [(c, cos(v, self.concept_vecs[c])) for c in concept_names]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    def decode_prompt(self, concepts: list, model_name: str) -> str:
        """Build a prompt for the LLM to reconstruct from concept tensor."""
        sig = ", ".join(f"{c}({w:.2f})" for c, w in concepts)
        return (
            f"A message was compressed into these semantic weights:\n"
            f"  {sig}\n\n"
            f"Reconstruct the original message. Be specific and natural.\n"
            f"One sentence. Do NOT mention the weights or concepts.\n"
            f"Original message:"
        )

    def format_tensor(self, concepts: list) -> str:
        """Human-readable tensor display."""
        bars = []
        for c, w in concepts:
            bar_len = int(w * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            bars.append(f"  {c:20s} {bar} {w:.2f}")
        return "\n".join(bars)


# ── Main ───────────────────────────────────────────────────
def main():
    # Parse args first (before any use of globals)
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    model_a_override = None
    model_b_override = None
    for arg in sys.argv[1:]:
        if arg.startswith("--models="):
            parts = arg.split("=", 1)[1].split(",")
            model_a_override = parts[0].strip()
            model_b_override = parts[1].strip() if len(parts) > 1 else None

    # Apply overrides
    global MODEL_A, MODEL_B
    if model_a_override:
        MODEL_A = model_a_override
    if model_b_override:
        MODEL_B = model_b_override

    print("=" * 64)
    print("  CMTIP Chat — Cross-Model Tensor Conversation")
    print(f"  Model A: {MODEL_A.split('/')[-1]}")
    print(f"  Model B: {MODEL_B.split('/')[-1]}")
    print(f"  Embeddings: {EMBED_MODEL}")
    print("=" * 64)

    # Initialize engine
    engine = CmtipEngine()

    # ── Conversation ────────────────────────────────────
    conversation = [
        ("A", "The production database is throwing connection pool errors. We need to investigate."),
        ("B", "I checked the logs — the pool exhausted after that bulk import job. We can increase max_connections."),
        ("A", "Good catch. Set it to 200 temporarily and I'll optimize the import batch size."),
        ("B", "Done. Pool is now at 200. I'm monitoring for any other issues."),
        ("A", "Thanks! The import is running smoothly now with smaller batches."),
    ]

    for turn, (speaker, msg) in enumerate(conversation, 1):
        # Which models are encoding vs decoding
        encoder_model = MODEL_A if speaker == "A" else MODEL_B
        decoder_model = MODEL_B if speaker == "A" else MODEL_A
        encoder_name = encoder_model.split('/')[-1]
        decoder_name = decoder_model.split('/')[-1]

        # Step 1: Encode to concept tensor
        concepts = engine.encode(msg, top_n=8)

        # Step 2: Decode via the OTHER model
        prompt = engine.decode_prompt(concepts, decoder_name)
        decoded = chat(decoder_model, prompt, MAX_TOKENS_DECODE)
        decoded = decoded.strip()
        # Clean up common artifacts
        for prefix in ["Original message:", "The original message:", "Message:"]:
            if decoded.lower().startswith(prefix.lower()):
                decoded = decoded[len(prefix):].strip()

        # Step 3: Measure fidelity
        e = embed_batch([msg, decoded])
        fidelity = cos(e[0], e[1]) if e else 0

        # ── Display ─────────────────────────────────────
        print(f"\n─── Turn {turn} ───")
        print(f"[{speaker}] {encoder_name}: {msg}")
        if verbose:
            print(f"      Tensor ({len(concepts)} concepts):")
            print(engine.format_tensor(concepts))
        else:
            top3 = ", ".join(f"{c}({w:.2f})" for c, w in concepts[:3])
            print(f"      → {top3}")
        print(f"[{'B' if speaker == 'A' else 'A'}] {decoder_name} decoded: {decoded}")
        print(f"      fidelity: {fidelity:.3f}")

    # ── Summary ────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Conversation complete.")
    print(f"  This used ONLY free Nvidia API models.")
    print(f"  Each message: text → {len(concept_names)}-dim concept vector → text")
    print("=" * 64)


if __name__ == "__main__":
    main()
