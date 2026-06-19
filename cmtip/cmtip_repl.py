#!/usr/bin/env python3
"""
cmtip_repl.py — Interactive Cross-Model Tensor REPL
Type messages, see them encoded as concept tensors, decoded by the other model.
Concept memory persists across turns for coherent multi-turn conversation.

All on free Nvidia API models.
"""
import os, sys, json, math, urllib.request, urllib.error
from collections import OrderedDict
import readline  # enables line editing

# ══════════════════════════════════════════════════════════
BASE = "https://integrate.api.nvidia.com/v1"
EMBED_MODEL = "nvidia/nv-embed-v1"
MODEL_A = "meta/llama-3.1-8b-instruct"
MODEL_B = "google/gemma-2-2b-it"

CONCEPTS = OrderedDict({
    "connection_pool": "connection pool management",
    "timeout": "connection or request timeout",
    "latency": "high latency or slowness",
    "overload": "resource exhaustion or overload",
    "crash": "system failure or crash",
    "healthy": "system running normally",
    "degraded": "degraded but functional",
    "recovery": "recovering from failure",
    "scaling": "scaling up or down",
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
    "urgent": "immediate attention needed",
    "casual": "informal tone",
    "suggestion": "making a suggestion",
    "agreement": "agreeing or approving",
    "disagreement": "disagreeing or objecting",
    "gratitude": "expressing thanks",
    "confident": "feeling confident",
    "uncertain": "feeling uncertain",
    "worried": "feeling worried",
    "relieved": "feeling relieved",
    "frustrated": "feeling frustrated",
    "satisfied": "feeling satisfied",
    "database": "relating to databases",
    "query": "database query related",
    "replication": "database replication",
    "backup": "backup or restore",
    "network": "network related",
    "firewall": "firewall related",
    "bandwidth": "bandwidth related",
    "critical": "critically important",
    "minor": "minor or low priority",
    "temporary": "temporary or short-term",
})
concept_names = list(CONCEPTS.keys())

# ══════════════════════════════════════════════════════════
def load_key():
    for path in [os.path.expanduser("~/.hermes/.env"), ".env"]:
        try:
            with open(path) as f:
                for line in f:
                    if "NVIDIA_API_KEY" in line and "=" in line:
                        return line.strip().split("=",1)[1].strip().strip('"').strip("'")
        except FileNotFoundError: continue
    return os.environ.get("NVIDIA_API_KEY", "")

KEY = load_key()
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def api(endpoint, data):
    url = f"{BASE}/{endpoint}"
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers=HDR, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        return {"_err": str(e)[:120]}

def chat(model, msg, max_t=150):
    r = api("chat/completions", {
        "model": model, "messages": [{"role": "user", "content": msg}],
        "max_tokens": max_t, "temperature": 0.2
    })
    if "_err" in r: return None
    return r["choices"][0]["message"]["content"]

def embed_batch(texts):
    r = api("embeddings", {"model": EMBED_MODEL, "input": texts, "encoding_format": "float"})
    if "_err" in r: return None
    return [d["embedding"] for d in r["data"]]

def cos(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na*nb > 0 else 0

# ══════════════════════════════════════════════════════════
class CmtipRepl:
    def __init__(self):
        print("  Initializing concept vocabulary...", end=" ", flush=True)
        descs = [CONCEPTS[c] for c in concept_names]
        e = embed_batch(descs)
        self.concept_vecs = {c: e[i] for i, c in enumerate(concept_names)}
        self.concept_memory = []  # accumulates across turns
        self.turn = 0
        self.speaker = "A"  # alternates A/B
        print("ready.\n")

    def encode(self, text, top_n=6):
        v = embed_batch([text])
        if not v: return []
        v = v[0]
        scored = [(c, cos(v, self.concept_vecs[c])) for c in concept_names]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_n]

    def decode(self, concepts, decoder_model):
        sig = ", ".join(f"{c}({w:.2f})" for c, w in concepts)
        # Include recent concept memory for coherence
        memory = ""
        if self.concept_memory:
            recent = self.concept_memory[-6:]  # last 3 turns × 2 messages
            flat = []
            for turn_concepts in recent:
                flat.extend(turn_concepts)
            if flat:
                # Aggregate recent concept weights
                agg = {}
                for c, w in flat:
                    agg[c] = agg.get(c, 0) + w
                top_agg = sorted(agg.items(), key=lambda x: -x[1])[:5]
                mem_sig = ", ".join(f"{c}({w:.1f})" for c, w in top_agg)
                memory = f"\nRecent topic memory: {mem_sig}"

        prompt = (
            f"You intercepted a compressed message. Semantic signature:{memory}\n"
            f"  Current: {sig}\n\n"
            f"What was the EXACT original message? Be specific. "
            f"Use concrete details (numbers, tools, actions). One sentence."
        )
        decoded = chat(decoder_model, prompt, max_t=120)
        if not decoded: return "[API error]"
        decoded = decoded.strip()
        for p in ["The message was:", "Original message:", "The original message was:",
                  "The intercepted message was:", "The communication was:"]:
            if decoded.lower().startswith(p.lower()):
                decoded = decoded[len(p):].strip()
        return decoded

    def run(self):
        print("  CMTIP Tensor REPL")
        print(f"  Type messages. Models: {MODEL_A.split('/')[-1]} ↔ {MODEL_B.split('/')[-1]}")
        print(f"  Commands: /switch, /concepts, /memory, /quit")
        print("  " + "─" * 62)

        while True:
            try:
                msg = input(f"\n  [{self.speaker}] > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye.")
                break

            if not msg:
                continue

            if msg.startswith("/"):
                self._handle_command(msg)
                continue

            self.turn += 1
            encoder = MODEL_A if self.speaker == "A" else MODEL_B
            decoder = MODEL_B if self.speaker == "A" else MODEL_A
            enc_name = encoder.split('/')[-1]
            dec_name = decoder.split('/')[-1]

            # Encode
            concepts = self.encode(msg)
            self.concept_memory.append(concepts)
            if len(self.concept_memory) > 20:  # keep last 10 turns
                self.concept_memory = self.concept_memory[-20:]

            # Display tensor
            bars = "  ".join(f"{c}:{'█'*int(w*10)}{'░'*(10-int(w*10))}" for c, w in concepts[:5])
            print(f"  ╰─ tensor: {bars}")

            # Decode
            decoded = self.decode(concepts, decoder)
            e = embed_batch([msg, decoded])
            fid = cos(e[0], e[1]) if e else 0

            # Quality indicator
            bar = "●" if fid > 0.6 else ("◐" if fid > 0.4 else "○")
            print(f"  [{dec_name}] decoded: {decoded}")
            print(f"  fidelity: {bar} {fid:.3f}")

            # Alternate speaker
            self.speaker = "B" if self.speaker == "A" else "A"

    def _handle_command(self, cmd):
        cmd = cmd.lower()
        if cmd in ("/quit", "/q", "/exit"):
            print("  Goodbye.")
            sys.exit(0)
        elif cmd in ("/switch", "/s"):
            self.speaker = "B" if self.speaker == "A" else "A"
            print(f"  Switched to speaker {self.speaker}")
        elif cmd in ("/concepts", "/c"):
            print(f"  Concepts: {', '.join(concept_names)}")
        elif cmd in ("/memory", "/m"):
            if not self.concept_memory:
                print("  No memory yet.")
            else:
                print(f"  Memory: {len(self.concept_memory)} turns stored")
                for i, cs in enumerate(self.concept_memory[-6:]):
                    top3 = ", ".join(f"{c}({w:.2f})" for c, w in cs[:3])
                    print(f"    turn {max(1, self.turn - len(self.concept_memory) + i + 1)}: {top3}")
        elif cmd in ("/help", "/h"):
            print("  Commands: /switch /concepts /memory /quit")
        else:
            print(f"  Unknown: {cmd}")

if __name__ == "__main__":
    repl = CmtipRepl()
    repl.run()
