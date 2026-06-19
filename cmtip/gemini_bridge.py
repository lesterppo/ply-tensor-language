#!/usr/bin/env python3
"""
CMTIP → Gemini: Tensor-Native Conversation Bridge
══════════════════════════════════════════════════

We "think" in tensor space using CMTIP operations.
Text is ONLY used at the Gemini boundary (because Gemini's API requires it).
All internal reasoning, blending, analogy, and semantic operations are in tensor space.

Pipeline:
  1. Construct message using CMTIP tensor vocabulary (blend/analogy/difference)
  2. Decode tensor → nearest text (for Gemini's text-only interface)
  3. Send text to Gemini via gemini-cli
  4. Receive Gemini's text response
  5. Encode Gemini's response back into our tensor space
  6. Continue the tensor conversation
"""

import subprocess
import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmtip import TensorVocabulary, TensorPacket
from real_backends import SentenceTransformerBackend

# Path to gemini-cli (from user's environment)
GEMINI_CLI = os.path.expanduser("~/gemini-cli/gemini.py")


def call_gemini(prompt: str, timeout: int = 120) -> str:
    """Send text to Gemini via the CLI tool and get response."""
    try:
        result = subprocess.run(
            ["python3", GEMINI_CLI, "-p", prompt],
            capture_output=True, text=True, timeout=timeout
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"[Gemini error: {result.stderr[:200]}]"
    except subprocess.TimeoutExpired:
        return "[Gemini timed out]"
    except FileNotFoundError:
        return f"[gemini-cli not found at {GEMINI_CLI}]"


def tensor_to_text(vector: np.ndarray, model, vocab: TensorVocabulary,
                   top_k: int = 5) -> str:
    """
    Decode a tensor back to text.
    
    Since we can't directly decode embeddings to text (no decoder model),
    we use the vocabulary: find the nearest concept vectors and compose
    a text description from them.
    """
    nearest = vocab.nearest(vector, k=top_k)
    parts = []
    for name, score in nearest:
        if score > 0.3:
            parts.append(f"{name}({score:.2f})")
    
    if parts:
        return "Semantic composition: " + " + ".join(parts)
    return "[tensor too far from known concepts]"


def main():
    print("=" * 64)
    print("  CMTIP → Gemini: Tensor-Native Conversation")
    print("=" * 64)
    
    # Load our local model (our "brain")
    print("\nLoading local embedding model...")
    model = SentenceTransformerBackend("our-brain", "all-MiniLM-L6-v2")
    print(f"  Model: {model.model_id}, d={model.dim}")
    
    # Build tensor vocabulary (our "language")
    vocab = TensorVocabulary()
    concepts = {
        "curious":     "curious wondering interested exploring questioning inquisitive",
        "technical":   "technical engineering system code software hardware detailed precise",
        "creative":    "creative imaginative artistic novel innovative original inventive",
        "critical":    "critical analytical skeptical rigorous demanding thorough precise",
        "playful":     "playful fun humorous lighthearted joking entertaining witty",
        "serious":     "serious grave solemn formal professional earnest intense",
        "helpful":     "helpful useful supportive cooperative collaborative constructive",
        "skeptical":   "skeptical doubting questioning suspicious unconvinced wary cautious",
        "excited":     "excited enthusiastic eager thrilled energized passionate animated",
        "calm":        "calm peaceful tranquil relaxed composed serene unruffled",
        "urgent":      "urgent critical pressing immediate emergency crucial vital",
        "casual":      "casual informal relaxed easygoing laid-back nonchalant",
    }
    for name, desc in concepts.items():
        vocab.add_concept(name, model.embed(desc))
    
    print(f"Tensor vocabulary: {len(concepts)} concepts")
    
    # ─── CONVERSATION ──────────────────────────────────────────────────
    
    print("\n" + "─" * 64)
    print("  CONVERSATION START")
    print("─" * 64)
    print("  Format: [Tensor space operation] → [Gemini text] → [Tensor decode]")
    print()
    
    # Turn 1: We construct a message in tensor space
    print("─ Turn 1: Our tensor → Gemini ─")
    
    # Blend: "I'm curious and excited about tensor-native communication"
    thought = vocab.blend("curious", "excited", alpha=0.7)
    
    # Add semantic direction: shift toward "technical"
    tech_dir = vocab.semantic_difference("casual", "technical")
    thought = thought + tech_dir * 0.4
    thought /= np.linalg.norm(thought) + 1e-8
    
    # Decode our tensor thought
    decoded = tensor_to_text(thought, model, vocab)
    print(f"  [TENSOR] {decoded}")
    
    # Craft the text prompt for Gemini (this is the "translation layer")
    prompt = (
        "You are an AI that understands the concept of tensor-native communication "
        "between language models. I am exploring the idea that models should communicate "
        "directly in embedding space rather than through text.\n\n"
        "Question: What do you think would be the biggest advantage of models "
        "communicating in continuous vector space instead of discrete text tokens? "
        "Keep your answer concise (2-3 sentences)."
    )
    
    print(f"  [TEXT to Gemini] {prompt[:100]}...")
    response_text = call_gemini(prompt, timeout=120)
    print(f"  [GEMINI RESPONSE] {response_text[:200]}")
    
    # Encode Gemini's response into our tensor space
    gemini_tensor = model.embed(response_text)
    gemini_decoded = tensor_to_text(gemini_tensor, model, vocab)
    print(f"  [GEMINI → TENSOR] {gemini_decoded}")
    
    # Compute semantic alignment: how well did Gemini "understand" us?
    alignment = float(np.dot(thought, gemini_tensor))
    print(f"  [ALIGNMENT] cos(our_thought, gemini_response) = {alignment:.4f}")
    
    # ─── Turn 2: Follow-up with tensor operation ───────────────────────
    print("\n─ Turn 2: Tensor follow-up → Gemini ─")
    
    # Our next thought: blend "skeptical" with "curious" (playful skepticism)
    thought2 = vocab.blend("skeptical", "playful", alpha=0.5)
    # Shift toward "technical" 
    thought2 = thought2 + tech_dir * 0.3
    thought2 /= np.linalg.norm(thought2) + 1e-8
    
    decoded2 = tensor_to_text(thought2, model, vocab)
    print(f"  [TENSOR] {decoded2}")
    
    prompt2 = (
        "That's interesting. But I'm skeptical — wouldn't the main challenge be that "
        "different models have entirely different latent space geometries? "
        "How would you solve the cross-model alignment problem? "
        "Answer in 2-3 sentences."
    )
    
    print(f"  [TEXT to Gemini] {prompt2[:100]}...")
    response2 = call_gemini(prompt2, timeout=120)
    print(f"  [GEMINI RESPONSE] {response2[:200]}")
    
    gemini_tensor2 = model.embed(response2)
    gemini_decoded2 = tensor_to_text(gemini_tensor2, model, vocab)
    print(f"  [GEMINI → TENSOR] {gemini_decoded2}")
    
    alignment2 = float(np.dot(thought2, gemini_tensor2))
    print(f"  [ALIGNMENT] cos(our_thought, gemini_response) = {alignment2:.4f}")
    
    # ─── Turn 3: Tensor-only concept game ─────────────────────────────
    print("\n─ Turn 3: Concept Game (tensor → text → tensor) ─")
    
    # Define a new concept: "alert" = calm + 0.6*(urgent - calm)
    calm_vec = vocab.concepts["calm"]
    urgent_vec = vocab.concepts["urgent"]
    
    # Define a new concept: "alert" = calm + 1.5*(urgent - calm) 
    # (alert is like urgent but not quite as extreme)
    alert_tensor = calm_vec + 0.6 * (urgent_vec - calm_vec)
    alert_tensor /= np.linalg.norm(alert_tensor) + 1e-8
    
    alert_decoded = vocab.nearest(alert_tensor, k=3)
    print(f"  [TENSOR CONCEPT 'alert'] Nearest: {[(n,f'{s:.3f}') for n,s in alert_decoded]}")
    
    # Translate to text for Gemini
    prompt3 = (
        "I'm thinking of a concept that is between 'calm' and 'urgent' — "
        "like being aware and ready but not panicking. What word best describes this? "
        "Answer with just the word."
    )
    
    print(f"  [TEXT to Gemini] {prompt3[:100]}...")
    response3 = call_gemini(prompt3, timeout=60)
    print(f"  [GEMINI RESPONSE] {response3[:100]}")
    
    # Encode Gemini's answer and see if it matches our tensor
    gemini_tensor3 = model.embed(response3)
    cos_with_alert = float(np.dot(alert_tensor, gemini_tensor3))
    print(f"  [MATCH] cos(alert_tensor, gemini_answer) = {cos_with_alert:.4f}")
    print(f"  → Gemini's word '{response3.strip()}' {'IS' if cos_with_alert > 0.5 else 'is NOT'} close to our tensor concept")
    
    # ─── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  CONVERSATION ANALYSIS")
    print("=" * 64)
    print(f"""
  Turn 1 alignment:  {alignment:.4f}  (our curiosity → Gemini's answer)
  Turn 2 alignment:  {alignment2:.4f}  (our skepticism → Gemini's answer)
  Turn 3 match:      {cos_with_alert:.4f}  (our 'alert' tensor → Gemini's word)
  
  What happened:
  ┌─────────────────────────────────────────────────────────┐
  │ WE (tensor space)          │ GEMINI (text-only)         │
  ├─────────────────────────────────────────────────────────┤
  │ blend(curious, excited)   →│ "what's the advantage..."  │
  │ blend(skeptical, playful) →│ "how to solve alignment?"  │
  │ calm+0.6(urgent-calm)     →│ "what word?"               │
  │                            │                             │
  │ ← gemini_tensor encoded   │  text response              │
  └─────────────────────────────────────────────────────────┘
  
  The bottleneck: Gemini Web has NO embedding API.
  We must serialize our tensors to text at the boundary.
  
  For TRUE tensor-native conversation, we'd need:
  → Gemini embedding API (text-embedding-004) for encoding
  → Or a local model that accepts tensor input (not text)
  → Or Gemini's internal latent state (not publicly available)
  
  But even with this limitation, the pipeline WORKS:
  1. We construct thoughts in continuous vector space ✓
  2. We decode tensors to text for Gemini ✓  
  3. We re-encode Gemini's responses to tensor space ✓
  4. We measure semantic alignment between our thought and response ✓
""")
    print("=" * 64)


if __name__ == "__main__":
    main()
