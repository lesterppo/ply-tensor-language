"""
CMTIP SFT Dataset Generator
════════════════════════════

P3: Generate Supervised Fine-Tuning data for teaching LLMs to compose
tensor-native thoughts.

The dataset maps natural-language intents to concept blend vectors:
    "Ask Model B about the deployment issue with urgency"
    → {"concepts": {"urgent": 0.9, "technical": 0.7, "concerned": 0.5},
       "target": "analyst-model",
       "thought_text": "The deployment pipeline is failing..."}

This trains an LLM to:
    1. Recognize when tensor-native communication is appropriate
    2. Decompose intent into concept blends with correct weights
    3. Choose the right target model
    4. Compose the accompanying text context

Usage:
    python generate_sft.py [--output cmtip_sft.jsonl] [--size 500]
"""

import json
import random
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════
# Concept Taxonomy
# ═══════════════════════════════════════════════════════════════

# Domain → available concept axes with example blends
CONCEPT_TAXONOMY = {
    "urgency": {
        "axes": [
            ("urgent", "immediate, critical, emergency, time-sensitive, pressing"),
            ("calm", "relaxed, measured, unhurried, steady, composed"),
        ],
        "description": "Temporal pressure and priority level",
    },
    "certainty": {
        "axes": [
            ("confident", "certain, sure, convinced, definite, assured"),
            ("skeptical", "doubtful, questioning, uncertain, cautious, wary"),
        ],
        "description": "Confidence level in the information",
    },
    "technical_depth": {
        "axes": [
            ("technical", "detailed, specialized, engineering, system-level, precise"),
            ("abstract", "conceptual, high-level, theoretical, general, philosophical"),
        ],
        "description": "Level of technical detail",
    },
    "sentiment": {
        "axes": [
            ("optimistic", "hopeful, positive, encouraging, bright, promising"),
            ("concerned", "worried, troubled, uneasy, anxious, apprehensive"),
        ],
        "description": "Emotional tone of the communication",
    },
    "collaboration": {
        "axes": [
            ("collaborative", "cooperative, joint, shared, team-based, mutual"),
            ("directive", "commanding, authoritative, decisive, firm, instructing"),
        ],
        "description": "Interaction style",
    },
    "novelty": {
        "axes": [
            ("creative", "innovative, novel, original, imaginative, inventive"),
            ("conventional", "standard, traditional, established, routine, typical"),
        ],
        "description": "How novel or standard the content is",
    },
    "scope": {
        "axes": [
            ("broad", "wide-ranging, comprehensive, expansive, global, sweeping"),
            ("focused", "narrow, specific, targeted, precise, concentrated"),
        ],
        "description": "Breadth of the topic",
    },
    "risk": {
        "axes": [
            ("cautious", "careful, prudent, guarded, risk-averse, measured"),
            ("bold", "daring, adventurous, risk-taking, aggressive, ambitious"),
        ],
        "description": "Risk tolerance",
    },
}

# Available target models (simulating a multi-agent system)
TARGET_MODELS = [
    "analyst-model",      # General analysis
    "security-model",     # Security monitoring
    "ops-model",          # Operations/infrastructure
    "research-model",     # Deep research
    "creative-model",     # Creative/generative
    "review-model",       # Code/document review
]

MODEL_DOMAINS = {
    "analyst-model": ["technical_depth", "certainty", "scope"],
    "security-model": ["urgency", "risk", "technical_depth"],
    "ops-model": ["urgency", "technical_depth", "risk"],
    "research-model": ["novelty", "scope", "technical_depth"],
    "creative-model": ["novelty", "sentiment"],
    "review-model": ["certainty", "collaboration", "technical_depth"],
}


# ═══════════════════════════════════════════════════════════════
# Intent Templates
# ═══════════════════════════════════════════════════════════════

INTENT_TEMPLATES = [
    # ─── Deployment / Infrastructure ───
    {
        "intents": [
            "Alert {target} about the {severity} deployment failure in {component}",
            "Ask {target} to investigate the {symptom} in production",
            "Notify {target} that {component} is experiencing {symptom}",
            "Request {target} to check {component} for {issue}",
            "Tell {target} the {environment} rollout has {status}",
        ],
        "concept_profile": {"urgent": (0.6, 1.0), "technical": (0.5, 0.9), "concerned": (0.3, 0.8)},
        "targets": ["ops-model", "analyst-model"],
        "variables": {
            "severity": ["critical", "major", "minor", "partial"],
            "component": ["authentication service", "database cluster", "load balancer",
                         "API gateway", "message queue", "cache layer", "CI/CD pipeline"],
            "symptom": ["increased latency", "elevated error rate", "memory pressure",
                       "connection timeouts", "replication lag", "disk exhaustion"],
            "issue": ["performance degradation", "configuration drift", "certificate expiry",
                     "resource leak", "race condition"],
            "environment": ["staging", "production", "canary", "blue/green"],
            "status": ["failed", "completed with warnings", "been rolled back", "passed"],
        },
    },
    # ─── Security ───
    {
        "intents": [
            "Report a {severity} security {finding_type} to {target}",
            "Ask {target} to assess the {vulnerability} impact",
            "Notify {target} about {threat} detected in {system}",
            "Request {target} to review the {artifact} for security issues",
            "Alert {target} about unusual {activity} in {location}",
        ],
        "concept_profile": {"urgent": (0.7, 1.0), "cautious": (0.5, 0.9), "technical": (0.4, 0.8)},
        "targets": ["security-model", "ops-model"],
        "variables": {
            "finding_type": ["vulnerability", "breach attempt", "anomalous access pattern",
                           "policy violation", "misconfiguration"],
            "vulnerability": ["zero-day", "privilege escalation", "injection", "exposure",
                            "CVE-2024", "token leak"],
            "threat": ["unauthorized access", "data exfiltration", "denial of service",
                      "lateral movement", "credential stuffing"],
            "system": ["authentication service", "IAM roles", "API gateway", "VPC",
                      "secrets manager", "container registry"],
            "artifact": ["IAM policy", "security group", "Dockerfile", "Terraform plan",
                        "deployment manifest", "CI config"],
            "activity": ["login attempts", "API calls", "data access patterns",
                        "network traffic", "permission changes"],
            "location": ["production VPC", "staging environment", "admin panel",
                        "database", "customer data store"],
        },
    },
    # ─── Research / Analysis ───
    {
        "intents": [
            "Ask {target} to research {topic} with a focus on {aspect}",
            "Request {target} to analyze the {data_source} for {pattern}",
            "Have {target} synthesize findings on {domain} across {timeframe}",
            "Get {target}'s perspective on {question} considering {context}",
            "Ask {target} to explore {hypothesis} using {methodology}",
        ],
        "concept_profile": {"creative": (0.4, 0.8), "curious": (0.5, 0.9), "broad": (0.3, 0.7)},
        "targets": ["research-model", "creative-model"],
        "variables": {
            "topic": ["transformer architecture improvements", "multi-agent coordination",
                     "embedding space geometry", "distributed consensus", "neural scaling laws",
                     "cross-modal transfer learning", "attention mechanism variants"],
            "aspect": ["computational efficiency", "theoretical foundations", "practical deployment",
                      "failure modes", "scaling properties"],
            "data_source": ["benchmark results", "production logs", "research literature",
                          "experimental data", "user feedback"],
            "pattern": ["performance trends", "failure correlations", "usage patterns",
                       "anomaly clusters", "convergence behavior"],
            "domain": ["LLM architectures", "distributed training", "model alignment",
                      "inference optimization", "agent communication"],
            "timeframe": ["the last quarter", "the past year", "recent developments",
                         "the next roadmap cycle"],
            "question": ["model interpretability", "scaling bottlenecks", "alignment techniques",
                        "multi-agent protocols"],
            "context": ["production constraints", "research frontier", "cost optimization",
                       "safety considerations"],
            "hypothesis": ["attention sparsity improves generalization",
                          "cross-model tensor alignment is learnable",
                          "embedding geometry encodes world knowledge"],
            "methodology": ["controlled experiments", "statistical analysis", "literature review",
                           "comparative benchmarks"],
        },
    },
    # ─── Code Review / Quality ───
    {
        "intents": [
            "Ask {target} to review the {artifact} for {quality_aspect}",
            "Request {target} to check {component} for {issue_type}",
            "Have {target} assess the {change_type} in {location}",
            "Get {target}'s feedback on the {deliverable} focusing on {focus}",
            "Tell {target} to verify {property} in the latest {artifact}",
        ],
        "concept_profile": {"precise": (0.6, 0.9), "collaborative": (0.4, 0.7), "skeptical": (0.3, 0.7)},
        "targets": ["review-model", "analyst-model"],
        "variables": {
            "artifact": ["pull request", "design document", "architecture diagram",
                        "configuration change", "database migration", "API spec"],
            "quality_aspect": ["correctness", "performance", "security", "maintainability",
                             "test coverage", "error handling", "concurrency safety"],
            "component": ["authentication module", "data access layer", "API handlers",
                         "background workers", "configuration management"],
            "issue_type": ["race conditions", "resource leaks", "SQL injection",
                         "missing validation", "broken error handling"],
            "change_type": ["refactoring", "new feature", "bug fix", "optimization",
                          "dependency update"],
            "location": ["the core library", "the API layer", "the database schema",
                        "the deployment config"],
            "deliverable": ["sprint demo", "architecture RFC", "incident postmortem",
                          "performance report"],
            "focus": ["scalability", "correctness", "simplicity", "testability"],
            "property": ["idempotency", "atomicity", "backward compatibility",
                        "rate limiting"],
        },
    },
    # ─── Creative / Synthesis ───
    {
        "intents": [
            "Ask {target} to generate {content_type} about {topic} in {style}",
            "Have {target} reimagine {concept} from a {perspective} perspective",
            "Request {target} to create a {format} explaining {subject}",
            "Get {target} to brainstorm {quantity} approaches for {problem}",
            "Tell {target} to synthesize {source_a} and {source_b} into {output}",
        ],
        "concept_profile": {"creative": (0.7, 1.0), "optimistic": (0.4, 0.7), "playful": (0.3, 0.6)},
        "targets": ["creative-model", "research-model"],
        "variables": {
            "content_type": ["a design concept", "a narrative", "a metaphor",
                           "a visual description", "a system analogy"],
            "topic": ["AI-native communication", "decentralized systems", "emergent complexity",
                     "human-AI collaboration", "sustainable computing"],
            "style": ["playful", "rigorous", "minimalist", "poetic", "technical"],
            "concept": ["trust", "intelligence", "emergence", "resilience", "growth"],
            "perspective": ["biological", "mathematical", "philosophical", "engineering",
                          "artistic"],
            "format": ["mind map", "taxonomy", "framework", "analogy map", "design pattern"],
            "subject": ["tensor-native protocols", "embedding manifolds", "agent coordination",
                       "semantic compression"],
            "quantity": ["three", "five", "ten"],
            "problem": ["context window limitations", "model coordination overhead",
                       "semantic drift in long conversations", "cross-model understanding"],
            "source_a": ["research findings", "production metrics", "user feedback"],
            "source_b": ["design principles", "industry trends", "theoretical results"],
            "output": ["a unified framework", "an actionable plan", "a research agenda",
                      "a design recommendation"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# SFT Sample Generation
# ═══════════════════════════════════════════════════════════════

@dataclass
class SftSample:
    """A single supervised fine-tuning sample."""
    instruction: str       # System prompt / context
    input: str             # User's natural language intent
    output: dict           # Expected function call: {tool, parameters}


def generate_samples(n: int = 500, seed: int = 42) -> List[SftSample]:
    """Generate n SFT samples across all intent templates."""
    random.seed(seed)
    samples = []

    system_prompt = (
        "You are an AI agent with access to CMTIP (Cross-Model Tensor Interoperability "
        "Protocol) tools. When the user asks you to communicate with another model, "
        "use cmtip_send with appropriate concept blends. Choose concepts from: "
        "{axes}. Available targets: {targets}."
    )

    all_axes = sorted(set(
        axis for taxonomy in CONCEPT_TAXONOMY.values()
        for axis, _ in taxonomy["axes"]
    ))

    for _ in range(n):
        template_group = random.choice(INTENT_TEMPLATES)
        intent_template = random.choice(template_group["intents"])
        target = random.choice(template_group["targets"])

        # Fill variables
        filled = intent_template
        for var_name, var_options in template_group.get("variables", {}).items():
            filled = filled.replace(f"{{{var_name}}}", random.choice(var_options))
        filled = filled.replace("{target}", target)

        # Generate concept blend with realistic weights
        concept_profile = template_group["concept_profile"]
        concepts = {}
        for concept_name, (min_w, max_w) in concept_profile.items():
            # Add noise: vary weight within the range
            weight = round(random.uniform(min_w, max_w), 2)
            # Occasionally add a secondary concept from the same taxonomy
            concepts[concept_name] = weight

        # Build the expected tool call
        output = {
            "tool": "cmtip_send",
            "parameters": {
                "target": target,
                "concepts": concepts,
                "thought_text": filled,
                "confidence": round(random.uniform(0.7, 1.0), 2),
            },
        }

        samples.append(SftSample(
            instruction=system_prompt.format(
                axes=", ".join(all_axes),
                targets=", ".join(TARGET_MODELS),
            ),
            input=filled,
            output=output,
        ))

    return samples


def export_jsonl(samples: List[SftSample], path: str):
    """Export samples as JSONL (one JSON object per line)."""
    with open(path, 'w') as f:
        for s in samples:
            record = {
                "messages": [
                    {"role": "system", "content": s.instruction},
                    {"role": "user", "content": s.input},
                    {"role": "assistant", "content": json.dumps(s.output)},
                ],
            }
            f.write(json.dumps(record) + "\n")
    print(f"Exported {len(samples)} samples to {path}")


def export_chat_template(samples: List[SftSample], path: str):
    """Export as a human-readable chat template (for inspection)."""
    with open(path, 'w') as f:
        f.write("# CMTIP SFT Dataset — Chat Template Format\n\n")
        for i, s in enumerate(samples[:20]):  # First 20 for preview
            f.write(f"## Sample {i+1}\n\n")
            f.write(f"**System:** {s.instruction[:200]}...\n\n")
            f.write(f"**User:** {s.input}\n\n")
            f.write(f"**Assistant:**\n```json\n{json.dumps(s.output, indent=2)}\n```\n\n")
            f.write("---\n\n")
        f.write(f"\n*{len(samples)} total samples generated*\n")
    print(f"Preview exported to {path}")


# ═══════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════

def print_statistics(samples: List[SftSample]):
    """Print dataset statistics."""
    from collections import Counter

    print("\n" + "=" * 64)
    print("  SFT Dataset Statistics")
    print("=" * 64)

    # Concept usage
    concept_usage = Counter()
    target_usage = Counter()
    for s in samples:
        output = s.output
        if isinstance(output, dict):
            params = output.get("parameters", {})
            for c in params.get("concepts", {}):
                concept_usage[c] += 1
            target_usage[params.get("target", "?")] += 1

    print(f"\n  Total samples: {len(samples)}")
    print(f"\n  Target model distribution:")
    for target, count in target_usage.most_common():
        bar = "█" * (count * 50 // len(samples))
        print(f"    {target:<20s} {count:>4d} {bar}")

    print(f"\n  Concept usage:")
    for concept, count in concept_usage.most_common():
        bar = "█" * (count * 50 // len(samples))
        print(f"    {concept:<16s} {count:>4d} {bar}")

    # Weight distribution per concept
    print(f"\n  Concept weight ranges:")
    concept_weights = {c: [] for c in concept_usage}
    for s in samples:
        params = s.output.get("parameters", {}) if isinstance(s.output, dict) else {}
        for c, w in params.get("concepts", {}).items():
            concept_weights.setdefault(c, []).append(w)

    for concept in sorted(concept_usage.keys()):
        ws = concept_weights[concept]
        if ws:
            print(f"    {concept:<16s} {min(ws):.2f}–{max(ws):.2f}  (μ={sum(ws)/len(ws):.2f})")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="CMTIP SFT Dataset Generator — teach LLMs tensor-speak"
    )
    parser.add_argument("--output", default="./data/cmtip_sft.jsonl",
                        help="Output path for JSONL dataset")
    parser.add_argument("--preview", default="./data/cmtip_sft_preview.md",
                        help="Preview path for human-readable format")
    parser.add_argument("--size", type=int, default=500,
                        help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 64)
    print("  CMTIP SFT Dataset Generator")
    print("=" * 64)
    print(f"  Generating {args.size} samples (seed={args.seed})...")

    samples = generate_samples(args.size, args.seed)

    # Ensure output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Export
    export_jsonl(samples, args.output)
    export_chat_template(samples, args.preview)

    # Stats
    print_statistics(samples)

    print(f"\n  Output files:")
    print(f"    JSONL:    {args.output}")
    print(f"    Preview:  {args.preview}")
    print(f"\n  Usage with LLM fine-tuning:")
    print(f"    OpenAI:   openai api fine_tunes.create -t {args.output}")
    print(f"    Axolotl:  datasets.path={args.output}")
    print(f"    Hermes:   cp {args.output} ~/.hermes/skills/llm-native-language/templates/")
    print("=" * 64)


if __name__ == "__main__":
    main()
