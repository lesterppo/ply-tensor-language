#!/usr/bin/env python3
"""
Generate 5,000+ diverse paraphrase training pairs for CMTIP adapter.
Uses template-based generation with systematic synonym variation.
Covers 8 semantic domains for robust cross-model alignment.
"""

import json
import os
import random

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

random.seed(42)


# ═══════════════════════════════════════════════════════════════
# Synonym tables for paraphrase generation
# ═══════════════════════════════════════════════════════════════

SYNONYMS = {
    "increase": ["rise", "grow", "climb", "surge", "escalate", "ramp up", "expand"],
    "decrease": ["fall", "drop", "decline", "shrink", "diminish", "reduce", "contract"],
    "fast": ["rapid", "quick", "swift", "speedy", "brisk", "accelerated"],
    "slow": ["sluggish", "gradual", "lethargic", "delayed", "prolonged"],
    "large": ["substantial", "significant", "considerable", "massive", "enormous", "vast"],
    "small": ["minor", "slight", "modest", "negligible", "marginal", "tiny"],
    "problem": ["issue", "concern", "difficulty", "complication", "malfunction", "defect"],
    "fix": ["resolve", "address", "rectify", "remedy", "correct", "patch", "repair"],
    "important": ["critical", "crucial", "vital", "essential", "key", "pivotal"],
    "difficult": ["challenging", "demanding", "arduous", "complex", "intricate"],
    "easy": ["simple", "straightforward", "trivial", "effortless", "basic"],
    "good": ["positive", "favorable", "beneficial", "advantageous", "promising"],
    "bad": ["negative", "adverse", "detrimental", "unfavorable", "problematic"],
    "start": ["begin", "initiate", "commence", "launch", "kick off", "trigger"],
    "stop": ["halt", "cease", "terminate", "suspend", "discontinue", "abort"],
    "improve": ["enhance", "optimize", "boost", "upgrade", "refine", "strengthen"],
    "cause": ["trigger", "induce", "provoke", "bring about", "lead to", "result in"],
    "detect": ["identify", "discover", "spot", "notice", "observe", "flag"],
    "complete": ["finish", "finalize", "conclude", "wrap up", "accomplish"],
    "fail": ["break", "malfunction", "crash", "go down", "become unresponsive"],
    "require": ["need", "demand", "necessitate", "call for", "entail"],
    "provide": ["supply", "deliver", "furnish", "offer", "make available"],
    "ensure": ["guarantee", "secure", "verify", "confirm", "make certain"],
    "reduce": ["lower", "cut", "minimize", "bring down", "diminish", "curb"],
    "achieve": ["attain", "reach", "accomplish", "realize", "obtain"],
    "support": ["assist", "aid", "help", "back", "facilitate", "enable"],
    "prevent": ["avoid", "block", "stop", "thwart", "avert", "deter"],
    "allow": ["permit", "enable", "let", "authorize", "grant"],
    "monitor": ["track", "watch", "observe", "oversee", "supervise", "survey"],
    "manage": ["handle", "control", "administer", "oversee", "direct"],
    "damage": ["harm", "impair", "compromise", "degrade", "undermine"],
    "protect": ["safeguard", "defend", "shield", "secure", "guard"],
    "delay": ["postpone", "defer", "push back", "put off", "reschedule"],
    "exceed": ["surpass", "go beyond", "outstrip", "overtake", "beat"],
    "analyze": ["examine", "investigate", "study", "assess", "evaluate", "inspect"],
    "implement": ["deploy", "roll out", "put in place", "execute", "apply"],
    "maintain": ["preserve", "sustain", "keep up", "uphold", "retain"],
    "establish": ["set up", "create", "form", "institute", "found"],
    "eliminate": ["remove", "get rid of", "erase", "wipe out", "strip"],
    "generate": ["produce", "create", "yield", "output", "synthesize"],
    "consume": ["use", "utilize", "expend", "deplete", "drain"],
    "distribute": ["spread", "allocate", "disseminate", "disperse", "share"],
    "collect": ["gather", "accumulate", "amass", "aggregate", "compile"],
    "verify": ["validate", "confirm", "check", "authenticate", "corroborate"],
    "restore": ["recover", "reinstate", "bring back", "reestablish", "revive"],
    "optimize": ["fine-tune", "tweak", "streamline", "perfect", "polish"],
    "coordinate": ["orchestrate", "organize", "synchronize", "align", "harmonize"],
    "transform": ["convert", "change", "reshape", "revolutionize", "overhaul"],
    "integrate": ["combine", "merge", "unify", "consolidate", "blend"],
    "isolate": ["separate", "quarantine", "detach", "disconnect", "sever"],
    "automate": ["mechanize", "robotize", "program", "script", "systematize"],
    "accelerate": ["speed up", "hasten", "quicken", "expedite", "fast-track"],
}


def fill_template(template):
    """Fill a template with random slot values."""
    import re
    result = template
    unfilled = []
    for match in re.finditer(r'\{(\w+)\}', template):
        key = match.group(1)
        if key in SLOTS:
            result = result.replace(f'{{{key}}}', random.choice(SLOTS[key]), 1)
        else:
            unfilled.append(key)
    # Remove any remaining unfilled {placeholders}
    for key in unfilled:
        result = result.replace(f'{{{key}}}', '')
    # Clean up double spaces and trailing/leading spaces
    result = ' '.join(result.split())
    return result


def paraphrase(text):
    """Generate a different-surface-form paraphrase of text."""
    words = text.split()
    if len(words) < 3:
        return text + " (rephrased)"
    
    new_words = []
    changed = False
    for w in words:
        clean = w.strip('.,;:!?()[]{}"\'-').lower()
        suffix = ''
        prefix = ''
        for ch in w:
            if ch in '.,;:!?()[]{}"\'-':
                suffix += ch
            else:
                break
        # Actually get the prefix right
        real_prefix = ''
        real_suffix = ''
        for i, ch in enumerate(w):
            if ch.isalpha():
                real_prefix = w[:i]
                break
        else:
            real_prefix = ''
        for i in range(len(w)-1, -1, -1):
            if w[i].isalpha():
                real_suffix = w[i+1:]
                break
        else:
            real_suffix = ''
        
        if clean in SYNONYMS and random.random() < 0.6:
            replacement = random.choice(SYNONYMS[clean])
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            new_words.append(real_prefix + replacement + real_suffix)
            changed = True
        else:
            new_words.append(w)
    
    result = ' '.join(new_words)
    
    # Also apply structural paraphrases
    if random.random() < 0.3 or not changed:
        result = result.replace(" due to ", " caused by ")
        result = result.replace(" because of ", " as a result of ")
        result = result.replace(" was detected in ", " appeared in ")
        result = result.replace(" alerted on ", " triggered an alert for ")
        result = result.replace(" is showing ", " exhibits ")
        result = result.replace(" after ", " following ")
        result = result.replace(" before ", " prior to ")
        changed = True
    
    if not changed:
        # Force a change
        result = result.replace("The ", "A ", 1) if result.startswith("The ") else "The " + result[2:] if result[:2] in ("A ", "An") else result
        result = result.replace(" a ", " one ") if " a " in result else result
    
    return result.strip()


# ═══════════════════════════════════════════════════════════════
# Template-based pair generation across domains
# ═══════════════════════════════════════════════════════════════

TEMPLATES = {
    "sysadmin": [
        "The {component} {action} due to {cause}",
        "{component} {action} because of {cause}",
        "After {event}, the {component} started {action}",
        "We observed {symptom} on {component}",
        "{component} is showing {symptom}",
        "The {component} {action} during {scenario}",
        "{monitoring} alerted on {symptom} for {component}",
        "A {severity} {problem} was detected in {component}",
        "The {component} {action} after {threshold} was exceeded",
        "{remediation} was applied to {component} after {symptom}",
    ],
    "software": [
        "The {feature} was {action} to {system}",
        "{system} now supports {feature}",
        "Version {version} of {system} includes {feature}",
        "The {feature} {action} user experience by {metric}",
        "We {action} the {feature} based on user feedback",
        "The {feature} was {action} due to {reason}",
        "{system} {action} {metric} after the update",
        "Users reported {issue} with the {feature}",
        "The {feature} {action} when {condition}",
        "{system} {action} the {feature} in the latest release",
    ],
    "business": [
        "{company} {action} {metric} in {period}",
        "The {department} reported {metric} {action} by {amount}",
        "{company} announced {action} of {initiative}",
        "The {initiative} {action} {metric} by {amount}",
        "Market conditions {action} the {sector} industry",
        "{company} {action} its {strategy} to address {challenge}",
        "The {event} {action} consumer confidence in {market}",
        "{company} exceeded {target} by {amount} in {period}",
        "Analysts predict {trend} for {sector} in {period}",
        "{company} {action} partnership with {partner}",
    ],
    "science": [
        "Researchers {action} that {finding}",
        "A new study {action} the link between {factor} and {outcome}",
        "The experiment {action} that {finding}",
        "Scientists {action} a {discovery} in {field}",
        "The {discovery} could {action} {application}",
        "Research in {field} {action} our understanding of {concept}",
        "The study found {finding} when examining {population}",
        "{technique} revealed {finding} about {subject}",
        "The {method} approach {action} previous results",
        "Findings suggest that {hypothesis}",
    ],
    "medicine": [
        "The patient presented with {symptom} in the {region}",
        "Treatment with {drug} {action} {outcome} by {amount}",
        "Clinical trial results {action} {benefit} for {population}",
        "The {condition} {action} following {intervention}",
        "Screening for {condition} {action} detection rates",
        "Patients receiving {treatment} showed {outcome}",
        "The {procedure} was {action} in {population} patients",
        "Researchers {action} a new biomarker for {disease}",
        "The incidence of {disease} {action} over {period}",
        "{intervention} {action} the risk of {outcome}",
    ],
    "finance": [
        "{asset} prices {action} following {event}",
        "The {index} {action} {amount} after {catalyst}",
        "{institution} {action} its {position} on {asset}",
        "Trading volume {action} during {period}",
        "The yield on {instrument} {action} to {level}",
        "{currency} {action} against {other_currency}",
        "Analysts {action} their {target} for {company}",
        "The {sector} sector {action} in {period}",
        "{event} {action} market sentiment",
        "{regulation} {action} the {market} landscape",
    ],
    "daily_life": [
        "The {event} {action} my plans for {time_period}",
        "I {action} the {activity} because of {reason}",
        "The weather {action} our {activity} on {day}",
        "We decided to {action} our {plan} due to {reason}",
        "The {item} was {condition} when I checked it",
        "My experience with {service} was {quality}",
        "The journey to {place} {action} longer than expected",
        "I found the {thing} to be {quality} and {quality2}",
        "The {place} was {condition} when we arrived",
        "We spent the {time_period} {activity} at {location}",
    ],
    "abstract": [
        "{concept} is the foundation of {field}",
        "Without {concept}, {outcome} would be impossible",
        "The essence of {concept} lies in {attribute}",
        "True {concept} requires {requirement}",
        "{concept} and {concept2} are deeply interconnected",
        "The pursuit of {concept} drives human {activity}",
        "We measure {concept} by its impact on {measure}",
        "The relationship between {concept} and {concept2} is {nature}",
        "Understanding {concept} means accepting {truth}",
        "{concept} emerges when {condition} is met",
    ],
}


SLOTS = {
    "component": ["server", "database", "load balancer", "API gateway", "cache", "message queue",
                  "storage volume", "firewall", "DNS resolver", "container", "microservice",
                  "authentication service", "logging pipeline", "backup system", "CDN"],
    "action": ["failed", "crashed", "slowed down", "became unresponsive", "timed out",
               "restarted unexpectedly", "returned errors", "reached capacity", "lost connectivity",
               "degraded", "overheated", "exhausted resources", "experienced latency spikes"],
    "cause": ["a memory leak", "high traffic volume", "a configuration error", "a network partition",
              "disk failure", "a race condition", "exhausted connection pool", "a bad deployment",
              "an expired certificate", "a DDoS attack", "power fluctuation", "hardware degradation"],
    "symptom": ["elevated error rates", "increased latency", "connection timeouts", "memory exhaustion",
                "CPU throttling", "disk I/O saturation", "packet loss", "slow query performance",
                "cache misses", "authentication failures", "data corruption", "replication lag"],
    "monitoring": ["Prometheus", "Grafana", "Datadog", "PagerDuty", "CloudWatch", "Nagios",
                   "New Relic", "Splunk", "ELK stack", "Zabbix"],
    "severity": ["critical", "major", "minor", "severe", "moderate", "catastrophic", "high-priority"],
    "remediation": ["A hotfix", "A configuration rollback", "Traffic failover", "Resource scaling",
                    "Cache clearing", "Connection pool reset", "A circuit breaker", "Rate limiting"],
    "threshold": ["memory limit", "CPU quota", "connection limit", "disk capacity", "timeout value",
                  "rate limit", "error budget", "latency SLO"],
    "feature": ["dark mode", "real-time sync", "two-factor authentication", "search functionality",
               "export to PDF", "collaborative editing", "offline mode", "push notifications",
               "API versioning", "SSO integration", "audit logging", "drag and drop"],
    "system": ["the platform", "the application", "the dashboard", "the mobile app",
               "the backend service", "the admin panel", "the CLI tool", "the web interface"],
    "metric": ["performance", "reliability", "user satisfaction", "response time", "throughput",
               "conversion rate", "retention", "engagement", "efficiency", "accuracy"],
    "company": ["Acme Corp", "TechGiant Inc", "StartupX", "MegaBank", "CloudCo",
               "DataFlow Systems", "GreenEnergy Ltd", "PharmaPlus", "RetailCorp"],
    "initiative": ["digital transformation", "sustainability program", "market expansion",
                   "cost reduction", "product launch", "brand refresh", "talent acquisition"],
    "finding": ["the treatment is effective", "climate change is accelerating",
               "the theory holds under extreme conditions", "new species were discovered",
               "the material exhibits superconductivity", "gene expression varies by environment"],
    "disease": ["diabetes", "hypertension", "Alzheimer's", "asthma", "arthritis",
               "depression", "migraine", "osteoporosis", "anemia", "bronchitis"],
    "concept": ["trust", "wisdom", "courage", "patience", "creativity", "resilience",
               "empathy", "integrity", "curiosity", "discipline", "gratitude", "humility"],
}


def generate_pairs(n=6000):
    """Generate n paraphrase pairs across all domains."""
    pairs = []
    
    templates_per_domain = n // (len(TEMPLATES) * 3) + 1
    
    for domain, tmpls in TEMPLATES.items():
        for _ in range(templates_per_domain):
            for tmpl in tmpls:
                if len(pairs) >= n:
                    break
                
                # Fill template with random values
                text_a = fill_template(tmpl)
                
                # Paraphrase: same meaning, different wording
                text_b = paraphrase(text_a)
                
                # Occasionally restructure the paraphrase
                if random.random() < 0.3:
                    text_b = paraphrase(paraphrase(text_a))
                
                pairs.append((text_a.strip(), text_b.strip()))
            
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break
    
    # Shuffle
    random.shuffle(pairs)
    
    # Deduplicate
    seen = set()
    unique = []
    for a, b in pairs:
        key = a[:50]
        if key not in seen:
            seen.add(key)
            unique.append((a, b))
    
    return unique[:n]


def main():
    n_target = 5000
    pairs = generate_pairs(n_target * 2)  # Over-generate for dedup
    
    # Split: 80% train, 10% dev, 10% test
    random.shuffle(pairs)
    n_train = int(len(pairs) * 0.8)
    n_dev = int(len(pairs) * 0.1)
    
    train = pairs[:n_train]
    dev = pairs[n_train:n_train+n_dev]
    test = pairs[n_train+n_dev:]
    
    data = {
        "train": [{"a": a, "b": b} for a, b in train],
        "dev": [{"a": a, "b": b} for a, b in dev],
        "test": [{"a": a, "b": b} for a, b in test],
    }
    
    out = os.path.join(DATA_DIR, "training_pairs.json")
    with open(out, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {len(pairs):,} diverse paraphrase pairs")
    print(f"  Train: {len(train):,}")
    print(f"  Dev:   {len(dev):,}")
    print(f"  Test:  {len(test):,}")
    print(f"  Saved: {out}")
    
    # Show samples
    print(f"\nSample pairs:")
    for i, (a, b) in enumerate(train[:5]):
        print(f"  [{i+1}] A: {a}")
        print(f"      B: {b}")
        print()


if __name__ == "__main__":
    main()
