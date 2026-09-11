from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from agent.classify import classify_keyword
from agent.config import ROOT, load_taxonomy
from agent.gate import tier1_gate
from retrieval.hybrid import default_index

CANNED = (
    "Thanks for contacting British Airways. A member of the team will review "
    "your booking and reply as soon as possible."
)


def trivial(text: str, taxonomy: dict | None = None) -> dict:
    taxonomy = taxonomy or load_taxonomy()
    intent = taxonomy.get("majority_intent", "general_query")
    return {
        "intent": intent,
        "status": "auto",
        "reply": CANNED,
        "reason": "trivial_majority_always_auto",
        "justification": "Always majority intent, always send canned reply, never escalate.",
        "confidence": 1.0,
    }


def simple(text: str) -> dict:
    gate = tier1_gate(text)
    if gate.escalate:
        return {
            "intent": "escalate",
            "status": "escalate",
            "reply": "",
            "reason": gate.rule,
            "justification": f"regex {gate.rule}",
            "confidence": 1.0,
        }
    pred = classify_keyword(text)
    hits = default_index().search(text, k=1, min_score=0.0)
    reply = hits[0].company_text if hits else CANNED
    return {
        "intent": pred.intent,
        "status": "auto",
        "reply": reply,
        "reason": "keyword_nn",
        "justification": f"keyword={pred.rationale}; nn={'hit' if hits else 'none'}",
        "confidence": pred.confidence,
    }


def load_gold(path: Path | None = None) -> list[dict]:
    path = path or ROOT / "eval" / "golden_set.csv"
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def intent_histogram(rows: list[dict]) -> Counter:
    return Counter(r["gold_intent"] for r in rows)
