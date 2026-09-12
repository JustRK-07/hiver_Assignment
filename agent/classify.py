"""Intent classifier: keyword scores, optional Groq LLM overlay.

Keywords keep the 15-minute no-key path alive and tag the retrieval index
offline. When GROQ_API_KEY is set, classify() prefers the LLM and falls
back to keywords if the JSON is malformed or the label is off-taxonomy.
Few-shot examples are BA-specific; do not copy them to another brand.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from agent.config import load_config, load_taxonomy
from agent.llm import chat_json


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    source: str
    rationale: str


def _keyword_score(text: str, keywords: list[str]) -> int:
    t = text.lower()
    hits = 0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw.lower())}\b", t):
            hits += 1
        elif kw.lower() in t:
            hits += 1
    return hits


def classify_keyword(text: str) -> IntentPrediction:
    taxonomy = load_taxonomy()
    intents = taxonomy["intents"]
    scores = []
    for item in intents:
        score = _keyword_score(text, item.get("keywords") or [])
        scores.append((score, item["id"]))
    scores.sort(reverse=True)
    best, intent = scores[0]
    second = scores[1][0] if len(scores) > 1 else 0
    if best == 0:
        return IntentPrediction(
            taxonomy.get("majority_intent", "general_query"),
            0.35,
            "keyword",
            "no keyword hits; majority intent",
        )
    conf = min(0.93, 0.45 + 0.15 * best + 0.08 * max(0, best - second))
    return IntentPrediction(intent, conf, "keyword", f"keyword hits={best}")


def classify_llm(text: str) -> IntentPrediction | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    taxonomy = load_taxonomy()
    labels = [i["id"] for i in taxonomy["intents"]]
    shots = [
        ("BA0273 delayed 3 hours sitting at the gate with no info", "flight_delay"),
        ("flight cancelled can you rebook me to Edinburgh tomorrow", "cancellation_rebook"),
        ("my suitcase never arrived in Cape Town tracking says still LHR", "baggage"),
        ("I want a refund and EU261 compensation for the delay", "refund_compensation"),
        ("Manage My Booking is broken I cannot check in", "booking_checkin"),
        ("paid for extra-legroom seats and they are gone", "seats_upgrade"),
        ("Avios expired can you reinstate my points", "loyalty_avios"),
        ("breakfast bar for a 6 hour afternoon flight is appalling", "inflight_product"),
        ("T5 ground staff ignored us at the bag drop", "airport_staff"),
        ("is a driving licence enough ID Heathrow to Glasgow", "general_query"),
    ]
    shot_txt = "\n".join(f"- {q} => {y}" for q, y in shots)
    prompt = f"""Classify this British Airways customer tweet into exactly one intent.
Intents: {", ".join(labels)}
If multiple issues, pick the one the customer wants action on first.

Few-shot:
{shot_txt}

Tweet: {text}

Return JSON: {{"intent": "...", "confidence": 0.0-1.0, "rationale": "one sentence"}}"""
    cfg = load_config()
    data = chat_json(prompt, model=cfg["models"]["classifier"])
    intent = data.get("intent")
    if intent not in labels:
        return None
    conf = float(data.get("confidence", 0.5))
    return IntentPrediction(intent, conf, "llm", data.get("rationale", ""))


def classify(text: str) -> IntentPrediction:
    try:
        llm = classify_llm(text)
    except Exception:
        # Rate-limit / outage: still produce a label so the harness finishes.
        llm = None
    if llm:
        return llm
    return classify_keyword(text)
