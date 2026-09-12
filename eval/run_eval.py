"""Evaluation harness: trivial + simple + system vs gold.

Headline metrics:
  - intent macro-F1 / accuracy (skip rows whose gold_intent is unused)
  - escalation precision / recall / F1 (positive class = escalate)
  - auto-rate (share of tickets the system would send)
  - retrieval hit rate when gold has relevant_company_tweet_id

Tau curve is computed WITHOUT generation so you can pick τ cheaply.
Use --tau-classifier llm (needs GROQ_API_KEY) before freezing config.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agent.config import ROOT
from agent.pipeline import handle_ticket_dict
from eval.baselines import load_gold, simple, trivial

SYSTEMS = {
    "trivial": lambda t: trivial(t),
    "simple": lambda t: simple(t),
    "system": lambda t: handle_ticket_dict(t),
}


def binary_prf(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(int(t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum(int(t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum(int(t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lab in labels:
        yt = [int(y == lab) for y in y_true]
        yp = [int(y == lab) for y in y_pred]
        f1s.append(binary_prf(yt, yp)["f1"])
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(name: str, gold: list[dict]) -> dict:
    rows = []
    y_int_t, y_int_p = [], []
    y_esc_t, y_esc_p = [], []
    retrieve_hits = 0
    retrieve_n = 0
    for g in gold:
        pred = SYSTEMS[name](g["customer_text"])
        gold_esc = g["gold_escalate"].lower() in {"1", "true", "yes"}
        pred_esc = pred["status"] == "escalate"
        y_esc_t.append(int(gold_esc))
        y_esc_p.append(int(pred_esc))
        # Intent F1: if the agent escalated via Tier-1 it reports intent="escalate".
        # Map that to general_query so we don't invent an 11th class.
        if g["gold_intent"] != "escalate":
            y_int_t.append(g["gold_intent"])
            y_int_p.append(pred["intent"] if pred["intent"] != "escalate" else "general_query")
        retrieved = pred.get("retrieved_ids") or []
        gold_rel = (g.get("relevant_company_tweet_id") or "").strip()
        if gold_rel:
            retrieve_n += 1
            if gold_rel in retrieved:
                retrieve_hits += 1
        rows.append(
            {
                "id": g["id"],
                "customer_text": g["customer_text"],
                "gold_intent": g["gold_intent"],
                "pred_intent": pred["intent"],
                "gold_escalate": gold_esc,
                "pred_status": pred["status"],
                "reason": pred["reason"],
                "reply": pred.get("reply", ""),
                "retrieved_ids": "|".join(retrieved),
            }
        )
    esc = binary_prf(y_esc_t, y_esc_p)
    hit_rate = retrieve_hits / retrieve_n if retrieve_n else None
    return {
        "system": name,
        "n": len(gold),
        "intent_macro_f1": macro_f1(y_int_t, y_int_p) if y_int_t else 0.0,
        "intent_accuracy": (
            sum(int(a == b) for a, b in zip(y_int_t, y_int_p)) / len(y_int_t) if y_int_t else 0.0
        ),
        "escalation": esc,
        "auto_rate": sum(int(r["pred_status"] == "auto") for r in rows) / len(rows),
        "retrieval_hit_rate": hit_rate,
        "retrieval_n": retrieve_n,
        "rows": rows,
    }


def tau_curve(gold: list[dict], taus: list[float], classifier: str = "keyword") -> list[dict]:
    """Sweep confidence using the tier-1 gate + chosen classifier (no generation).

    classifier="keyword": deterministic, no API key. Confidence is lumpy.
    classifier="llm": Groq per gold row — this is the curve to pick tau from.
    """
    import os

    from agent.classify import classify_keyword, classify_llm
    from agent.gate import tier1_gate

    if classifier == "llm" and not os.getenv("GROQ_API_KEY"):
        print(
            "WARNING: classifier='llm' requested but GROQ_API_KEY is not set; "
            "falling back to keyword classifier."
        )
        classifier = "keyword"

    cache: dict[str, tuple[bool, float]] = {}
    for g in gold:
        text = g["customer_text"]
        gate = tier1_gate(text)
        if gate.escalate:
            cache[g["id"]] = (True, 1.0)
            continue
        if classifier == "llm":
            try:
                pred = classify_llm(text)
            except Exception:
                pred = None
            conf = pred.confidence if pred else classify_keyword(text).confidence
        else:
            conf = classify_keyword(text).confidence
        cache[g["id"]] = (False, conf)

    out = []
    for tau in taus:
        y_t, y_p = [], []
        for g in gold:
            gold_esc = g["gold_escalate"].lower() in {"1", "true", "yes"}
            gate_escalate, conf = cache[g["id"]]
            pred_esc = True if gate_escalate else conf < tau
            y_t.append(int(gold_esc))
            y_p.append(int(pred_esc))
        stats = binary_prf(y_t, y_p)
        stats["tau"] = tau
        stats["classifier"] = classifier
        # Auto-send precision: among predicted-auto, how often gold said auto.
        auto_ok = sum(int(t == 0 and p == 0) for t, p in zip(y_t, y_p))
        auto_pred = sum(int(p == 0) for p in y_p)
        stats["auto_send_precision"] = auto_ok / auto_pred if auto_pred else 0.0
        out.append(stats)
    return out


def pick_tau(curve: list[dict], min_auto_precision: float = 0.85) -> float:
    """Highest auto-send volume (lowest tau) that still clears auto-send precision.

    If nothing clears the floor, fall back to the tau with best auto-send precision.
    """
    eligible = [c for c in curve if c["auto_send_precision"] >= min_auto_precision]
    if eligible:
        # Among those safe enough, pick the one with highest escalation F1.
        best = max(eligible, key=lambda c: (c["f1"], c["auto_send_precision"]))
        return float(best["tau"])
    return float(max(curve, key=lambda c: c["auto_send_precision"])["tau"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "runs" / "latest.json")
    ap.add_argument(
        "--tau-classifier",
        choices=["keyword", "llm"],
        default="keyword",
        help="Fit the escalation tau curve on this classifier. Use llm before freezing tau.",
    )
    ap.add_argument("--skip-system", action="store_true", help="Only baselines + tau (cheap).")
    args = ap.parse_args()
    gold = load_gold()
    names = ["trivial", "simple"] if args.skip_system else list(SYSTEMS)
    summary = {name: evaluate(name, gold) for name in names}
    curve = tau_curve(gold, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85], classifier=args.tau_classifier)
    suggested = pick_tau(curve)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in summary.items()},
        "tau_curve_escalation": curve,
        "suggested_tau": suggested,
        "gold_n": len(gold),
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred_path = args.out.with_suffix(".predictions.csv")
    fieldnames = [
        "system",
        "id",
        "customer_text",
        "gold_intent",
        "pred_intent",
        "gold_escalate",
        "pred_status",
        "reason",
        "reply",
        "retrieved_ids",
    ]
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for name, result in summary.items():
            for row in result["rows"]:
                w.writerow({"system": name, **row})
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out} and {pred_path}")
    print(f"suggested_tau={suggested} (from {args.tau_classifier} curve)")


if __name__ == "__main__":
    main()
