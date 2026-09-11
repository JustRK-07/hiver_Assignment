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
        if g["gold_intent"] != "escalate":
            y_int_t.append(g["gold_intent"])
            y_int_p.append(pred["intent"] if pred["intent"] != "escalate" else "general_query")
        rows.append(
            {
                "id": g["id"],
                "gold_intent": g["gold_intent"],
                "pred_intent": pred["intent"],
                "gold_escalate": gold_esc,
                "pred_status": pred["status"],
                "reason": pred["reason"],
                "reply": pred.get("reply", ""),
            }
        )
        if g.get("relevant_company_tweet_id"):
            retrieve_n += 1
    esc = binary_prf(y_esc_t, y_esc_p)
    return {
        "system": name,
        "n": len(gold),
        "intent_macro_f1": macro_f1(y_int_t, y_int_p) if y_int_t else 0.0,
        "intent_accuracy": (
            sum(int(a == b) for a, b in zip(y_int_t, y_int_p)) / len(y_int_t) if y_int_t else 0.0
        ),
        "escalation": esc,
        "auto_rate": sum(int(r["pred_status"] == "auto") for r in rows) / len(rows),
        "rows": rows,
        "retrieval_hit_rate_note": (
            "Full retrieval hit rate needs labelled relevant ids; "
            f"{retrieve_n} gold rows currently carry a relevant id."
        ),
    }


def tau_curve(gold: list[dict], taus: list[float]) -> list[dict]:
    """Sweep confidence using keyword classifier + gate only (no generation)."""
    from agent.classify import classify_keyword
    from agent.gate import tier1_gate

    out = []
    for tau in taus:
        y_t, y_p = [], []
        for g in gold:
            gold_esc = g["gold_escalate"].lower() in {"1", "true", "yes"}
            gate = tier1_gate(g["customer_text"])
            pred_esc = True
            if not gate.escalate:
                pred = classify_keyword(g["customer_text"])
                pred_esc = pred.confidence < tau
            y_t.append(int(gold_esc))
            y_p.append(int(pred_esc))
        stats = binary_prf(y_t, y_p)
        stats["tau"] = tau
        out.append(stats)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "runs" / "latest.json")
    args = ap.parse_args()
    gold = load_gold()
    summary = {name: evaluate(name, gold) for name in SYSTEMS}
    curve = tau_curve(gold, [0.35, 0.45, 0.55, 0.65, 0.75, 0.85])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in summary.items()},
        "tau_curve_escalation": curve,
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pred_path = args.out.with_suffix(".predictions.csv")
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "system",
                "id",
                "gold_intent",
                "pred_intent",
                "gold_escalate",
                "pred_status",
                "reason",
                "reply",
            ],
        )
        w.writeheader()
        for name, result in summary.items():
            for row in result["rows"]:
                w.writerow({"system": name, **row})
    print(json.dumps(payload, indent=2))
    print(f"wrote {args.out} and {pred_path}")


if __name__ == "__main__":
    main()
