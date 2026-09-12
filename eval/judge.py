"""LLM-as-judge. Different model family than the generator.

Do not put this on the send path. After it runs, re-label ~50 verdicts in
eval/judge_human.csv and compute Cohen's κ with eval/agreement.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agent.config import ROOT, load_config
from agent.llm import chat_json

RUBRIC = """You are scoring a British Airways Twitter support agent.
Score 1-5 independently (integers only):
- groundedness: 5 = reply only uses evidence a historical BA agent said; 1 = invents refunds/EU261/upgrades
- helpfulness: 5 = customer knows the next step; 1 = empty apology or "DM us" with no path
- tone: 5 = professional; 1 = sarcastic, blaming, or hollow over-apology
- escalation_correctness: 5 if system status matches gold should_escalate; 1 if they disagree
Return JSON with those four keys plus short_reason.
"""


def judge_row(ticket: str, reply: str, status: str, gold_escalate: str) -> dict:
    cfg = load_config()
    prompt = f"""{RUBRIC}
Ticket: {ticket}
System status: {status}
Gold should_escalate: {gold_escalate}
Reply: {reply or "(empty, escalated)"}
"""
    data = chat_json(prompt, model=cfg["models"]["judge"])
    return {
        "groundedness": data.get("groundedness"),
        "helpfulness": data.get("helpfulness"),
        "tone": data.get("tone"),
        "escalation_correctness": data.get("escalation_correctness"),
        "short_reason": data.get("short_reason", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, default=ROOT / "eval" / "runs" / "latest.predictions.csv")
    ap.add_argument("--system", default="system")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "runs" / "judge.jsonl")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()
    rows = [r for r in csv.DictReader(args.predictions.open(encoding="utf-8")) if r["system"] == args.system]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows[: args.limit]):
            scores = judge_row(
                row.get("customer_text") or "",
                row.get("reply") or "",
                row["pred_status"],
                row["gold_escalate"],
            )
            rec = {**row, **scores}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"judged {i+1}/{min(args.limit, len(rows))} {row['id']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
