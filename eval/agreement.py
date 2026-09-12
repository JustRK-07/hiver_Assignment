"""Cohen's κ and % agreement between LLM-as-judge and human re-labels.

Human file: eval/judge_human.csv with columns id + the four rubric scores
(human_groundedness, ...). Escalation is binarized: score >= 4 is 'correct'.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agent.config import ROOT

DIMS = ("groundedness", "helpfulness", "tone", "escalation_correctness")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    if n == 0:
        return 0.0
    po = sum(int(x == y) for x, y in zip(a, b)) / n
    labels = sorted(set(a) | set(b))
    pe = 0.0
    for lab in labels:
        pe += (a.count(lab) / n) * (b.count(lab) / n)
    if abs(1 - pe) < 1e-9:
        return 1.0 if po == 1 else 0.0
    return (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", type=Path, default=ROOT / "eval" / "runs" / "judge.jsonl")
    ap.add_argument("--human", type=Path, default=ROOT / "eval" / "judge_human.csv")
    args = ap.parse_args()
    judge = {json.loads(line)["id"]: json.loads(line) for line in args.judge.read_text(encoding="utf-8").splitlines() if line.strip()}
    human_rows = list(csv.DictReader(args.human.open(encoding="utf-8")))
    report = {}
    for dim in DIMS:
        j, h = [], []
        for row in human_rows:
            jrow = judge.get(row["id"])
            if not jrow or jrow.get(dim) in (None, ""):
                continue
            try:
                j.append(int(float(jrow[dim])))
                h.append(int(float(row[f"human_{dim}"])))
            except (TypeError, ValueError):
                continue
        if dim == "escalation_correctness":
            j_bin = [int(x >= 4) for x in j]
            h_bin = [int(x >= 4) for x in h]
            report[dim] = {
                "n": len(j),
                "exact_agreement": sum(int(x == y) for x, y in zip(j, h)) / len(j) if j else 0,
                "binary_kappa": cohen_kappa(j_bin, h_bin),
            }
        else:
            report[dim] = {
                "n": len(j),
                "exact_agreement": sum(int(x == y) for x, y in zip(j, h)) / len(j) if j else 0,
                "within_1": sum(int(abs(x - y) <= 1) for x, y in zip(j, h)) / len(j) if j else 0,
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
