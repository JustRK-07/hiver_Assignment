"""Build a labeling WORKSHEET for eval/golden_set.csv — candidates only.

This script does NOT decide gold_intent / gold_escalate / notes for you.
It reads data/pairs_cluster.csv, buckets rows against the same quota table
in DECISIONS.md (10 intents, multi-issue, explicit-human, legal/fraud,
self-harm, sarcasm, ambiguous-via-dm-deflect), excludes rows already used
in eval/golden_set.csv, and writes a CSV with the tweet text pre-filled
and every label column BLANK for you to type by hand.

Usage:
    PYTHONPATH=. python -m scripts.build_gold_candidates

Output:
    eval/golden_set_candidates.csv  (pick rows from here, delete the rest,
                                      fill in the blank columns yourself,
                                      then append the finished rows into
                                      eval/golden_set.csv as g051...g150)
"""

from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

from agent.config import ROOT, load_taxonomy
from agent.gate import tier1_gate

SEED = 42  # fixed so the candidate list is reproducible — note this in DECISIONS.md
PAIRS_CSV = ROOT / "data" / "pairs_cluster.csv"
GOLD_CSV = ROOT / "eval" / "golden_set.csv"
OUT_CSV = ROOT / "eval" / "golden_set_candidates.csv"

# Per-bucket cap: how many candidates to SHOW you, not how many to keep.
# Kept generous so you have real choice instead of rubber-stamping the first hit.
CAP_PER_INTENT = 20
CAP_MULTI = 25
CAP_SARCASM = 25
CAP_HUMAN = 20
CAP_LEGAL_FRAUD = 20
CAP_AMBIGUOUS = 25
CAP_GENERAL = 15

# Heuristic only — for surfacing candidates, NOT for labeling them.
# Praise-shaped phrasing that, in an airline-complaint corpus, is very
# often sarcastic. You judge each one; do not trust this list.
SARCASM_PATTERNS = re.compile(
    r"""
    thanks\ for\ nothing | thanks\ a\ lot\b | great\ (job|service|customer\ service) |
    well\ done\b | brilliant\b | top\ marks | 10/10 | 5\ star | five\ star |
    love\ (that|how|it\ when) | really\ helpful\b | so\ helpful\b | cheers\ for\ that |
    amazing\ service | fantastic\ service | wonderful\ service
    """,
    re.IGNORECASE | re.VERBOSE,
)


def load_used_texts() -> set[str]:
    if not GOLD_CSV.exists():
        return set()
    with GOLD_CSV.open(encoding="utf-8") as f:
        return {row["customer_text"].strip() for row in csv.DictReader(f)}


def load_pairs() -> list[dict]:
    with PAIRS_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def matched_intents(text: str, taxonomy: dict) -> list[str]:
    t = text.lower()
    hits = []
    for item in taxonomy["intents"]:
        if item["id"] == "general_query":
            continue
        for kw in item.get("keywords") or []:
            if kw.lower() in t:
                hits.append(item["id"])
                break
    return hits


def main() -> None:
    random.seed(SEED)
    taxonomy = load_taxonomy()
    used = load_used_texts()
    rows = load_pairs()

    buckets: dict[str, list[dict]] = defaultdict(list)
    self_harm_hits: list[dict] = []

    for row in rows:
        text = (row.get("customer_text") or "").strip()
        if not text or text in used:
            continue

        gate = tier1_gate(text)
        specific = matched_intents(text, taxonomy)
        is_multi = len(specific) >= 2
        is_dm_deflect = row.get("is_dm_deflect") == "True"

        candidate = {
            "customer_tweet_id": row.get("customer_tweet_id", ""),
            "customer_text": text,
            "is_dm_deflect": row.get("is_dm_deflect", ""),
            "substantive": row.get("substantive", ""),
        }

        if gate.rule == "self_harm":
            self_harm_hits.append(candidate)
            continue  # handled separately below, not mixed into normal buckets

        if gate.rule == "request_human":
            c = dict(candidate, bucket="explicit_human", matched_signal="gate:request_human",
                     suggested_intent=specific[0] if specific else "general_query")
            buckets["explicit_human"].append(c)
            continue

        if gate.rule in ("legal", "fraud"):
            c = dict(candidate, bucket="legal_fraud", matched_signal=f"gate:{gate.rule}",
                     suggested_intent=specific[0] if specific else "general_query")
            buckets["legal_fraud"].append(c)
            continue

        if is_multi:
            c = dict(candidate, bucket="multi_issue", matched_signal="+".join(specific),
                      suggested_intent=specific[0])
            buckets["multi_issue"].append(c)
            continue

        if SARCASM_PATTERNS.search(text):
            c = dict(candidate, bucket="sarcasm_or_praise_complaint", matched_signal="sarcasm_phrase",
                      suggested_intent=specific[0] if specific else "general_query")
            buckets["sarcasm_or_praise_complaint"].append(c)
            continue

        if is_dm_deflect:
            c = dict(candidate, bucket="ambiguous_dm_deflect_only", matched_signal="is_dm_deflect=True",
                      suggested_intent=specific[0] if specific else "general_query")
            buckets["ambiguous_dm_deflect_only"].append(c)
            continue

        if len(specific) == 1:
            intent = specific[0]
            c = dict(candidate, bucket=f"intent:{intent}", matched_signal="keyword",
                      suggested_intent=intent)
            buckets[f"intent:{intent}"].append(c)
            continue

        if not specific:
            c = dict(candidate, bucket="intent:general_query", matched_signal="no_keyword_hit",
                      suggested_intent="general_query")
            buckets["intent:general_query"].append(c)

    caps = {
        "multi_issue": CAP_MULTI,
        "sarcasm_or_praise_complaint": CAP_SARCASM,
        "explicit_human": CAP_HUMAN,
        "legal_fraud": CAP_LEGAL_FRAUD,
        "ambiguous_dm_deflect_only": CAP_AMBIGUOUS,
        "intent:general_query": CAP_GENERAL,
    }

    out_rows = []
    print("=== candidate counts per bucket (before capping) ===")
    for name, items in sorted(buckets.items()):
        cap = caps.get(name, CAP_PER_INTENT)
        random.shuffle(items)
        kept = items[:cap]
        print(f"{name:35s} found={len(items):4d}  showing={len(kept):3d}")
        out_rows.extend(kept)

    print(f"\nself_harm regex hits in this brand's subsample: {len(self_harm_hits)}")
    if self_harm_hits:
        print("Real matches found — review them yourself, do not auto-accept:")
        for c in self_harm_hits[:5]:
            print(f"  tweet {c['customer_tweet_id']}: {c['customer_text'][:100]}")
    else:
        print(
            "No real self-harm-pattern tweets in this subsample (expected — rare in "
            "airline complaints). Write 1-2 constructed examples yourself for this "
            "bucket, same as the constructed edge cases already in your first 50 rows. "
            "Mark them 'constructed' in the notes column."
        )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "bucket", "matched_signal", "suggested_intent", "customer_tweet_id",
        "customer_text", "is_dm_deflect", "substantive",
        # blank — you fill these in by hand:
        "id", "gold_intent", "gold_escalate", "gold_escalate_reason", "notes",
        "relevant_company_tweet_id",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({
                "bucket": r["bucket"],
                "matched_signal": r["matched_signal"],
                "suggested_intent": r["suggested_intent"],
                "customer_tweet_id": r["customer_tweet_id"],
                "customer_text": r["customer_text"],
                "is_dm_deflect": r["is_dm_deflect"],
                "substantive": r["substantive"],
                "id": "", "gold_intent": "", "gold_escalate": "",
                "gold_escalate_reason": "", "notes": "",
                "relevant_company_tweet_id": "",
            })

    print(f"\nwrote {len(out_rows)} candidates to {OUT_CSV}")
    print(
        "\nNext steps (do these yourself, this is the part graders will ask about):\n"
        "  1. Open the csv, read each customer_text.\n"
        "  2. Delete rows you don't want. Keep enough per bucket to hit the quota\n"
        "     table in DECISIONS.md (~7-8/intent, ~10 multi-issue, ~8 sarcasm,\n"
        "     ~8 explicit-human, ~6 legal/fraud/self-harm, ~8 ambiguous).\n"
        "  3. Fill in gold_intent, gold_escalate, gold_escalate_reason, notes yourself.\n"
        "  4. Number the kept rows id=g051...g150 in the order you finalize them.\n"
        "  5. Append the finished rows to eval/golden_set.csv and delete this candidates file.\n"
    )


if __name__ == "__main__":
    main()
