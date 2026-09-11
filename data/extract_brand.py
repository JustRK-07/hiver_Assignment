#!/usr/bin/env python3
"""Extract British_Airways customer↔company pairs from the Kaggle twcs.csv."""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

BRAND = "British_Airways"
DM_RE = re.compile(r"\b(dm|direct message|please dm|inbox us|private message)\b", re.I)


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract(src: Path, brand: str) -> list[dict]:
    company, need = [], set()
    with src.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["author_id"] != brand:
                continue
            company.append(row)
            if row.get("in_response_to_tweet_id"):
                need.add(row["in_response_to_tweet_id"])
    parents = {}
    with src.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["tweet_id"] in need:
                parents[row["tweet_id"]] = row
    pairs = []
    for c in company:
        p = parents.get(c.get("in_response_to_tweet_id") or "")
        if not p or p["inbound"].lower() != "true":
            continue
        customer, company_text = clean(p["text"]), clean(c["text"])
        if len(customer.split()) < 4:
            continue
        dm = bool(DM_RE.search(c["text"] or ""))
        pairs.append(
            {
                "customer_tweet_id": p["tweet_id"],
                "company_tweet_id": c["tweet_id"],
                "customer_text": customer,
                "company_text": company_text,
                "created_at": p["created_at"],
                "is_dm_deflect": dm,
                "substantive": (not dm) and len(company_text.split()) >= 12,
            }
        )
    return pairs


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Path to twcs.csv")
    ap.add_argument("--brand", default=BRAND)
    ap.add_argument("--index-n", type=int, default=3500)
    ap.add_argument("--cluster-n", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    random.seed(args.seed)
    pairs = extract(args.src, args.brand)
    subst = [p for p in pairs if p["substantive"]]
    other = [p for p in pairs if not p["substantive"]]
    random.shuffle(subst)
    random.shuffle(other)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "pairs_index.csv", subst[: args.index_n])
    cluster = subst[: args.cluster_n] + other[:500]
    seen, uniq = set(), []
    for row in cluster:
        key = row["customer_text"].lower()[:160]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
    write_csv(args.out_dir / "pairs_cluster.csv", uniq)
    print(f"pairs={len(pairs)} substantive={len(subst)} index={min(args.index_n,len(subst))} cluster={len(uniq)}")


if __name__ == "__main__":
    main()
