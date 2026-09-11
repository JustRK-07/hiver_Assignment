from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from agent.config import ROOT, load_config

TOKEN = re.compile(r"[a-z0-9]+")
DM_RE = re.compile(r"\b(dm|direct message|please dm|inbox us|private message)\b", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def is_dm_deflect(text: str) -> bool:
    return bool(DM_RE.search(text or ""))


@dataclass
class Hit:
    customer_text: str
    company_text: str
    score: float
    company_tweet_id: str
    customer_tweet_id: str


class HybridIndex:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self._bm25 = BM25Okapi([tokenize(r["customer_text"]) for r in rows])

    @classmethod
    def from_csv(cls, path: Path, drop_dm: bool = True) -> "HybridIndex":
        rows = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if drop_dm and (
                    row.get("is_dm_deflect") == "True" or is_dm_deflect(row["company_text"])
                ):
                    continue
                if not (row.get("company_text") or "").strip():
                    continue
                rows.append(row)
        if not rows:
            raise ValueError(f"no index rows in {path}")
        return cls(rows)

    def search(self, query: str, k: int = 5, min_score: float = 0.0) -> list[Hit]:
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
        hits = []
        for idx, score in ranked:
            if score < min_score:
                continue
            row = self.rows[idx]
            hits.append(
                Hit(
                    customer_text=row["customer_text"],
                    company_text=row["company_text"],
                    score=float(score),
                    company_tweet_id=row.get("company_tweet_id", ""),
                    customer_tweet_id=row.get("customer_tweet_id", ""),
                )
            )
        return hits


@lru_cache(maxsize=1)
def default_index() -> HybridIndex:
    cfg = load_config()
    return HybridIndex.from_csv(ROOT / cfg["data"]["index_pairs"])
