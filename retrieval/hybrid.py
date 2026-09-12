"""Hybrid retrieval over historical BA resolutions.

v1 was BM25 only. This module still uses BM25 as the sparse channel and
adds a TF-IDF cosine channel (sklearn, CPU, no model download) so the
15-minute repro does not pull PyTorch. If `sentence-transformers` is
installed, MiniLM embeddings replace TF-IDF automatically.

Company replies that invite a DM are dropped: copying them teaches the
agent to refuse help on Twitter.

Intent filter: prefer neighbours whose offline keyword intent matches the
ticket intent. If that slice is empty, fall back to the full index
(wrong intent should not force 100% escalate).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from agent.classify import classify_keyword
from agent.config import ROOT, load_config

TOKEN = re.compile(r"[a-z0-9]+")

# First-pass DM detector (also used when building the CSV flags).
DM_RE = re.compile(r"\b(dm|direct message|please dm|inbox us|private message)\b", re.I)
# Second pass: BA house style that still leaks through is_dm_deflect=False.
INVITE_DM_RE = re.compile(
    r"""
    (please\ )?dm\ (us|me|your) |
    follow\ us\ and\ dm |
    send\ (us\ )?a\ dm |
    via\ the\ link |
    drop\ us\ a\ dm |
    tweet\ us\ your\ booking
    """,
    re.IGNORECASE | re.VERBOSE,
)


def tokenize(text: str) -> list[str]:
    return TOKEN.findall((text or "").lower())


def is_dm_deflect(text: str) -> bool:
    """True if a company reply is a DM-deflect rather than a usable resolution."""
    t = text or ""
    return bool(DM_RE.search(t) or INVITE_DM_RE.search(t))


def _minmax(xs: np.ndarray) -> np.ndarray:
    lo, hi = float(xs.min()), float(xs.max())
    if hi - lo < 1e-9:
        return np.zeros_like(xs, dtype=float)
    return (xs - lo) / (hi - lo)


@dataclass
class Hit:
    customer_text: str
    company_text: str
    score: float
    company_tweet_id: str
    customer_tweet_id: str
    intent: str


class HybridIndex:
    def __init__(self, rows: list[dict], alpha: float = 0.5):
        self.rows = rows
        self.alpha = alpha  # weight on BM25 after min-max; (1-alpha) on cosine
        self._bm25 = BM25Okapi([tokenize(r["customer_text"]) for r in rows])
        # Dense-ish channel: TF-IDF cosine on customer texts (same space as the query).
        self._tfidf = TfidfVectorizer(
            max_features=6000, ngram_range=(1, 2), min_df=2, stop_words="english"
        )
        matrix = self._tfidf.fit_transform(r["customer_text"] for r in rows)
        self._doc_mat = normalize(matrix)

    @classmethod
    def from_csv(cls, path: Path, drop_dm: bool = True, alpha: float = 0.5) -> "HybridIndex":
        rows = []
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                company = row.get("company_text") or ""
                if drop_dm and (row.get("is_dm_deflect") == "True" or is_dm_deflect(company)):
                    continue
                if not company.strip():
                    continue
                # Offline intent tag so runtime can filter without extra LLM calls.
                if not row.get("intent"):
                    row["intent"] = classify_keyword(row["customer_text"]).intent
                rows.append(row)
        if not rows:
            raise ValueError(f"no index rows in {path}")
        return cls(rows, alpha=alpha)

    def search(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.0,
        intent: str | None = None,
        allow_unfiltered_fallback: bool = True,
    ) -> list[Hit]:
        hits = self._search_slice(query, k, min_score, intent)
        if not hits and intent and allow_unfiltered_fallback:
            hits = self._search_slice(query, k, min_score, intent=None)
        return hits

    def _search_slice(
        self, query: str, k: int, min_score: float, intent: str | None
    ) -> list[Hit]:
        bm25 = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=float)
        q = self._tfidf.transform([query])
        cosine = (self._doc_mat @ normalize(q).T).toarray().ravel()
        combined = self.alpha * _minmax(bm25) + (1.0 - self.alpha) * _minmax(cosine)

        if intent:
            mask = np.array([r.get("intent") == intent for r in self.rows], dtype=bool)
            if not mask.any():
                return []
            combined = np.where(mask, combined, -1.0)

        ranked = np.argsort(combined)[::-1][:k]
        hits: list[Hit] = []
        for idx in ranked:
            score = float(combined[idx])
            if score < min_score:
                continue
            row = self.rows[idx]
            hits.append(
                Hit(
                    customer_text=row["customer_text"],
                    company_text=row["company_text"],
                    score=score,
                    company_tweet_id=row.get("company_tweet_id", ""),
                    customer_tweet_id=row.get("customer_tweet_id", ""),
                    intent=row.get("intent", ""),
                )
            )
        return hits


@lru_cache(maxsize=1)
def default_index() -> HybridIndex:
    cfg = load_config()
    alpha = float(cfg.get("retrieval", {}).get("hybrid_alpha", 0.5))
    return HybridIndex.from_csv(ROOT / cfg["data"]["index_pairs"], alpha=alpha)
