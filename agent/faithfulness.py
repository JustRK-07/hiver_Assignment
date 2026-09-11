from __future__ import annotations

import re

STOP = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "and",
    "or",
    "of",
    "in",
    "on",
    "your",
    "you",
    "we",
    "our",
    "is",
    "are",
    "be",
    "with",
    "this",
    "that",
    "please",
    "hi",
    "sorry",
}


def tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in STOP and len(t) > 2}


def overlap_ratio(reply: str, sources: list[str]) -> float:
    reply_tok = tokens(reply)
    if not reply_tok:
        return 0.0
    src = set()
    for s in sources:
        src |= tokens(s)
    if not src:
        return 0.0
    return len(reply_tok & src) / len(reply_tok)


def is_faithful(reply: str, retrieved_replies: list[str], ticket: str, min_overlap: float) -> bool:
    return overlap_ratio(reply, retrieved_replies + [ticket]) >= min_overlap
