"""Tier-1 deterministic escalation. Rule name is the reason.

This runs BEFORE any LLM. A tweet about a lawsuit should not wait on Groq
and should not be auto-replied from a similar-looking refund thread.
Patterns are intentionally narrow (`kill myself`, not `kill`) so g033
(delay + "killing my schedule") does not false-positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

FLAGS = re.IGNORECASE | re.VERBOSE

RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "self_harm",
        re.compile(
            r"""
            \b(
                kill\ myself | suicide | want\ to\ die | self[-\s]?harm |
                ending\ it\ all
            )\b
            """,
            FLAGS,
        ),
    ),
    (
        "fraud",
        re.compile(
            r"""
            \b(
                stolen\ card | fraudulent | identity\ theft |
                someone\ booked\ in\ my\ name | hacked\ my\ account
            )\b
            """,
            FLAGS,
        ),
    ),
    (
        "legal",
        re.compile(
            r"""
            \b(
                solicitor | lawyer | lawsuit | sue\ you | court\ claim |
                legal\ action | legal\ claim | small\ claims | gdpr\ complaint
            )\b
            """,
            FLAGS,
        ),
    ),
    (
        "request_human",
        re.compile(
            r"""
            \b(
                speak\ to\ (a\ )?(human|person|manager|agent|someone) |
                real\ (person|human) |
                supervisor |
                call\ me |
                phone\ number |
                picking\ up\ your\ phone |
                I\ want\ a\ (human|person|manager)
            )\b
            """,
            FLAGS,
        ),
    ),
]


@dataclass
class GateResult:
    escalate: bool
    rule: str | None
    snippet: str | None = None


def tier1_gate(text: str) -> GateResult:
    for name, pattern in RULES:
        match = pattern.search(text or "")
        if match:
            return GateResult(True, name, match.group(0))
    return GateResult(False, None)
