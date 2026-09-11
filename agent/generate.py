from __future__ import annotations

import os

from agent.config import load_config
from agent.llm import chat
from retrieval.hybrid import Hit


def _template_reply(ticket: str, hits: list[Hit]) -> str:
    if not hits:
        return ""
    best = hits[0].company_text.strip()
    # Strip agent sign-offs like "^Jane" stay; they are historical.
    return (
        "Based on how we have handled similar cases: "
        f"{best} If this does not match your booking, a colleague will take it from here."
    )


def generate_reply(ticket: str, intent: str, hits: list[Hit]) -> tuple[str, str]:
    """Returns (reply, source)."""
    evidence = "\n".join(
        f"- Similar customer: {h.customer_text[:220]}\n  Historical reply: {h.company_text[:280]}"
        for h in hits
    )
    if os.getenv("GROQ_API_KEY"):
        cfg = load_config()
        prompt = f"""You are drafting a British Airways Twitter support reply.
Intent: {intent}
Incoming tweet: {ticket}

Use ONLY facts and actions present in the historical replies below.
Do not invent refunds, EU261 eligibility, rebooking, or policy.
Do not tell the customer to DM unless a historical reply does and no other action is available.
Keep it under 80 words, professional, no hashtags.

Evidence:
{evidence}

Write the reply only."""
        return chat(prompt, model=cfg["models"]["generator"]).strip(), "llm"
    return _template_reply(ticket, hits), "template"
