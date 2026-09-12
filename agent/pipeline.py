"""Runtime ticket loop.

Order is the product: never draft a reply before the safety gate, never send
a draft the lexical check cannot support from retrieved BA text.

  ticket
    → Tier-1 regex (self-harm / fraud / legal / explicit human)
    → intent + confidence (LLM if GROQ_API_KEY else keywords)
    → escalate if confidence < τ
    → hybrid retrieve, intent-filtered, then unfiltered fallback
    → escalate if still empty
    → generate (LLM or historical template)
    → lexical faithfulness → fail → escalate, never send the bad draft
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from agent.classify import classify
from agent.config import load_config
from agent.faithfulness import is_faithful
from agent.gate import tier1_gate
from agent.generate import generate_reply
from retrieval.hybrid import default_index


@dataclass
class AgentOutput:
    intent: str
    status: str  # auto | escalate
    reply: str
    reason: str
    justification: str
    confidence: float
    retrieved_ids: list[str] = field(default_factory=list)


def handle_ticket(text: str) -> AgentOutput:
    cfg = load_config()
    tau = float(cfg["gate"]["confidence_tau"])
    top_k = int(cfg["retrieval"]["top_k"])
    min_score = float(cfg["retrieval"]["min_score"])
    min_overlap = float(cfg["gate"]["faithfulness_min_overlap"])
    fallback = bool(cfg["retrieval"].get("fallback_unfiltered", True))

    gate = tier1_gate(text)
    if gate.escalate:
        return AgentOutput(
            intent="escalate",
            status="escalate",
            reply="",
            reason=gate.rule or "tier1",
            justification=f"Tier-1 rule '{gate.rule}' matched '{gate.snippet}'.",
            confidence=1.0,
        )

    pred = classify(text)
    if pred.confidence < tau:
        return AgentOutput(
            intent=pred.intent,
            status="escalate",
            reply="",
            reason="low_confidence",
            justification=(
                f"{pred.source} intent={pred.intent} confidence={pred.confidence:.2f} "
                f"< tau={tau:.2f}. {pred.rationale}"
            ),
            confidence=pred.confidence,
        )

    index = default_index()
    hits = index.search(
        text,
        k=top_k,
        min_score=min_score,
        intent=pred.intent,
        allow_unfiltered_fallback=fallback,
    )

    if not hits:
        return AgentOutput(
            intent=pred.intent,
            status="escalate",
            reply="",
            reason="empty_retrieval",
            justification="No similar historical resolutions above the score floor.",
            confidence=pred.confidence,
        )

    reply, gen_src = generate_reply(text, pred.intent, hits)
    sources = [h.company_text for h in hits]
    retrieved_ids = [h.company_tweet_id for h in hits if h.company_tweet_id]
    if not is_faithful(reply, sources, text, min_overlap):
        return AgentOutput(
            intent=pred.intent,
            status="escalate",
            reply="",
            reason="faithfulness_fail",
            justification=(
                f"Draft from {gen_src} failed lexical overlap check against retrieved replies."
            ),
            confidence=pred.confidence,
            retrieved_ids=retrieved_ids,
        )

    return AgentOutput(
        intent=pred.intent,
        status="auto",
        reply=reply,
        reason="grounded_reply",
        justification=(
            f"{pred.source} intent={pred.intent} conf={pred.confidence:.2f}; "
            f"retrieved {len(hits)} threads; generator={gen_src}."
        ),
        confidence=pred.confidence,
        retrieved_ids=retrieved_ids,
    )


def handle_ticket_dict(text: str) -> dict:
    return asdict(handle_ticket(text))
