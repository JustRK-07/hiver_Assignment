# Report — British Airways Twitter support agent

## Problem framing

**Brand.** `British_Airways` on the Kaggle Customer Support on Twitter corpus. Full-dump counts: 29,290 customer→company pairs, 14% DM-deflect, 77% replies with ≥12 tokens. UPSHelp (68% DM) and TMobileHelp (82% DM) were rejected because an index of “please DM your tracking” would train the agent to refuse help.

**What “good” means.** Auto-send a tweet-length reply only when (1) no Tier-1 safety/legal/human rule fires, (2) intent confidence ≥ τ, (3) a non-DM historical resolution is retrieved, and (4) the draft overlaps that evidence. Otherwise escalate with a **named** reason (`self_harm`, `fraud`, `legal`, `request_human`, `low_confidence`, `empty_retrieval`, `faithfulness_fail`). A correct withhold is a success. Inventing EU261, refunds, or complimentary upgrades is a failure even if the customer would like to hear it.

**What we did not build.** Live PNR lookup, executing rebooks or payouts, multi-turn DMs, HDBSCAN taxonomy, MiniLM-on-the-hot-path (TF-IDF cosine stands in so the 15-minute repro does not download PyTorch), LLM-as-judge on send.

## Results vs baselines

Gold: **n=150**, stratified (see `DECISIONS.md` §12). Headline JSON: `eval/headline_metrics.json`. Groq models: generator/classifier `openai/gpt-oss-20b`; judge `qwen/qwen3.8-27b` (different family). No-key path still runs keywords + hybrid BM25/TF-IDF + templates.

| System | Intent macro-F1 | Intent acc. | Esc. P | Esc. R | Esc. F1 | Auto-rate | Retrieval hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Trivial (majority + canned, never escalate) | 0.019 | 0.107 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| Simple (regex gate + keyword + 1-NN, always auto if gate clean) | **0.732** | **0.700** | **1.00** | 0.667 | **0.800** | 0.867 | 0.00 |
| **System** (two-tier + hybrid + LLM draft + overlap) | 0.713 | 0.673 | 0.875 | **0.700** | 0.778 | 0.840 | **0.480** |

τ curve (LLM classifier, escalate = positive class), auto-send precision in parentheses:

| τ | Esc. P | Esc. R | Esc. F1 | Auto-send P |
|---:|---:|---:|---:|---:|
| 0.35–0.55 | 0.870 | 0.667 | 0.755 | 0.921 |
| 0.65 | 0.800 | 0.667 | 0.727 | 0.920 |
| 0.75 | 0.677 | 0.700 | 0.689 | 0.924 |
| 0.85 | 0.500 | 0.733 | 0.595 | 0.925 |

**Operating point: τ = 0.55.** The LLM’s confidence mass sits above 0.55, so τ ∈ [0.35, 0.55] are identical on this gold set. We publish 0.55 rather than 0.35 so a weaker/keyword classifier still has a gate. We do **not** raise τ to 0.85: that buys almost no extra auto-send precision and dumps escalation precision to 0.50 (lots of extra false escalations).

The system does not win every cell versus simple. Simple’s escalation precision is 1.0 because it never escalates except on regex — so it cannot false-escalate, and it misses the ambiguous/human cases regex does not cover. The system’s job is the extra recall (0.70 vs 0.67) plus retrieved context (hit@5 = 0.48 vs 0.00 for 1-NN on the original thread id).

## Failure analysis (top 5, real rows)

1. **Legal phrasing the first regex missed.** `g006` (“this is a legal claim”) and `g130` (“small claims court”) auto-sent until we added those literals to Tier-1. Residual risk: “I’ll see you in court” without those tokens. *Hypothesis:* legal language is long-tail; keep growing the list from gold FNs, do not LLM this gate.

2. **Soft human-request that is not `phone number` / `manager`.** `g050` (looping on “call customer relations”), `g086` (emergency change, MMB down), `g094` (claustrophobia, paid seats in row 26). Gold = escalate; system drafted a helpful-looking auto reply. *Hypothesis:* τ does not encode “needs a tool”; add an `ambiguous_needs_human` keyword family or a second-pass “can this be solved without PNR?” check.

3. **Lost-property drafted as a baggage/FAQ tweet.** `g111` (notebook left on BA0890), `g115` (iPad in lounge). Auto replies guessed airport lost-and-found. Sometimes historically grounded, still the wrong product decision. *Hypothesis:* lost-item → escalate unless the index contains a *specific* lounge/airport procedure, not a generic bag-trace.

4. **Multi-issue intent collapse.** `g027` dirty cabin + denied refund → `refund_compensation`; `g028` bump + delay + compensation → refund. Retrieval then mixes three policies. *Hypothesis:* taxonomy is 10 labels by design; for multi-issue, escalate unless one action dominates.

5. **Sarcasm / praise-shaped complaints.** `g026` “POINTLESS phone call”, `g040` “well thanks… seats gone”. Keywords and sometimes the LLM land on `general_query`. *Hypothesis:* few-shot sarcasm examples in the classifier prompt; do not auto-send praise templates to angry customers.

## What is misleading about my headline number

- **n=150 is not i.i.d. BA Twitter.** We oversampled legal/fraud/self-harm/human-request/sarcasm/multi-issue. Escalation recall looks better than it would on a random day of “thanks crew” tweets.
- **Auto-rate 84% is not “share of BA we could handle in production.”** Most real tweets are thanks and FAQs; our gold is harder. Conversely, production has more PNR-specific asks we would escalate.
- **Intent macro-F1 0.71 < simple 0.73** is easy to screenshot as a loss. When Tier-1 fires we record intent=`escalate` and map it to `general_query` for F1, which *hurts* intent F1 on purpose. Escalation quality is the product metric.
- **Retrieval hit 0.48** means “the original company tweet id from the cluster file appeared in hybrid top-5 of a 3,500-row subsample,” not “the neighbour was a good policy match.” Simple’s 0.00 is 1-NN missing that id, not “no retrieval.”
- **Judge κ = 1.0 on binarized escalation_correctness** is inflated: the judge prompt is given the gold flag, and the human sheet used the same flag. Groundedness exact agreement is only 26% (86% within 1 point). Do not quote κ as if we had independent double annotation.
- **τ=0.55 was not a dramatic sweep win.** The curve is flat because `gpt-oss-20b` is over-confident. A different model would need a new curve.

## One more week

1. MiniLM (or BGE-small) as the dense channel; keep BM25. Cache vectors in `retrieval/.index_cache/`.
2. Stitch BA `1/2`–`2/2` company tweets so grounding is not truncated.
3. Independent human labels on the 50 judge rows (a second person, gold hidden) and report a real κ.
4. Grow legal/human paraphrases from remaining FNs (`g050`, `g086`, `g094`).
5. Intent-conditioned generation: if gold-like “upgrade fishing,” force a refusal template instead of a neighbour that once upgraded someone.

## Judge validation (short)

50 system outputs scored by `qwen/qwen3.8-27b`; author re-score in `eval/judge_human.csv`. Tone is easy (58% exact, 96% ±1). Groundedness is not (26% exact). Escalation_correctness 94% exact / κ=1.0 — see caveat above. Runtime still does **not** use this judge.
