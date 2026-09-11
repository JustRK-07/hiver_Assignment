# Report (draft — fill after n=150 gold and a full judge pass)

## Problem framing

**Brand:** British Airways Twitter care (`British_Airways`).

**Good** means: auto-send only when the intent is one of the 10 reviewed labels, a similar historical resolution exists that is not a DM-deflect, and the draft does not invent refunds/rebooks/upgrades. Everything else escalates with a **named** reason (`self_harm`, `fraud`, `legal`, `request_human`, `low_confidence`, `empty_retrieval`, `faithfulness_fail`).

**Not built:** live booking tools, multi-turn DM, executing compensation, HDBSCAN taxonomy, MiniLM embeddings (v1 is BM25), LLM-as-judge on the send path.

## Results vs baselines

Run `PYTHONPATH=. python -m eval.run_eval` and paste the JSON here.

On the 50-row **starter** gold set, treat numbers as directional. The trivial baseline always auto-sends a canned reply (escalation recall = 0). The simple baseline shares Tier-1 regex but never uses τ or faithfulness.

## Failure analysis (hypotheses — replace with eval/runs examples)

1. **Multi-issue tickets.** Delay + compensation + bump (g028) collapses to one label; retrieval may mix three policies.
2. **Sarcasm.** “Well thanks, seats are gone” (g040) looks like praise to bag-of-words.
3. **Legal bolted onto an operational ask.** g006/g036 must escalate even if refund/baggage retrieval is strong.
4. **Self-harm with travel content.** g037 is a cancel tweet; gate must win.
5. **False DM-like drafts.** If a non-DM neighbour still says “follow us and DM”, overlap checks can still pass. Filter remaining invite-to-DM language in a later index pass.

## What is misleading about my headline number

- n=50 is oversampled on edge cases, so escalation recall looks better than production Twitter.
- Auto-rate on this set is not “share of BA tweets we could handle”; most real tweets are thanks / seat FAQs.
- Intent macro-F1 ignores that `general_query` is a dump label.
- Template replies can pass overlap by copying a historical tweet that is on-topic but wrong for *this* PNR.
- τ=0.55 is not yet the precision/recall operating point from a 150-row curve.

## Next week

1. Hand-label to 150 with the same strata; freeze the sample list in `eval/`.
2. Fit τ on a precision/recall curve; pick high precision for auto-send.
3. Tag index rows with intent; enable filtered retrieval + unfiltered fallback.
4. Add MiniLM embeddings; keep BM25.
5. Run judge on 40–60 rows; I re-label and report % agreement / Cohen’s κ.
6. Second-pass DM language filter on company text.
