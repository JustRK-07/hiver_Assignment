# Decision log

1. **Brand = British_Airways**, not UPS or T-Mobile. Counted inbound customer → company pairs on the full `twcs.csv`. BA: 29,290 pairs, 14% DM-deflect, 77% substantive. UPS was 68% “DM your tracking”. T-Mobile was 82% DM. Grounding on DM-heavy brands would teach the agent to refuse help.

2. **Subsample instead of full 3M tweets.** Assignment says graders will not run the full dump. Index = 3,500 substantive BA pairs; cluster = 2,934 deduped customer texts. Seed 7.

3. **Strip @handles and URLs** before clustering and indexing. They dominate TF-IDF otherwise.

4. **Drop DM-deflect company replies from the index**, plus a second-pass `follow us and DM` / `via the link` filter. Otherwise nearest-neighbour copies “please DM your booking ref”.

5. **KMeans + hand merge, not HDBSCAN.** MiniBatchKMeans k=10 on TF-IDF produced one 1,305-tweet junk bucket and mixed delay/cancel. Taxonomy in `intents.yaml` is reviewed labels (10 intents), not cluster ids.

6. **Tier-1 is regex, not an LLM.** Self-harm, fraud, legal, explicit human request fire before classification. Reason = rule name. After gold FNs we added `legal claim`, `small claims`, `speak to someone`, `picking up your phone`.

7. **τ = 0.55 from the LLM precision/recall curve, not a guess.** On n=150 the curve is flat for τ ∈ [0.35, 0.55] (auto-send precision 0.921). We keep 0.55 so keyword fallback still gates. We refuse τ=0.85 even though auto-send precision ticks up: escalation precision collapses to 0.50.

8. **Intent-filtered hybrid retrieval with unfiltered fallback.** Each index row is keyword-tagged offline. Search BM25 + TF-IDF cosine (α=0.55). If the intent slice is empty, search the full index, then escalate.

9. **Faithfulness at runtime is lexical overlap, not LLM-as-judge.** Judge family is reserved for `eval/judge.py`. Runtime judge doubles cost and is untrusted (groundedness exact agreement 26%).

10. **One API: Groq.** Llama chat SKUs were 404 on this key. Classifier/generator = `openai/gpt-oss-20b`. Judge = `qwen/qwen3.8-27b` (Qwen vs OpenAI). Disk cache `.cache/llm.json` (gitignored) makes the second eval pass ~20s. No-key path: keywords + hybrid + templates.

11. **Trivial baseline never escalates** (majority intent + canned reply). **Simple baseline** = same regex gate + keyword intent + 1-NN historical reply, always auto if gate is clean.

12. **Gold sampling n=150.** g001–g050: starter, mixed real tweets + constructed legal/self-harm. g051–g148: read from `eval/golden_set_candidates.csv` (seed-42 worksheet from `scripts/build_gold_candidates.py`). Labels were **not** copied from `suggested_intent` blindly — see `eval/finalize_gold.py`. g149–g150: constructed legal + self-harm because those regexes had almost no natural hits in the BA subsample. Not i.i.d. from Twitter, on purpose.

13. **Do not auto-send complimentary upgrades or EU261 eligibility.** Generator instructions forbid inventing policy.

14. **“kill” is not a self-harm rule.** Pattern is `kill myself` / suicide / want to die. Gold g033 exists so we do not over-gate.

15. **Dense channel is TF-IDF cosine, not MiniLM, in v1.** Hybrid still exists (sparse BM25 + vector cosine) without a 400MB download, which would break the 15-minute README. MiniLM is a one-week swap behind the same `HybridIndex.search` API.
