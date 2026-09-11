# Decision log

1. **Brand = British_Airways**, not UPS or T-Mobile. Counted inbound customer → company pairs on the full `twcs.csv`. BA: 29,290 pairs, 14% DM-deflect, 77% substantive. UPS was 68% “DM your tracking”. T-Mobile was 82% DM. Grounding on DM-heavy brands would teach the agent to refuse help.

2. **Subsample instead of full 3M tweets.** Assignment says graders will not run the full dump. Index = 3,500 substantive BA pairs; cluster = 2,934 deduped customer texts. Seed 7.

3. **Strip @handles and URLs** before clustering and indexing. They dominate TF-IDF otherwise.

4. **Drop DM-deflect company replies from the index.** Otherwise nearest-neighbour copies “please DM your booking ref”.

5. **KMeans + hand merge, not HDBSCAN.** MiniBatchKMeans k=10 on TF-IDF produced one 1,305-tweet junk bucket and mixed delay/cancel. Taxonomy in `intents.yaml` is reviewed labels (10 intents), not cluster ids.

6. **Tier-1 is regex, not an LLM.** Self-harm, fraud, legal, explicit human request fire before classification. Reason = rule name. Avoids paying a model to notice “I will sue you”.

7. **τ lives in config and is swept in `eval.run_eval`.** 0.55 is a placeholder operating point (prefer precision of auto-send). Re-fit on 150+ gold; do not guess.

8. **If intent-filtered retrieval is empty, retry unfiltered, then escalate.** Wrong intent should not force 100% escalate. Current BM25 index is not intent-tagged yet, so runtime is unfiltered BM25; tagging the index is a next-week item.

9. **Faithfulness at runtime is lexical overlap, not LLM-as-judge.** Judge family is reserved for `eval/judge.py`. Runtime judge doubles cost and is untrusted until κ is measured.

10. **One optional API: Groq.** Classifier/generator default `llama-3.1-8b-instant`. Judge should be a different family (`mixtral-8x7b-32768` in config). Harness runs with zero keys via keywords + BM25 + template.

11. **Trivial baseline never escalates** (majority intent + canned reply). **Simple baseline** = same regex gate + keyword intent + 1-NN historical reply, always auto if gate is clean. That makes “we escalate more than simple” visible on the PR curve.

12. **Gold sampling (starter n=50).** From `pairs_cluster.csv` plus constructed edge strings that still look like tweets: at least 3 per intent, then oversample request-human, legal, fraud, self-harm, sarcasm, multi-issue. Not i.i.d. from Twitter — on purpose. Expand to 150 with the same strata.

13. **Do not auto-send complimentary upgrades or EU261 eligibility.** Generator instructions forbid inventing policy; historical tweets often waffle. Escalating those is correct.

14. **“kill” is not a self-harm rule.** Pattern is `kill myself` / suicide / want to die. Gold g033 exists so we do not over-gate.

15. **Embeddings deferred.** MiniLM can replace BM25 later; first skeleton is CPU, no download, 15-minute repro. Hybrid BM25+embeddings is the planned retrieval, not the committed v1 index.
