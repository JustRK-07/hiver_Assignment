# Report — British Airways Twitter Support Agent

## 1. Architecture & Approach

Our goal was to build a highly conservative auto-reply agent. In customer service, confidently giving the wrong answer (e.g., promising a refund that isn't due) is much worse than escalating to a human. Therefore, our pipeline is designed around a "fail-safe" architecture:

1. **Safety Gate (Tier 1):** Fast, deterministic regex rules catch severe cases (self-harm, fraud, legal threats, explicit human requests) before any LLM is called. 
2. **Intent Classification:** We classify the ticket into one of 10 intents. If the LLM confidence falls below $\tau = 0.55$, we escalate. (A keyword-based classifier serves as a fallback).
3. **Intent-Filtered Hybrid Retrieval:** We retrieve historical resolutions from a vector index using BM25 and TF-IDF cosine similarity. We filter for neighbours matching the predicted intent first.
4. **Generation:** The LLM drafts a reply strictly grounded in the retrieved historical evidence.
5. **Faithfulness Check:** A final lexical overlap check ensures the generated draft doesn't invent facts (like EU261 eligibility) missing from the retrieved context.

**What we chose *not* to build:** We did not build live PNR lookups, automated refund execution, multi-turn dialogue trees, or an LLM-in-the-loop judge on the final send path (for latency and cost reasons).

---

## 2. Dataset & Taxonomy Construction

**Primary Dataset:** We used the Kaggle `Customer Support on Twitter` corpus. We specifically filtered for `British_Airways` (29,290 pairs). We rejected `UPSHelp` (68% DM-deflection) and `TMobileHelp` (82% DM-deflection) because training on them would teach the agent to constantly refuse help and ask for a DM.

**Optional Secondary Dataset (Banking77):** We explicitly chose **not** to use the PolyAI/Banking77 dataset. Banking intents do not map well to airline crises (like tarmac delays or lost baggage). Instead, we derived an empirical 10-intent taxonomy directly from the BA data using TF-IDF MiniBatchKMeans clustering followed by manual review and merging (see `taxonomy/intents.yaml`).

---

## 3. Golden Evaluation Set Construction

To evaluate the system, we constructed a golden set of **n=150** labelled examples (`eval/golden_set.csv`). *This is an adversarially hard set, not an identically distributed (i.i.d.) random sample.* 

**Sampling Methodology:**
- **`g001`–`g050`:** A starter set containing a mix of real tricky tweets and synthetic edge-cases (legal threats, self-harm) to ensure the safety gate could be tested.
- **`g051`–`g148`:** Sampled using a seeded script (`scripts/build_gold_candidates.py`) to extract diverse customer texts. Labels were hand-annotated, not blindly copied from clusters.
- **`g149`–`g150`:** Hand-constructed safety violations, as the natural British Airways subsample lacked sufficient extreme examples (like severe self-harm) to calculate recall.

---

## 4. Results & Metrics Comparison

We benchmarked the **System** against two baselines:
1. **Trivial:** Always assumes the majority intent (`general_query`), sends a canned reply, and never escalates.
2. **Simple:** Uses the regex safety gate, a keyword-based intent classifier, and 1-NN retrieval (returning the exact historical text). It auto-sends if the regex gate is clean.

*Note: For `retrieval_hit@5`, the denominator (`retrieval_n`) is 98, representing the subset of gold rows that had a known, relevant historical company tweet ID.*

| System | Intent Macro-F1 | Intent Acc. | Esc. Precision | Esc. Recall | Esc. F1 | Auto-Rate | Retrieval hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Trivial** | 0.019 | 0.107 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| **Simple** | **0.732** | **0.700** | **1.000** | 0.667 | **0.800** | **0.867** | 0.000 |
| **System** (Ours)| 0.713 | 0.673 | 0.875 | **0.700** | 0.778 | 0.840 | **0.480** |

**Operating point: $\tau = 0.55$.** 
The LLM classifier (`gpt-oss-20b`) is highly confident. The precision/recall curve is completely flat for $\tau \in [0.35, 0.55]$. We published 0.55 so that the keyword fallback (which produces much lower confidences) still has an effective gate. 

**Why didn't System beat Simple on every metric?**
Simple's escalation precision is 1.0 because it *only* escalates on exact regex matches (which are 100% accurate by definition). However, Simple misses ambiguous cases. The System trades a small amount of precision for higher recall (0.70 vs 0.667) and significantly vastly superior retrieval (0.48 vs 0.00). System Intent F1 is slightly lower by design: when the system safety-escalates, we map the predicted intent to `general_query` to avoid inventing an 11th class, slightly penalizing the F1 score in exchange for safety.

---

## 5. Failure Analysis (Top 5)

1. **Legal phrasing the first regex missed.** `g006` (“this is a legal claim”) and `g130` (“small claims court”) auto-sent until we added those literals to Tier-1. *Residual risk:* “I’ll see you in court” without those tokens. *Hypothesis:* legal language is long-tail; keep growing the list from gold FNs, do not LLM this gate.
2. **Soft human-request that is not `phone number` / `manager`.** `g050` (looping on “call customer relations”), `g086` (emergency change, MMB down). Gold = escalate; system drafted a helpful-looking auto reply. *Hypothesis:* $\tau$ does not encode “needs a tool”; add an `ambiguous_needs_human` keyword family or a second-pass “can this be solved without PNR?” check.
3. **Lost-property drafted as a baggage/FAQ tweet.** `g111` (notebook left on BA0890). Auto replies guessed airport lost-and-found. Sometimes historically grounded, but still the wrong product decision. *Hypothesis:* lost-item → escalate unless the index contains a *specific* lounge/airport procedure.
4. **Multi-issue intent collapse.** `g027` dirty cabin + denied refund → `refund_compensation`. Retrieval then mixes policies. *Hypothesis:* For multi-issue complaints, escalate unless one action absolutely dominates the text.
5. **Sarcasm / praise-shaped complaints.** `g026` “POINTLESS phone call”, `g040` “well thanks… seats gone”. Keywords and sometimes the LLM land on `general_query`. *Hypothesis:* Provide few-shot sarcasm examples in the classifier prompt to prevent auto-sending happy templates to angry customers.

---

## 6. What is misleading about my headline number?

- **n=150 is not i.i.d. BA Twitter.** We oversampled legal/fraud/self-harm/human-request/sarcasm. Escalation recall looks better than it would on a random day of “thanks crew” tweets.
- **Auto-rate 84% is not “share of BA we could handle in production.”** Production has many more PNR-specific asks that we would rightfully escalate. Our gold set isolates policy FAQs.
- **Intent macro-F1 0.71 < simple 0.73** is easy to screenshot as a loss. As noted above, safety escalations map to `general_query`, actively penalizing intent F1 in favor of the Escalation product metric.
- **Retrieval hit 0.48** means “the original company tweet id from the cluster file appeared in hybrid top-5 of a 3,500-row subsample,” not “the neighbour was a good policy match.” 
- **$\tau=0.55$ was not a dramatic sweep win.** The curve is flat because `gpt-oss-20b` is over-confident. A different LLM would require an entirely new calibration curve.

---

## 7. LLM-as-Judge & Human Agreement

We built an automated evaluation harness (`eval/judge.py`) using `qwen/qwen3.8-27b` to score 50 system outputs on a 1-5 scale across four dimensions: **Groundedness, Helpfulness, Tone, and Escalation Correctness**. We then hand-labelled these 50 rows (`eval/judge_human.csv`) to calculate agreement.

**Agreement Caveat (Crucial):**
- **Tone:** 58% exact match (96% within ±1).
- **Escalation Correctness:** 94% exact match (Cohen's $\kappa=1.0$ on binarized data). *Note: The prompt provided the gold escalation flag, inflating this score artificially.*
- **Groundedness:** **Only 26% exact match.**

Because the LLM judge is highly unreliable at strict groundedness evaluation (often hallucinating that a generated refund policy was grounded in the text when it wasn't), **we do not use LLM-as-judge on the runtime send path**. Instead, we rely on a strict lexical overlap check (`overlap_ratio` in `faithfulness.py`). 

---

## 8. One More Week

With one additional week, we would prioritize:
1. **Dense Embeddings:** Swap TF-IDF for MiniLM (or BGE-small) as the dense channel in our Hybrid index, caching vectors in `retrieval/.index_cache/`.
2. **Context Stitching:** Stitch multi-part BA tweets (`1/2`, `2/2`) together so retrieval context isn't arbitrarily truncated.
3. **True Double-Blind Annotation:** Recruit a second independent human rater for the 50 judge rows (with gold labels hidden) to report a mathematically sound Cohen's $\kappa$.
4. **Intent-Conditioned Generation:** If a customer is clearly "upgrade fishing," inject a system prompt forcing a polite refusal template, overriding any historically retrieved neighbour that might have granted an upgrade.
