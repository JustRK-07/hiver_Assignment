# Hiver support agent — British Airways

Conservative auto-reply agent: classify a tweet, ground a draft in historical BA replies, or escalate with a named reason.

**Headline (n=150 gold, Groq):** system escalation F1 **0.78** (P 0.88 / R 0.70) vs trivial **0.00** vs simple **0.80**; retrieval hit@5 **0.48** vs simple **0.00**. Full table and caveats: [`report.md`](report.md). JSON: [`eval/headline_metrics.json`](eval/headline_metrics.json).

## Brand (verified on `twcs.csv`)

| Brand | Pairs | DM-deflect | Substantive |
|---|---:|---:|---:|
| British_Airways | 29,290 | 14% | 77% |
| UPSHelp (rejected) | 17,765 | 68% | mostly “DM tracking” |
| TMobileHelp (rejected) | 34,215 | 82% | 16% substantive |

Subsample in-repo: `data/pairs_index.csv` (3,500), `data/pairs_cluster.csv` (2,934).

## Auto-send only if all hold

1. No Tier-1 rule (self-harm, fraud, legal, explicit human).
2. Intent confidence ≥ τ = **0.55** (LLM PR curve; see report — the curve is flat on [0.35, 0.55]).
3. Hybrid retrieval (BM25 + TF-IDF cosine) returns a non-DM neighbour (intent slice, then unfiltered fallback).
4. Draft passes lexical overlap vs retrieved text.

## Reproduce headline numbers (<15 min after the first Groq pass)

Python 3.11+. First Groq pass is slower (rate limits); `.cache/llm.json` is local-only. Graders without a key get the keyword/template path — numbers will differ; `eval/headline_metrics.json` is the quoted run.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY=...          # optional; omit for the no-key smoke test
PYTHONPATH=. python -m eval.run_eval --tau-classifier llm
PYTHONPATH=. python scripts/run_ticket.py "BA0273 delayed three hours at the gate"
```

Judge + agreement (needs key):

```bash
PYTHONPATH=. python -m eval.judge --limit 50
PYTHONPATH=. python -m eval.agreement
```

Rebuild the labeling worksheet (does **not** write gold labels):

```bash
PYTHONPATH=. python -m scripts.build_gold_candidates
```

Dataset citation: [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (thoughtvector). Optional intent set: PolyAI Banking77 — not used in the submitted BA taxonomy.

## Layout

```
data/          pairs + extract_brand.py
taxonomy/      intents.yaml + cluster.py
retrieval/     hybrid BM25 + TF-IDF cosine
agent/         gate → classify → retrieve → generate → overlap
eval/          golden_set.csv (n=150), baselines, judge, run_eval
report.md      assignment writeup
DECISIONS.md   non-obvious calls
```

Do not commit `.env`. Rotate any key that was pasted into chat.
