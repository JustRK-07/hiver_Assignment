# Hiver support agent — British Airways

Turn noisy Twitter threads into a **conservative** auto-reply agent: classify intent, ground a draft in historical BA replies, or escalate with a named reason.

Headline metric on the current 50-row starter gold set is **escalation F1 and intent macro-F1 vs two baselines**, not “how often we auto-send.”

## Brand (verified from `twcs.csv`)

| Brand | Customer↔company pairs | DM-deflect replies | Substantive replies (≥12 tokens, not DM) |
|---|---:|---:|---:|
| British_Airways | 29,290 | 14.0% | 77.2% |
| UPSHelp (rejected) | 17,765 | 67.5% | too many “please DM tracking” |
| TMobileHelp (rejected) | 34,215 | 81.8% | 15.5% |

Committed subsample: 3,500 substantive pairs in `data/pairs_index.csv` (grounding) and 2,934 cluster texts in `data/pairs_cluster.csv`.

## What this agent will auto-send

Only when **all** of these hold:

1. No Tier-1 rule (self-harm, fraud, legal, explicit human request).
2. Intent confidence ≥ τ (`config.yaml`, currently **0.55** until the gold set is 150+ and the PR curve is re-fit).
3. Hybrid (BM25) retrieval returns at least one non-DM historical resolution.
4. Draft passes a **lexical** faithfulness overlap check (not an LLM judge).

Otherwise: `status=escalate` and `reason` is the rule name.

Out of scope: live PNR lookup, refunds as actions, multi-turn DMs, executing rebooks.

## 15-minute reproduction

Python 3.11+. No API key required for the harness (keyword classifier + BM25 + template reply).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m eval.run_eval
PYTHONPATH=. python scripts/run_ticket.py "BA0273 delayed three hours at the gate and nobody is telling us anything"
```

Optional Groq for LLM classify / generate / judge:

```bash
export GROQ_API_KEY=...
PYTHONPATH=. python -m eval.judge --limit 40
```

Rebuild pairs from the full Kaggle file (not required for graders):

```bash
PYTHONPATH=. python data/extract_brand.py --src /path/to/twcs.csv
PYTHONPATH=. python taxonomy/cluster.py
```

## Layout

```
data/          pairs + extract_brand.py
taxonomy/      intents.yaml (reviewed) + cluster.py
retrieval/     BM25 index (DM replies dropped)
agent/         gate → classify → retrieve → generate → overlap check
eval/          golden_set.csv, baselines.py, judge.py, run_eval.py
report.md      assignment writeup (fill after 150 labels)
DECISIONS.md   10–15 non-obvious calls
```

## Gold set status

`eval/golden_set.csv` is a **50-example starter**, stratified across 10 intents with oversampled legal/fraud/self-harm/human-request/sarcasm/multi-issue. Target is **150 hand labels**. Sampling note is in `DECISIONS.md`.
