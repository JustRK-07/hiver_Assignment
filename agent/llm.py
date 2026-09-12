"""Groq chat wrapper: dotenv, retries, on-disk cache.

The cache is what makes `eval.run_eval` re-runnable in <15 minutes after the
first paid pass. Keys are (model, prompt hash) so a prompt tweak busts cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

from agent.config import ROOT, load_config

load_dotenv(ROOT / ".env")

CACHE_PATH = ROOT / ".cache" / "llm.json"

# In-memory cache — loaded from disk on first use, written only when a new
# entry is added. This avoids re-reading and re-parsing the JSON file on
# every call to chat() during an eval run.
_CACHE: dict | None = None


def _get_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(CACHE_PATH.read_text(encoding="utf-8")) if CACHE_PATH.exists() else {}
    return _CACHE


def _save_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(_CACHE), encoding="utf-8")


def chat(prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    cfg = load_config()
    model = model or cfg["models"]["generator"]
    key = hashlib.sha256(f"{model}\n{temperature}\n{prompt}".encode()).hexdigest()
    cache = _get_cache()
    if key in cache:
        return cache[key]

    url = cfg["models"]["groq_base_url"].rstrip("/") + "/chat/completions"
    last_err: Exception | None = None
    for attempt in range(6):
        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=90,
            )
            if resp.status_code == 429:
                last_err = RuntimeError("429 rate limited")
                time.sleep(min(60, 3 ** attempt))
                continue
            if not resp.ok:
                last_err = RuntimeError(f"{resp.status_code} {resp.text[:400]}")
                time.sleep(2 ** attempt)
                continue
            # resp.ok is True here — no raise_for_status() needed.
            text = resp.json()["choices"][0]["message"]["content"]
            cache[key] = text
            _save_cache()
            return text
        except requests.RequestException as exc:
            last_err = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Groq chat failed after retries: {last_err}")



def chat_json(prompt: str, model: str | None = None) -> dict:
    raw = chat(prompt + "\nRespond with JSON only.", model=model, temperature=0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
