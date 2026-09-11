from __future__ import annotations

import json
import os
import re

import requests
from dotenv import load_dotenv

from agent.config import load_config

load_dotenv()


def chat(prompt: str, model: str | None = None, temperature: float = 0.1) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    cfg = load_config()
    model = model or cfg["models"]["generator"]
    url = cfg["models"]["groq_base_url"].rstrip("/") + "/chat/completions"
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def chat_json(prompt: str, model: str | None = None) -> dict:
    raw = chat(prompt + "\nRespond with JSON only.", model=model, temperature=0)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
