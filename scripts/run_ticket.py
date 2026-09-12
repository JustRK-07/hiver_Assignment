#!/usr/bin/env python3
"""Run one ticket through the agent. Example:

    PYTHONPATH=. python scripts/run_ticket.py "BA0273 delayed three hours at the gate"
"""

from __future__ import annotations

import argparse
import json

from agent.pipeline import handle_ticket_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Customer tweet text")
    ap.add_argument("--file", help="Read tweet from a text file")
    args = ap.parse_args()
    text = args.text
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    if not text:
        raise SystemExit("pass tweet text or --file")
    print(json.dumps(handle_ticket_dict(text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
