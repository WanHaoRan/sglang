#!/usr/bin/env python3
"""List real user prompts of a transcript: index, timestamp, text (truncated). usage: prompts.py <jsonl> [maxlen]"""
import sys, json
src = sys.argv[1]
mx = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
SKIP = ('<system-reminder>', '<task-notification>', '<local-command', '<command-')
with open(src, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "user":
            continue
        if o.get("isMeta"):
            continue
        msg = o.get("message") or {}
        c = msg.get("content")
        texts = []
        if isinstance(c, str):
            texts = [c]
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        real = [t for t in texts if t.strip() and not t.lstrip().startswith(SKIP)]
        if not real:
            continue
        j = "\n".join(real)
        tag = "SUMMARY" if (o.get("isCompactSummary") or j.startswith("This session is being continued")) else "PROMPT"
        print(f"=== [{i}] {o.get('timestamp','')} {tag} len={len(j)} cwd={o.get('cwd','')} ===")
        print(j[:mx] if tag == "PROMPT" else j[:400])
        print()
