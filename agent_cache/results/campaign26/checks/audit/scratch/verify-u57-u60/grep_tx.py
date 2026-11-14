"""Search user and assistant TEXT blocks of the transcript for a regex, read-only.

usage: grep_tx.py REGEX [LO] [HI] [CTX]
Prints line index, timestamp, role and the match with context.
"""
import json
import re
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
pat = re.compile(sys.argv[1], re.I)
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 220

with open(SRC, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        if '"type":"user"' not in line and '"type":"assistant"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        msg = o.get("message") or {}
        content = msg.get("content")
        texts = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(b.get("text", ""))
        for text in texts:
            for m in pat.finditer(text):
                s = max(0, m.start() - ctx)
                e = min(len(text), m.end() + ctx)
                print("%d %s %s: ...%s..." % (i, t, o.get("timestamp", ""), text[s:e].replace("\n", " ")))
