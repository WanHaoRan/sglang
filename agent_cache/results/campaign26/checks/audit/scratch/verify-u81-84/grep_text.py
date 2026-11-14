"""Grep assistant text blocks (and user prompts) of a transcript for a regex since a date (read only)."""
import json
import re
import sys

path = sys.argv[1]
pat = re.compile(sys.argv[2])
since = sys.argv[3] if len(sys.argv) > 3 else "2026-11-11"
width = int(sys.argv[4]) if len(sys.argv) > 4 else 500
roles = sys.argv[5].split(",") if len(sys.argv) > 5 else ["assistant", "user"]

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if not pat.search(line):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        if t not in roles:
            continue
        if d.get("isSidechain"):
            continue
        ts = d.get("timestamp", "")
        if ts < since:
            continue
        msg = d.get("message", {})
        c = msg.get("content")
        texts = []
        if isinstance(c, str):
            texts.append(("str", c))
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") == "text":
                    texts.append(("text", b.get("text", "")))
        for kind, tx in texts:
            for m in pat.finditer(tx):
                a = max(0, m.start() - width)
                b = min(len(tx), m.end() + width)
                print("=== line %d @ %s %s/%s (textlen %d, match at %d) ===" % (i, ts, t, kind, len(tx), m.start()))
                print(tx[a:b].replace("\n", "\n    "))
                print()
