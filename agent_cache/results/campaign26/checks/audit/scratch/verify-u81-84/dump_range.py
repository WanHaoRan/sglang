"""Dump user prompts and assistant text blocks in a line range of a transcript (read only)."""
import json
import sys

path = sys.argv[1]
lo = int(sys.argv[2])
hi = int(sys.argv[3])
cap = int(sys.argv[4]) if len(sys.argv) > 4 else 6000

with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        if t not in ("user", "assistant"):
            continue
        if d.get("isSidechain"):
            continue
        ts = d.get("timestamp", "")
        c = d.get("message", {}).get("content")
        if isinstance(c, str):
            if t == "user":
                print("=== line %d @ %s USER(str) len %d ===" % (i, ts, len(c)))
                print(c[:cap])
                print()
        elif isinstance(c, list):
            for b in c:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    tx = b.get("text", "")
                    if tx.strip():
                        print("=== line %d @ %s %s(text) len %d ===" % (i, ts, t.upper(), len(tx)))
                        print(tx[:cap])
                        print()
