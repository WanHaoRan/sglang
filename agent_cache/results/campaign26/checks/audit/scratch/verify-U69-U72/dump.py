#!/usr/bin/env python3
"""Read-only: dump assistant text and user text between two line indexes of a transcript.
Usage: dump.py <jsonl> <lo> <hi> [maxchars]
"""
import json, sys
src, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mx = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
with open(src, "r", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx < lo:
            continue
        if idx > hi:
            break
        try:
            o = json.loads(line)
        except Exception:
            continue
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        ts = o.get("timestamp") or ""
        c = (o.get("message") or {}).get("content")
        if isinstance(c, str):
            if t == "user" and not o.get("isMeta"):
                print("=== L%d %s USER\n%s\n" % (idx, ts, c[:mx]))
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text":
                txt = (b.get("text") or "").strip()
                if txt:
                    print("=== L%d %s %s\n%s\n" % (idx, ts, t.upper(), txt[:mx]))
