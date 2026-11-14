#!/usr/bin/env python3
"""Read-only: keyword hits inside tool_result blocks (and everything else) between two line indexes.
Usage: kwres.py <jsonl> <regex> <lo> <hi> [ctx]
"""
import json, re, sys
src, pat, lo, hi = sys.argv[1], re.compile(sys.argv[2], re.I), int(sys.argv[3]), int(sys.argv[4])
ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 300
def walk(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, list):
        for y in x:
            yield from walk(y)
    elif isinstance(x, dict):
        for k, y in x.items():
            if k in ("signature",):
                continue
            yield from walk(y)
with open(src, "r", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx < lo:
            continue
        if idx > hi:
            break
        if not pat.search(line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp") or ""
        t = o.get("type")
        n = 0
        for s in walk(o.get("message") or o):
            for m in pat.finditer(s):
                a, b = max(0, m.start() - ctx), min(len(s), m.end() + ctx)
                print("L%d %s %s side=%s :: ...%s..." % (idx, ts, t, o.get("isSidechain"), s[a:b].replace("\n", " | ")))
                n += 1
                if n >= 6:
                    break
            if n >= 6:
                break
