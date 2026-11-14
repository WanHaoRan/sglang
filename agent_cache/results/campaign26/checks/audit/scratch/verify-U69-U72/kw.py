#!/usr/bin/env python3
"""Read-only: find keyword hits in assistant text / tool inputs / user text of a transcript.
Usage: kw.py <jsonl> <regex> [min_ts] [ctx]
Prints line index, timestamp, role/kind and a context window around each hit (at most 3 per block).
"""
import json, re, sys
src, pat = sys.argv[1], re.compile(sys.argv[2], re.I)
min_ts = sys.argv[3] if len(sys.argv) > 3 else ""
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 220
def emit(idx, ts, kind, s):
    n = 0
    for m in pat.finditer(s):
        a, b = max(0, m.start() - ctx), min(len(s), m.end() + ctx)
        print("L%d %s %s :: ...%s..." % (idx, ts, kind, s[a:b].replace("\n", " | ")))
        n += 1
        if n >= 3:
            break
with open(src, "r", errors="replace") as f:
    for idx, line in enumerate(f):
        if not pat.search(line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp") or ""
        if ts < min_ts:
            continue
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        if o.get("isSidechain"):
            continue
        c = (o.get("message") or {}).get("content")
        if isinstance(c, str):
            emit(idx, ts, t + ":str", c)
            continue
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                emit(idx, ts, t + ":text", b.get("text") or "")
            elif bt == "tool_use":
                emit(idx, ts, t + ":tool_use:" + str(b.get("name")), json.dumps(b.get("input"), ensure_ascii=False))
            elif bt == "tool_result":
                pass  # skip tool results (file contents) to keep to what was said or written
