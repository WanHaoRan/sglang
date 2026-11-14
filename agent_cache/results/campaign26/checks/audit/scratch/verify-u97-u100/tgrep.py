#!/usr/bin/env python3
"""Read-only grep over a session transcript: prints line index, timestamp, role/kind and a snippet around each match.
usage: tgrep.py <jsonl> <regex> [min_line] [max_line] [ctx_chars] [roles]
roles: comma list among user,assistant,tool_use,tool_result,thinking (default: user,assistant)
"""
import json, re, sys

src = sys.argv[1]
rx = re.compile(sys.argv[2], re.I | re.S)
lo = int(sys.argv[3]) if len(sys.argv) > 3 else 0
hi = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
ctx = int(sys.argv[5]) if len(sys.argv) > 5 else 300
roles = set((sys.argv[6] if len(sys.argv) > 6 else "user,assistant").split(","))

def pieces(o):
    typ = o.get("type")
    msg = o.get("message") or {}
    c = msg.get("content")
    if isinstance(c, str):
        if typ in roles:
            yield typ, c
        return
    if not isinstance(c, list):
        return
    for b in c:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text" and typ in roles:
            yield typ, b.get("text", "")
        elif bt == "thinking" and "thinking" in roles:
            yield "thinking", b.get("thinking", "")
        elif bt == "tool_use" and "tool_use" in roles:
            yield "tool_use:" + str(b.get("name")), json.dumps(b.get("input"), ensure_ascii=False)
        elif bt == "tool_result" and "tool_result" in roles:
            cc = b.get("content")
            if isinstance(cc, list):
                cc = "\n".join(x.get("text", "") for x in cc if isinstance(x, dict))
            yield "tool_result", str(cc)

with open(src, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        if not rx.search(line):
            # cheap prefilter on the raw line (escaped text still matches plain words)
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp", "")
        for kind, text in pieces(o):
            for m in rx.finditer(text):
                a = max(0, m.start() - ctx)
                b = min(len(text), m.end() + ctx)
                snip = text[a:b].replace("\n", " \\n ")
                print(f"[{i}] {ts} {kind}: ...{snip}...")
                print()
