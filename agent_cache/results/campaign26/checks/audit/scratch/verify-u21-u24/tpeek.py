#!/usr/bin/env python3
"""tpeek.py LO HI [maxchars [kinds]]
Print the text-like blocks of transcript lines LO..HI (read-only)."""
import json
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
lo, hi = int(sys.argv[1]), int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
kinds = set((sys.argv[4] if len(sys.argv) > 4 else "text,user").split(","))
with open(SRC, encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i < lo:
            continue
        if i > hi:
            break
        try:
            o = json.loads(line)
        except Exception:
            continue
        typ = o.get("type")
        if typ not in ("user", "assistant"):
            continue
        msg = o.get("message", {}) or {}
        c = msg.get("content")
        if isinstance(c, str):
            c = [{"type": "text", "text": c}]
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                k = "user" if typ == "user" else "text"
                t = b.get("text", "")
            elif bt == "thinking":
                k = "thinking"
                t = b.get("thinking", "")
            elif bt == "tool_use":
                k = "tool_use"
                t = json.dumps(b.get("input", {}), ensure_ascii=False)
            elif bt == "tool_result":
                k = "tool_result"
                cc = b.get("content")
                if isinstance(cc, list):
                    cc = "\n".join(
                        x.get("text", "")
                        for x in cc
                        if isinstance(x, dict) and x.get("type") == "text"
                    )
                t = str(cc)
            else:
                continue
            if k not in kinds:
                continue
            side = "sidechain " if o.get("isSidechain") else ""
            print(f"=== [{i}] {side}{k} {o.get('timestamp','')} (len {len(t)}) ===")
            print(t[:mx])
            print()
