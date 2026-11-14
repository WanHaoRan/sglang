#!/usr/bin/env python3
"""Dump a line range of the transcript in readable form (text, thinking flag, tool calls, tool results truncated).

usage: dump_range.py START END [maxchars_per_block]
"""
import json
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
start = int(sys.argv[1])
end = int(sys.argv[2])
mx = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
show_thinking = len(sys.argv) > 4 and sys.argv[4] == "think"


def cut(s, n=mx):
    s = s if isinstance(s, str) else json.dumps(s, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ...[+%d chars]" % (len(s) - n)


with open(SRC, "r", errors="replace") as f:
    for idx, line in enumerate(f):
        if idx < start:
            continue
        if idx > end:
            break
        try:
            o = json.loads(line)
        except Exception:
            print("L%d BADJSON" % idx)
            continue
        t = o.get("type")
        ts = o.get("timestamp") or ""
        if t == "assistant":
            for b in (o.get("message") or {}).get("content") or []:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    print("L%d %s ASSISTANT TEXT: %s\n" % (idx, ts, cut(b.get("text") or "")))
                elif bt == "thinking":
                    if show_thinking:
                        print("L%d %s THINKING: %s\n" % (idx, ts, cut(b.get("thinking") or "")))
                elif bt == "tool_use":
                    print("L%d %s TOOL_USE %s: %s\n" % (idx, ts, b.get("name"), cut(json.dumps(b.get("input"), ensure_ascii=False))))
        elif t == "user":
            c = (o.get("message") or {}).get("content")
            if isinstance(c, str):
                print("L%d %s USER(str, meta=%s): %s\n" % (idx, ts, o.get("isMeta"), cut(c)))
            elif isinstance(c, list):
                for b in c:
                    if not isinstance(b, dict):
                        continue
                    if b.get("type") == "tool_result":
                        cc = b.get("content")
                        if isinstance(cc, list):
                            cc = "\n".join((x.get("text") or "") if isinstance(x, dict) and x.get("type") == "text" else "[%s]" % (x.get("type") if isinstance(x, dict) else "?") for x in cc)
                        print("L%d %s TOOL_RESULT: %s\n" % (idx, ts, cut(cc or "")))
                    elif b.get("type") == "text":
                        print("L%d %s USER(text, meta=%s): %s\n" % (idx, ts, o.get("isMeta"), cut(b.get("text") or "")))
        elif t in ("system",):
            print("L%d %s SYSTEM: %s\n" % (idx, ts, cut(json.dumps({k: o.get(k) for k in ("subtype", "content", "level")}, ensure_ascii=False), 400)))
        elif t == "queue-operation":
            print("L%d %s QUEUE-OP %s: %s\n" % (idx, ts, o.get("operation"), cut(o.get("content") or "", 300)))
