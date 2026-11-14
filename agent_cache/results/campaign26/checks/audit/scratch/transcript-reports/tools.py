#!/usr/bin/env python3
"""Count tool_use names since the cut; dump AskUserQuestion / Artifact / Workflow / Agent / WebSearch / WebFetch calls briefly."""
import collections
import json
import os

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
OUT = os.path.dirname(os.path.abspath(__file__))
CUT = "2026-11-11T00:00"
WATCH = {"AskUserQuestion", "Artifact", "Workflow", "Agent", "WebSearch", "WebFetch", "TaskStop", "SendMessage"}

cnt = collections.Counter()
ids = {}
fo = open(os.path.join(OUT, "tools_watch.txt"), "w")
with open(SRC, "r", errors="replace") as f:
    for idx, line in enumerate(f):
        if '"tool_use"' not in line and '"tool_result"' not in line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ts = o.get("timestamp") or ""
        if ts < CUT:
            continue
        t = o.get("type")
        c = (o.get("message") or {}).get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if t == "assistant" and b.get("type") == "tool_use":
                n = b.get("name")
                cnt[n] += 1
                if n in WATCH:
                    ids[b.get("id")] = n
                    s = json.dumps(b.get("input"), ensure_ascii=False)
                    fo.write("=== L%d %s TOOL_USE %s\n%s\n\n" % (idx, ts, n, s[:2500] + (" ...[+%d]" % (len(s) - 2500) if len(s) > 2500 else "")))
            elif t == "user" and b.get("type") == "tool_result" and b.get("tool_use_id") in ids:
                cc = b.get("content")
                if isinstance(cc, list):
                    cc = "\n".join((x.get("text") or "") if isinstance(x, dict) and x.get("type") == "text" else "[blk]" for x in cc)
                cc = cc or ""
                fo.write("=== L%d %s TOOL_RESULT for %s\n%s\n\n" % (idx, ts, ids[b.get("tool_use_id")], cc[:1500] + (" ...[+%d]" % (len(cc) - 1500) if len(cc) > 1500 else "")))
fo.close()
print(cnt.most_common())
