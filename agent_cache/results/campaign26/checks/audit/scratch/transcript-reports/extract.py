#!/usr/bin/env python3
"""Read-only extraction from the session transcript.

Writes, under the scratch folder:
  user_prompts.txt   every real user prompt (all dates), with line index + timestamp
  reports.txt        assistant text blocks > 300 chars from 2026-11-11T00:00Z on
  reports_short.txt  assistant text blocks <= 300 chars from 2026-11-11 on (for context)
  index.tsv          line_idx, timestamp, kind, length (for all of the above)
"""
import json
import os
import sys

SRC = "/home/wanhr/.claude/projects/-home-wanhr-sglang/7c93b1d5-dc1c-431d-8d3e-efb4ae23a30d.jsonl"
OUT = os.path.dirname(os.path.abspath(__file__))
CUT = "2026-11-11T00:00"

SKIP_PREFIX = ("<system-reminder>", "<task-notification>", "<local-command", "<command-")


def user_texts(obj):
    """Return list of real prompt texts in a user line (may be empty)."""
    if obj.get("isMeta"):
        return []
    msg = obj.get("message") or {}
    c = msg.get("content")
    out = []
    if isinstance(c, str):
        t = c.strip()
        if t and not t.startswith(SKIP_PREFIX):
            out.append(t)
    elif isinstance(c, list):
        for b in c:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "tool_result":
                continue
            if b.get("type") == "text":
                t = (b.get("text") or "").strip()
                if t and not t.startswith(SKIP_PREFIX):
                    out.append(t)
    return out


def main():
    n = 0
    types = {}
    fu = open(os.path.join(OUT, "user_prompts.txt"), "w")
    fr = open(os.path.join(OUT, "reports.txt"), "w")
    fs = open(os.path.join(OUT, "reports_short.txt"), "w")
    fi = open(os.path.join(OUT, "index.tsv"), "w")
    first_ts = None
    last_ts = None
    with open(SRC, "r", errors="replace") as f:
        for idx, line in enumerate(f):
            n += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                fi.write("%d\t\tBADJSON\t%d\n" % (idx, len(line)))
                continue
            t = obj.get("type")
            types[t] = types.get(t, 0) + 1
            ts = obj.get("timestamp") or ""
            if ts:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
            if t == "user":
                side = obj.get("isSidechain")
                for txt in user_texts(obj):
                    fu.write("=== L%d %s side=%s len=%d\n%s\n\n" % (idx, ts, side, len(txt), txt))
                    fi.write("%d\t%s\tUSER\t%d\n" % (idx, ts, len(txt)))
            elif t == "assistant":
                if ts < CUT:
                    continue
                side = obj.get("isSidechain")
                msg = obj.get("message") or {}
                c = msg.get("content")
                if not isinstance(c, list):
                    continue
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "text":
                        txt = (b.get("text") or "").strip()
                        if not txt:
                            continue
                        if len(txt) > 300:
                            fr.write("=== L%d %s side=%s len=%d\n%s\n\n" % (idx, ts, side, len(txt), txt))
                            fi.write("%d\t%s\tREPORT\t%d\n" % (idx, ts, len(txt)))
                        else:
                            fs.write("=== L%d %s side=%s len=%d\n%s\n\n" % (idx, ts, side, len(txt), txt))
                            fi.write("%d\t%s\tSHORT\t%d\n" % (idx, ts, len(txt)))
    fu.close(); fr.close(); fs.close(); fi.close()
    sys.stdout.write("lines=%d types=%s first=%s last=%s\n" % (n, types, first_ts, last_ts))


if __name__ == "__main__":
    main()
